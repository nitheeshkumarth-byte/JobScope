"""Tests for the dashboard's resume-parsing helpers and its per-account state.

The skill extractor is the highest-traffic regex in the project: whatever it
pulls out of the CV becomes the user's search query and match criteria, and a
silent miss here degrades every downstream result. These lock in the shapes it
has to survive — colon lines, pipe-separated table rows, bullets, headings.

The second half covers the accounts layer: profiles are per-user rows in SQLite
now, and the agent bundle is built per (account, session) rather than stored in
one process-wide slot. Getting that wrong is a data leak, so the isolation tests
are as load-bearing as the parsing ones.
"""

import json as _json
import asyncio
import re
import time
import types
from pathlib import Path

import pytest

import agent
import auth
import dashboard as d
import filters


# --------------------------------------------------------------------------
# _technical_skill_rows — reading the CV's explicit skills block
# --------------------------------------------------------------------------

def test_skills_block_after_a_colon_heading():
    cv = "Jane Doe\nData Analyst\n\nTechnical Skills: Python, SQL, Power BI\n\n"
    cv += "Education\nBSc Statistics\n"
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    assert "python" in got and "sql" in got and "power bi" in got


def test_skills_block_as_a_pipe_separated_table_row():
    cv = "Jane Doe\nTechnical Skills|Python|SQL|Power BI|Tableau\nEducation\nBSc\n"
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    for want in ("python", "sql", "power bi", "tableau"):
        assert want in got, (want, got)


def test_skills_block_as_bullet_lines():
    cv = ("Jane Doe\nTECHNICAL SKILLS\n"
          "- Python, SQL\n"
          "- Power BI, Tableau\n"
          "PROJECTS\nSome project\n")
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    assert "python" in got and "power bi" in got
    # The block must STOP at the next section heading.
    assert not any("project" in g for g in got)


def test_generic_skills_heading_is_a_last_resort():
    cv = "Jane Doe\nSkills\nPython, SQL\n"
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    assert "python" in got and "sql" in got


def test_strict_heading_beats_a_generic_one():
    cv = "Skills\nSomething irrelevant\n\nTechnical Skills: Python, SQL\n"
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    assert "python" in got
    assert "irrelevant" not in got


def test_no_skills_block_returns_nothing():
    assert d._technical_skill_rows("Jane Doe\nA professional with experience.\n") == []
    assert d._technical_skill_rows("") == []


def test_skill_filler_phrases_are_stripped():
    cv = "Technical Skills: Proficient in Python, Working knowledge of SQL\n"
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    assert got == ["python", "sql"]


def test_non_skill_words_are_not_skills():
    cv = "Technical Skills: Python, SQL, etc, languages\n"
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    assert "python" in got and "sql" in got
    for junk in ("etc", "languages", "skills"):
        assert junk not in got


def test_a_group_label_glued_to_its_first_skill_is_not_itself_a_skill():
    """The PDF text layer prints "•Programming Languages:Python, Java" - label
    against the first item, colon and no space. Splitting only on commas turned
    that into the single keyword "Programming Languages Python", and the search
    went out for a phrase no job ad contains. Same for "Web Technologies HTML".
    """
    cv = ("Technical Skills\n"
          "•Programming Languages:Python, PHP, JavaScript\n"
          "•Web Technologies:HTML, CSS\n"
          "•AI / ML:RAG, LlamaIndex\n"
          "•Cloud/DevOps:AWS, Docker\n"
          "•Data Analysis & Visualization:Pandas, NumPy\n"
          "•Soft skills:Communication, Team work\n"
          "Experience\nA job\n")
    got = [s.lower() for s in d._technical_skill_rows(cv)]
    assert got == ["python", "php", "javascript", "html", "css",
                   "rag", "llamaindex", "aws", "docker", "pandas", "numpy",
                   "communication", "team work"]
    for phrase in ("programming languages", "web technologies",
                   "programming languages python", "web technologies html",
                   "ai / ml", "cloud/devops", "data analysis", "soft skills"):
        assert phrase not in got


def test_a_colon_inside_a_real_skill_keeps_both_sides():
    """Only something that reads like a section label is dropped. A skill that
    happens to contain a colon must survive intact."""
    got = [s.lower() for s in d._technical_skill_rows(
        "Technical Skills\nNode:JS, C++, SQL: ANSI\n")]
    assert "node" in got and "js" in got
    assert "c++" in got


# --------------------------------------------------------------------------
# _merge_skills
# --------------------------------------------------------------------------

def test_merge_prefers_explicit_skills_and_dedupes():
    out = d._merge_skills(["Python", "SQL"], ["python", "Tableau"])
    assert out[0] == "Python" and out[1] == "SQL"
    assert out.count("Python") + out.count("python") == 1
    assert "Tableau" in out


def test_merge_drops_a_skill_already_covered_by_an_earlier_one():
    """'Excel' is redundant once 'Advanced Excel' is in the explicit list."""
    out = d._merge_skills(["Advanced Excel"], ["excel", "sql"])
    assert out == ["Advanced Excel", "sql"]


def test_merge_is_capped():
    out = d._merge_skills([f"skill{i}" for i in range(100)])
    assert len(out) <= 60


# --------------------------------------------------------------------------
# _lexicon_skills / extract_skills (strict mode, no LLM)
# --------------------------------------------------------------------------

def test_lexicon_skills_only_returns_literal_mentions():
    got = d._lexicon_skills("I use Python and SQL every day.")
    assert "python" in got and "sql" in got
    assert "tableau" not in got          # not in the text -> not claimed


# --------------------------------------------------------------------------
# Role suggestion + location inference
# --------------------------------------------------------------------------

def test_suggest_roles_matches_skills_to_a_title():
    roles = d.suggest_roles(["sql", "python", "pandas", "power bi"])
    assert roles
    assert roles[0]["role"] == "Data Analyst"
    assert roles[0]["score"] > 0


def test_suggest_roles_with_no_known_skills():
    assert d.suggest_roles(["zzz", "qqq"]) == []


def test_infer_location_finds_an_indian_city():
    out = d.infer_location("Jane Doe\nBangalore, India\nPython developer")
    assert out["city"] == "Bangalore"
    assert out["country"] == "India"


def test_infer_location_on_an_unknown_cv():
    out = d.infer_location("Jane Doe\nSomewhere else entirely")
    assert out["city"] == ""
    assert "not found" in out["location"] or out["location"] == "global"


def test_infer_location_detects_remote():
    assert d.infer_location("Fully remote, work from home")["remote_only"]


# --------------------------------------------------------------------------
# detect_links
# --------------------------------------------------------------------------

def test_detect_links_pulls_github_and_linkedin():
    out = d.detect_links("Jane Doe\nhttps://github.com/janedoe\n"
                         "https://www.linkedin.com/in/janedoe/")
    assert out["github"].endswith("github.com/janedoe")
    assert out["linkedin"].endswith("linkedin.com/in/janedoe")


def test_detect_links_skips_social_and_asset_urls():
    out = d.detect_links("mail me x@y.com or see https://twitter.com/jane "
                         "or https://example.com/photo.png")
    assert "portfolio" not in out


def test_detect_links_handles_no_links():
    assert d.detect_links("Jane Doe\nPython developer") == {}
    assert d.detect_links("") == {}


# The real CVs are LaTeX-built, so the text layer hands over a two-column
# contact table: no scheme on the URLs, and pipe characters around them. This is
# the shape that made GitHub detection fail on every uploaded PDF.

def test_detect_links_finds_a_scheme_less_github_in_a_table_row():
    line = ("www.linkedin.com/in/jane-doe-123|github.com/janedoe|")
    out = d.detect_links("Jane Doe\n" + line)
    assert out["github"] == "https://github.com/janedoe"
    assert out["linkedin"] == "https://www.linkedin.com/in/jane-doe-123"


def test_detect_links_adds_the_scheme_it_finds_missing():
    out = d.detect_links("github.com/janedoe linkedin.com/in/janedoe")
    assert out["github"].startswith("https://")
    assert out["linkedin"].startswith("https://")


def test_detect_links_rejoins_a_url_split_across_lines():
    out = d.detect_links("Jane Doe\nhttps://github.com/\njanedoe\nSkills: Python")
    assert out["github"] == "https://github.com/janedoe"


def test_detect_links_keeps_a_split_url_out_of_the_portfolio_slot():
    out = d.detect_links("https://github.com/\njanedoe")
    assert "portfolio" not in out


def test_detect_links_ignores_tech_names_that_look_like_domains():
    """A protocol-less match would happily read "Node.js" as a website, and a
    bogus URL printed on the resume is worse than none."""
    out = d.detect_links("Backend: Node.js, Express.js, Nest.js, GraphQL\n"
                         "Email: jane.doe@company.th\n"
                         "Angular 19, Tailwind CSS")
    assert out == {}


def test_detect_links_does_not_read_the_domain_out_of_an_email_address():
    out = d.detect_links("+91 7997457091|jane.doe@company.th|Quiddity Ltd")
    assert "portfolio" not in out


def test_detect_links_prefers_the_first_github_it_sees():
    out = d.detect_links("github.com/first\ngithub.com/second")
    assert out["github"].endswith("/first")


def test_detect_links_ignores_a_url_glued_to_an_email_address():
    out = d.detect_links("mail http://weird@site.example/x or "
                         "https://realportfolio.dev/")
    assert out.get("portfolio") == "https://realportfolio.dev/"


def test_detect_links_strips_trailing_punctuation():
    out = d.detect_links("visit https://janedoe.dev/.")
    assert out["portfolio"] == "https://janedoe.dev/"


# --------------------------------------------------------------------------
# PDF link annotations: where a clickable CV link actually lives
# --------------------------------------------------------------------------

class _Action:
    """A PDF link action. The URI hangs off the /URI key, as pypdf exposes it."""

    def __init__(self, uri):
        self._uri = uri

    def get(self, key):
        return {"/URI": self._uri}.get(key)


class _Annot:
    """Stands in for a page's link annotation. This is why plain text extraction
    misses the URL: it lives in the /A action, not in the drawn text."""

    def __init__(self, uri=None, broken=False):
        self._uri = uri
        self._broken = broken

    def get_object(self):
        if self._broken:
            raise ValueError("malformed annotation")
        return {"/A": _Action(self._uri)}


class _Page:
    def __init__(self, text, annots=()):
        self._text = text
        self._annots = list(annots)

    def extract_text(self):
        return self._text

    def get(self, key):
        return self._annots if key == "/Annots" else None


@pytest.fixture
def pdf_upload(monkeypatch):
    """Run _extract_text against a fake pypdf whose pages we control."""
    def _run(pages):
        import sys
        import types

        fake = types.ModuleType("pypdf")
        fake.PdfReader = lambda _bio: types.SimpleNamespace(pages=pages)
        monkeypatch.setitem(sys.modules, "pypdf", fake)
        return d._extract_text(b"%PDF-1.4 fake", "cv.pdf")
    return _run


def test_pdf_link_annotations_are_appended_to_the_text(pdf_upload):
    """A CV whose visible text only says "LinkedIn" still has to yield a URL."""
    page = _Page("Jane Doe\nPortfolio\nLinkedIn\n", [
        _Annot("https://github.com/janedoe"),
        _Annot("https://www.linkedin.com/in/janedoe"),
    ])
    text = pdf_upload([page])

    assert "Jane Doe" in text                 # the body is not disturbed
    assert d.detect_links(text)["github"] == "https://github.com/janedoe"
    assert d.detect_links(text)["linkedin"].endswith("in/janedoe")


def test_pdf_with_no_link_annotations_is_unchanged(pdf_upload):
    assert pdf_upload([_Page("Jane Doe\nPython developer")]) == \
        "Jane Doe\nPython developer"


def test_a_malformed_annotation_does_not_break_extraction(pdf_upload):
    page = _Page("Jane Doe", [_Annot(broken=True), _Annot("https://x.dev/")])
    assert "Jane Doe" in pdf_upload([page])


def test_a_page_without_annotations_key_is_tolerated(pdf_upload):
    class _Bare:
        def extract_text(self):
            return "Jane Doe"

        def get(self, key):
            raise KeyError("/Annots")

    assert "Jane Doe" in pdf_upload([_Bare()])


# --------------------------------------------------------------------------
# _parse_skill_list (LLM-output parser)
# --------------------------------------------------------------------------

def test_parse_skill_list_strips_bullets_and_dedupes():
    out = [s.lower() for s in d._parse_skill_list("- Python\nSQL\npython, Tableau")]
    assert "python" in out and "sql" in out and "tableau" in out
    assert out.count("python") == 1


def test_parse_skill_list_rejects_junk():
    assert d._parse_skill_list("") == []
    assert d._parse_skill_list("2024") == []


# --------------------------------------------------------------------------
# _extract_text
# --------------------------------------------------------------------------

def test_extract_text_reads_plain_and_utf16():
    assert d._extract_text(b"hello world", "cv.txt") == "hello world"
    assert "skills" in d._extract_text("Technical Skills: Python".encode("utf-16"),
                                       "cv.md").lower()


def test_extract_text_never_raises_on_binary():
    assert isinstance(d._extract_text(bytes(range(256)), "cv.pdf"), str)


# --------------------------------------------------------------------------
# _recent_errors — now spans several days, not just today
# --------------------------------------------------------------------------

def test_recent_errors_spans_multiple_days(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(d, "LOG_DIR", str(log_dir))

    today = time.strftime("%Y%m%d", time.localtime())
    yesterday = time.strftime("%Y%m%d", time.localtime(time.time() - 86400))
    (log_dir / f"errors-{today}.jsonl").write_text(
        _json.dumps({"event": "new"}) + "\n", encoding="utf-8")
    (log_dir / f"errors-{yesterday}.jsonl").write_text(
        _json.dumps({"event": "old"}) + "\n", encoding="utf-8")

    got = d._recent_errors("errors", n=10)
    events = {r["event"] for r in got}
    assert events == {"new", "old"}, got
    # newest first
    assert got[0]["event"] == "new"


def test_recent_errors_respects_the_limit(tmp_path, monkeypatch):
    import json as _json
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(d, "LOG_DIR", str(log_dir))
    today = time.strftime("%Y%m%d")
    (log_dir / f"errors-{today}.jsonl").write_text(
        "\n".join(_json.dumps({"event": f"e{i}"}) for i in range(50)),
        encoding="utf-8")
    assert len(d._recent_errors("errors", n=5)) == 5


def test_recent_errors_with_no_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "LOG_DIR", str(tmp_path / "nope"))
    assert d._recent_errors("errors", n=5) == []


# --------------------------------------------------------------------------
# Per-account profiles: the store is keyed by owner, not a shared dict
# --------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point both modules at a throwaway SQLite file with the schema created."""
    path = str(tmp_path / "users.db")
    monkeypatch.setattr(auth, "DB_FILE", path)
    monkeypatch.setattr(d.auth, "DB_FILE", path)
    auth.init_db()
    return path


@pytest.fixture
def owner(temp_db):
    return auth.create_user("owner@example.com", "hunter2hunter2", owner=True)


def test_a_new_account_starts_with_one_empty_default_session(owner):
    profiles = auth.list_profiles(owner["id"])
    assert [p["id"] for p in profiles] == ["default"]
    assert profiles[0]["skills"] == []
    assert profiles[0]["work_modes"] == []


def test_a_second_account_cannot_see_the_first_ones_sessions(owner):
    other = auth.create_user("other@example.com", "hunter2hunter2")
    auth.save_profile(owner["id"], "p1", {"name": "Owner CV", "skills": ["SQL"]})
    # the other account only ever sees its own default
    assert [p["id"] for p in auth.list_profiles(other["id"])] == ["default"]
    # and cannot read the owner's record by guessing the id
    assert auth.get_profile(other["id"], "p1") is None
    assert auth.get_profile(owner["id"], "p1")["skills"] == ["SQL"]


def test_profiles_round_trip_the_filter_fields(owner):
    d._patch_profile(owner["id"], "default", work_modes=["hybrid"],
                     regions=["latam"], skills=["SQL", "Power BI"])
    rec = auth.get_profile(owner["id"], "default")
    assert rec["work_modes"] == ["hybrid"]
    assert rec["regions"] == ["latam"]
    assert rec["skills"] == ["SQL", "Power BI"]


def test_patch_profile_leaves_unmentioned_fields_alone(owner):
    d._patch_profile(owner["id"], "default", skills=["SQL"])
    d._patch_profile(owner["id"], "default", regions=["europe"])
    rec = auth.get_profile(owner["id"], "default")
    assert rec["skills"] == ["SQL"]          # not clobbered by the second patch
    assert rec["regions"] == ["europe"]


def test_the_default_session_cannot_be_deleted_but_others_can(owner):
    pid = auth.create_profile(owner["id"], "Scratch")
    assert auth.delete_profile(owner["id"], "default") is False
    assert auth.delete_profile(owner["id"], pid) is True
    assert auth.get_profile(owner["id"], pid) is None


def test_an_unknown_profile_id_falls_back_to_the_accounts_own_default(owner):
    """The old code rewrote unknown ids to a SHARED 'default', which handed one
    person's session to another. It must now stay inside this account."""
    d._patch_profile(owner["id"], "default", roles=["Data Analyst"])
    rec = d._profile_for(owner["id"], "does-not-exist")
    assert rec["id"] == "default"
    assert rec["roles"] == ["Data Analyst"]


# --------------------------------------------------------------------------
# _cfg_from_profile — one account's filters never bleed into another's
# --------------------------------------------------------------------------

def test_cfg_from_profile_restores_the_saved_filters(owner):
    from agent import AgentConfig
    d._patch_profile(owner["id"], "default",
                     roles=["Data Analyst"], cities=["Bengaluru", "Pune"],
                     skills=["SQL"], work_modes=["onsite", "hybrid"],
                     regions=["europe"], remote_only=True)
    cfg = d._cfg_from_profile(AgentConfig(),
                              auth.get_profile(owner["id"], "default"))
    # canonical chip order, not the order they happened to be saved in
    assert cfg.work_modes == ["hybrid", "onsite"]
    assert cfg.regions == ["europe"]
    assert cfg.preferred_cities == ["Bengaluru", "Pune"]
    # an on-site hunt must search a city, not the literal "Remote"
    assert cfg.indeed_location == "Bengaluru"


def test_cfg_from_profile_tolerates_profiles_saved_before_the_filters(owner):
    from agent import AgentConfig
    auth.save_profile(owner["id"], "default",
                      {"name": "Old", "roles": ["Data Analyst"],
                       "cities": ["Pune"], "remote_only": False})
    cfg = d._cfg_from_profile(AgentConfig(),
                              auth.get_profile(owner["id"], "default"))
    assert cfg.work_modes == [] and cfg.regions == []
    assert cfg.indeed_location == "Pune"        # legacy remote_only=False rule


def test_a_stale_remote_only_in_a_saved_profile_cannot_come_back(owner):
    """The outside-India-only rule is gone, but profiles written before the
    removal still carry remote_only=true. Reading it back is what kept the
    behaviour alive: opening one of those sessions silently started dropping
    every India-tied job again. The flag must be inert on load, not merely
    defaulted off for new profiles."""
    from agent import AgentConfig
    d._patch_profile(owner["id"], "default", roles=["Data Analyst"],
                     cities=[], work_modes=[], remote_only=True)
    snap = auth.get_profile(owner["id"], "default")
    assert snap["remote_only"] is True, "the file itself is left alone"

    cfg = d._cfg_from_profile(AgentConfig(), snap)
    assert cfg.remote_only is False
    # The flag was also what re-narrowed a location search, so the board
    # query and the India filter have to come back clean too.
    assert filters.india_rule_applies(cfg.remote_only, cfg.work_modes) is False
    assert filters.boards_location(cfg.work_modes, cfg.preferred_cities,
                                   cfg.remote_only, cfg.countries) == "Remote"
    from agent import board_tool_args
    assert board_tool_args(cfg)["remote_only"] is False


@pytest.mark.anyio
async def test_each_accounts_bundle_carries_its_own_seen_scope(owner):
    """The scraper's "already shown" memory is shared per process, so it has
    to be keyed by account or the first person to see a posting makes it a
    repeat for everyone else. The bundle is cached per (account, profile), so
    the scope has to be stamped where that cache key is built."""
    from types import SimpleNamespace

    async def fake_create(cfg, runtime=None):
        async def _never():  # pragma: no cover - the run is not exercised here
            yield {}
        return SimpleNamespace(cfg=cfg, astream=_never)
    d.app.state.runtime = object()
    original, d.create_agent = d.create_agent, fake_create
    d._bundle_cache().clear()
    try:
        b1 = await d._bundle_for(owner["id"], "default", {"roles": []})
        assert b1.cfg.seen_scope == f"u{owner['id']}"
        # The cache must not re-derive a different scope for the same account.
        b2 = await d._bundle_for(owner["id"], "default", {"roles": []})
        assert b2 is b1
    finally:
        d.create_agent = original
        d._bundle_cache().clear()




def test_cfg_from_profile_keeps_the_perf_picks_per_account(owner):
    """model/num_ctx used to live only in a process-wide bundle, so one person
    changing them changed it for everyone. They belong to the profile now."""
    from agent import AgentConfig
    d._patch_profile(owner["id"], "default", model="llama3.2", num_ctx=4096,
                     days_back=30, max_results=25)
    cfg = d._cfg_from_profile(AgentConfig(),
                              auth.get_profile(owner["id"], "default"))
    assert (cfg.model, cfg.num_ctx) == ("llama3.2", 4096)
    assert (cfg.days_back, cfg.max_results) == (30, 25)


def test_cfg_from_profile_does_not_inherit_another_profiles_search(owner):
    from agent import AgentConfig
    d._patch_profile(owner["id"], "p1", roles=["Data Analyst"],
                     cities=["Bengaluru"], work_modes=["onsite"])
    fresh = d._cfg_from_profile(AgentConfig(), {"roles": [], "cities": []})
    assert fresh.preferred_cities == []
    assert fresh.work_modes == []


# --------------------------------------------------------------------------
# The agent bundle is per (account, session), not one global slot
# --------------------------------------------------------------------------

class _FakeBundle:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tools = []


@pytest.fixture
def stub_agent(monkeypatch):
    """create_agent compiles a graph per config. Record what it was handed and
    hand back a cheap stand-in so the tests never boot Ollama."""
    seen = []

    async def _fake_create_agent(cfg=None, runtime=None):
        seen.append(cfg)
        return _FakeBundle(cfg)

    monkeypatch.setattr(d, "create_agent", _fake_create_agent)
    # _bundle_for reads the shared MCP runtime off app.state; it is never used
    # by the stub, it just has to exist.
    monkeypatch.setattr(d.app.state, "runtime", object(), raising=False)
    monkeypatch.setattr(d.app.state, "bundles", {}, raising=False)
    d.app.state.bundles = {}
    return seen


def test_bundle_for_builds_and_then_reuses_one_graph_per_account_session(
        temp_db, owner, stub_agent):
    d._patch_profile(owner["id"], "default", roles=["Data Analyst"])
    snap = auth.get_profile(owner["id"], "default")

    async def _go():
        first = await d._bundle_for(owner["id"], "default", snap)
        second = await d._bundle_for(owner["id"], "default", snap)
        return first, second

    first, second = asyncio.run(_go())
    assert first is second                  # cached, not recompiled
    assert len(stub_agent) == 1


def test_two_accounts_never_share_a_graph(temp_db, owner, stub_agent):
    """The regression this whole refactor exists for: one process-wide bundle
    meant two signed-in users shared a search context."""
    other = auth.create_user("other@example.com", "hunter2hunter2")
    d._patch_profile(owner["id"], "default", roles=["Data Analyst"])
    d._patch_profile(other["id"], "default", roles=["Backend Engineer"])

    async def _go():
        a = await d._bundle_for(owner["id"], "default",
                                auth.get_profile(owner["id"], "default"))
        b = await d._bundle_for(other["id"], "default",
                                auth.get_profile(other["id"], "default"))
        return a, b

    a, b = asyncio.run(_go())
    assert a is not b
    assert a.cfg.target_role == "Data Analyst"
    assert b.cfg.target_role == "Backend Engineer"


def test_dropping_a_bundle_forces_a_rebuild(temp_db, owner, stub_agent):
    d._patch_profile(owner["id"], "default", roles=["Data Analyst"])
    snap = auth.get_profile(owner["id"], "default")

    async def _go():
        await d._bundle_for(owner["id"], "default", snap)
        d._drop_bundle(owner["id"], "default")
        return await d._bundle_for(owner["id"], "default", snap)

    asyncio.run(_go())
    assert len(stub_agent) == 2


# --------------------------------------------------------------------------
# Run history is per account
# --------------------------------------------------------------------------

def test_run_history_is_kept_per_account():
    d.RUNS.clear()
    d._remember_run({"message": "a"}, 1)
    d._remember_run({"message": "b"}, 1)
    d._remember_run({"message": "c"}, 2)
    assert [r["message"] for r in d._user_runs_all(1)] == ["b", "a"]
    assert [r["message"] for r in d._user_runs_all(2)] == ["c"]


def test_run_history_is_capped_per_account():
    d.RUNS.clear()
    for i in range(d.MAX_RUNS_KEPT + 5):
        d._remember_run({"message": str(i)}, 1)
    assert len(d._user_runs_all(1)) == d.MAX_RUNS_KEPT


# --------------------------------------------------------------------------
# The old profiles.json is adopted by the first account
# --------------------------------------------------------------------------

def test_the_legacy_profiles_file_is_adopted_by_the_first_account(tmp_path,
                                                                monkeypatch):
    legacy = tmp_path / "profiles.json"
    legacy.write_text(_json.dumps({
        "default": {"id": "default", "name": "Samsudheen", "roles": ["AI Engineer"],
                    "cities": ["Hyderabad"], "skills": ["SQL"],
                    "work_modes": ["remote"], "regions": ["europe"],
                    "remote_only": True, "resume_text": "CV text"},
        "a942592f99": {"id": "a942592f99", "name": "Second CV",
                       "roles": ["Data Analyst"], "skills": []},
    }), encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))
    path = str(tmp_path / "users.db")
    monkeypatch.setattr(auth, "DB_FILE", path)
    auth.init_db()

    user = auth.create_user("first@example.com", "hunter2hunter2", owner=True)
    moved = d._migrate_legacy_profiles(user["id"])
    assert moved == 2
    mine = auth.list_profiles(user["id"])
    # The account opens on the blank template; the old CVs are kept beside it.
    assert [p["id"] for p in mine][0] == "default"
    assert d._is_pristine(auth.get_profile(user["id"], "default"))

    imported = [p for p in mine if p.get("migrated_from") == "default"]
    assert len(imported) == 1
    assert imported[0]["resume_text"] == "CV text"
    assert imported[0]["name"] == "Samsudheen"
    # Imported CVs are references, not the session to hunt in.
    assert auth.get_profile(user["id"], imported[0]["id"])["id"] != "default"
    assert auth.get_profile(user["id"], "a942592f99")["name"] == "Second CV"


def test_the_blank_default_survives_migration_and_can_be_used(tmp_path,
                                                              monkeypatch):
    """A new account must land on an empty session, not on an imported CV."""
    legacy = tmp_path / "profiles.json"
    legacy.write_text(_json.dumps({
        "default": {"id": "default", "name": "Old CV", "roles": ["AI Engineer"],
                    "cities": ["Hyderabad"], "skills": ["SQL", "Python"],
                    "resume_text": "a lot of cv text"},
    }), encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()

    user = auth.create_user("first@example.com", "hunter2hunter2", owner=True)
    d._migrate_legacy_profiles(user["id"])

    blank = d._profile_for(user["id"], "default")
    for field in ("resume_text", "roles", "skills", "cities", "work_modes",
                  "regions", "github_url"):
        assert not blank.get(field), f"{field} should start empty"

    # The imported CV is still reachable, one click away, with its own data.
    imported = [p for p in auth.list_profiles(user["id"])
                if p.get("migrated_from") == "default"][0]
    kept = d._profile_for(user["id"], imported["id"])
    assert kept["resume_text"] == "a lot of cv text"
    assert kept["roles"] == ["AI Engineer"]
    assert kept["github_url"] is None or kept["github_url"] == ""


def test_migration_is_idempotent_for_the_re_id_default(tmp_path, monkeypatch):
    """Re-running the import must not pile up duplicate copies of the CV."""
    legacy = tmp_path / "profiles.json"
    legacy.write_text(_json.dumps(
        {"default": {"id": "default", "name": "Old CV", "skills": ["SQL"],
                     "resume_text": "cv text"}}), encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()

    user = auth.create_user("first@example.com", "hunter2hunter2", owner=True)
    assert d._migrate_legacy_profiles(user["id"]) == 1
    assert d._migrate_legacy_profiles(user["id"]) == 0
    assert d._migrate_legacy_profiles(user["id"]) == 0
    assert len(auth.list_profiles(user["id"])) == 2      # blank + one import


def test_empty_legacy_records_are_not_imported(tmp_path, monkeypatch):
    """A legacy entry with nothing in it would only add sidebar noise."""
    legacy = tmp_path / "profiles.json"
    legacy.write_text(_json.dumps({
        "default": {"id": "default", "name": "Empty", "skills": [], "roles": [],
                    "cities": [], "resume_text": ""},
        "abc123": {"id": "abc123", "name": "Real", "skills": ["SQL"],
                   "resume_text": "text"},
    }), encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()

    user = auth.create_user("first@example.com", "hunter2hunter2", owner=True)
    assert d._migrate_legacy_profiles(user["id"]) == 1
    assert auth.get_profile(user["id"], "abc123")["name"] == "Real"


def test_migration_never_overwrites_an_already_migrated_profile(tmp_path,
                                                               monkeypatch):
    """Re-running the migration must not roll a session back to its old copy."""
    legacy = tmp_path / "profiles.json"
    legacy.write_text(_json.dumps(
        {"default": {"id": "default", "name": "Old", "skills": ["SQL"]}}),
        encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))
    path = str(tmp_path / "users.db")
    monkeypatch.setattr(auth, "DB_FILE", path)
    auth.init_db()

    user = auth.create_user("first@example.com", "hunter2hunter2", owner=True)
    d._migrate_legacy_profiles(user["id"])
    # Fill in the imported copy, not the blank default.
    imported = [p for p in auth.list_profiles(user["id"])
                if p.get("migrated_from") == "default"][0]
    d._patch_profile(user["id"], imported["id"], skills=["Python"])
    assert d._migrate_legacy_profiles(user["id"]) == 0
    assert auth.get_profile(user["id"], imported["id"])["skills"] == ["Python"]


def test_migration_survives_a_corrupt_legacy_file(tmp_path, monkeypatch):
    legacy = tmp_path / "profiles.json"
    legacy.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))
    path = str(tmp_path / "users.db")
    monkeypatch.setattr(auth, "DB_FILE", path)
    auth.init_db()
    user = auth.create_user("first@example.com", "hunter2hunter2", owner=True)
    assert d._migrate_legacy_profiles(user["id"]) == 0


# --------------------------------------------------------------------------
# An account whose `default` was already polluted by the old build
# --------------------------------------------------------------------------

def test_a_polluted_default_is_moved_aside_not_deleted(tmp_path, monkeypatch):
    """Accounts created before the blank-default fix adopted somebody's CV AS
    `default`, so login opened a session that was already full of skills. The
    contents must be preserved in the sidebar and `default` reset."""
    monkeypatch.setattr(d, "PROFILES_FILE", str(tmp_path / "nope.json"))
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    user = auth.create_user("first@example.com", "hunter2hunter2", owner=True)
    uid = user["id"]

    auth.save_profile(uid, "default", {
        "id": "default", "name": "Samsudheen I",
        "skills": ["python", "sql"], "roles": ["Data Analyst"],
        "cities": ["Hyderabad"], "resume_text": "cv text",
        "github_url": "https://github.com/someone"})

    moved = d._rehome_polluted_default(uid)
    assert moved, "the old contents should have been re-homed"

    default = auth.get_profile(uid, "default")
    for field in ("resume_text", "skills", "roles", "cities", "github_url"):
        assert not default.get(field), f"{field} should be blank on login"
    assert default["remote_only"] is False

    kept = auth.get_profile(uid, moved)
    assert kept["skills"] == ["python", "sql"]
    assert kept["roles"] == ["Data Analyst"]
    assert kept["migrated_from"] == "default"


def test_rehoming_the_default_runs_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "PROFILES_FILE", str(tmp_path / "nope.json"))
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    uid = auth.create_user("first@example.com", "hunter2hunter2", owner=True)["id"]
    auth.save_profile(uid, "default", {"id": "default", "skills": ["sql"]})

    assert d._rehome_polluted_default(uid)
    assert d._rehome_polluted_default(uid) == ""
    assert len(auth.list_profiles(uid)) == 2      # blank + the one import


def test_a_session_the_user_filled_in_themselves_is_left_alone(tmp_path, monkeypatch):
    """Once the re-home has happened once, whatever the user puts in their own
    default session is theirs and must survive the next login."""
    monkeypatch.setattr(d, "PROFILES_FILE", str(tmp_path / "nope.json"))
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    uid = auth.create_user("first@example.com", "hunter2hunter2", owner=True)["id"]
    auth.save_profile(uid, "default", {"id": "default", "skills": ["sql"]})
    d._rehome_polluted_default(uid)

    d._patch_profile(uid, "default", skills=["rust"], roles=["Systems Engineer"])
    assert d._rehome_polluted_default(uid) == ""
    assert auth.get_profile(uid, "default")["skills"] == ["rust"]


def test_a_fresh_account_default_is_already_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    uid = auth.create_user("first@example.com", "hunter2hunter2", owner=True)["id"]
    rec = auth.get_profile(uid, "default")
    assert rec["skills"] == [] and rec["roles"] == [] and rec["cities"] == []
    assert rec["regions"] == [] and rec["remote_only"] is False
    assert d._rehome_polluted_default(uid) == ""


# --------------------------------------------------------------------------
# The run must always say something
# --------------------------------------------------------------------------

class _FakeGraph:
    """A stand-in graph that reproduces the two ways a reply can arrive:
    a real LLM (streamed chunks) and the template summary (one whole message)."""

    def __init__(self, events):
        self.events = events
        self.cfg = agent.AgentConfig()
        self.tools = []
        self.app = self

    def astream(self, payload, stream_mode=None, config=None):
        events = self.events

        async def _gen():
            for mode, data in events:
                yield mode, data
        return _gen()


def _frames(events, monkeypatch):
    monkeypatch.setattr(d, "_log", lambda *a, **k: None)
    graph = _FakeGraph(events)
    out = []

    async def _drive():
        async for frame in d._run_stream("find me jobs", "run1", time.time(),
                                         graph, 1, "default", "a@b.com"):
            out.append(frame)
    asyncio.run(_drive())
    return [_json.loads(f[len("data: "):]) for f in out]


def test_the_template_summary_reaches_the_browser(monkeypatch):
    """The bug: SUMMARY_MODE=template makes ZERO LLM calls, so the reply is a
    plain AIMessage that only ever shows up in the "updates" stream. The
    dashboard read tool_calls from that node and ignored its content, so every
    run logged token_chars: 0 and the user got an empty bubble."""
    from langchain_core.messages import AIMessage
    reply = "I found 3 roles. Here they are:\n- Data Analyst at Acme"
    events = [("updates", {"agent": {"messages": [
        AIMessage(content="", tool_calls=[{"name": "search_job_boards",
                                           "args": {}, "id": "c0"}])]}}),
        ("updates", {"agent": {"messages": [AIMessage(content=reply)]}})]
    frames = _frames(events, monkeypatch)
    text = "".join(f.get("text", "") for f in frames if f.get("type") == "token")
    assert reply in text


def test_llm_chunks_are_not_sent_twice(monkeypatch):
    """In llm mode the text arrives as chunks AND as the node's final message.
    Emitting both would double every reply."""
    from langchain_core.messages import AIMessage, AIMessageChunk
    events = [("messages", (AIMessageChunk(content="Hello ", id="m1"), {})),
              ("messages", (AIMessageChunk(content="there", id="m1"), {})),
              ("updates", {"agent": {"messages": [
                  AIMessage(content="Hello there", id="m1")]}})]
    frames = _frames(events, monkeypatch)
    text = "".join(f.get("text", "") for f in frames if f.get("type") == "token")
    assert text == "Hello there"


def test_a_run_that_produced_nothing_still_says_something(monkeypatch):
    """No jobs and no summary must never render as an empty bubble again."""
    frames = _frames([("updates", {"agent": {"messages": []}})], monkeypatch)
    text = "".join(f.get("text", "") for f in frames if f.get("type") == "token")
    assert "could not build a summary" in text
    assert any(f.get("type") == "status" and f.get("stage") == "done" for f in frames)



# --------------------------------------------------------------------------
# Startup: no global bundle any more
# --------------------------------------------------------------------------

def test_lifespan_opens_the_database_and_builds_no_global_bundle(monkeypatch):
    async def _fake_create_runtime():
        return object()

    monkeypatch.setattr(d, "create_runtime", _fake_create_runtime)
    monkeypatch.setattr(d, "auth", types.SimpleNamespace(
        init_db=lambda: None, DB_FILE=":memory:"))

    async def _drive():
        fake_app = types.SimpleNamespace(state=types.SimpleNamespace())
        gen = d.lifespan.__wrapped__(fake_app)
        await gen.__anext__()
        st = fake_app.state
        captured = (hasattr(st, "bundle"), dict(st.bundles))
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
        return captured

    has_global_bundle, cache = asyncio.run(_drive())
    assert has_global_bundle is False        # one shared bundle would leak
    assert cache == {}


def test_config_request_accepts_and_defaults_the_new_fields():
    plain = d.ConfigRequest(model="llama3.1")
    assert plain.work_modes is None and plain.regions is None
    assert plain.countries is None
    picked = d.ConfigRequest(model="llama3.1", work_modes=["onsite"],
                             regions=["europe"], countries=["germany"])
    assert picked.work_modes == ["onsite"] and picked.regions == ["europe"]
    assert picked.countries == ["germany"]


def test_a_profile_round_trips_its_countries():
    saved = {"roles": ["Junior Data Analyst"], "cities": [], "skills": ["sql"],
             "work_modes": ["onsite"], "regions": ["europe"],
             "countries": ["Germany", "Japan"], "remote_only": False}
    cfg = d._cfg_from_profile(agent.AgentConfig(), saved)
    assert cfg.countries == ["Germany", "Japan"]
    # a country only becomes the boards' location when there is no city
    assert cfg.indeed_location == "Germany"


def test_a_profile_without_countries_filters_on_nothing():
    # a session saved before this feature must behave exactly as it did
    cfg = d._cfg_from_profile(agent.AgentConfig(),
                              {"roles": ["Junior Data Analyst"],
                               "cities": ["Pune"], "remote_only": False})
    assert cfg.countries == []
    assert cfg.indeed_location == "Pune"


def test_a_profile_drops_countries_it_does_not_recognise():
    cfg = d._cfg_from_profile(agent.AgentConfig(),
                              {"roles": ["Junior Data Analyst"],
                               "countries": ["Atlantis", "uk"]})
    assert cfg.countries == ["United Kingdom"]


def test_the_settings_payload_offers_a_searchable_country_list():
    # the picker searches this list, so every entry needs a label and the region
    # it belongs to, and the keys must be the canonical names filters uses
    opts = [{"key": k, "label": k, "region": v[0]}
            for k, v in filters.COUNTRIES.items()]
    assert len(opts) > 50
    assert {o["key"] for o in opts} == set(filters.COUNTRIES)
    assert all(o["region"] in filters.REGIONS for o in opts)


def test_a_detected_country_is_offered_but_not_applied():
    # Seeding the chip would make a brand-new session's first search come back
    # empty, because a country chip is an active filter.
    seed = d.infer_location("Berlin, Germany. 5 years Python.")
    assert seed["country"] == "Germany"
    # a CV naming two countries says nothing about where the author lives
    both = d.infer_location("Worked in France, based in the United Kingdom.")
    assert both["country"] == ""


def test_the_drawer_has_a_clear_all_for_every_filter_group():
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    for box in ("roleChipsEdit", "cityChipsEdit", "workModeChips",
                "countryChips", "skillChipsEdit"):
        assert f'data-clear="{box}"' in html, f"{box} has no Clear all"


def test_the_region_filter_and_legacy_remote_toggle_are_gone():
    """Both were retired: the region chips narrowed a search the user could not
    reason about, and the remote-outside-India checkbox defaulted to ON."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert "Legacy: remote jobs" not in html
    assert "outside India" not in html
    assert 'id="cfgRemote"' not in html
    assert 'id="regionChips"' not in html
    # the country picker is the one geography control that remains
    assert 'id="countryChips"' in html


def test_the_role_list_is_a_dropdown_that_needs_an_explicit_add():
    """The list suggests roles; it must not apply one behind the user's back."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert 'id="roleSelect"' in html
    assert 'id="roleAddBtn"' in html
    # The old free-text box is what let a role be typed straight in.
    assert 'id="roleAdd"' not in html
    # And the catalog is served, not hardcoded only in the browser.
    assert len(d.JOB_ROLE_OPTIONS) > 40
    assert any("Developer" in r for r in d.JOB_ROLE_OPTIONS)
    assert len(set(d.JOB_ROLE_OPTIONS)) == len(d.JOB_ROLE_OPTIONS)


def test_the_role_dropdown_is_a_real_animated_combobox():
    """A native <select> renders its popup in the OS, so it can carry no
    transition and no filter. The replacement has to be a real combobox:
    filterable, keyboard-navigable, and it must still stage rather than
    apply the choice."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    # The combobox contract, not just an element that happens to be styled.
    assert 'role="combobox"' in html
    assert 'aria-haspopup="listbox"' in html
    assert 'aria-expanded="false"' in html
    assert 'aria-controls="roleList"' in html
    assert 'role="listbox"' in html
    assert 'id="roleSearch"' in html
    # The native select is what could not be animated, so it must be gone.
    assert '<select id="roleSelect"' not in html
    # Keyboard support: arrows move, Enter takes, Escape backs out.
    for key in ("ArrowDown", "ArrowUp", "Enter", "Escape"):
        assert f'"{key}"' in html, f"the dropdown ignores {key}"
    assert "function markRoleCursor(" in html
    # The choice is staged into a hidden field; only the button commits it.
    assert 'id="rolePick"' in html
    add = html.split('$("roleAddBtn").onclick')[1][:600]
    assert 'appendRoleChip($("rolePick").value' in add or "appendRoleChip(v)" in add
    # Picking must not call appendRoleChip by itself.
    pick = html.split("function pickRole(")[1].split("\n}")[0]
    assert "appendRoleChip" not in pick, "picking a role applied it immediately"
    # The list stagger is what makes the panel feel built rather than dumped.
    assert "@keyframes roleddItem" in html
    assert "animation-delay: calc(var(--i, 0) * 13ms)" in html
    # Capping the index: at ~66 entries an uncapped stagger finishes long
    # after the user has read the top of the list.
    assert 'Math.min(i, 14)' in html


def test_cities_are_a_filterable_dropdown_backed_by_a_server_catalog():
    """Cities were a bare text box. The picker is now the same filtered menu as
    countries, with the catalog served from filters.py so the list cannot drift
    away from the geography rules that run against a chosen city."""
    import re
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert 'id="citySearch"' in html
    assert 'id="cityMenu"' in html
    # The old free-text-only row is gone, including its id.
    assert 'id="cityAdd"' not in html
    assert 'id="cityAddBtn"' not in html
    # The menu is the same pattern the country picker uses.
    assert "function cityMenu(" in html
    assert "function closeCityMenu(" in html
    assert re.search(r'<div class="menu" id="cityMenu"[^>]*hidden>', html)
    # combobox semantics, same as the role dropdown.
    assert 'aria-haspopup="listbox"' in html
    assert 'aria-expanded="false"' in html
    for key in ("Escape", "Enter"):
        assert f'e.key === "{key}"' in html, f"the city menu ignores {key}"

    # Served, not hardcoded only in the browser, and populated by the drawer.
    assert len(d.filters.known_cities()) > 100
    assert "loadCityOptions(r);" in html
    assert "state.regionByCity" in html


def test_the_city_catalog_covers_every_city_the_cv_reader_can_infer():
    """A CV is seeded with its city automatically. If the reader can infer a
    city the catalog has never heard of, the picker would show that saved city
    as unrecognised and the user would think their upload was dropped."""
    for city in d.INDIAN_CITIES:
        assert city.title() in d.filters.CITY_OPTIONS, \
            f"{city} can be inferred from a CV but is not in the city catalog"


def test_every_catalogued_city_maps_to_a_real_region_key():
    """The picker labels each city with its region. A key that is not in REGIONS
    renders as a blank column, so the label would silently stop appearing."""
    for city, region in d.filters.CITY_OPTIONS.items():
        assert region in d.filters.REGIONS, f"{city} points at unknown region {region!r}"
    # And the catalog is genuinely deduplicated / alphabetical to list.
    names = d.filters.known_cities()
    assert names == sorted(names, key=str.lower)
    assert len(set(names)) == len(names)


def test_every_id_the_script_touches_exists_in_the_markup():
    """A $("id") that returns null is a TypeError on the next property access.

    gateShow() had `$("gResetMsg").textContent = ...` while the element's id was
    `resetMsg`, so every single gate render threw halfway through - including the
    render that api() triggers on any 401. Everything after that line in
    gateShow() silently stopped happening. This walks the whole script so the
    next typo is a test failure instead of a half-rendered sign-in screen.
    """
    import re
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    script = re.search(r"(?s)<script>(.*)</script>", html).group(1)
    markup = html[: html.index("<script>")]
    # deck* are built at runtime by the deck renderer, so they are not in the
    # initial markup by design.
    runtime_only = {"deckCount", "deckDots", "deckNext", "deckPrev", "deckStage"}
    referenced = set(re.findall(r'\$\("([^"]+)"\)', script)) - runtime_only
    present = set(re.findall(r'\bid="([^"]+)"', markup))
    assert not (referenced - present), \
        f"script reads ids that do not exist: {sorted(referenced - present)}"
    # And no id is defined twice, which makes getElementById order-dependent.
    for i in present:
        assert len(re.findall(r'\bid="%s"' % re.escape(i), markup)) == 1, \
            f"duplicate id {i}"


def test_the_session_list_can_scroll_instead_of_pushing_the_footer_away():
    """The list element was `class="slist"` but the sizing rule was written for
    `.sbody`, so the list had no flex, no height and no overflow at all. In a
    fixed-height column panel that means every session past the fold was
    unreachable and the footer was pushed off-screen - which is what made the
    whole sidebar look dead, including a "New" session that was created but
    scrolled out of sight."""
    import re
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert 'class="slist" id="sessList"' in html
    # No rule may still target the retired name (the prose in a comment is fine).
    assert not re.search(r"^\.sbody\b", html, re.M), \
        "a CSS rule still targets the retired .sbody"
    slist = html.split(".slist {")[1].split("}")[0]
    # min-height:0 is the load-bearing part: a flex child defaults to
    # min-height:auto, so overflow-y:auto alone will not shrink it.
    for decl in ("flex: 1 1 auto", "min-height: 0", "overflow-y: auto"):
        assert decl in slist, f".slist is missing {decl}"
    # The head and foot must not be squeezed by the list growing.
    assert "flex: none" in html.split(".shead {")[1].split("}")[0]
    assert "flex: none" in html.split(".sfoot {")[1].split("}")[0]
    # A new session has to be scrolled into view, or creating one looks like a
    # no-op on a long list.
    assert "function scrollSessionIntoView(" in html
    assert "scrollSessionIntoView(r.id)" in html


def test_no_markup_class_is_left_without_a_style_or_a_script_reference():
    """The .slist bug was a class name in the markup with no matching rule. That
    is invisible in a code review and to the test suite, so it gets checked
    mechanically: anything in a class="" attribute has to be styled or driven.
    """
    import re
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    script = re.search(r"(?s)<script>(.*)</script>", html).group(1)
    markup = html[: html.index("<script>")]
    css = markup.split("<style>")[1].split("</style>")[0]
    styled = set(re.findall(r"\.([A-Za-z][\w-]*)", css))
    used = {c for attr in re.findall(r'class="([^"]+)"', markup) for c in attr.split()}
    orphans = sorted(c for c in used if c not in styled and c not in script)
    assert not orphans, f"unstyled, unreferenced classes: {orphans}"


def test_the_sidebar_shows_the_summary_the_server_already_sends():
    """/api/profiles returns skills / resume_chars / github per session. A list of
    rows called "New session" with no other detail is untellable, so the meta line
    puts that data on screen."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    rows = html.split("async function loadSessions(")[1].split("\nasync function")[0]
    for field in ("p.resume_chars", "p.skills", "p.github"):
        assert field in rows, f"the sidebar ignores {field}"
    assert 'm.className = "smeta"' in rows
    assert 'text.className = "stext"' in rows


def test_reset_lives_in_the_top_bar_not_only_the_sidebar():
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    header = html.split("</header>")[0]
    assert 'id="resetAllBtn"' in header, "Reset is not in the navbar"
    # Still bound, and still guarded by the typed-RESET confirmation.
    assert '$("resetAllBtn").onclick = resetEverything' in html
    confirm = html.split("async function resetEverything(")[1].split("\n}")[0]
    assert "confirm(" in confirm and '!== "RESET"' in confirm


def test_the_gate_has_a_designed_backdrop_and_motion():
    """The sign-in screen is the first thing anyone sees, so it gets a real
    backdrop, a stagger and a few transitions - all of which have to be
    reachable from the markup rather than left as an idea in a comment."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    # Layered backdrop: a drifting aurora and a masked grid.
    assert ".gate::before" in html
    assert ".gate::after" in html
    assert "@keyframes aurora" in html
    assert "@keyframes gridDrift" in html
    assert "mask-composite: exclude" in html, "the panel hairline needs a border-only mask"
    # A mark, a feature row and the board ticker, all actually in the markup.
    assert 'class="gatemark"' in html
    assert 'class="gatefeats"' in html
    assert 'class="gateticker"' in html
    assert "@keyframes ticker" in html
    # Glass panel.
    assert "backdrop-filter: blur(" in html
    # Staggered entrance driven by the existing .in class, not a timer.
    assert ".gate.in .gatemark" in html
    assert ".gate.in #gateSign > *" in html
    assert "@keyframes gateRowIn" in html
    # Interaction feedback: errors shake, buttons shine, check-inbox ticks.
    assert ".gerr:not([hidden])" in html
    assert "@keyframes shake" in html
    assert "@keyframes tickIn" in html
    # The theme must be switchable before signing in, and both toggles have
    # to stay in step so the header cannot revert it on first paint.
    assert 'id="gTheme"' in html
    assert "$(\"gTheme\").onclick = toggleTheme;" in html
    assert "function applyTheme(" in html


def test_the_gate_decorations_stop_for_reduced_motion():
    """The new backdrop, mark, ticker and list stagger all loop or stagger.
    The global duration clamp does not cover animation-delay, so the
    options would crawl in one at a time with no motion at all - these
    selectors have to be named explicitly."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    block = html.split("@media (prefers-reduced-motion: reduce)")[-1]
    for looping in (".gate::before", ".gate::after", ".gatemark .ring",
                    ".gatemark .core", ".gateticker .gtrack", ".gatecheck::before"):
        assert looping in block, f"{looping} keeps animating under reduced motion"
    assert ".roledd.open .roledd-opt { animation: none !important; }" in block, \
        "the list stagger uses animation-delay, which the duration clamp misses"
    # The tick is a one-shot but it is a flourish on a message that can be
    # read immediately, so it is dropped with the rest of the motion.
    assert ".gatecheck::after" in block


def test_the_country_picker_is_searchable_and_submitted():
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert 'id="countrySearch"' in html
    assert 'id="countryMenu"' in html
    # the selection has to reach the server or the filter never runs
    assert "countries: currentCountries()" in html
    assert "renderCountryChips" in html


# --------------------------------------------------------------------------
# Auth gate: the two bugs that made logging in look broken
# --------------------------------------------------------------------------

def _html():
    return (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Launcher: an occupied port
#
# Starting a second copy used to die on WinError 10048 / "address already in
# use". That message names neither the culprit nor the obvious fix, and the very
# common case is simply "the one you started ten minutes ago is still up".
# --------------------------------------------------------------------------

def _serve_once(body: bytes):
    """A throwaway HTTP server on a free port; returns (server, port)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_an_occupied_port_is_explained_instead_of_raising_winerror():
    srv, port = _serve_once(b"<title>JobScope</title>")
    try:
        msg = d.already_serving("127.0.0.1", port)
        assert msg and "already running" in msg
        assert str(port) in msg, "the message must name the port"
        assert "Stop-Process" in msg, "and say how to stop the running copy"
        # ASCII only: this prints to a Windows console whose default code page
        # is cp1252, and an em-dash turns into a replacement glyph there.
        assert msg.isascii()
    finally:
        srv.shutdown()


def test_a_port_held_by_something_else_is_not_mistaken_for_jobscope():
    srv, port = _serve_once(b"<title>Some other dev tool</title>")
    try:
        msg = d.already_serving("127.0.0.1", port)
        assert msg and "not JobScope" in msg
        assert "already running" not in msg
    finally:
        srv.shutdown()


def test_a_free_port_reports_nothing_to_stop():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert d.already_serving("127.0.0.1", free) is None


def test_a_hidden_element_is_actually_hidden():
    # The confirm-password row is markup-hidden, but the stylesheet set
    # display:flex on .addrow, which beat the [hidden] default and left the
    # "Confirm password" field sitting on the plain sign-in form.
    assert "[hidden]" in _html()
    rule = next(ln for ln in _html().splitlines()
                if "[hidden]" in ln and "display" in ln)
    assert "display: none" in rule
    assert "important" in rule, "a later display rule would win without !important"


def test_signing_in_dismisses_the_gate():
    # Otherwise a successful login still showed the gate until a manual reload.
    body = next(b for b in _html().split("function signedIn(")[1:2])
    start = body[: body.index("\n}")]
    assert "gateHide()" in start, \
        "signedIn() must hide the gate, not only repaint the account label"


# --------------------------------------------------------------------------
# The deck tells the truth about repeats and blocked sources
# --------------------------------------------------------------------------

def test_the_deck_reports_repeats_and_blocked_sources():
    """Two silent failures made a working search look broken.

    The board tool now reports how many listings were already shown and which
    sources refused to answer; the deck has to render both, or the JSON carries
    warnings nobody ever sees.
    """
    html = _html()
    assert "function deckNotes(" in html
    body = html.split("function deckNotes(")[1].split("\n}")[0]
    assert "p.repeats" in body, "repeat count is never rendered"
    assert "p.blocked_reasons" in body, "blocked sources are never rendered"
    # The deck must receive the whole payload, not just the jobs array.
    assert 'case "jobs":' in html
    assert "renderDeck(obj.payload)" in html, \
        "renderDeck needs the payload's repeat/blocked metadata, not jobs alone"
    # A plain array is still accepted so no caller has to change shape.
    deck = html.split("function renderDeck(")[1].split("\n  const slot")[0]
    assert "payload.jobs" in deck


def test_the_previous_runs_deck_is_actually_removed_before_a_new_run():
    """A hunt aimed at a different board must not leave the last run's cards
    on screen looking like this run's answer.

    It called renderDeck([]), which did nothing: an array has no `.jobs`, so
    renderDeck took the non-deck branch and returned at `if (!isDeck) return`.
    Neither the DOM node nor state.deck was cleared, so the old deck survived
    a run that found nothing AND a run that failed before its first frame.
    """
    html = _html()
    assert "function clearDeck(" in html, "there is no way to clear the deck"
    clear = html.split("function clearDeck(")[1].split("\n}")[0]
    assert "state.deck = []" in clear, "the deck state outlives the run"
    assert ".deck" in clear and ".remove()" in clear, "the deck element stays on screen"

    # The run-start path must use it, not renderDeck([]).
    send = html.split("async function sendMessage(")[1]
    start = send[:send.index("appendMsg(\"user\"")]
    assert "clearDeck()" in start, "a new run does not clear the previous deck"
    assert "renderDeck([])" not in start, \
        "renderDeck([]) is a no-op for a bare array and must not be used to clear"
    # And the "nothing came back" path must still be reachable, so a run that
    # genuinely finds nothing says so.
    assert 'No listings came back' in html


def test_a_new_session_is_told_it_has_no_role_yet():
    """An empty role list is the real state of a fresh account and needs a
    visible explanation, or it just looks like the app failed to load."""
    html = _html()
    assert 'id="roleEmpty"' in html
    assert "function syncRoleEmpty(" in html
    # The hint has to follow every path that changes the chip list, including
    # "Clear all", which empties the box without firing the chip ✕ handler.
    assert "syncRoleEmpty();" in html.split("function syncClearButtons(")[1][:600]


def test_reduced_motion_is_respected():
    """The animations are decorative, so a reader who asks for less motion
    must actually get none of them - including the infinite pulses."""
    html = _html()
    blocks = html.split("@media (prefers-reduced-motion: reduce)")
    assert len(blocks) > 1, "no reduced-motion query at all"
    # The global clamp near the top of the stylesheet is what stops the
    # decorative loops; the later block is the new entrance motion.
    global_block = blocks[1].split("}")[1]
    assert "animation-duration" in global_block
    assert "transition-duration" in global_block
    assert "!important" in global_block, \
        "without !important the later rules keep the motion"
    for looping in ("ringpulse", "typing i", "spin"):
        assert looping in global_block or "animation-duration" in global_block

    new_block = blocks[2].split("@keyframes")[0]
    assert "animation: none" in new_block
    assert "!important" in new_block, \
        "the looping pulses need !important to actually be stopped"
    # The endless loops are named by their selectors, not their keyframes:
    # the live-node ring, the typing dots and the spinner.
    for looping in (".node.live .pill::after", ".typing i", ".spin"):
        assert looping in new_block, f"{looping} keeps animating under reduced motion"


def test_the_gate_animates_in_and_out_without_js_timing():
    """The fade is driven by a class, not a timer: a setTimeout to re-hide the
    gate is the classic way to leave it stuck on screen."""
    html = _html()
    assert "class=\"gate in\"" not in html, "the class is added by gateShow"
    show = html.split("function gateShow(")[1].split("\nfunction gateHide")[0]
    assert 'classList.add("in")' in show
    assert "offsetWidth" in show, "re-showing needs a reflow to re-trigger"
    hide = html.split("function gateHide(")[1].split("\n}")[0]
    assert "setTimeout" not in hide, "a timer can leave the gate stuck open"


def test_the_dev_reset_link_actually_switches_the_form():
    # With AUTH_DEV_LINKS=1 the server returns the link it would have mailed.
    # Showing the new-password fields while leaving the form in "forgot" mode
    # meant the next click re-posted the email and the reset never completed.
    body = _html().split("if (r.dev_link) {")[-1]
    block = body[: body.index("\n  }")]
    assert 'GATE.mode = "setpw"' in block
    assert "GATE.token" in block, "the link's token has to be kept to be sent"
    # and the submit path must prefer it over an absent ?token= in the URL
    submit = _html().split('$("gateReset").onsubmit')[1][:900]
    assert "GATE.token" in submit


# --------------------------------------------------------------------------
# Responsive layout
#
# There is no browser in this environment, so the layout cannot be screenshotted.
# What *can* be checked mechanically is the class of bug that produced the
# problem in the first place: a rule that is absent, ordered wrongly, or
# silently lost. Each test below pins a specific failure that was reachable on a
# real device, so a later edit that reintroduces it fails here.
# --------------------------------------------------------------------------

def _css():
    """Just the stylesheet - not the JS, which contains its own "<style>" string."""
    return _html().split("<style>")[1].split("</style>")[0]


def _media_blocks(css):
    """{query: body} for every @media in the sheet, innermost query included."""
    out = {}
    for m in re.finditer(r"@media([^{]*)\{", css):
        q = re.sub(r"\s+", " ", m.group(1).strip())
        depth, i = 1, m.end()
        while depth and i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        out.setdefault(q, []).append(css[m.end() : i - 1])
    return out


def test_the_stylesheet_is_structurally_sound():
    """Braces and comment markers balanced. A stray brace in a media query
    silently voids every rule after it, and the page then looks broken in a way
    no unit test would otherwise notice."""
    css = _css()
    assert css.count("{") == css.count("}"), "unbalanced braces in the stylesheet"
    assert css.count("/*") == css.count("*/"), "unclosed CSS comment"


def test_the_transcript_scrolls_itself_instead_of_pushing_the_composer_away():
    """.app is a flex column of circuit / transcript / composer. Without
    flex + min-height:0 on the transcript it just grew, so on any short viewport
    the composer - the only way to send anything - was pushed off-screen.
    This was not caught because nothing in the markup changed."""
    css = _css()
    assert re.search(r"main\.chat\s*\{[^}]*flex:\s*1", css), "main.chat is not a flex child"
    assert re.search(r"main\.chat\s*\{[^}]*min-height:\s*0", css), (
        "main.chat needs min-height:0 or the flex floor stops it shrinking"
    )
    assert re.search(r"main\.chat\s*\{[^}]*overflow-y:\s*auto", css)


def test_the_shell_uses_a_dynamic_viewport_height_with_a_fallback():
    """100vh is the *tallest* viewport on mobile, so with the browser's URL bar
    collapsed the composer sat underneath it. 100dvh tracks the visible area;
    the bare 100vh has to stay ahead of it for browsers that lack dvh."""
    app = re.search(r"\.app\s*\{([^}]*)\}", _css()).group(1)
    assert "min-height: 100vh" in app, "the non-dvh fallback is missing"
    assert "100dvh" in app, "mobile browsers will hide the composer behind the URL bar"
    assert app.index("100vh") < app.index("100dvh"), (
        "the fallback must be declared first or supporting browsers ignore it"
    )
    # Anchored so it cannot match the "100vh" inside "min-height: 100vh".
    assert not re.search(r"(?<![-\w])height:\s*100vh", app), (
        "a fixed height clips the controls rather than letting the page grow"
    )


def test_the_header_is_capped_so_it_does_not_stretch_on_an_ultrawide_screen():
    """The header is outside .app, so it stayed full-bleed while the column it
    controls was centred at 900px: the brand drifted to the far left and the
    buttons to the far right, far from the content."""
    css = _css()
    assert "header > * { max-width: 100%; }" in css, (
        "a long label or button can otherwise force the page wider than the screen"
    )
    assert "@media (min-width: 1600px)" in css, "no wide-screen cap on the header"


def test_the_pipeline_cannot_force_a_horizontal_scrollbar():
    """The five nodes are space-between with nowrap labels, so at phone width
    they overflowed their container - and because .app is a shrink-to-fit block
    that overflow scrolled the *whole page* sideways, taking the composer and
    every other control with it. The pills have to shrink before that happens."""
    blocks = _media_blocks(_css())
    narrow = next(b for q in blocks if q == "(max-width: 760px)" for b in blocks[q])
    assert re.search(r"\.node \.pill\s*\{[^}]*width:\s*3\dpx", narrow), (
        "the node pills do not shrink on a narrow screen"
    )
    assert re.search(r"\.nodes\s*\{[^}]*padding:\s*10px 8px", narrow)


def test_the_gate_can_scroll_on_a_landscape_phone():
    """.gate is overflow:hidden and the sign-in card is taller than a 360px-tall
    viewport, so the card was clipped with the submit button unreachable - an
    outright lockout, and the one responsive bug with a hard functional cost."""
    css = _css()
    blocks = _media_blocks(css)
    short = "".join(blocks.get("(max-height: 560px)", []))
    assert "@media (max-height: 560px)" in css, "no short-viewport handling at all"
    assert re.search(r"\.gate\s*\{[^}]*overflow-y:\s*auto", short), (
        "the gate still clips its own content on a landscape phone"
    )
    # Decoration goes first; the form and its submit button must survive.
    for deco in (".gatemark", ".gatefeats", ".gateticker"):
        assert re.search(re.escape(deco) + r"\s*\{[^}]*display:\s*none", short), (
            f"{deco} should yield its space on a short screen"
        )
    assert ".gatefoot" in short, "the footer is not tightened for short screens"


def test_the_gate_ticker_is_driven_by_the_scraper_not_hardcoded():
    """The marquee used to carry its own copy of the board list and drifted:
    it advertised Wellfound, StepStone, Bayt and Jooble - none of which this
    app can scrape - while omitting four boards that do work. The names now
    come from the scraper's registry, so there is one list with one owner."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")

    # The track element ships empty and is filled from /api/config.
    assert 'id="gateTrack"' in html
    assert "function fillBoardList" in html
    assert "fillBoardList(r.boards" in html, "fillBoardList is never called"

    # Nothing may be baked into the markup inside the track.
    track = html.split('id="gateTrack"')[1].split("</div>")[0]
    for ghost in ("Wellfound", "StepStone", "Bayt", "Jooble"):
        assert ghost not in track, f"{ghost} is not a board this app can scrape"

    # The count must reflect what a sweep really probes, not a fixed number.
    assert "9 job boards, one search" not in html
    assert "defaults.length" in html

    # The pipeline popup reads from the same place.
    assert 'id="boardsPopup"' in html
    assert "pop.textContent = boards.join" in html


def test_the_gate_board_ticker_loads_before_anyone_signs_in():
    """The sign-in ticker renders pre-auth, but the only caller of
    fillBoardList was loadConfig, which 401s. So the first screen a visitor
    ever saw had an empty ticker and claimed only "Job boards, one search"."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")

    # A public fetch, called at init rather than from signedIn().
    assert "function loadPublicBoards" in html
    assert '"/api/boards"' in html
    assert "loadPublicBoards();" in html
    # It must be called at init, i.e. before bootAuth settles, and not only
    # from inside signedIn (which is where the authenticated config lives).
    boot = html.split("/* ---------- init ---------- */")[1]
    assert "loadPublicBoards()" in boot.split("bootAuth()")[0], (
        "loadPublicBoards must be called before bootAuth to beat the gate")
    signed_in = html.split("function signedIn(st)")[1].split("\n}")[0]
    assert "loadPublicBoards" not in signed_in


def test_the_public_board_list_leaks_nothing_about_the_user():
    """It is unauthenticated, so it has to be provably free of user state -
    the gate cannot use /api/config, but it must not become a way around login."""
    out = asyncio.run(d.public_boards())
    assert set(out) == {"boards", "default_boards"}, out.keys()

    src = Path(d.__file__).read_text(encoding="utf-8")
    body = src.split("async def public_boards(")[1].split("\n\n")[0]
    for leak in ("ctx", "cookie", "auth.", "_profile_for", "resume", "query",
                 "email", "token"):
        assert leak not in body, f"/api/boards must not touch {leak!r}"


def test_the_public_board_list_matches_the_scrapers_own_registry():
    """Both boards and defaults, so the gate counts a sweep honestly."""
    out = asyncio.run(d.public_boards())
    import mcp_server_indeed_scraper as scraper
    assert out["boards"] == list(scraper.BOARDS)
    assert out["default_boards"] == list(scraper.default_boards())
    # A sweep probes fewer boards than exist; that gap is what the count shows.
    assert 0 < len(out["default_boards"]) <= len(out["boards"])


def test_config_endpoint_exposes_the_real_board_registry():
    """The list has to come from the scraper, so a board added there shows up
    in the UI without a second edit."""
    src = Path(d.__file__).read_text(encoding="utf-8")
    assert "def _board_names()" in src
    assert '"boards": _board_names()' in src
    assert '"default_boards": _default_board_names()' in src
    # And it really does read the scraper rather than a local literal.
    assert "list(_s.BOARDS)" in src
    assert "list(_s.default_boards())" in src

    names = d._board_names()
    assert "Indeed" in names and "LinkedIn" in names
    defaults = d._default_board_names()
    assert set(defaults) < set(names), "some boards must be skippable by default"
    for skipped in ("Glassdoor", "Foundit", "Naukri"):
        assert skipped in names, f"{skipped} must stay reachable by name"
        assert skipped not in defaults, f"{skipped} costs every hunt"


def test_a_skipped_board_renders_dimmed_not_hidden():
    """Hiding it would read as "unsupported"; it is only not searched by
    default."""
    html = (Path(d.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert ".gateticker span.skipped" in html
    assert "indexOf(b) !== -1" in html


def test_the_sidebar_and_drawers_fit_a_phone_screen():
    """A 262px sidebar is 82% of a 320px phone. That is a panel, not a drawer."""
    blocks = _media_blocks(_css())
    phone = "".join(blocks.get("(max-width: 600px)", []))
    assert re.search(r"\.sidebar\s*\{[^}]*width:\s*min\(300px, 88vw\)", phone)
    assert re.search(r"\.drawer\s*\{[^}]*width:\s*100vw", phone), (
        "the settings drawer keeps its desktop side-panel width on a phone"
    )


def test_the_header_controls_fit_a_narrow_screen():
    """Five controls, the brand and a status pill in a nowrap row do not fit at
    phone width. Two things are needed: the row may wrap, and the decorative
    bits drop rather than overflow."""
    css = _css()
    header = re.search(r"header\s*\{([^}]*)\}", css).group(1)
    assert "flex-wrap: wrap" in header, "the header cannot wrap, so it overflows"
    blocks = _media_blocks(css)
    assert "(max-width: 1040px)" in blocks, (
        "nothing sheds load before the header overflows"
    )
    tablet = "".join(blocks["(max-width: 1040px)"])
    assert ".status .modeltag" in tablet and "display: none" in tablet, (
        "the model name is the first thing to drop"
    )
    phone = "".join(blocks.get("(max-width: 600px)", []))
    assert re.search(r"\.acct\s*\{[^}]*display:\s*none", phone), (
        "the account address pushes the controls off-screen at 320px"
    )


def test_touch_devices_do_not_depend_on_hover():
    """On a phone there is no hover, so anything revealed only by :hover is
    unreachable. The session and chip delete buttons were both partially
    transparent and the pipeline tooltips were hover/focus-only."""
    blocks = _media_blocks(_css())
    assert "(hover: none)" in blocks, "no hover-capability handling for touch"
    hoverless = "".join(blocks["(hover: none)"])
    assert ".srow .sdel" in hoverless, "the session delete button needs hover to be seen"
    assert ".chips .chip .x" in hoverless, "the chip remove button is unreachable on touch"
    assert ".node .pop" in hoverless, "the hover-only tooltips still block the touch UI"
    coarse = "".join(blocks.get("(pointer: coarse)", []))
    assert ".sendbtn" in coarse, "the send button stays a 38px touch target"


def test_notched_phones_get_their_safe_area_insets():
    """Without these the fixed sidebar and drawers sit flush to the screen edge,
    putting their headers under the notch and their footers under the home bar."""
    css = _css()
    assert "@supports (padding: env(safe-area-inset-" in css
    supports = css.split("@supports (padding: env(safe-area-inset-left)")[1]
    for target in (".sidebar", ".drawer", ".app", ".toast"):
        assert target in supports.split("}")[0] or target in supports, (
            f"{target} ignores the safe-area insets"
        )


def test_long_strings_cannot_push_the_page_sideways():
    """A confirmation URL, a company name or a listing title with no spaces was
    the other source of a full-page horizontal scroll."""
    css = _css()
    assert re.search(
        r"\.bubble[^{]*\{[^}]*overflow-wrap:\s*anywhere", css
    ), "long strings in the transcript can widen the page"


def test_the_responsive_overrides_come_after_the_components_they_override():
    """Ordering is load-bearing here. These are plain overrides, so the block
    has to come *after* the component rules; .gate is redefined ~340 lines below
    where the responsive block used to sit, which silently voided the
    landscape-phone fix. A plain string check on the banner comment would not
    survive someone re-adding a component rule after it."""
    css = _css()
    banner = css.index("Responsive layout")
    for component in (
        ".gate { position: fixed; inset: 0; z-index: 90; overflow: hidden",
        ".deck { margin: 2px 42px 18px",
        ".sidebar {",
        ".drawer {",
    ):
        assert css.index(component) < banner, (
            f"{component!r} is defined after the responsive block and will beat it"
        )
    # And it still has to be inside the stylesheet, not after it.
    assert css.rstrip().endswith("overflow-wrap: anywhere; }"), (
        "the responsive block is no longer the last thing in the stylesheet"
    )


# --------------------------------------------------------------------------
# Reset leaves the "already shown" memory behind
#
# The scraper marks postings the account has seen so repeat hunts lead with
# something new. That memory is per account, correctly - but reset_everything
# never touched it, so every posting stayed marked and the first hunt after a
# reset came back with nothing fresh, falling back to repeats. That reads as a
# broken search rather than a clean slate.