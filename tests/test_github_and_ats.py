"""Tests for the GitHub-project pipeline and the ATS scorer.

Each test here corresponds to a defect found by running the code against the
real profile (nitheeshkumarth-byte) and the real generated documents, not to an
assumption:

  * `messages-noreply`-style noise, forks and archived repos must not be
    treated as projects.
  * A README with an unresolved merge marker has its title *below* the marker;
    cleaning line-by-line previously dropped it and surfaced a pip requirement
    as the project description.
  * "python-multipart is required by FastAPI" is an install note, not a
    project description.
  * "rest" must not match "restaurant".
  * Structure must start from full marks, or every clean resume scores 8/16.
  * Date ranges are written "Mar 2023 -- Present" by the generator, so a
    pattern accepting only a single hyphen never matches real output.
"""
import pytest

import ats_score as A
import github_projects as gp


# ------------------------------------------------------- username parsing ----

@pytest.mark.parametrize("text,expected", [
    ("https://github.com/nitheeshkumarth-byte", "nitheeshkumarth-byte"),
    ("http://www.github.com/some-user", "some-user"),
    ("github.com/some-user", "some-user"),
    ("github.com/some-user/resume/blob/main/cv.pdf", "some-user"),
    ("GitHub: github.com/another-user", "another-user"),
    ("Nitheesh Kumar (@nitheeshkumarth-byte)", "nitheeshkumarth-byte"),
])
def test_username_comes_from_the_cv_text(text, expected):
    assert gp.github_username(text) == expected


@pytest.mark.parametrize("text", [
    "",
    "no link here",
    "reach me at foo@gmail.com",
    "AI/ML Engineer at Quiddity Infotech",
    "github.com",
    "https://gitlab.com/some-user",
])
def test_non_github_text_yields_no_username(text):
    assert gp.github_username(text) == ""


def test_a_username_is_not_inferred_from_a_job_title():
    """The word "engineer" must never become a GitHub handle."""
    assert gp.github_username("Engineer at Acme") == ""


@pytest.mark.parametrize("name", ["features", "explore", "pricing", "topics"])
def test_reserved_github_paths_are_not_usernames(name):
    """github.com/features is a page, not a profile; it must not be fetched."""
    assert gp.is_reserved(name)


# ----------------------------------------------------------- token policy ----

def test_public_repos_work_without_a_token(monkeypatch):
    """A token is a rate-limit upgrade, not a requirement."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert gp.has_token() is False
    assert "Authorization" not in gp._headers()


def test_a_token_is_used_when_present(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    assert gp.has_token() is True
    assert gp._headers()["Authorization"] == "Bearer ghp_test"


def test_a_blank_token_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert gp.has_token() is False


# ----------------------------------------------------------- repo scoring ----

def _repo(**kw):
    base = {"name": "x", "description": "", "language": "", "topics": [],
            "fork": False, "archived": False, "stargazers_count": 0}
    base.update(kw)
    return base


def test_a_matching_repo_outscores_an_unrelated_posting():
    repo = _repo(name="rag-pipeline", description="RAG service with FastAPI",
                 language="Python")
    jd = "Python engineer with RAG and FastAPI experience"
    assert gp.score_repo(repo, jd) > gp.score_repo(repo, "Chef de Partie, Lyon")


def test_a_repo_with_no_signal_scores_zero():
    assert gp.score_repo(_repo(name="dotfiles", description="my config"),
                         "Python RAG engineer") == 0.0


def test_an_empty_posting_scores_zero():
    assert gp.score_repo(_repo(name="rag", description="rag"), "") == 0.0


def test_jd_vocabulary_outweighs_repo_vocabulary():
    """A term the JD itself uses is the requirement; the repo's is evidence."""
    jd = "We need Rust engineers."
    only_repo = _repo(name="rust-toolkit", description="helpers")
    both = _repo(name="rust-toolkit", description="rust helpers for the stack")
    assert gp.score_repo(both, jd) >= gp.score_repo(only_repo, jd)


# ------------------------------------------------------------ readme clean ----

def test_badges_and_images_are_removed():
    out = gp._clean_readme("![build](https://img.shields.io/x.svg)\n# Real Title\n")
    assert "shields.io" not in out
    assert "Real Title" in out


def test_installation_sections_are_dropped():
    out = gp._clean_readme("# T\n## Installation\npip install foo\n## Usage\nrun it\n")
    assert "pip install" not in out


def test_an_unresolved_merge_keeps_the_resolved_side():
    """A conflicted README has its title BELOW the marker.

    Cleaning line-by-line discarded the first 40 lines and left a requirements
    fragment as the project description, which is how
    "`python-multipart` is required by FastAPI" ended up on a resume.
    """
    conflicted = (
        "<<<<<<< HEAD\n"
        "# Old title\n"
        "stale content\n"
        "=======\n"
        "# Minimal RAG Project\n"
        "A from-scratch RAG pipeline.\n"
        ">>>>>>> branch\n"
    )
    out = gp._clean_readme(conflicted)
    assert "Minimal RAG Project" in out
    assert "<<<<<<<" not in out and "=======" not in out
    assert "stale content" not in out
    assert gp._first_heading(out) == "Minimal RAG Project"


def test_an_install_note_is_never_a_project_description():
    readme = ("# Project\n"
              "(`python-multipart` is required by FastAPI for `UploadFile`\n"
              "handling.)\n")
    summary = gp._summarise("proj", "", readme)
    assert "required by FastAPI" not in summary


def test_the_repo_description_is_preferred_over_the_readme():
    summary = gp._summarise("proj", "A curated one-line description here",
                            "# Heading\n\nfirst prose line that is long enough\n")
    assert summary == "A curated one-line description here"


def test_a_readme_with_only_boilerplate_falls_back_to_the_repo_name():
    assert gp._summarise("my-project", "", "## License\nMIT\n") == "my-project"


# ----------------------------------------------------- the no-invention guard ----

def test_a_summary_unsupported_by_the_readme_is_rejected():
    """The central safety property.

    `_bullet` refuses a line that shares no vocabulary with the README it came
    from. Without it, summarising the wrong repo would print a capability the
    candidate never claimed — the same failure mode as inventing experience.
    """
    readme = "A RAG pipeline using pgvector and FastAPI."
    assert gp._bullet("RAG pipeline with FastAPI", readme, {"rag", "fastapi"})
    assert gp._bullet("Kubernetes operator for air traffic control", readme,
                      {"kubernetes"}) == ""


def test_an_empty_summary_produces_no_bullet():
    assert gp._bullet("", "some readme", {"python"}) == ""


def test_a_long_summary_is_truncated_not_invented():
    out = gp._bullet("RAG pipeline " + "with FastAPI and pgvector " * 12,
                     "RAG pipeline with FastAPI and pgvector", {"rag"})
    assert len(out) <= 165
    assert out.endswith("...")


# ----------------------------------------------------------- find_projects ----

def test_no_username_returns_an_empty_result_without_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: pytest.fail("no request for no username"))
    out = gp.find_projects("", "Python RAG engineer")
    assert out["projects"] == [] and out["repos_scanned"] == 0


def test_a_reserved_path_returns_an_empty_result_without_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: pytest.fail("no request for /features"))
    assert gp.find_projects("https://github.com/features", "Python")["projects"] == []


def test_a_missing_profile_yields_no_projects_and_no_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_CACHE_DIR", str(tmp_path))

    class _Resp:
        status_code = 404
        def raise_for_status(self): pass
        def json(self): return {"message": "Not Found"}
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    out = gp.find_projects("https://github.com/no-such-user-xyz", "Python RAG")
    assert out["projects"] == [] and out["error"] == ""


def test_a_network_failure_never_raises(monkeypatch, tmp_path):
    """A GitHub outage must degrade the resume, not fail the request."""
    monkeypatch.setenv("GITHUB_CACHE_DIR", str(tmp_path))
    import requests

    def _boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(requests, "get", _boom)
    out = gp.find_projects("https://github.com/some-user", "Python")
    assert out["projects"] == []


def test_a_rate_limit_is_reported_so_the_user_can_add_a_token(monkeypatch,
                                                              tmp_path):
    monkeypatch.setenv("GITHUB_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    class _Resp:
        status_code = 403
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    out = gp.find_projects("https://github.com/some-user", "Python")
    assert "GITHUB_TOKEN" in out["error"]


def test_forks_and_archived_repos_are_excluded(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    payload = [
        {"name": "fork-of-x", "description": "rag", "fork": True},
        {"name": "old-thing", "description": "rag", "archived": True},
        {"name": "homework", "description": "rag"},
        {"name": "real-rag", "description": "RAG service", "language": "Python",
         "stargazers_count": 2, "html_url": "https://github.com/u/real-rag"},
    ]

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    assert [r["name"] for r in gp.list_repos("u")] == ["real-rag"]


def test_readmes_are_only_fetched_for_matching_repos(monkeypatch, tmp_path):
    """Requests must scale with matches, not with the whole profile."""
    monkeypatch.setenv("GITHUB_CACHE_DIR", str(tmp_path))
    seen = []

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return [
                {"name": "chef-recipes", "description": "pasta", "language": None},
                {"name": "rag-service", "description": "RAG with FastAPI",
                 "language": "Python", "html_url": "u/rag-service"},
            ]
    import base64
    import requests

    def _get(url, **kw):
        seen.append(url)
        if "/users/" in url:
            return _Resp()
        import base64 as b64
        body = "# RAG Service\n\nA RAG service with FastAPI and pgvector.\n"
        return type("R", (), {
            "status_code": 200,
            "json": lambda self, body=body: {
                "content": b64.b64encode(body.encode()).decode()},
        })()
    monkeypatch.setattr(requests, "get", _get)
    out = gp.find_projects("https://github.com/u", "RAG FastAPI engineer",
                           max_projects=3, max_readmes=2)
    assert [p["name"] for p in out["projects"]] == ["rag-service"]
    assert not any("chef-recipes" in u for u in seen)


# ---------------------------------------------------------------- ATS score ----

JD = ("Python engineer with FastAPI, Docker, AWS, PostgreSQL, CI/CD, "
      "microservices, REST APIs, pytest, problem solving and cross-functional "
      "collaboration.")

# A reference document with no layout problems, so the structure component can
# be asserted at full marks. It carries roughly one page of real content: the
# word-count check fires below ~120 words, so a four-line fixture would fail for
# a reason that has nothing to do with structure.
GOOD = r"""
\section*{Summary}
Python backend engineer with four years building and operating production
services for fintech and healthcare products, from first schema to on-call.
\section*{Skills}
Languages: Python, JavaScript, SQL. Backend: FastAPI, Django, REST APIs,
microservices. Data: PostgreSQL, MongoDB, Redis. Cloud: AWS, Docker, Kubernetes,
Terraform, GitHub Actions CI/CD. Testing: pytest, integration suites.
\section*{Experience}
Senior Backend Engineer, Acme -- Mar 2023 -- Present
- Built a FastAPI service handling 12000 requests per minute, cutting p99
  latency 35% by adding query caching and connection pooling.
- Deployed to AWS with Docker and GitHub Actions CI/CD, reducing deploy time 60%
  and making every release reversible with a single command.
- Migrated PostgreSQL to a sharded schema across three regions, improving
  query time 40% and removing the nightly maintenance window.
- Led cross-functional collaboration between three teams to define the public
  API contract, and wrote the RFC that other services now depend on.
Backend Engineer, Beta -- Jul 2021 -- Feb 2023
- Rebuilt a monolithic reporting job as asynchronous workers, halving the
  overnight run and eliminating the manual retry script.
- Cut infrastructure spend 22% by rightsizing instances and adding caching in
  front of the two most expensive queries.
\section*{Projects}
- Retrieval augmented generation pipeline using FastAPI, pgvector and pytest,
  answering support questions from internal documentation.
\section*{Education}
B.Tech Computer Science, 2020
\section*{Certifications}
AWS Solutions Architect
\section*{Languages}
English, Telugu
github.com/someuser linkedin.com/in/someuser
"""


def test_a_strong_resume_scores_well():
    r = A.score_resume(GOOD, JD)
    assert r["total"] >= 70, r
    assert r["band"] in ("good", "strong")


def test_an_empty_resume_scores_low_without_raising():
    r = A.score_resume("", JD)
    assert 0 <= r["total"] < 40
    assert r["issues"]


def test_every_component_is_always_present():
    """The UI must never have to handle a missing key."""
    for text, jd in ((GOOD, JD), ("", ""), ("x" * 50, JD)):
        b = A.score_resume(text, jd)["breakdown"]
        assert set(b) == {"contact", "sections", "structure", "dates",
                          "keywords", "evidence"}
        assert sum(v["max"] for v in b.values()) == 100


def test_a_clean_single_column_resume_gets_full_structure_marks():
    """Structure starts at full and subtracts.

    Starting it at a fixed 8 capped every well-formed resume at 8/16, so a
    document with no layout problems still looked half-broken.
    """
    r = A.score_resume(GOOD, JD)
    assert r["breakdown"]["structure"]["score"] == 16.0


def test_a_table_is_penalised():
    clean = A.score_resume(GOOD, JD)["breakdown"]["structure"]["score"]
    tabled = A.score_resume(GOOD.replace("\\section*{Skills}",
                                         "\\begin{tabular}{ll}\n\\section*{Skills}"),
                            JD)["breakdown"]["structure"]["score"]
    assert tabled < clean
    assert any("table" in i.lower() for i in A.score_resume(
        GOOD + "\n\\begin{tabular}{ll}a&b\\end{tabular}", JD)["issues"])


def test_graphics_are_penalised():
    r = A.score_resume(GOOD + "\n\\includegraphics[width=2cm]{logo.png}", JD)
    assert any("graphic" in i.lower() or "image" in i.lower()
               for i in r["issues"])


def test_latex_date_ranges_are_recognised():
    """The generator writes 'Mar 2023 -- Present'.

    A pattern accepting only a single hyphen or an en dash never matched, so
    every real generated resume reported 'no employment dates'.
    """
    r = A.score_resume(GOOD, JD)
    assert r["breakdown"]["dates"]["score"] >= 10
    assert not any("No employment dates" in i for i in r["issues"])


def test_bare_years_score_above_no_dates_at_all():
    years = A.score_resume("Experience\nEngineer, Acme 2021 - 2023\n", JD)
    none = A.score_resume("Experience\nEngineer, Acme\n", JD)
    assert (years["breakdown"]["dates"]["score"]
            > none["breakdown"]["dates"]["score"])


def test_ambiguous_numeric_dates_are_flagged():
    r = A.score_resume(GOOD + "\n01/02/25 - present\n", JD)
    assert any("ambiguous" in i.lower() for i in r["issues"])


def test_keywords_come_from_the_posting_not_a_fixed_list():
    assert A.extract_keywords("") == []
    ml = A.extract_keywords("Machine learning engineer, NLP")
    assert "machine learning" in ml
    # A multi-word phrase wins over its component words.
    assert "machine" not in ml and "learning" not in ml
    # A role outside the tech vocabulary contributes nothing rather than noise.
    assert A.extract_keywords("Chef de Partie, French cuisine") == []


def test_keyword_matching_is_word_bounded():
    """'rest' must not match 'restaurant'."""
    kws = ["rest"]
    assert A._keyword_hits("I enjoy restaurant food", kws) == []
    assert A._keyword_hits("built a REST API", kws) == ["rest"]


def test_go_does_not_match_going():
    assert A._keyword_hits("always going places", ["go"]) == []


def test_a_better_resume_outscores_a_worse_one():
    thin = r"\section*{Experience}\nresponsible for tasks\n"
    assert A.score_resume(GOOD, JD)["total"] > A.score_resume(thin, JD)["total"]


def test_a_resume_with_no_jd_still_scores_the_structural_parts():
    r = A.score_resume(GOOD, "")
    assert r["breakdown"]["keywords"]["score"] == 0.0
    assert r["breakdown"]["contact"]["score"] > 0


def test_issues_are_actionable_not_just_numbers():
    r = A.score_resume("", JD)
    assert r["issues"]
    assert all(len(i) > 25 for i in r["issues"])


def test_scoring_is_deterministic():
    assert A.score_resume(GOOD, JD) == A.score_resume(GOOD, JD)


# ------------------------------------------- the scorer reads both documents ----

HTML = """<div class="rh-center"><div class="rh-name">Jane Doe</div>
jane@example.com | +1 555 010 0100 | Austin, TX
<a href="https://www.linkedin.com/in/jane">linkedin</a>
<a href="https://github.com/jane">github</a></div>
<h2>Objective</h2>
Backend engineer with six years of Python and PostgreSQL work.
<h2>Technical Skills</h2>
<ul><li><strong>Languages:</strong> Python, SQL
<li><strong>Cloud:</strong> AWS, Docker</ul>
<h2>Experience</h2>
<p>Senior Engineer, Acme -- Mar 2023 -- Present</p>
<ul><li>Cut p99 latency 35% by adding caching to the FastAPI service
<li>Migrated PostgreSQL to a sharded schema, 40% faster</ul>
<p>Engineer, Beta -- Jul 2021 -- Feb 2023</p>
<ul><li>Owned the REST API used by two internal clients</ul>
<h2>Education</h2>
<p>BSc Computer Science, 2019</p>
"""


def test_the_rendered_html_is_scored_not_only_the_latex_source():
    """The preview is what the user looks at, so it is what gets scored.

    Scoring the HTML with LaTeX-only patterns reported "No 'summary' section",
    "No 'experience' section" and "No recognisable headings" on a document that
    had all three.
    """
    r = A.score_resume(HTML, JD)
    joined = " ".join(r["issues"])
    for heading in ("summary", "experience", "skills"):
        assert f"No '{heading}' section" not in joined
    assert "No recognisable headings" not in joined
    assert r["breakdown"]["sections"]["score"] >= 11.0


def test_both_representations_of_one_resume_score_alike():
    """The .tex and its rendered preview are the same document. A seven-point
    spread between them was an artefact of which representation was passed in.

    The LaTeX here mirrors HTML exactly - same contact block, same skills, same
    bullets - because a fixture that quietly drops the header would compare two
    different documents and prove nothing.
    """
    tex = (
        r"\begin{center}" "\n"
        r"Jane Doe" "\n"
        r"jane@example.com \quad | \quad +1 555 010 0100 \quad | \quad Austin, TX"
        "\n"
        r"linkedin.com/in/jane \quad | \quad github.com/jane" "\n"
        r"\end{center}" "\n"
        r"\section*{Objective}" "\n"
        r"Backend engineer with six years of Python and PostgreSQL work." "\n"
        r"\section*{Technical Skills}" "\n"
        r"Languages: Python, SQL" "\n"
        r"Cloud: AWS, Docker" "\n"
        r"\section*{Experience}" "\n"
        r"Senior Engineer, Acme -- Mar 2023 -- Present" "\n"
        r"- Cut p99 latency 35% by adding caching to the FastAPI service" "\n"
        r"- Migrated PostgreSQL to a sharded schema, 40% faster" "\n"
        r"Engineer, Beta -- Jul 2021 -- Feb 2023" "\n"
        r"- Owned the REST API used by two internal clients" "\n"
        r"\section*{Education}" "\n"
        r"BSc Computer Science, 2019")
    a, b = A.score_resume(tex, JD), A.score_resume(HTML, JD)
    for part in ("contact", "sections", "dates"):
        assert a["breakdown"][part]["score"] == b["breakdown"][part]["score"], part
    assert abs(a["total"] - b["total"]) <= 4


def test_the_dates_component_can_actually_reach_its_maximum():
    """It is worth 16 and the old code stopped at 10, so a resume with perfect
    dates still looked 6 points short on the one component it had got right."""
    r = A.score_resume(HTML, JD)
    assert r["breakdown"]["dates"]["max"] == 16.0
    assert r["breakdown"]["dates"]["score"] == 16.0


def test_dated_roles_outscore_a_single_dated_role():
    """Coverage is what the remaining points are for: a parser reads a role
    with no date in it as a role it cannot place."""
    one = HTML.replace("Engineer, Beta -- Jul 2021 -- Feb 2023", "Beta")
    assert (A.score_resume(HTML, JD)["breakdown"]["dates"]["score"]
            > A.score_resume(one, JD)["breakdown"]["dates"]["score"])


def test_a_closed_timeline_loses_the_currency_points():
    r = A.score_resume(HTML.replace("-- Present", "-- 2024"), JD)
    assert any("Present" in i for i in r["issues"])
    assert r["breakdown"]["dates"]["score"] < 16.0


def test_html_entities_do_not_hide_a_heading():
    r = A.score_resume("<h2>Technical &amp; Other Skills</h2>Python", JD)
    assert "No 'skills' section" not in " ".join(r["issues"])


def test_a_repo_whose_terms_the_posting_never_uses_scores_zero():
    """The defect that let an unrelated project be selected.

    Repo technology counted for 1 point whether or not the JD mentioned it, so a
    restaurant-themed demo repository scored 5.5 against "Python RAG engineer" -
    enough to be picked and printed on the resume.
    """
    repo = _repo(name="pasta-recipes",
                 description="Restaurant menu planner in JavaScript")
    assert gp.score_repo(repo, "Python engineer with RAG and FastAPI") == 0.0


def test_the_relevant_repo_is_the_one_that_survives():
    jd = "Python engineer with RAG and FastAPI experience"
    relevant = _repo(name="rag-service", description="RAG service with FastAPI",
                     language="Python")
    noise = _repo(name="pasta-recipes",
                  description="Restaurant menu planner in JavaScript")
    ranked = sorted([(noise, gp.score_repo(noise, jd)),
                     (relevant, gp.score_repo(relevant, jd))],
                    key=lambda p: -p[1])
    assert ranked[0][0]["name"] == "rag-service"
