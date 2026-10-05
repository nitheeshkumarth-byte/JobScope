"""Tests for the agent-side helpers: query building, payload handling and the
summary template.

These cover the pieces of agent.py that are pure functions — the graph itself
needs Ollama and a live MCP scrape, so it is exercised by `python agent.py`
instead.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from agent import (AgentConfig, JOBS_MARKER, _blocked_marker,
                   _board_role_phrase, _extract_struct_from,
                   _fallback_node_note, _fallback_note, _keep_job,
                   _silent_senders, _summary_payload, _template_summary,
                   _text_content, board_tool_args, default_task_messages,
                   is_hunt_request, requested_board, requested_source,
                   search_query_from, system_message)


def _cfg(**over):
    base = AgentConfig()
    base.__dict__.update(over)
    return base


def _state(*texts, tools=()):
    """A minimal AgentState stand-in: the graph only ever reads `messages`."""
    messages = [HumanMessage(content=t) for t in texts]
    messages += [ToolMessage(content=json.dumps(t), name=n, tool_call_id=str(i))
                 for i, (n, t) in enumerate(tools)]
    return {"messages": messages}


# --------------------------------------------------------------------------
# search_query_from
# --------------------------------------------------------------------------

def test_query_is_the_role_when_no_skills():
    assert search_query_from(_cfg(target_role="Data Analyst")) == "Data Analyst"


def test_query_keeps_at_most_two_skills():
    """Five skills stuffed into a board query returns worse results than the
    role alone, and then get ranked by any single matching word."""
    cfg = _cfg(target_role="Data Analyst",
               target_roles=["Data Analyst"],
               skills=["python", "sql", "pandas", "excel", "power bi"])
    query = search_query_from(cfg)
    assert query.startswith("Data Analyst")
    # role + at most 2 skills
    assert query == "Data Analyst power bi python"
    for dropped in ("sql", "pandas", "excel"):
        assert dropped not in query


def test_query_prefers_multiword_skills():
    """'power bi' narrows a search far more than the one-word 'excel'."""
    cfg = _cfg(target_role="Data Analyst", target_roles=["Data Analyst"],
               skills=["excel", "power bi", "tableau"])
    assert search_query_from(cfg) == "Data Analyst power bi tableau"


def test_query_skips_skills_already_in_the_role():
    cfg = _cfg(target_role="Data Analyst", target_roles=["Data Analyst"],
               skills=["data", "analyst", "sql"])
    assert search_query_from(cfg) == "Data Analyst sql"


def test_query_survives_empty_and_garbage_skills():
    cfg = _cfg(target_role="Data Analyst", target_roles=["Data Analyst"],
               skills=["", "  ", None, "ab"])
    assert search_query_from(cfg) == "Data Analyst"


def test_query_with_no_role_never_sends_the_baked_in_example_role():
    """A session with no roles must not silently hunt for the demo role.

    Fresh accounts start with an empty role list, and falling back to
    AgentConfig.target_role ("Junior Data Analyst / entry-level...") made every
    new user search somebody else's job title.
    """
    cfg = _cfg(target_role="", target_roles=[])
    query = search_query_from(cfg)
    assert query
    assert "Analyst" not in query
    assert "Junior" not in query


def test_query_uses_the_users_own_role_when_they_have_one():
    cfg = _cfg(target_role="", target_roles=["Data Engineer"])
    assert search_query_from(cfg) == "Data Engineer"


# --------------------------------------------------------------------------
# _board_role_phrase — a role with notes on it matches nothing
# --------------------------------------------------------------------------

def test_a_trailing_qualifier_is_not_searched_for():
    """These role strings are built by the CV extractor, so they carry notes.

    Handed to a board verbatim, "Python Developer / entry-level, remote" is one
    search term that matches no posting at all, because boards AND every term
    in the query.
    """
    assert _board_role_phrase("Python Developer / entry-level, remote") \
        == "Python Developer"
    assert _board_role_phrase("Machine Learning Engineer - Remote") \
        == "Machine Learning Engineer"
    assert _board_role_phrase("Full Stack Developer, Intern") \
        == "Full Stack Developer"
    assert _board_role_phrase("Data Analyst | entry level") == "Data Analyst"


def test_a_bracketed_note_is_never_searched_for():
    assert _board_role_phrase("Senior Software Engineer (backend)") \
        == "Senior Software Engineer"
    assert _board_role_phrase("QA Automation Engineer [contract]") \
        == "QA Automation Engineer"
    assert _board_role_phrase("Business Analyst (Remote - US only)") \
        == "Business Analyst"


def test_a_role_title_is_never_truncated():
    """The trim is trailing-qualifier removal, not "cut at the first space".

    Dropping everything after the first separator destroyed real multi-word
    titles and the second half of compound ones.
    """
    assert _board_role_phrase("Data Scientist") == "Data Scientist"
    assert _board_role_phrase("Senior Software Engineer") \
        == "Senior Software Engineer"
    assert _board_role_phrase("Full-Stack Developer") == "Full-Stack Developer"
    # A separator inside the title is an alternate spelling, not a note.
    assert _board_role_phrase("DevOps/Cloud Engineer") == "DevOps Cloud Engineer"
    assert _board_role_phrase("C# / .NET Developer") == "C# .NET Developer"


def test_a_level_requirement_is_not_part_of_the_title():
    assert _board_role_phrase("React Developer, 2+ years experience") \
        == "React Developer"
    assert _board_role_phrase("Data Analyst, 0-1 years") == "Data Analyst"
    # ...but a level word inside a real title is kept.
    assert _board_role_phrase("Senior Data Analyst") == "Senior Data Analyst"


def test_cv_typos_are_corrected_so_the_query_still_hits():
    """A misspelled skill or role finds nothing at all on a board."""
    assert _board_role_phrase("Java Devloper") == "Java Developer"
    assert _board_role_phrase("Python Phyton Developer") == "Python Developer"
    assert _board_role_phrase("Fullstack Developer") == "Full stack Developer"


def test_an_unrecognised_word_is_never_rewritten():
    """Only known slips are corrected; a real title word is passed through."""
    assert _board_role_phrase("Rust Engineer") == "Rust Engineer"
    assert _board_role_phrase("Kubernetes Operator") == "Kubernetes Operator"
    # Capitalisation is the user's, not the correction's.
    assert _board_role_phrase("JAVA DEVLOPER") == "JAVA DEVELOPER"


def test_a_role_that_is_only_a_constraint_survives():
    """Nothing to trim must not produce an empty query."""
    assert _board_role_phrase("Remote")
    assert _board_role_phrase("Hybrid, Remote")
    assert _board_role_phrase("") == ""


def test_the_annotated_role_reaches_the_board_as_a_clean_title():
    cfg = _cfg(target_role="Python Developer / entry-level, remote",
               target_roles=["Python Developer / entry-level, remote"],
               skills=["python", "docker", "excel", "fastapi", "aws"])
    # Two distinctive skills, multi-word/longer first; the role is clean.
    assert search_query_from(cfg) == "Python Developer fastapi docker"


def test_the_agent_asks_instead_of_searching_when_no_role_is_set():
    """A fresh account has no role, and the agent must not invent one."""
    cfg = _cfg(target_role="", target_roles=[])
    msg = system_message(cfg).content
    assert "none set yet" in msg
    assert "Junior Data Analyst" not in msg


# --------------------------------------------------------------------------
# _text_content — MCP returns content as a list of blocks, not a str
# --------------------------------------------------------------------------

def test_text_content_normalizes_block_lists():
    assert _text_content([{"type": "text", "text": "hello"}]) == "hello"
    assert _text_content("plain") == "plain"
    assert _text_content([{"name": "x"}]) == "x"
    assert _text_content(["a", "b"]) == "a\nb"
    assert _text_content(None) == "None"


# --------------------------------------------------------------------------
# keep_job
# --------------------------------------------------------------------------

def test_keep_job_uses_cfg_remote_only():
    cfg = _cfg(remote_only=True)
    assert not _keep_job({"title": "Data Analyst", "location": "Pune",
                          "link": "https://x.com/1"}, cfg)
    assert _keep_job({"title": "Data Analyst", "location": "Pune",
                      "link": "https://x.com/1"}, _cfg(remote_only=False))


# --------------------------------------------------------------------------
# work_modes plumbing. The region filter and the legacy remote_outside_india
# flag were both retired: the settings UI no longer offers them and nothing
# forwards them, so a stale value in an old profile must not quietly narrow a
# search the user can no longer see.
# --------------------------------------------------------------------------

def _job(loc):
    return {"title": "Data Analyst", "location": loc, "link": "https://x.com/1"}


def test_keep_job_keeps_an_india_tied_role_by_default():
    """Nothing drops India-tied listings unless the user asked for it."""
    cfg = _cfg()
    assert _keep_job(_job("Pune, India"), cfg) is True
    assert _keep_job(_job("Remote, India"), cfg) is True


def test_keep_job_keeps_everywhere_when_no_country_is_picked():
    cfg = _cfg(remote_only=False, work_modes=[], countries=[])
    assert _keep_job(_job("Pune, India"), cfg)
    assert _keep_job(_job("Remote, US"), cfg)
    assert _keep_job(_job("Berlin, Germany"), cfg)


def test_keep_job_passes_the_country_filter_from_cfg():
    cfg = _cfg(remote_only=False, countries=["Germany"])
    assert _keep_job(_job("Dülmen, Germany"), cfg) is True
    assert _keep_job(_job("Tokyo, Japan"), cfg) is False
    # a worldwide listing is in every country
    assert _keep_job(_job("Remote, Worldwide"), cfg) is True


def test_board_tool_args_carries_the_country_filter():
    cfg = _cfg(regions=["europe"], countries=["Germany", "Japan"])
    args = board_tool_args(cfg, "")
    assert args["countries"] == ["Germany", "Japan"]


def test_board_tool_args_never_sends_a_retired_region():
    """A region left in a saved profile must not reach the scraper."""
    args = board_tool_args(_cfg(regions=["europe", "asia-pacific"]), "")
    assert "regions" not in args


def test_board_tool_args_omits_countries_when_unset():
    # the scraper treats a missing filter as "don't filter", so an unset
    # country list must stay out of the call entirely
    assert "countries" not in board_tool_args(_cfg(), "")


def test_system_message_mentions_the_country_filter():
    msg = system_message(_cfg(countries=["germany"])).content
    assert "Country wanted: Germany" in msg


def test_search_uses_the_country_when_there_is_no_city():
    # on-site + no city: the country becomes the place the boards are pointed at
    args = board_tool_args(_cfg(work_modes=["onsite"], remote_only=False,
                                preferred_cities=[], countries=["Poland"]), "")
    assert args["location"] == "Poland"


def test_a_city_beats_a_country_for_the_boards_location():
    # a city is more specific, so it is what the boards are pointed at
    args = board_tool_args(_cfg(work_modes=["onsite"], remote_only=False,
                                preferred_cities=["Krakow"],
                                countries=["Poland"]), "")
    assert args["location"] == "Krakow"


def test_keep_job_handles_a_missing_cfg():
    assert _keep_job(_job("Remote, US"), None) is True


def test_agent_config_defaults_to_the_documented_defaults():
    cfg = AgentConfig()
    assert cfg.work_modes == []      # [] means "don't filter", not "remote only"
    assert cfg.regions == []
    # The retired flag used to default to True, which dropped every India-tied
    # posting with nothing on screen asking for it.
    assert cfg.remote_only is False


def test_system_message_states_the_selected_filters():
    """The model has to know the user's constraints, otherwise it re-adds the
    very listings the filters removed."""
    text = system_message(_cfg(target_role="Data Analyst", work_modes=["onsite", "hybrid"],
                               countries=["Poland"])).content
    assert "On-site / Office" in text and "Hybrid" in text
    assert "Poland" in text


def test_system_message_never_claims_the_user_lives_in_india():
    """Geography comes from the user's own filters, not from an assumption."""
    text = system_message(_cfg()).content
    assert "lives in India" not in text
    assert "NOT based in India" not in text


def test_board_tool_args_carries_the_filters_to_the_scraper():
    """If these don't reach the tool, the chips change the UI and nothing else."""
    args = board_tool_args(_cfg(target_role="Data Analyst",
                                work_modes=["onsite"],
                                preferred_cities=["Bengaluru"]))
    assert args["work_modes"] == ["onsite"]
    assert args["location"] == "Bengaluru"      # on-site -> a real place to search


def test_board_tool_args_omits_unset_filters():
    """An unconfigured profile must produce the exact pre-feature call."""
    args = board_tool_args(_cfg(target_role="Data Analyst"))
    assert "work_modes" not in args and "regions" not in args
    # "Remote" is this app's default search scope and is unchanged. What is
    # gone is the India rule riding along with it: remote_only now reports
    # False, so nothing is dropped for being India-tied.
    assert args["location"] == "Remote"
    assert args["remote_only"] is False


def test_an_unqualified_search_no_longer_drops_india_tied_listings():
    """The bug this closes: remote_only defaulted to True, so every search
    silently discarded India-tied postings with no control on screen for it."""
    args = board_tool_args(_cfg(target_role="Data Analyst"), "")
    assert _keep_job(_job("Pune, India"), _cfg()) is True
    assert args["remote_only"] is False


def test_board_tool_args_still_detects_a_named_board():
    args = board_tool_args(_cfg(target_role="Data Analyst"), "only linkedin please")
    assert args["board"] == "LinkedIn"


def test_the_seen_memory_is_scoped_to_the_account():
    """The board scraper keys its "already shown" memory so one person's hunt
    does not make every posting a repeat for everybody else. The scope has to
    reach the tool, or the fresh-first ranking is a shared global."""
    # A bare run with no account keeps the shared bucket: the tool's own
    # default, so the call is unchanged for CLI users.
    assert "seen_scope" not in board_tool_args(_cfg(target_role="Data Analyst"))
    args = board_tool_args(_cfg(target_role="Data Analyst", seen_scope="u7"))
    assert args["seen_scope"] == "u7"


@pytest.mark.parametrize("text", [
    "search the gmail alerts",
    "check my inbox for jobs",
    "just email me the matches",
    "job alerts from naukri",
    "send me the emails",
    "look in my mail",
    "gmail jobs",
])
def test_gmail_is_searched_only_when_the_user_asks_for_it(text):
    assert requested_source(_state(text)) == "gmail"


@pytest.mark.parametrize("text", [
    "find python jobs",
    "only linkedin",
    "remote work please",
    "look on cutshort",
    "software engineer roles",
    "show me openings",
    "naukri and linkedin",
])
def test_a_normal_hunt_searches_the_boards_only(text):
    """No Gmail search unless it was requested.

    The old fan-out fired the board scraper AND both alert senders on every
    hunt, which is what the user saw as "two job search emails running" - and
    the Gmail results were computed even when the boards answered, then thrown
    away by the prep node.
    """
    assert requested_source(_state(text)) == "boards"


def test_a_bare_job_word_is_not_read_as_a_request_for_email():
    """"job" alone must not trigger Gmail, or every hunt becomes an email hunt."""
    for text in ("find me a job", "job openings", "new jobs today", "jobs"):
        assert requested_source(_state(text)) == "boards"


def test_board_tool_args_searches_the_role_not_the_board_keyword():
    """Typing "linkedin" picks the source; it must not become the query, or the
    scraper searches LinkedIn for the literal word "linkedin"."""
    args = board_tool_args(_cfg(target_role="Data Analyst"), "linkedin")
    assert args["board"] == "LinkedIn"
    assert args["query"] == "Data Analyst"


def test_a_bare_board_name_counts_as_a_hunt_request():
    """Regression: "linkedin" matched no verb in HUNT_HINT, so the deterministic
    fan-out was skipped and the LLM called the Gmail tool instead - the user saw
    no job cards."""
    assert is_hunt_request("linkedin")
    assert is_hunt_request("LinkedIn")
    assert is_hunt_request("only linkedin please")
    assert is_hunt_request("find remote jobs")
    # still not a hunt
    assert not is_hunt_request("thanks, that was helpful")
    assert not is_hunt_request("")


def test_first_user_message_points_the_tool_at_the_right_place():
    """An on-site hunt has to search a city, not the literal string 'Remote'."""
    assert "'Bengaluru'" in default_task_messages(
        _cfg(target_role="Data Analyst", work_modes=["onsite"],
             preferred_cities=["Bengaluru"]))[1][1]
    assert "'Remote'" in default_task_messages(
        _cfg(target_role="Data Analyst", work_modes=["remote"]))[1][1]


# --------------------------------------------------------------------------
# JOBS_JSON round-trip
# --------------------------------------------------------------------------

def test_extract_struct_from_parses_the_marker_block():
    payload = {"total": 1, "jobs": [{"title": "T", "link": "https://x.com/1"}]}
    text = "human summary\n\n" + JOBS_MARKER + "\n" + json.dumps(payload)
    assert _extract_struct_from(text) == payload["jobs"]


def test_extract_struct_from_tolerates_garbage():
    assert _extract_struct_from("no marker here") == []
    assert _extract_struct_from(JOBS_MARKER + "\nnot json") == []


def test_summary_payload_round_trips():
    jobs = [{"title": "Data Analyst", "company": "Acme", "location": "Remote",
             "source": "Remotive", "link": "https://x.com/1"}]
    content = _summary_payload(jobs, "note")
    assert content.startswith("note")
    assert JOBS_MARKER in content
    assert _extract_struct_from(content) == jobs


def test_template_summary_renders_every_job_with_its_link():
    jobs = [
        {"title": "Data Analyst", "company": "Acme", "location": "Remote",
         "source": "Remotive", "link": "https://x.com/1"},
        {"title": "BI Analyst", "company": "", "location": "Berlin",
         "source": "Arbeitnow", "link": "https://x.com/2"},
    ]
    out = _template_summary(_summary_payload(jobs, "note"))
    assert "Data Analyst" in out and "Acme" in out
    assert "https://x.com/1" in out and "https://x.com/2" in out
    assert "2 matching role(s)" in out
    assert JOBS_MARKER not in out   # marker must not leak into the UI


def test_template_summary_handles_no_jobs():
    out = _template_summary(_summary_payload([], "note"))
    assert "no matching listings" in out


def test_template_summary_survives_malformed_payload():
    assert _template_summary("plain text, no marker") == "plain text, no marker"


# --------------------------------------------------------------------------
# Why an empty hunt came back empty. An empty result and a blocked board are
# different problems with different fixes, and the note must not blame a block
# that never happened.
# --------------------------------------------------------------------------

def test_blocked_marker_reads_a_real_wall():
    assert _blocked_marker("Indeed returned HTTP 403") == "http 403"
    assert _blocked_marker("all boards were blocked by a captcha") == \
        "all boards were blocked"


def test_blocked_marker_prefers_the_structured_blocked_list():
    text = ("Job search across boards:\n[TOTAL] 0 listing(s)\n" + JOBS_MARKER +
            json.dumps({"blocked": ["Naukri", "Glassdoor"],
                        "blocked_reasons": {"Naukri": "needs a login",
                                            "Glassdoor": "login wall"}}))
    assert _blocked_marker(text) == "needs a login"


def test_blocked_marker_names_every_blocked_board_without_reasons():
    text = JOBS_MARKER + json.dumps({"blocked": ["Indeed", "LinkedIn"]})
    assert _blocked_marker(text) == "Indeed, LinkedIn"


def test_blocked_marker_stays_quiet_when_the_payload_lists_no_block():
    text = ("[Indeed] 0 (nothing matched this query)\n[TOTAL] 0 listing(s)\n"
            + JOBS_MARKER + json.dumps(
                {"blocked": [], "sources": {"Indeed": 0}}))
    assert _blocked_marker(text) == ""


def test_blocked_marker_stays_quiet_when_boards_merely_had_no_matches():
    assert _blocked_marker("[TOTAL] 0 listing(s)\nNo job links found.") == \
        "no job links"  # prose fallback still catches a raised-tool message


def test_blocked_marker_stays_quiet_on_a_healthy_but_empty_scrape():
    assert _blocked_marker("Job search across boards (all work types):\n"
                           "[Indeed] 0\n[TOTAL] 0 listing(s)") == ""


def test_silent_senders_names_the_alert_sources_that_were_silent():
    texts = [
        "No emails found from 'careers-noreply@google.com' in the last 60 days.",
        "No emails found from 'noreply@other.com' in the last 60 days.",
    ]
    assert _silent_senders(texts) == ["careers-noreply@google.com",
                                      "noreply@other.com"]


def test_silent_senders_is_empty_when_an_alert_carried_listings():
    assert _silent_senders(["--- Subject: Data Analyst | ... ---"]) == []


def test_silent_senders_dedupes_a_repeated_sender():
    text = "No emails found from 'a@b.com' in the last 60 days."
    assert _silent_senders([text, text]) == ["a@b.com"]


# --------------------------------------------------------------------------
# The note itself. The bug these lock down: the old fallback always announced
# "here is fallback data from Gmail job-alert emails" and then, one line later,
# printed "(no matching listings found)".
# --------------------------------------------------------------------------

def test_note_promises_gmail_only_when_gmail_actually_has_listings():
    note, blocked = _fallback_note("", [], gmail_total=0, gmail_kept=2)
    assert "fallback data from Gmail" in note
    assert "https link exactly as given" in note
    assert blocked == ""


def test_note_never_promises_gmail_data_it_does_not_have():
    note, _ = _fallback_note("", [], gmail_total=0, gmail_kept=0)
    assert "fallback data from Gmail" not in note
    assert "Nothing came back" in note
    assert "Do not invent listings or links" in note


def test_note_names_the_alert_source_that_was_silent():
    note, _ = _fallback_note("", ["careers-noreply@google.com"],
                             gmail_total=0, gmail_kept=0)
    assert "no job-alert email arrived from careers-noreply@google.com" in note
    assert "Nothing came back" in note


def test_note_distinguishes_filtered_from_silent_gmail():
    note, _ = _fallback_note("", [], gmail_total=3, gmail_kept=0)
    assert "3 Gmail listing(s) were all filtered out" in note
    assert "no job-alert email arrived" not in note


def test_note_reports_a_real_block_rather_than_guessing_one():
    note, blocked = _fallback_note("http 403", [], 0, 0)
    assert "boards were blocked (http 403)" in note
    assert blocked == "http 403"


def test_note_does_not_claim_a_block_when_the_boards_merely_ran_empty():
    note, blocked = _fallback_note("", [], 0, 0)
    assert "were blocked" not in note
    assert blocked == ""


def test_note_says_boards_were_empty_rather_than_blocked_when_gmail_has_data():
    note, _ = _fallback_note("", [], gmail_total=4, gmail_kept=1)
    assert "came back empty" in note
    assert "were blocked" not in note


def test_note_reports_block_and_silence_together():
    note, _ = _fallback_note("captcha", ["a@b.com"], 0, 0)
    assert "the boards were blocked (captcha)" in note
    assert "no job-alert email arrived from a@b.com" in note


def test_legacy_fallback_note_is_honest_when_gmail_is_empty():
    note = _fallback_node_note(["a@b.com"], 0)
    assert "fallback job data" not in note
    assert "no job-alert email arrived from a@b.com" in note
    assert "Do not invent listings or links" in note


def test_legacy_fallback_note_reports_filtered_listings():
    note = _fallback_node_note([], 4)
    assert "4 listing(s) the alerts contained were all filtered out" in note


# --------------------------------------------------------------------------
# "only <board>" means only that board, from every source
# --------------------------------------------------------------------------

def test_a_named_board_is_recognised_from_the_users_own_words():
    assert requested_board(_state("find backend jobs on linkedin only")) == "LinkedIn"
    assert requested_board(_state("only indeed please")) == "Indeed"
    assert requested_board(_state("hunt for python roles")) == ""


def test_a_named_board_reaches_the_scraper_as_the_board_argument():
    args = board_tool_args(_cfg(), "only linkedin jobs please")
    assert args["board"] == "LinkedIn"


def test_an_unrestricted_hunt_leaves_the_board_unset():
    """Omitting it is what makes the scraper fan out over every board."""
    assert "board" not in board_tool_args(_cfg(), "find me a python job")


def test_a_restricted_payload_says_why_it_is_empty_instead_of_going_elsewhere():
    """The fallback reads job-alert emails from several boards at once, so
    falling back after a named board came back empty answers a question the
    user did not ask. The reason has to survive into the deck payload or the
    UI can only say "nothing found"."""
    payload = _summary_payload([], "note", blocked_reason="linkedin: did not respond")
    assert _extract_struct_from(payload) == []
    assert "linkedin: did not respond" in payload
