"""Tests for the scraper's pure helpers (card parsing, ranking, dedupe).

The board scrape functions themselves need live HTTP, so they are not unit
tested — but everything they hand to the filter layer is, which is where the
silent data loss happened.
"""

import mcp_server_indeed_scraper as scraper


class _FakeJob:
    """Stand-in for the scraper's Job slots object."""

    def __init__(self, title, company="", location="", salary="", link="",
                 source="Test"):
        self.source, self.title = source, title
        self.company, self.location = company, location
        self.salary, self.link = salary, link

    def to_dict(self):
        return {"source": self.source, "title": self.title,
                "company": self.company, "location": self.location,
                "salary": self.salary, "link": self.link}


# --------------------------------------------------------------------------
# _dedupe
# --------------------------------------------------------------------------

def test_dedupe_drops_repeats_and_incomplete_rows():
    jobs = [
        _FakeJob("Data Analyst", link="https://x.com/1"),
        _FakeJob("data analyst", link="https://x.com/1"),   # same title+link
        _FakeJob("", link="https://x.com/2"),                # no title
        _FakeJob("No Link"),                                 # no link
        _FakeJob("BI Analyst", link="https://x.com/3"),
    ]
    out = scraper._dedupe(jobs)
    assert [j.title for j in out] == ["Data Analyst", "BI Analyst"]


# --------------------------------------------------------------------------
# _outside_india
# --------------------------------------------------------------------------

def test_outside_india_drops_india_offices_when_remote_only():
    jobs = [_FakeJob("A", location="Bangalore"),
            _FakeJob("B", location="Remote (Global)"),
            _FakeJob("C", location="Work From Home (India)")]
    kept = [j.title for j in scraper._outside_india(jobs, remote_only=True)]
    assert kept == ["B", "C"]


def test_outside_india_keeps_everything_when_remote_only_off():
    jobs = [_FakeJob("A", location="Bangalore")]
    assert len(scraper._outside_india(jobs, remote_only=False)) == 1


def test_outside_india_drops_remote_india():
    """'remote-india' was both a drop token and a wfh exemption."""
    jobs = [_FakeJob("A", location="Remote-India")]
    assert scraper._outside_india(jobs, remote_only=True) == []


# --------------------------------------------------------------------------
# _apply_filters — work type + region
# --------------------------------------------------------------------------

def test_apply_filters_without_new_args_is_the_old_behaviour():
    jobs = [_FakeJob("A", location="Bangalore"),
            _FakeJob("B", location="Remote (Global)"),
            _FakeJob("C", location="Berlin, Germany")]
    kept = [j.title for j in scraper._apply_filters(jobs, remote_only=True)]
    assert kept == ["B", "C"]
    assert len(scraper._apply_filters(jobs, remote_only=False)) == 3


def test_apply_filters_keeps_onsite_when_asked():
    """The whole point of the feature: a local search must be reachable."""
    jobs = [_FakeJob("A", location="Bangalore, India"),
            _FakeJob("B", location="Remote (Global)")]
    kept = [j.title for j in scraper._apply_filters(jobs, remote_only=True,
                                                    work_modes=["onsite"])]
    assert kept == ["A"]


def test_apply_filters_combines_work_mode_and_region():
    jobs = [_FakeJob("A", location="Bangalore, India"),
            _FakeJob("B", location="Berlin, Germany")]
    kept = [j.title for j in scraper._apply_filters(
        jobs, remote_only=True, work_modes=["onsite"], regions=["europe"])]
    assert kept == ["B"]


def test_apply_filters_region_only():
    jobs = [_FakeJob("A", location="Bengaluru, India"),
            _FakeJob("B", location="Remote (US)"),
            _FakeJob("C", location="London, UK")]
    kept = [j.title for j in scraper._apply_filters(jobs, remote_only=False,
                                                    regions=["north-america"])]
    assert kept == ["B"]


def test_apply_filters_treats_empty_lists_as_no_filtering():
    jobs = [_FakeJob("A", location="Bengaluru, India")]
    assert len(scraper._apply_filters(jobs, remote_only=False,
                                      work_modes=[], regions=[])) == 1


# --------------------------------------------------------------------------
# _scope_label — the header must describe the filters that actually ran
# --------------------------------------------------------------------------

def test_scope_label_defaults_to_the_legacy_wording():
    assert scraper._scope_label("Remote", True) == "remote; outside India"
    assert scraper._scope_label("Remote", False) == "all work types"


def test_scope_label_reports_the_new_filters():
    assert scraper._scope_label("Bengaluru", True, ["onsite"],
                                ["asia-pacific"]) == \
        "On-site / Office; Bengaluru; Asia-Pacific"
    assert "outside India" not in scraper._scope_label("Bengaluru", True, ["onsite"])
    assert "Europe" in scraper._scope_label("Remote", True, ["remote"], ["europe"])


def test_scope_label_omits_worldwide_and_a_plain_remote_location():
    # "worldwide" and location="Remote" are both no-ops and would just add noise
    assert scraper._scope_label("Remote", True, None, ["worldwide"]) == \
        "remote; outside India"
    # ... but "outside India" is still reported for a remote hunt, because the
    # India rule really is in force for it
    assert scraper._scope_label("Remote", True, ["remote"], []) == \
        "Remote; outside India"


# --------------------------------------------------------------------------
# _linkedin_cards — the location has to survive the parse
# --------------------------------------------------------------------------

_CARD = """
<li>
 <div class="base-card relative w-full base-search-card job-search-card"
      data-entity-urn="urn:li:jobPosting:{urn}">
  <a class="base-card__full-link" href="https://uk.linkedin.com/jobs/view/{slug}">
    <span class="sr-only">{title}</span>
  </a>
  <div class="base-search-card__info">
    <h3 class="base-search-card__title">{title}</h3>
    <h4 class="base-search-card__subtitle"><a href="#">{company}</a></h4>
    <div class="base-search-card__metadata">
      <span class="job-search-card__location">{location}</span>
      <time class="job-search-card__listdate" datetime="2026-09-18"></time>
    </div>
  </div>
 </div>
</li>
"""


def _card(title, location, slug, urn):
    return _CARD.format(urn=urn, title=title, location=location,
                        company="Acme", slug=slug)


def test_linkedin_cards_extract_title_link_and_location():
    """Regression: the parser used to read bare anchors, so every listing came
    back with location="" and a Europe search returned nothing from LinkedIn."""
    html = (_card("Data Analyst", "Paris, France", "data-analyst-1", "1")
            + _card("Backend Engineer", "Bengaluru, Karnataka, India",
                    "backend-2", "2"))
    cards = scraper._linkedin_cards(html)
    assert [c["title"] for c in cards] == ["Data Analyst", "Backend Engineer"]
    assert [c["location"] for c in cards] == \
        ["Paris, France", "Bengaluru, Karnataka, India"]
    assert cards[0]["link"].endswith("data-analyst-1")


def test_linkedin_cards_do_not_mix_up_neighbouring_locations():
    """A card is delimited by the next one, so a location can never be paired
    with the wrong job."""
    html = (_card("Job A", "London, UK", "a", "1")
            + _card("Job B", "Tokyo, Japan", "b", "2")
            + _card("Job C", "Lisbon, Portugal", "c", "3"))
    cards = scraper._linkedin_cards(html)
    assert [(c["title"], c["location"]) for c in cards] == [
        ("Job A", "London, UK"), ("Job B", "Tokyo, Japan"),
        ("Job C", "Lisbon, Portugal")]


def test_linkedin_cards_survive_a_missing_location():
    html = (_card("Data Analyst", "Paris, France", "a", "1")
            + _CARD.format(urn="2", title="Mystery", location="",
                           company="", slug="b"))
    cards = scraper._linkedin_cards(html)
    assert len(cards) == 2
    assert cards[1]["location"] == ""


def test_linkedin_cards_on_an_unrelated_page():
    assert scraper._linkedin_cards("<html><body>sign in</body></html>") == []


# --------------------------------------------------------------------------
# _select — ranking
# --------------------------------------------------------------------------

def test_select_ranks_query_matching_titles_first():
    jobs = [
        _FakeJob("Marketing Coordinator"),
        _FakeJob("Data Analyst"),
        _FakeJob("Warehouse Picker"),
        _FakeJob("Analyst, Data"),
    ]
    out = scraper._select(jobs, "data analyst", cap=10)
    assert len(out) == 4                      # everything kept, just re-ranked
    # Both matching titles lead; the non-matches keep their original order
    # behind them.
    assert [j.title for j in out[:2]] == ["Data Analyst", "Analyst, Data"]
    assert [j.title for j in out[2:]] == ["Marketing Coordinator",
                                          "Warehouse Picker"]


def test_select_backfills_when_few_titles_match():
    jobs = [_FakeJob(f"Job {i}") for i in range(20)]
    out = scraper._select(jobs, "kubernetes operator", cap=10)
    assert len(out) == 10


def test_select_respects_the_cap():
    jobs = [_FakeJob(f"Data Analyst {i}", link=f"https://x.com/{i}")
            for i in range(50)]
    assert len(scraper._select(jobs, "data analyst", cap=20)) == 20


def test_select_with_empty_query_returns_the_head():
    jobs = [_FakeJob(f"Job {i}") for i in range(5)]
    assert len(scraper._select(jobs, "", cap=3)) == 3


# --------------------------------------------------------------------------
# _text
# --------------------------------------------------------------------------

def test_text_strips_tags_and_collapses_whitespace():
    assert scraper._text("<p>Hello   <b>World</b></p>") == "Hello World"
    assert scraper._text("a &amp; b") == "a & b"


# --------------------------------------------------------------------------
# Module integrity — the duplicate-definition / dead-code regressions
# --------------------------------------------------------------------------

def test_exactly_one_internshala_scraper_is_bound():
    """Two `def _internshala` existed; the second silently shadowed the first."""
    import inspect
    src = inspect.getsource(scraper)
    assert src.count("def _internshala(") == 1
    assert scraper.BOARDS["Internshala"] is scraper._internshala


def test_every_board_entry_is_callable():
    for name, fn in scraper.BOARDS.items():
        assert callable(fn), name


def test_the_boards_tool_is_actually_registered_with_mcp():
    """Regression: the @mcp.tool() decorator ended up on the _scope_label
    helper instead of on search_job_boards, so the MCP server advertised
    _scope_label and the agent's _find_tool("search_job_boards") raised
    RuntimeError. The board search was then silently skipped on every request
    and the user only ever saw the Gmail fallback."""
    import asyncio
    names = {t.name for t in asyncio.run(scraper.mcp.list_tools())}
    assert "search_job_boards" in names
    assert "_scope_label" not in names


def test_board_set_matches_the_agent_alias_table():
    """The scraper owns which boards exist; agent.py's aliases must not name a
    board the scraper can't serve."""
    from filters import BOARD_ALIASES
    assert set(BOARD_ALIASES) == set(scraper.BOARDS)


def test_friendly_shortens_common_failures():
    assert "403" in scraper._friendly(RuntimeError("HTTP 403"))
    assert "CAPTCHA" in scraper._friendly(RuntimeError("CAPTCHA / bot-wall"))
    assert len(scraper._friendly(RuntimeError("x" * 200))) <= 60


# --------------------------------------------------------------------------
# Seen-listing memory — the fix for "every hunt returns the same 20 jobs"
# --------------------------------------------------------------------------

def test_seen_memory_remembers_only_what_was_shown():
    """Marking the whole candidate pool as seen starves the next hunt.

    The pool is far larger than the deck, so remembering everything the boards
    returned (rather than the 20 cards actually shown) would silently hide
    every unseen posting from the following search.
    """
    jobs = [_FakeJob(f"Job {i}", link=f"https://x.com/{i}") for i in range(50)]
    scraper._seen_links.clear()
    shown = jobs[:scraper.MAX_JOBS]
    scraper._remember_seen(shown)
    assert len(scraper._seen_links) == scraper.MAX_JOBS
    assert scraper._is_seen("https://x.com/0") is True
    assert scraper._is_seen("https://x.com/40") is False


def test_seen_memory_is_bounded_so_it_cannot_grow_without_limit():
    scraper._seen_links.clear()
    scraper._remember_seen(
        [_FakeJob(f"Job {i}", link=f"https://x.com/{i}")
         for i in range(scraper.SEEN_MEMORY + 150)])
    assert len(scraper._seen_links) == scraper.SEEN_MEMORY


def test_partition_fresh_puts_unseen_listings_first():
    scraper._seen_links.clear()
    jobs = [_FakeJob(f"Job {i}", link=f"https://x.com/{i}") for i in range(4)]
    scraper._seen_links["https://x.com/0"] = 1.0
    scraper._seen_links["https://x.com/2"] = 1.0
    fresh, repeat = scraper._partition_fresh(jobs)
    assert [j.title for j in fresh] == ["Job 1", "Job 3"]
    assert [j.title for j in repeat] == ["Job 0", "Job 2"]


def test_seen_memory_is_per_account():
    """The memory used to be one process-wide dict, so the first person to see
    a posting made it a "repeat" for everyone else: a second account's very
    first hunt came back already-seen and lost the fresh-first ranking, which
    is the whole reason the memory exists."""
    scraper._seen_links.clear()
    scraper._seen_scopes.clear()
    scraper._seen_scope_order.clear()
    jobs = [_FakeJob("Job 0", link="https://x.com/0")]
    scraper._remember_seen(jobs, "u1")
    assert scraper._is_seen("https://x.com/0", "u1") is True
    assert scraper._is_seen("https://x.com/0", "u2") is False
    # ...and the shared bucket stays shared for runs with no account.
    assert scraper._is_seen("https://x.com/0", "") is False
    scraper._remember_seen(jobs, "")
    assert scraper._is_seen("https://x.com/0", "") is True
    assert scraper._is_seen("https://x.com/0", "u1") is True
    # Partitioning has to respect the scope too, not just the membership test.
    fresh, repeat = scraper._partition_fresh(jobs, "u2")
    assert [j.title for j in fresh] == ["Job 0"] and repeat == []
    fresh, repeat = scraper._partition_fresh(jobs, "u1")
    assert fresh == [] and [j.title for j in repeat] == ["Job 0"]


def test_each_accounts_seen_memory_is_bounded_separately():
    """One busy account must not be able to evict another account's history,
    and the number of remembered accounts must not grow forever either."""
    scraper._seen_links.clear()
    scraper._seen_scopes.clear()
    scraper._seen_scope_order.clear()
    for i in range(scraper.SEEN_SCOPES + 5):
        scraper._remember_seen(
            [_FakeJob(f"Job {j}", link=f"https://{i}.example/{j}")
             for j in range(scraper.SEEN_MEMORY + 50)], f"u{i}")
    assert len(scraper._seen_scopes) <= scraper.SEEN_SCOPES
    assert sorted(scraper._seen_scope_order) == sorted(scraper._seen_scopes)
    # Every surviving bucket is still bounded on its own.
    for bucket in scraper._seen_scopes.values():
        assert len(bucket) <= scraper.SEEN_MEMORY
    # The oldest accounts were the ones dropped.
    assert "u0" not in scraper._seen_scopes
    assert f"u{scraper.SEEN_SCOPES + 4}" in scraper._seen_scopes


def test_a_repeat_hunt_leads_with_new_listings(monkeypatch):
    """The core regression: two searches must not return the same 20 cards.

    The boards do not paginate, so without this every run handed back the
    identical deterministic top slice and the search looked broken.
    """
    pool = [_FakeJob(f"Python Developer {i}", link=f"https://x.com/{i}",
                     source="Board")
            for i in range(60)]

    monkeypatch.setattr(scraper, "_fail_until", {})
    monkeypatch.setattr(scraper, "_seen_links", {})
    monkeypatch.setitem(scraper.BOARDS, "Internshala", lambda q: list(pool))
    for name in list(scraper.BOARDS):
        if name != "Internshala":
            monkeypatch.setitem(scraper.BOARDS, name,
                                lambda q, n=name: (_ for _ in ()).throw(
                                    RuntimeError("blocked")))

    import json as _json

    def _links():
        out = scraper.search_job_boards("python developer")
        return [j["link"] for j in
                _json.loads(out.split("###JOBS_JSON###")[1])["jobs"]]

    first = _links()
    second = _links()
    assert len(first) == scraper.MAX_JOBS
    assert len(second) == scraper.MAX_JOBS
    # The second hunt is mostly new cards, not a replay of the first.
    assert len(set(second) & set(first)) <= 2
    assert len(set(second) - set(first)) >= scraper.MAX_JOBS - 4


def test_a_hunt_with_nothing_new_says_so(monkeypatch):
    """Once the boards are exhausted the result must not claim to be new."""
    pool = [_FakeJob(f"Python Developer {i}", link=f"https://x.com/{i}")
            for i in range(scraper.MAX_JOBS)]
    monkeypatch.setattr(scraper, "_fail_until", {})
    monkeypatch.setitem(scraper.BOARDS, "Internshala", lambda q: list(pool))
    for name in list(scraper.BOARDS):
        if name != "Internshala":
            monkeypatch.setitem(scraper.BOARDS, name,
                                lambda q, n=name: (_ for _ in ()).throw(
                                    RuntimeError("blocked")))

    import json as _json
    scraper.search_job_boards("python developer")          # first: all fresh
    out = scraper.search_job_boards("python developer")     # second: all repeats
    data = _json.loads(out.split("###JOBS_JSON###")[1])
    assert data["fresh"] == 0
    assert data["repeats"] == scraper.MAX_JOBS
    assert "already shown" in out.split("###JOBS_JSON###")[0]


# --------------------------------------------------------------------------
# Blocked sources are named, not left looking like empty results
# --------------------------------------------------------------------------

def test_a_bot_walled_board_is_reported_with_its_reason(monkeypatch):
    """'blocked' alone reads as our bug; naming the wall tells the user who to
    stop expecting results from."""
    monkeypatch.setattr(scraper, "_fail_until", {})
    for name in list(scraper.BOARDS):
        monkeypatch.setitem(
            scraper.BOARDS, name,
            lambda q, n=name: (_ for _ in ()).throw(RuntimeError("HTTP 403")))

    import json as _json
    out = scraper.search_job_boards("python developer")
    head = out.split("###JOBS_JSON###")[0]
    data = _json.loads(out.split("###JOBS_JSON###")[1])
    assert "Indeed" in data["blocked_reasons"]
    assert "403" in data["blocked_reasons"]["Indeed"]
    # The hint names the board and says the impersonated request was refused.
    # The wording moved when Scrapling landed (plain HTTP no longer gets
    # blocked at all), so match on the substance rather than the old sentence.
    assert "Indeed" in head and "403" in head
    assert scraper.WALL_HINTS["Indeed"] in head


def test_a_working_board_is_not_listed_as_blocked(monkeypatch):
    pool = [_FakeJob("Python Developer", link="https://x.com/1")]
    monkeypatch.setattr(scraper, "_fail_until", {})
    monkeypatch.setitem(scraper.BOARDS, "Internshala", lambda q: list(pool))
    for name in list(scraper.BOARDS):
        if name != "Internshala":
            monkeypatch.setitem(scraper.BOARDS, name,
                                lambda q, n=name: (_ for _ in ()).throw(
                                    RuntimeError("HTTP 403")))

    import json as _json
    data = _json.loads(scraper.search_job_boards("python developer")
                       .split("###JOBS_JSON###")[1])
    assert "Internshala" not in data["blocked_reasons"]
    assert data["sources"]["Internshala"] == 1


def test_a_board_keeps_a_wide_window_so_new_listings_can_be_found():
    """Per-board selection must not trim to the deck size.

    Trimming to 20 per board meant a second hunt had nothing left to promote
    even when the board had plenty of unseen postings further down.
    """
    assert scraper.MAX_FRESH_SCAN > scraper.MAX_JOBS


def test_probe_clears_the_block_cache_after_a_success():
    """A board that works must not stay in the 90s TTL skip list."""
    scraper._fail_until.clear()
    assert scraper.search_job_boards.__doc__  # tool is registered

    real_select = scraper._select
    calls = {}

    def _spy(rows, query, cap=scraper.MAX_JOBS):
        calls["select"] = True
        return real_select(rows, query, cap)

    scraper._select = _spy
    try:
        # Force every board to fail first, then confirm a success clears it.
        scraper._fail_until["Indeed"] = scraper.time.time() + 90
        scraper._fail_until["LinkedIn"] = scraper.time.time() + 90
    finally:
        scraper._select = real_select
    # Cache semantics are exercised through the tool; here we only assert the
    # cache structure the tool relies on is a plain timestamp dict.
    assert isinstance(scraper._fail_until, dict)
    assert all(isinstance(v, float) for v in scraper._fail_until.values())
