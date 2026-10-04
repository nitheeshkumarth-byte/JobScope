"""Tests for the Scrapling-backed board layer (boards.py) and its wiring.

Nothing here touches the network. The parser tests run against saved HTML
trimmed from what each board actually served on 2026-10-01, because a parser
tested only against hand-written markup merely proves it can parse the test.

The fixtures carry the weight: every board in this file is matched with
selectors over class names the *board* chooses, so these are really assertions
of the form "if the markup looks like this, we still get a listing out".
"""

import json as _json
import types

import pytest
from scrapling.parser import Selector

import boards
import mcp_server_indeed_scraper as scraper


def sel(html):
    return Selector(html)


# --------------------------------------------------------------------------
# Wall detection
#
# The original scraper matched the bare substring "captcha" anywhere in the body
# and declared the board blocked. Internshala embeds a reCAPTCHA site key in a
# script tag while serving 50 real internship cards, so that check threw away a
# working board. These pin the corrected behaviour.
# --------------------------------------------------------------------------

def test_an_embedded_recaptcha_key_is_not_a_wall():
    """Internshala ships this on a page with 50 real cards. Reporting it as
    blocked is a false positive that cost a working source."""
    page = sel('<html><body><script src="https://www.google.com/recaptcha/api.js'
               '?render=explicit&amp;k=6LcAbc"></script>'
               '<div class="individual_internship">Real card</div></body></html>')
    assert boards._looks_walled(page) is False


def test_a_real_interstitial_page_is_a_wall():
    page = sel("<html><body><h1>Verify you are a human</h1></body></html>")
    assert boards._looks_walled(page) is True


def test_a_js_shell_is_not_mistaken_for_a_wall():
    """Naukri answers HTTP 200 with an empty body. That is not a block - it is
    precisely the case the browser escalation exists for, so it must escalate
    rather than be reported as a challenge page."""
    assert boards._looks_walled(sel("<html><body></body></html>")) is False


# --------------------------------------------------------------------------
# Indeed
# --------------------------------------------------------------------------

INDEED_HTML = """
<html><body>
<div data-testid="slider_item"><div class="job_seen_beacon">
  <h3 class="jobTitle css-1o1rnx9">Senior Python Developer</h3>
  <div data-testid="company-name">Nitka Technologies</div>
  <div data-testid="text-location">Remote</div>
  <div data-testid="attribute_snippet_container">$95,000 - $130,000 a year</div>
  <a data-jk="a806" role="button" aria-label="full details of Senior Python"
     href="/rc/clk?jk=a806&amp;bb=xyz">x</a>
</div></div>
<div data-testid="slider_item"><div class="job_seen_beacon">
  <h3 class="jobTitle">Backend Engineer</h3>
  <div data-testid="company-name">Acme</div>
  <a data-jk="b2" href="/rc/clk?jk=b2">x</a>
</div></div>
</body></html>
"""


def test_indeed_parses_the_current_mosaic_markup():
    rows = boards.parse_indeed(sel(INDEED_HTML))
    assert len(rows) == 2
    first = rows[0]
    assert first.title == "Senior Python Developer"
    assert first.company == "Nitka Technologies"
    assert first.location == "Remote"
    assert first.salary == "$95,000 - $130,000 a year"
    assert first.link.startswith("https://www.indeed.com/rc/clk?jk=a806")
    # A relative href is made absolute, and the & in the query string survives.
    # Dropping the second parameter silently produces a dead Indeed link.
    assert "&bb=xyz" in first.link


def test_indeed_falls_back_to_the_aria_label_when_the_heading_is_missing():
    html = """<html><body><div data-testid="slider_item"><div class="job_seen_beacon">
      <div data-testid="company-name">Acme</div>
      <a data-jk="k1" aria-label="full details of Data Engineer"
         href="/rc/clk?jk=k1">x</a>
    </div></div></body></html>"""
    rows = boards.parse_indeed(sel(html))
    assert len(rows) == 1
    assert rows[0].title == "Data Engineer"


def test_indeed_does_not_double_count_the_nested_beacon():
    """Each card nests a .job_seen_beacon child. Counting the parent and the
    child as two cards would inflate the total and duplicate every listing."""
    rows = boards.parse_indeed(sel(INDEED_HTML))
    assert len(rows) == len({r.link for r in rows})


# --------------------------------------------------------------------------
# Glassdoor
#
# Every Glassdoor class name ends in a CSS-modules build hash that changes on
# each deploy. A selector written against the whole name returns zero the
# morning after a release, so these match on the stable prefix.
# --------------------------------------------------------------------------

GLASSDOOR_HTML = """
<html><body>
<div class="JobsList_jobListItem__wjTHv JobsList_selected__nFuMW">
  <div class="JobCard_jobCardLeftContent__cHcGe">
    <span class="EmployerProfile_compactEmployerName__9MGcV">DataAnnotation</span>
    <div class="JobCard_jobTitle__hL8Kd">Full Stack Engineer - AI Trainer</div>
    <div class="JobCard_location__qZbzN">Delhi</div>
    <div class="JobCard_salaryEstimate__ab12c">&#8377;7,195.34 Per hour</div>
    <a href="https://www.glassdoor.co.in/job-listing/full-stack-engineer-dataannotation-JV_KO0,30_1.0/">x</a>
  </div>
</div>
</body></html>
"""


def test_glassdoor_parses_and_decodes_entities():
    rows = boards.parse_glassdoor(sel(GLASSDOOR_HTML))
    assert len(rows) == 1
    row = rows[0]
    assert row.title == "Full Stack Engineer - AI Trainer"
    assert row.company == "DataAnnotation"
    assert row.location == "Delhi"
    # The rupee sign arrives as a numeric entity; it must survive as a real
    # character rather than staying "&#8377;" in the user's search results.
    assert "\u20b97,195.34" in row.salary
    assert row.link.endswith("JV_KO0,30_1.0/")


def test_glassdoor_survives_a_new_build_hash():
    """The whole point of the attribute-contains selectors: re-hashing every
    class must not change the outcome."""
    rehashed = GLASSDOOR_HTML
    for old, new in (("__hL8Kd", "__AAA1"), ("__qZbzN", "__AAA2"),
                     ("__ab12c", "__AAA3"), ("__9MGcV", "__AAA4"),
                     ("__cHcGe", "__AAA5"), ("__wjTHv", "__AAA6"),
                     ("__nFuMW", "__AAA7")):
        rehashed = rehashed.replace(old, new)
    assert len(boards.parse_glassdoor(sel(rehashed))) == 1


# --------------------------------------------------------------------------
# LinkedIn
# --------------------------------------------------------------------------

LINKEDIN_HTML = """
<html><body>
<li data-entity-urn="urn:li:jobPosting:1">
  <a class="base-card__full-link"
     href="https://fr.linkedin.com/jobs/view/x-at-y-1?a=1">x</a>
  <h3 class="base-search-card__title">D&eacute;veloppeur Back End Python</h3>
  <h4 class="base-search-card__subtitle">Confluences IT</h4>
  <span class="job-search-card__location">Toulouse, Occitanie, France</span>
</li>
</body></html>
"""


def test_linkedin_reads_the_declared_encoding():
    """Accented titles arrive as entities. Decoding as latin-1 or assuming
    cp1252 turns "Developpeur" into mojibake, which then fails every downstream
    keyword match and shows the user a broken title."""
    rows = boards.parse_linkedin(sel(LINKEDIN_HTML))
    assert len(rows) == 1
    assert rows[0].title == "D\xe9veloppeur Back End Python"
    assert rows[0].company == "Confluences IT"
    assert rows[0].location == "Toulouse, Occitanie, France"
    # The country subdomain encodes the geo the result was filtered to, so it
    # must not be rewritten to www.
    assert rows[0].link.startswith("https://fr.linkedin.com/")


def test_linkedin_works_without_the_entity_urn_wrapper():
    """LinkedIn has shipped markup variants with no data-entity-urn at all."""
    html = """<html><body><div class="base-card">
      <a href="https://in.linkedin.com/jobs/view/python-dev-at-acme-99">x</a>
      <h3 class="base-search-card__title">Python Developer</h3>
    </div></body></html>"""
    rows = boards.parse_linkedin(sel(html))
    assert len(rows) == 1
    assert rows[0].title == "Python Developer"


# --------------------------------------------------------------------------
# WeWorkRemotely
# --------------------------------------------------------------------------

WWR_HTML = """
<html><body>
<li class="wr_listing">
  <a href="/remote-jobs/cribl-sr-ux-designer">x</a>
  <h2 class="new-listing__header__title__text">Sr. UX Designer, Security</h2>
  <span class="new-listing__company-name">Cribl</span>
  <span class="new-listing__company-headquarters">Remote</span>
</li>
</body></html>
"""


def test_weworkremotely_makes_relative_links_absolute():
    rows = boards.parse_weworkremotely(sel(WWR_HTML))
    assert len(rows) == 1
    assert rows[0].title == "Sr. UX Designer, Security"
    assert rows[0].company == "Cribl"
    assert rows[0].link == "https://weworkremotely.com/remote-jobs/cribl-sr-ux-designer"


def test_an_entry_missing_its_title_or_link_is_dropped_not_half_emitted():
    """A card with no link is not clickable and one with no title is not
    matchable, so neither is worth showing."""
    html = """<html><body>
    <li class="wr_listing"><a href="/remote-jobs/orphan">x</a></li>
    <li class="wr_listing"><h2>No link here</h2></li>
    </body></html>"""
    assert boards.parse_weworkremotely(sel(html)) == []


def test_a_board_that_matches_nothing_returns_empty_rather_than_raising():
    """An empty list is the signal the caller escalates on. Raising here would
    turn a routine markup change into a crash instead of a browser retry."""
    assert boards.parse_indeed(sel("<html><body>nothing</body></html>")) == []


# --------------------------------------------------------------------------
# Which boards a default sweep probes
# --------------------------------------------------------------------------

def test_the_slow_always_empty_boards_are_not_probed_by_default():
    """Measured on a real hunt: Naukri escalates to a real browser, took 21s
    and still returned nothing because it wants a signed-in session. Glassdoor
    duplicates what the others already cover. Foundit needs a browser and 403s
    about half the time. All three stay reachable by name."""
    hunt = scraper.default_boards()
    for name in ("Naukri", "Glassdoor", "Foundit"):
        assert name not in hunt, f"{name} still taxes every hunt"
        assert name in scraper.BOARDS, f"{name} became unreachable by name"


def test_the_default_sweep_keeps_every_board_that_actually_returns_results():
    """The complement of the test above: the sweep must not quietly shrink.
    This is the check that fails if the exclusion list is used the wrong way
    round, or if a board is dropped by accident."""
    assert set(scraper.default_boards()) == {
        "Indeed", "LinkedIn", "Internshala", "WeWorkRemotely",
        "Remotive", "Arbeitnow",
    }


def test_the_two_keyless_json_feeds_stay_in_the_default_sweep():
    hunt = scraper.default_boards()
    assert "Remotive" in hunt, "a free JSON feed must never be dropped"
    assert "Arbeitnow" in hunt


def test_default_boards_returns_a_copy():
    """A caller mutating the returned dict must not be able to edit the
    registry every later sweep reads from."""
    hunt = scraper.default_boards()
    hunt["Indeed"] = None
    assert scraper.BOARDS["Indeed"] is not None


def test_hunt_order_reports_the_probed_boards_first():
    """The reporting loop walks the default set, so it must stay in registry
    order and must not re-add the skipped boards."""
    assert list(scraper.default_boards()) == [n for n in scraper.BOARDS
                                              if n in scraper.default_boards()]


def test_a_skipped_board_is_never_reported_as_blocked():
    """Reporting "Glassdoor: blocked (timed out)" and then listing Glassdoor as
    skipped says two contradictory things about a board that was deliberately
    never contacted."""
    import inspect
    src = inspect.getsource(scraper.search_job_boards)
    # The per-board loop must walk what was probed, not the whole registry.
    assert "for name, fn in default_boards().items():" in src


def test_naukri_hint_says_it_needs_an_account():
    """Naukri answers 200 with a shell and still renders nothing even in a real
    browser, because it wants a signed-in session. The hint must say that
    rather than blame JavaScript, or the user keeps retrying for nothing."""
    assert "account" in scraper.WALL_HINTS["Naukri"]


# --------------------------------------------------------------------------
# The sweep actually runs the default set
# --------------------------------------------------------------------------

@pytest.fixture
def stub_hunt(monkeypatch):
    """Stub only the boards a default sweep probes, and leave the excluded ones
    alone. If the sweep wrongly probed an excluded board, the real network
    scraper behind it would run and this test would be slow and flaky - so this
    also guards the default-selection wiring."""
    probed = []

    for name, fn in list(scraper.default_boards().items()):
        def fake(q, _name=name, _fn=fn):
            probed.append(_name)
            return [scraper.Job(_name, f"{_name} Listing",
                                link=f"https://x.test/{_name}")]

        monkeypatch.setitem(scraper.BOARDS, name, fake)
        scraper._fail_until.pop(name, None)
    return probed


def test_the_sweep_probes_the_default_set_and_names_what_it_skipped(stub_hunt):
    out = scraper.search_job_boards("python developer")
    head = out.split("###JOBS_JSON###")[0]
    data = _json.loads(out.split("###JOBS_JSON###")[1])

    assert set(stub_hunt) == set(scraper.default_boards())
    assert "Glassdoor" not in stub_hunt and "Foundit" not in stub_hunt
    assert data["total"] == len(scraper.default_boards())

    # A board missing from the results must be announced, or the user cannot
    # tell a deliberate skip from a broken source.
    assert "[skipped]" in head
    assert "ask for one by name" in head
    assert "Glassdoor" in head and "Foundit" in head


def test_a_skipped_board_can_still_be_requested_by_name(monkeypatch):
    """Excluding a board from the default sweep must not remove it."""
    probed = []
    monkeypatch.setitem(
        scraper.BOARDS, "Glassdoor",
        lambda q: (probed.append(q),
                   [scraper.Job("Glassdoor", "Glassdoor Listing",
                                link="https://x.test/gd")])[1])
    out = scraper.search_job_boards("python developer", board="Glassdoor")
    head, _, payload = out.partition("###JOBS_JSON###")
    data = _json.loads(payload)
    assert "limited to Glassdoor" in head
    assert probed == ["python developer"], "the named board was not probed"
    assert data["total"] == 1
    assert data["jobs"][0]["source"] == "Glassdoor"


def test_a_named_board_never_falls_back_to_a_different_board(stub_hunt, monkeypatch):
    """Naming one board must probe only that board. Falling back to Gmail or
    another board here is what made "collect from Indeed" quietly return
    unrelated listings."""
    monkeypatch.setitem(
        scraper.BOARDS, "Glassdoor",
        lambda q: [scraper.Job("Glassdoor", "GD", link="https://x.test/gd")])
    scraper.search_job_boards("python developer", board="Glassdoor")
    # stub_hunt recorded every board the sweep touched; asking for one by name
    # must not have dragged the default set in as well.
    assert set(stub_hunt) == set()


def test_only_the_excluded_boards_are_announced_as_skipped(stub_hunt):
    """Remotive, Arbeitnow and the rest are probed like everything else, so
    naming them as skipped would be wrong as well as confusing."""
    out = scraper.search_job_boards("python developer")
    head = out.split("###JOBS_JSON###")[0]
    skip_line = next(l for l in head.splitlines() if l.startswith("[skipped]"))
    for name in ("Glassdoor", "Foundit", "Naukri"):
        assert name in skip_line
    for name in ("Indeed", "LinkedIn", "Internshala", "WeWorkRemotely",
                 "Remotive", "Arbeitnow"):
        assert name not in skip_line, f"{name} is probed, not skipped"
    # A skipped board must not also be listed with a per-board status.
    for name in ("Glassdoor", "Foundit", "Naukri"):
        assert f"[{name}]" not in head.replace(skip_line, "")


# --------------------------------------------------------------------------
# "reached but empty" is not "blocked"
#
# This is the bug that made the dashboard report "all boards blocked" for a
# hunt that returned 259 listings: one board (WeWorkRemotely) answered fine and
# simply had no posting matching the query, and that was reported with the word
# "blocked" - which the UI greps for to decide whether the search failed.
# --------------------------------------------------------------------------

def _stub_sweep(monkeypatch, per_board):
    """Replace every default board with a stub returning a fixed listing list."""
    def make(rows):
        return lambda q, _rows=list(rows): list(_rows)

    for name in scraper.default_boards():
        monkeypatch.setitem(scraper.BOARDS, name,
                            make(per_board.get(name, [])))
        scraper._fail_until.pop(name, None)


def _wall(_q):
    raise RuntimeError("HTTP 403")


def test_a_quiet_board_is_not_reported_as_blocked(monkeypatch):
    """WeWorkRemotely had 0 matching postings on a hunt where five other boards
    returned 259. That must not read as a wall."""
    _stub_sweep(monkeypatch, {
        "Indeed": [scraper.Job("Indeed", "Python Dev", link="https://x.test/1")],
        "WeWorkRemotely": [],           # answered, nothing matched
    })
    out = scraper.search_job_boards("python developer")
    head, _, payload = out.partition("###JOBS_JSON###")
    data = _json.loads(payload)

    assert "WeWorkRemotely" in data["no_matches"]
    assert "WeWorkRemotely" not in data["blocked"], "a quiet board is not a wall"
    assert "WeWorkRemotely" not in data["blocked_reasons"]
    assert data["boards_worked"] is True
    # The prose must not contain the word either, because the UI greps for it.
    assert "WeWorkRemotely] blocked" not in head


def test_a_hunt_that_returned_listings_is_not_presented_as_blocked(monkeypatch):
    """End-to-end shape of the bug: the dashboard reads this text."""
    _stub_sweep(monkeypatch, {
        "Indeed": [scraper.Job("Indeed", "Python Dev", link="https://x.test/1")],
        "Remotive": [scraper.Job("Remotive", "Python Dev", link="https://x.test/2")],
        "WeWorkRemotely": [],
    })
    out = scraper.search_job_boards("python developer")
    head, _, payload = out.partition("###JOBS_JSON###")
    data = _json.loads(payload)
    assert data["total"] > 0
    # The UI's old test was: does this text contain "blocked" anywhere?
    import re
    assert not re.search(r"403|blocked|captcha|js wall|bot wall|no results",
                          head, re.I), head


def test_a_genuinely_walled_board_is_still_reported_as_blocked(monkeypatch):
    """The other half: a real 403 must still say so, or the fix would silence
    genuine failures."""
    # Stub the healthy boards first, then wall one of them: _stub_sweep writes
    # to every board in the default set, so the order matters.
    _stub_sweep(monkeypatch, {"LinkedIn": [
        scraper.Job("LinkedIn", "Python Dev", link="https://x.test/1")]})
    monkeypatch.setitem(scraper.BOARDS, "Indeed", _wall)

    out = scraper.search_job_boards("python developer")
    head, _, payload = out.partition("###JOBS_JSON###")
    data = _json.loads(payload)
    assert "Indeed" in data["blocked"]
    assert "403" in data["blocked_reasons"]["Indeed"]
    assert "Indeed] blocked" in head
    assert data["boards_worked"] is True, "other boards still worked"


def test_boards_that_answered_with_zero_are_still_reported_as_working(monkeypatch):
    """"Did the boards respond?" is a different question from "did they find
    anything?". A board answering "no Python jobs today" is a working board,
    and marking it a failure sends the user chasing a bug that isn't there."""
    _stub_sweep(monkeypatch, {"Remotive": [], "Arbeitnow": []})
    out = scraper.search_job_boards("python developer")
    _, _, payload = out.partition("###JOBS_JSON###")
    data = _json.loads(payload)
    assert data["total"] == 0
    assert data["blocked"] == []
    assert data["boards_worked"] is True, "reached, just empty"


def test_the_sources_map_does_not_call_a_quiet_board_blocked(monkeypatch):
    """"sources" is what the browser reads. Blanking every non-numeric outcome to
    the string "blocked" is what made the UI unable to distinguish "no matching
    postings" from "the site refused us" - and it left "sources" contradicting
    "blocked": [] in the very same payload."""
    _stub_sweep(monkeypatch, {})
    out = scraper.search_job_boards("python developer")
    _, _, payload = out.partition("###JOBS_JSON###")
    data = _json.loads(payload)
    assert data["total"] == 0
    assert data["blocked"] == []
    assert all(v == 0 for v in data["sources"].values()), data["sources"]
    assert set(data["no_matches"]) == set(data["sources"])


def test_only_a_hunt_where_every_board_walled_reports_failure(monkeypatch):
    for name in scraper.default_boards():
        monkeypatch.setitem(scraper.BOARDS, name, _wall)
        scraper._fail_until.pop(name, None)
    out = scraper.search_job_boards("python developer")
    _, _, payload = out.partition("###JOBS_JSON###")
    data = _json.loads(payload)
    assert data["boards_worked"] is False
    assert data["total"] == 0
    assert len(data["blocked"]) == len(scraper.default_boards())


def test_a_quiet_board_is_not_cached_as_broken(monkeypatch):
    """A board that answered and had no matches was previously remembered as
    failed for 90s, so the very next hunt skipped it as "blocked (cached)" and
    it could never contribute again."""
    _stub_sweep(monkeypatch, {"Indeed": [], "LinkedIn": [
        scraper.Job("LinkedIn", "Python Dev", link="https://x.test/1")]})
    scraper.search_job_boards("python developer")
    assert "Indeed" not in scraper._fail_until, \
        "a board with no matching postings must not be remembered as failed"


def test_a_really_broken_board_is_still_cached(monkeypatch):
    """The cache must survive for genuine failures, or every hunt would re-hit
    a wall."""
    _stub_sweep(monkeypatch, {"LinkedIn": [
        scraper.Job("LinkedIn", "Python Dev", link="https://x.test/1")]})
    monkeypatch.setitem(scraper.BOARDS, "Indeed", _wall)
    scraper.search_job_boards("python developer")
    assert "Indeed" in scraper._fail_until


def test_the_client_verdict_helper_reads_the_payload_not_prose():
    """static/index.html decides success/failure from this. It must key off the
    structured fields, not the word "blocked" in the prose."""
    html = open("static/index.html", encoding="utf-8").read()
    assert "function boardsVerdict" in html
    # The old heuristic must be gone from the tool_end handler.
    assert "all boards blocked — router will fall back" not in html
    assert "no results/i" not in html
    # And the payload it relies on must actually be produced.
    assert '"boards_worked"' in open("mcp_server_indeed_scraper.py",
                                     encoding="utf-8").read()


# --------------------------------------------------------------------------
# Scrapling-first wiring
# --------------------------------------------------------------------------

def test_scrapling_is_importable_in_this_environment():
    """If this regresses the app degrades silently to the legacy regex scrapers,
    which still get 403s - so assert it rather than assume it."""
    assert scraper._SCRAPLING_AVAILABLE is True
    assert boards.__file__


@pytest.mark.parametrize("name,slug", [
    ("Indeed", "indeed"), ("LinkedIn", "linkedin"), ("Naukri", "naukri"),
    ("Glassdoor", "glassdoor"), ("Foundit", "foundit"),
    ("Internshala", "internshala"), ("WeWorkRemotely", "weworkremotely"),
])
def test_the_registry_points_at_the_wrapper_not_the_legacy_scraper(name, slug):
    assert scraper.BOARDS[name] is getattr(scraper, "_" + slug)
    assert scraper.BOARDS[name] is not getattr(scraper, "_" + slug + "_legacy")


def test_listings_convert_to_the_existing_job_shape():
    """Ranking, filtering and the JSON payload downstream are unchanged, so the
    converter has to produce real Job objects with the same fields."""
    rows = [boards.Listing("Indeed", "Python Developer", "Acme", "Remote",
                           "$100k", "https://x.test/1")]
    jobs = scraper._listings_to_jobs("Indeed", rows)
    assert len(jobs) == 1
    job = jobs[0]
    assert (job.source, job.title, job.company, job.location, job.salary) == (
        "Indeed", "Python Developer", "Acme", "Remote", "$100k")
    assert job.to_dict()["link"] == "https://x.test/1"


def test_collect_falls_back_to_legacy_when_scrapling_returns_nothing(monkeypatch):
    """A throttled Scrapling response must not cost a board its results when a
    plainer request would have worked."""
    calls = []
    monkeypatch.setattr(boards, "scrape", lambda spec, query, **kw: [])
    monkeypatch.setattr(scraper, "_indeed_legacy",
                        lambda q: (calls.append(q),
                                   [scraper.Job("Indeed", "Legacy Hit",
                                                link="https://x.test/9")])[1])
    jobs = scraper._scrapling_collect("Indeed", "python developer",
                                      scraper._indeed_legacy)
    assert [j.title for j in jobs] == ["Legacy Hit"]
    assert calls == ["python developer"], "the legacy scraper was never reached"


def test_collect_prefers_scrapling_when_it_returns_rows(monkeypatch):
    monkeypatch.setattr(boards, "scrape", lambda spec, query, **kw: [
        boards.Listing("Indeed", "Scrapling Hit", link="https://x.test/1")])
    monkeypatch.setattr(scraper, "_indeed_legacy",
                        lambda q: pytest.fail("legacy must not be called"))
    jobs = scraper._scrapling_collect("Indeed", "q", scraper._indeed_legacy)
    assert [j.title for j in jobs] == ["Scrapling Hit"]


def test_collect_survives_scrapling_being_unavailable(monkeypatch):
    """The whole point of keeping the legacy scrapers."""
    monkeypatch.setattr(scraper, "_SCRAPLING_AVAILABLE", False)
    monkeypatch.setattr(scraper, "_indeed_legacy",
                        lambda q: [scraper.Job("Indeed", "Legacy",
                                               link="https://x.test/9")])
    jobs = scraper._scrapling_collect("Indeed", "q", scraper._indeed_legacy)
    assert [j.title for j in jobs] == ["Legacy"]# --------------------------------------------------------------------------
# How the query reaches the board
#
# Params are handed to Scrapling as a dict on purpose. Pre-encoding the query
# into the URL by hand double-encodes it: "c++ & rust" becomes
# "c%2B%2B%20%26%20rust" and is then percent-encoded a second time, so the
# board matches nothing and still answers 200. Nothing in the response
# distinguishes that from "this query has no results", so it has to be pinned by
# a test rather than diagnosed after the fact.
# --------------------------------------------------------------------------


@pytest.fixture
def captured_fetch(monkeypatch):
    """Record every boards.fetch() call instead of making one.

    Replaces fetch itself rather than Fetcher: scrape() escalates to a real
    Chromium when a page yields nothing parseable, and a test must never open a
    browser. Stubbing at the fetch boundary still pins the contract, which is
    what fetch was handed.
    """
    calls = []
    monkeypatch.setattr(boards, "fetch",
                        lambda url, **kw: (calls.append((url, kw)) or sel("<html/>")))
    monkeypatch.setattr(boards, "ThrottledOrUnparsed",
                        lambda *a, **k: None)
    return calls


@pytest.mark.parametrize("query", [
    "python developer",
    "c++ & rust",
    "C# developer",
    "data scientist / ml",
    "100% remote",
    "caf\u00e9 chef",
    "a|b",
    "q=1&x=2",
])
def test_the_query_reaches_the_board_as_raw_text(captured_fetch, query):
    boards.scrape(boards.SPECS["Indeed"], query)
    url, kw = captured_fetch[0]
    # Raw text in the dict: Scrapling performs the single encode.
    assert kw["params"]["q"] == query
    # Nothing pre-baked into the URL, which is what would encode it twice.
    assert "?" not in url, url


@pytest.mark.parametrize("name", ["Indeed", "LinkedIn", "Naukri", "Glassdoor"])
def test_every_board_that_takes_a_query_receives_it_unmangled(captured_fetch, name):
    boards.scrape(boards.SPECS[name], "c++ & rust")
    _url, kw = captured_fetch[0]
    params = kw.get("params") or {}
    assert params, f"{name} dropped the query entirely"
    assert "c++ & rust" in [str(v) for v in params.values()], params


def test_foundit_carries_its_search_in_the_url_not_a_param(captured_fetch):
    """Foundit is the odd one out: its spec URL already names the search, so it
    sends only a location param. Pinned so a future edit does not silently drop
    the query while still returning a healthy 200 with no results."""
    boards.scrape(boards.SPECS["Foundit"], "c++ & rust")
    url, kw = captured_fetch[0]
    assert kw["params"] == {"locations": "Remote"}
    assert "python-developer-jobs" in url


def test_a_board_with_no_search_parameter_sends_no_params(captured_fetch):
    """WeWorkRemotely is filtered client-side. Inventing params for it only
    invites a redirect that discards them."""
    boards.scrape(boards.SPECS["WeWorkRemotely"], "python developer")
    _url, kw = captured_fetch[0]
    assert kw.get("params") in (None, {}), kw.get("params")


def test_fetch_passes_the_params_dict_straight_to_scrapling(monkeypatch):
    """The other half: fetch() itself must not pre-encode. Scrapling is handed
    the same dict it was given, and the URL is untouched."""
    seen = {}

    class _Fake:
        @staticmethod
        def get(url, **kw):
            seen["url"] = url
            seen["kw"] = kw
            return types.SimpleNamespace(status=200, text="")

    monkeypatch.setattr(boards, "Fetcher", _Fake)
    boards.fetch("https://www.indeed.com/jobs",
                 params={"q": "c++ & rust", "l": "Remote"})
    assert seen["url"] == "https://www.indeed.com/jobs"
    assert seen["kw"]["params"] == {"q": "c++ & rust", "l": "Remote"}
    # The cheap path is impersonated; that is the whole reason it is cheap.
    assert seen["kw"].get("impersonate"), seen["kw"]
    assert seen["kw"].get("stealthy_headers") is True


# --------------------------------------------------------------------------
# Links are absolute
#
# parse_glassdoor passed href through raw while every other parser called
# _absolutise. Glassdoor serves root-relative /job-listing/ hrefs and is the
# board most likely to answer on a host other than the one requested, so its
# listings came back with no host - a dead link in the email the candidate
# sends. Caught by the registry walk below, which cannot go stale the way a
# hand-written list of parsers would.
# --------------------------------------------------------------------------


_CARD_HTML = {
    # Saved fixtures, reused rather than hand-written, so this walks the same
    # markup the board actually served rather than markup I invented.
    "Indeed": INDEED_HTML,
    "LinkedIn": LINKEDIN_HTML,
    "Glassdoor": GLASSDOOR_HTML,
    "WeWorkRemotely": WWR_HTML,
    "Naukri": '<div class="job-card-wrapper"><div class="job-card">'
              '<a class="title" href="/job-detail/x-1">Backend Engineer</a>'
              '</div></div>',
    "Foundit": '<div class="srp-list"><div><h2 class="title">'
               '<a href="/listing/x-1">Backend Engineer</a></h2></div></div>',
    "Internshala": '<div class="individual_internship">'
                   '<a class="internship_header_title" '
                   'href="/internship/x-1"><h4>Dev Intern</h4></a></div>',
}


def test_every_parser_absolutises_its_links():
    checked = []
    for name, spec in boards.SPECS.items():
        rows = spec.parser(sel(_CARD_HTML[name]))
        assert rows, f"{name}: the probe card did not parse at all"
        for row in rows:
            assert row.link.startswith("http"), f"{name}: {row.link!r}"
            checked.append(name)
    assert sorted(set(checked)) == sorted(boards.SPECS), checked


def test_a_relative_glassdoor_href_is_made_absolute():
    page = sel('<li data-testid="jobs-list-item-cel">'
               '<a data-testid="job-link" href="/job-listing/x/1">'
               '<div class="JobCard_jobTitle__hL8Kd">Data Engineer</div>'
               '</a></li>')
    out = boards.parse_glassdoor(page)
    assert out[0].title == "Data Engineer"
    assert out[0].link == "https://www.glassdoor.com/job-listing/x/1"


def test_a_protocol_relative_href_is_made_https():
    page = sel('<li data-testid="jobs-list-item-cel">'
               '<a data-testid="job-link" '
               'href="//www.glassdoor.com/job-listing/x/1">'
               '<div class="JobCard_jobTitle__hL8Kd">Data Engineer</div>'
               '</a></li>')
    out = boards.parse_glassdoor(page)
    assert out[0].link.startswith("https://"), out[0].link
# --------------------------------------------------------------------------
# Reset and the "already shown" memory
#
# The scraper marks postings the account has seen so a repeat hunt leads with
# something new. That memory is per account, correctly - but reset_everything
# never touched it, so every posting stayed marked and the first hunt after a
# reset came back with nothing fresh, falling back to repeats. That reads as a
# broken search rather than a clean slate.
# --------------------------------------------------------------------------


def test_forgetting_one_account_leaves_the_others_alone():
    """One person resetting must not hand their neighbours a flood of postings
    they had already dismissed. They share the process."""
    jobs = [scraper.Job("Indeed", "A", link="https://x.test/1"),
            scraper.Job("Indeed", "B", link="https://x.test/2")]
    scraper._remember_seen(jobs, "u1")
    scraper._remember_seen(jobs, "u2")
    try:
        assert scraper.forget_scope("u1") == 2
        assert not scraper._is_seen("https://x.test/1", "u1"), "u1 kept its memory"
        assert scraper._is_seen("https://x.test/1", "u2"), "u2 was collateral damage"
    finally:
        scraper.forget_scope("u1")
        scraper.forget_scope("u2")


def test_forgetting_an_unknown_account_is_a_no_op():
    assert scraper.forget_scope("u-does-not-exist") == 0


def test_forgetting_the_shared_bucket_does_not_touch_accounts():
    """The bare-CLI default bucket is not an account. forget_scope("") must
    refuse rather than fall through and evict something real."""
    scraper._remember_seen([scraper.Job("Indeed", "A", link="https://y.test/1")],
                           "u3")
    try:
        assert scraper.forget_scope("") == 0
        assert scraper._is_seen("https://y.test/1", "u3")
    finally:
        scraper.forget_scope("u3")


def test_a_forgotten_account_sees_its_postings_as_fresh_again():
    link = "https://z.test/1"
    scraper._remember_seen([scraper.Job("Indeed", "A", link=link)], "u4")
    try:
        fresh, repeat = scraper._partition_fresh(
            [scraper.Job("Indeed", "A", link=link)], "u4")
        assert not fresh and repeat, "precondition: should read as a repeat"
        scraper.forget_scope("u4")
        fresh, repeat = scraper._partition_fresh(
            [scraper.Job("Indeed", "A", link=link)], "u4")
        assert fresh and not repeat, "still a repeat after the reset"
    finally:
        scraper.forget_scope("u4")


def test_forget_scope_drops_the_lru_slot_so_the_scope_can_come_back():
    """seen_bucket re-creates a missing scope and re-registers it for LRU. If
    forget_scope left the stale name in _seen_scope_order, the cap would evict
    a live account while counting a dead one."""
    scraper._remember_seen([scraper.Job("Indeed", "A", link="https://w.test/1")],
                           "u5")
    scraper.forget_scope("u5")
    assert "u5" not in scraper._seen_scope_order
    # And using it again works, rather than tripping over the removed entry.
    scraper._remember_seen([scraper.Job("Indeed", "A", link="https://w.test/1")],
                           "u5")
    try:
        assert scraper._is_seen("https://w.test/1", "u5")
        assert "u5" in scraper._seen_scope_order
    finally:
        scraper.forget_scope("u5")
