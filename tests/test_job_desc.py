"""Background job-description fetching: URL canonicalisation, selector choice,
the cache, and the promise that a failure is remembered rather than retried.

The expensive thing here is a browser, so most of what is worth testing is the
bookkeeping that decides *whether* to launch one. These tests never launch one:
`fetch_sync` is exercised against stubbed fetchers so the suite stays fast and
does not need Chromium.
"""

import time

import pytest
from scrapling.parser import Selector

import auth
import boards
import dashboard as d
import job_desc as jd


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    yield
    jd.shutdown()


def _sel(body: str, attrs: str = 'class="show-more-less-html__markup"') -> Selector:
    return Selector(content=f"<html><body><div {attrs}>{body}</div></body></html>")


# Comfortably clear of DESC_MIN_CHARS, so these fixtures test the selector choice
# and the wall filter rather than accidentally testing the length gate. A body
# that fell short would be skipped and the scan would continue to the next
# selector, which is correct behaviour but not what these are here to check.
_JD = " ".join([
    "We are hiring a senior backend engineer to own the settlement pipeline that",
    "moves millions of events a day across three regions. You will write Python",
    "and Django services, design the Postgres schema, and keep Kafka, Redis and",
    "AWS honest under real load. You will be on call for what you ship, so we",
    "care much more about how you reason about failure modes and how you write",
    "down your decisions than about how many years you have written code. Every",
    "change gets reviewed as a team, and we interview for debugging judgement",
    "rather than trivia. If that sounds like work you would rather do than avoid,",
    "we would like to talk to you about the payments platform group in detail.",
])


# --------------------------------------------------------------------------
# URL canonicalisation: one posting, one cache row
# --------------------------------------------------------------------------

def test_tracking_parameters_are_stripped_from_the_cache_key():
    """The same posting reached from a search result and from a pasted link must
    not fetch twice - that is the whole cost saving the cache exists for."""
    assert jd.normalize_url(
        "https://www.linkedin.com/jobs/view/123/?ref=home&trk=public"
    ) == jd.normalize_url("https://www.linkedin.com/jobs/view/123/")


def test_the_fragment_is_not_part_of_the_key():
    assert jd.normalize_url("https://x.co/j/1#top") == "https://x.co/j/1"


def test_meaningful_query_parameters_survive():
    """Stripping is not allowed to collapse two different postings onto one row."""
    a = jd.normalize_url("https://x.co/jobs?id=1")
    b = jd.normalize_url("https://x.co/jobs?id=2")
    assert a != b
    assert "id=1" in a


def test_the_host_is_case_insensitive_but_the_path_is_not():
    assert jd.normalize_url("https://WWW.X.CO/J/1") == "https://www.x.co/J/1"
    assert jd.normalize_url("https://x.co/j/1") != jd.normalize_url("https://x.co/J/1")


@pytest.mark.parametrize("bad", ["", "   ", "javascript:alert(1)", "not a url",
                                 "/relative/path", None])
def test_a_non_http_url_has_no_cache_key(bad):
    """A key built from arbitrary user input would be a place to store junk, and
    a javascript: URL must never be handed to a fetcher."""
    assert jd.normalize_url(bad) == ""


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def test_the_most_specific_selector_wins():
    """Two posting bodies on one fixture page: the board-specific element is the
    description, the generic one is site furniture."""
    page = Selector(content=f"""<html><body>
      <div class="show-more-less-html__markup">{_JD}</div>
      <div id="jobDescriptionText">{'x' * 900}</div>
    </body></html>""")
    text, source = jd.extract(page)
    assert source == "linkedin"
    assert "settlement pipeline" in text
    assert "x" * 50 not in text


def test_a_block_page_is_never_cached_as_the_description():
    """The failure this guards against: a wall page can be longer than a real
    posting, and tailoring to "please verify you are human" is worse than not
    tailoring at all."""
    page = _sel("Please verify you are human. Unusual traffic detected from your "
                "computer network. " + _JD)
    text, source = jd.extract(page)
    assert text == "" and source == "none"


def test_a_page_with_no_description_is_a_miss_not_an_error():
    assert jd.extract(_sel("Apply now!")) == ("", "none")


def test_a_cookie_banner_is_too_short_to_be_mistaken_for_a_posting():
    assert jd.extract(_sel("We use cookies.")) == ("", "none")


def test_list_structure_survives_into_the_text():
    """Requirements and responsibilities are what tailoring reads, so the bullets
    have to survive as separate lines rather than collapsing into one paragraph."""
    page = _sel(f"<p>{_JD}</p><ul><li>Python</li><li>Django</li></ul>")
    text, _ = jd.extract(page)
    assert "Python" in text and "Django" in text
    assert text.count("\n") >= 2


def test_extraction_is_capped():
    page = _sel("word " * 20000)
    text, _ = jd.extract(page)
    assert len(text) <= jd.DESC_MAX_CHARS


def test_a_selector_that_throws_does_not_abort_the_whole_scan():
    class _Exploding:
        def css(self, sel):
            if "linkedin" in sel and "show-more-less" in sel:
                raise RuntimeError("parser blew up")
            return []

    assert jd.extract(_Exploding()) == ("", "none")


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------

def test_a_stored_description_reads_back_as_ready():
    jd.store("https://x.co/j/1", _JD, "linkedin")
    st = jd.status_of("https://x.co/j/1")
    assert st["status"] == "ready"
    assert st["chars"] == len(_JD)
    assert st["source"] == "linkedin"


def test_an_empty_result_is_cached_as_a_permanent_miss():
    """Otherwise every click on a walled posting launches another browser, which
    is how a client talks itself into being blocked for good."""
    jd.store("https://x.co/j/2", "", "none")
    assert jd.status_of("https://x.co/j/2")["status"] == "empty"


def test_a_stale_row_is_not_served_as_fresh():
    jd.store("https://x.co/j/3", _JD, "linkedin")
    _age("https://x.co/j/3", jd.DESC_TTL_SECONDS + 60)
    assert jd.status_of("https://x.co/j/3")["status"] == "pending"
    assert jd.is_fresh(jd.cached("https://x.co/j/3")) is False


def test_re_storing_replaces_the_previous_text():
    jd.store("https://x.co/j/4", _JD, "linkedin")
    jd.store("https://x.co/j/4", "short new text", "indeed")
    row = jd.cached("https://x.co/j/4")
    assert row["text"] == "short new text" and row["source"] == "indeed"


def test_two_accounts_share_one_row_because_the_text_is_public():
    """No user_id on this table on purpose: the description is public text about
    a posting, so caching it per account would mean paying for the same fetch
    once per person."""
    jd.store("https://x.co/j/5", _JD, "linkedin")
    with auth.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(job_desc_cache)")}
    assert "user_id" not in cols


def test_a_missing_table_is_treated_as_an_empty_cache_not_a_crash():
    """auth.init_db has not necessarily run; a resume must still build."""
    import sqlite3
    with sqlite3.connect(":memory:") as _:
        pass
    real = auth.DB_FILE
    try:
        auth.DB_FILE = ":memory:"
        assert jd.cached("https://x.co/j/6") is None
        assert jd.status_of("https://x.co/j/6")["status"] == "pending"
        jd.store("https://x.co/j/6", _JD, "linkedin")   # must not raise
    finally:
        auth.DB_FILE = real


def _age(url: str, seconds: float) -> None:
    with auth.connect() as conn:
        conn.execute("UPDATE job_desc_cache SET fetched_at = ? WHERE url = ?",
                     (time.time() - seconds, jd.normalize_url(url)))


# --------------------------------------------------------------------------
# the two-tier fetch
# --------------------------------------------------------------------------

def test_the_cheap_path_is_tried_first_and_a_browser_is_not(monkeypatch):
    calls = []

    def _cheap(url, **kw):
        calls.append(("cheap", kw.get("browser")))
        return _sel(_JD)

    monkeypatch.setattr(boards, "fetch", _cheap)
    text, source = jd.fetch_sync("https://x.co/j/7")
    assert source == "linkedin" and "settlement pipeline" in text
    assert calls == [("cheap", False)], "a browser ran despite the cheap path working"
    assert jd.cached("https://x.co/j/7")["text"]


def test_the_browser_is_only_escalated_to_when_the_cheap_path_finds_nothing(monkeypatch):
    seen = []

    def _fetch(url, **kw):
        seen.append(bool(kw.get("browser")))
        if kw.get("browser"):
            return _sel(_JD, attrs='id="jobDescriptionText"')
        raise boards.BoardWall("HTTP 403")

    monkeypatch.setattr(boards, "fetch", _fetch)
    text, source = jd.fetch_sync("https://x.co/j/8")
    assert seen == [False, True]
    assert source.startswith("indeed")
    assert text


def test_the_browser_does_not_wait_for_the_network_to_go_quiet(monkeypatch):
    """A posting page keeps polling from trackers long after its text is
    readable. Waiting for silence cost 131s against 8s for the same text."""
    seen = {}

    def _fetch(url, **kw):
        if kw.get("browser"):
            seen["network_idle"] = kw.get("network_idle")
            return _sel(_JD)
        raise boards.BoardWall("HTTP 403")

    monkeypatch.setattr(boards, "fetch", _fetch)
    jd.fetch_sync("https://x.co/j/9")
    assert seen.get("network_idle") is False


def test_a_total_failure_is_recorded_as_a_miss_rather_than_raised(monkeypatch):
    def _fetch(url, **kw):
        raise RuntimeError("chromium exploded")

    monkeypatch.setattr(boards, "fetch", _fetch)
    assert jd.fetch_sync("https://x.co/j/10") == ("", "none")
    # And the miss is remembered, so the next attempt reads the cache.
    assert jd.status_of("https://x.co/j/10")["status"] == "empty"


def test_a_non_http_url_never_reaches_a_fetcher(monkeypatch):
    monkeypatch.setattr(boards, "fetch",
                        lambda *a, **k: pytest.fail("fetched an unsafe URL"))
    assert jd.fetch_sync("javascript:alert(1)") == ("", "none")
    assert jd.fetch_sync("") == ("", "none")


def test_a_wall_from_the_browser_is_a_miss_not_a_description(monkeypatch):
    def _fetch(url, **kw):
        if kw.get("browser"):
            return _sel("Please verify you are human. Unusual traffic. " + _JD)
        raise boards.BoardWall("HTTP 403")

    monkeypatch.setattr(boards, "fetch", _fetch)
    assert jd.fetch_sync("https://x.co/j/11") == ("", "none")


# --------------------------------------------------------------------------
# ensure(): do not launch two browsers for one posting
# --------------------------------------------------------------------------

def test_ensure_serves_a_fresh_row_without_launching_anything(monkeypatch):
    jd.store("https://x.co/j/12", _JD, "linkedin")
    monkeypatch.setattr(jd, "fetch_sync",
                        lambda *a, **k: pytest.fail("refetched a fresh row"))
    assert jd.ensure("https://x.co/j/12")["status"] == "ready"


def test_one_url_in_flight_means_one_browser(monkeypatch):
    """Ten resumes for the same posting must not start ten Chromium instances."""
    started = []
    release = __import__("threading").Event()

    def _slow(url, **kw):
        started.append(url)
        release.wait(5)
        return _JD, "linkedin"

    monkeypatch.setattr(jd, "fetch_sync", _slow)
    for _ in range(10):
        jd.ensure("https://x.co/j/13")
    release.set()
    for _ in range(50):
        if started:
            break
        time.sleep(0.02)
    assert len(started) == 1


def test_ensure_reports_pending_for_a_url_it_has_never_seen(monkeypatch):
    monkeypatch.setattr(jd, "fetch_sync", lambda url, **kw: ("", "none"))
    assert jd.ensure("https://x.co/j/14")["status"] == "pending"


def test_ensure_on_an_unsafe_url_never_reaches_the_executor(monkeypatch):
    monkeypatch.setattr(jd, "fetch_sync",
                        lambda *a, **k: pytest.fail("fetched an unsafe URL"))
    assert jd.ensure("javascript:x")["status"] == "empty"


# --------------------------------------------------------------------------
# the route
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_smtp(monkeypatch):
    """No mail server, so registration self-confirms and sets a session cookie.

    Without this the fixture's account stays unverified and every request is a
    401, which would make the route tests pass for the wrong reason.
    """
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                 "SMTP_FROM", "AUTH_DEV_LINKS"):
        monkeypatch.setenv(name, "")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A signed-in client with the agent and runtime stubbed out.

    Stubbing fetch_sync globally matters here: /api/resume/gen calls
    job_desc.ensure() on every request, and without this each test would leave a
    real Chromium fetch running after it finished.
    """
    monkeypatch.setattr(d, "PROFILES_FILE", str(tmp_path / "no-profiles.json"))
    monkeypatch.setattr(jd, "fetch_sync", lambda url, **kw: ("", "none"))

    class _Bundle:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tools = []

    async def _agent(cfg=None, runtime=None):
        return _Bundle(cfg)

    async def _runtime():
        return object()

    monkeypatch.setattr(d, "create_agent", _agent)
    monkeypatch.setattr(d, "create_runtime", _runtime)
    monkeypatch.setattr(d.app.state, "bundles", {}, raising=False)
    d.app.state.bundles = {}
    monkeypatch.setattr(d.app.state, "runtime", object(), raising=False)

    from fastapi.testclient import TestClient
    with TestClient(d.app) as c:
        c.headers["host"] = "jobs.test"
        r = c.post("/api/auth/register",
                   json={"email": "desc@jobs.test", "password": "a-good-password"})
        assert r.status_code in (200, 202), r.text
        yield c


def test_the_poll_route_requires_a_session(client):
    """The cache is shared, but the endpoint is still account-gated: it is a
    server-side action that can spend a browser."""
    from fastapi.testclient import TestClient
    with TestClient(d.app) as anon:
        anon.headers["host"] = "jobs.test"
        assert anon.get("/api/resume/desc?link=https://x.co/j/15").status_code == 401


def test_the_poll_route_reports_a_cached_description(client):
    jd.store("https://x.co/j/16", _JD, "linkedin")
    r = client.get("/api/resume/desc?link=https://x.co/j/16").json()
    assert r["ok"] is True and r["status"] == "ready"
    assert r["chars"] == len(_JD) and "settlement pipeline" in r["text"]


def test_the_poll_route_rejects_a_missing_or_unsafe_link(client):
    for link in ("", "   ", "javascript:alert(1)"):
        r = client.get("/api/resume/desc?link=" + link).json()
        assert r["status"] == "empty" and r["chars"] == 0


def test_the_poll_route_does_not_launch_a_browser_for_a_cached_row(client, monkeypatch):
    jd.store("https://x.co/j/17", _JD, "linkedin")
    monkeypatch.setattr(jd, "fetch_sync",
                        lambda *a, **k: pytest.fail("refetched a cached row"))
    assert client.get("/api/resume/desc?link=https://x.co/j/17").json()["status"] == "ready"


def test_generation_reports_how_much_text_it_used(client, monkeypatch):
    """The client needs this to decide whether a later fetch actually found
    anything new, rather than offering a regenerate that changes nothing."""
    monkeypatch.setattr(jd, "fetch_sync", lambda url, **kw: ("", "none"))
    monkeypatch.setattr(d.resume_generator, "desc_snippet", lambda link: _JD)
    r = client.post("/api/resume/gen", json={
        "title": "Backend Engineer", "company": "Acme",
        "link": "https://x.co/j/18"}).json()
    assert r["ok"] is True
    assert r["desc_chars"] == len(_JD)
    assert r["jd_origin"] == "link"


def test_generation_prefers_a_cached_description_over_a_fresh_scrape(client, monkeypatch):
    """A posting someone already opened should cost no request at all."""
    jd.store("https://x.co/j/19", _JD, "linkedin")
    monkeypatch.setattr(jd, "fetch_sync", lambda url, **kw: ("", "none"))
    monkeypatch.setattr(d.resume_generator, "desc_snippet",
                        lambda link: pytest.fail("re-scraped a cached posting"))
    r = client.post("/api/resume/gen", json={
        "title": "Backend Engineer", "company": "Acme",
        "link": "https://x.co/j/19"}).json()
    assert r["ok"] is True and r["jd_origin"] == "cache"
    assert r["desc_chars"] == len(_JD)


def test_a_pasted_description_is_never_overridden_by_the_cache(client, monkeypatch):
    jd.store("https://x.co/j/20", _JD, "linkedin")
    monkeypatch.setattr(jd, "fetch_sync", lambda url, **kw: ("", "none"))
    r = client.post("/api/resume/gen", json={
        "title": "Backend Engineer", "link": "https://x.co/j/20",
        "jd_text": "A completely different posting the user typed in."}).json()
    assert r["jd_origin"] == "pasted"
