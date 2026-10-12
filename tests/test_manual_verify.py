"""Tests for manual bot-wall verification (manual_verify.py + its two routes).

The point of this feature is that a human clears a captcha the automated tiers
cannot, so the tests here never open a browser: the Scrapling seams
(_fetch_page, _solve_sync, _cleared) are patched, and the route tests drive the
real router with a stubbed agent exactly like test_auth_routes does.
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

import auth
import dashboard as d
import manual_verify as mv


@pytest.fixture(autouse=True)
def clean_jobs(monkeypatch):
    """Every test starts with an empty job registry and a known env."""
    monkeypatch.delenv("JOBSCOPE_MANUAL_WALL", raising=False)
    monkeypatch.delenv("JOBSCOPE_MANUAL_SECONDS", raising=False)
    # No mail server, so registration self-confirms and the gate is a plain
    # sign-in - the same default test_auth_routes uses, for the same reason.
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                 "SMTP_FROM", "AUTH_DEV_LINKS"):
        monkeypatch.delenv(name, raising=False)
    mv.reset()
    yield
    mv.reset()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Node:
    def __init__(self, text):
        self._text = text

    def get_all_text(self):
        return self._text


class FakePage:
    """Just enough of a Scrapling Selector for _looks_walled / _visible_text.

    A page whose body carries a wall marker is a wall; one with a long, neutral
    body is a cleared listing page; an empty body is a shell.
    """

    def __init__(self, body=""):
        self._body = body

    def css(self, selector):
        if selector == "body" and self._body:
            return [_Node(self._body)]
        return []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_off_disables_both_the_endpoint_and_inline(monkeypatch):
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "off")
    assert mv.enabled() is False
    assert mv.inline_enabled() is False


def test_on_forces_both_on_even_with_no_display(monkeypatch):
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "on")
    assert mv.enabled() is True
    assert mv.inline_enabled() is True


def test_auto_enables_the_endpoint_but_never_inline(monkeypatch):
    # auto follows display availability for the endpoint, but inline solving
    # (which blocks a running search) is opt-in only.
    monkeypatch.setattr(mv, "_display_available", lambda: True)
    assert mv.enabled() is True
    assert mv.inline_enabled() is False


def test_a_typo_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "yees")
    monkeypatch.setattr(mv, "_display_available", lambda: True)
    assert mv.enabled() is True
    assert mv.inline_enabled() is False


def test_seconds_is_clamped():
    os.environ["JOBSCOPE_MANUAL_SECONDS"] = "5"
    try:
        assert mv._seconds() == mv.MIN_SECONDS
    finally:
        del os.environ["JOBSCOPE_MANUAL_SECONDS"]


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def test_resolve_known_board_returns_url_and_params():
    url, params = mv.resolve("Indeed", "python developer")
    assert url.startswith("https://www.indeed.com")
    assert params == {"q": "python developer", "l": "Remote"}


def test_resolve_unknown_board_is_none():
    assert mv.resolve("Bogus") is None
    assert mv.resolve("") is None


# ---------------------------------------------------------------------------
# Clearing detection
# ---------------------------------------------------------------------------

def test_a_challenge_page_is_never_cleared():
    page = FakePage("please verify you are a human before continuing")
    assert mv._cleared(page, "") is False


def test_a_substantial_blank_body_is_cleared():
    page = FakePage("x" * (mv.MIN_BODY_CHARS + 50))
    assert mv._cleared(page, "") is True


def test_a_shell_body_is_not_cleared():
    assert mv._cleared(FakePage(""), "") is False
    assert mv._cleared(FakePage("short"), "") is False


def test_a_parser_hit_is_cleared_even_with_a_short_body(monkeypatch):
    import boards
    monkeypatch.setitem(boards.PARSERS, "Fake", lambda page: [object()])
    assert mv._cleared(FakePage(""), "Fake") is True


# ---------------------------------------------------------------------------
# _wait_for_human
# ---------------------------------------------------------------------------

def test_wait_returns_as_soon_as_cleared(monkeypatch):
    calls = {"n": 0}

    def cleared(page, board):
        calls["n"] += 1
        return calls["n"] >= 3

    monkeypatch.setattr(mv, "_cleared", cleared)
    page = FakePage()
    page.wait_for_timeout = lambda ms: None
    ticks = []
    assert mv._wait_for_human(page, "", 30, ticks.append) is True
    assert calls["n"] == 3
    assert ticks  # on_tick fired at least once


def test_wait_gives_up_at_the_deadline(monkeypatch):
    monkeypatch.setattr(mv, "_cleared", lambda page, board: False)
    page = FakePage()
    page.wait_for_timeout = lambda ms: None
    assert mv._wait_for_human(page, "", 0, lambda r: None) is False


# ---------------------------------------------------------------------------
# _solve_sync
# ---------------------------------------------------------------------------

def test_solve_sync_reads_the_page_once_it_is_clear(monkeypatch):
    monkeypatch.setattr(mv, "_fetch_page",
                        lambda url, params, action, secs: FakePage("y" * 500))
    monkeypatch.setattr(mv, "_extract",
                        lambda page, board: [{"title": "Backend Dev"}])
    out = mv._solve_sync("https://x.test/jobs", "", "", lambda: None, lambda r: None)
    assert out["ok"] is True
    assert out["listings"] == [{"title": "Backend Dev"}]


def test_solve_sync_reports_a_page_that_is_still_walled(monkeypatch):
    monkeypatch.setattr(
        mv, "_fetch_page",
        lambda url, params, action, secs: FakePage("verify you are a human"))
    monkeypatch.setattr(mv, "_extract", lambda page, board: [{"title": "X"}])
    out = mv._solve_sync("https://x.test/jobs", "", "", lambda: None, lambda r: None)
    assert out["ok"] is False
    assert "still showing" in out["detail"]


def test_solve_sync_reports_a_page_with_no_cards(monkeypatch):
    monkeypatch.setattr(mv, "_fetch_page",
                        lambda url, params, action, secs: FakePage("z" * 500))
    monkeypatch.setattr(mv, "_extract", lambda page, board: [])
    out = mv._solve_sync("https://x.test/jobs", "", "", lambda: None, lambda r: None)
    assert out["ok"] is False
    assert "no job cards" in out["detail"]


def test_solve_sync_never_raises_when_the_browser_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no chromium here")

    monkeypatch.setattr(mv, "_fetch_page", boom)
    out = mv._solve_sync("https://x.test/jobs", "", "", lambda: None, lambda r: None)
    assert out["ok"] is False
    assert "could not open" in out["detail"].lower()


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

def test_start_runs_in_background_and_reaches_done(monkeypatch):
    monkeypatch.setattr(mv, "_solve_sync",
                        lambda url, board, query, on_open, on_tick: {
                            "ok": True, "detail": "", "listings": [{"title": "A"}]})
    job = mv.start(board="Indeed", query="python")
    assert job["state"] in ("starting", "waiting", "done")
    final = mv.wait(job["id"], timeout=5)
    assert final["state"] == "done"
    assert final["count"] == 1
    assert final["listings"] == [{"title": "A"}]


def test_a_failed_solve_marks_the_job_blocked_or_failed(monkeypatch):
    monkeypatch.setattr(mv, "_solve_sync",
                        lambda url, board, query, on_open, on_tick: {
                            "ok": False, "listings": [],
                            "detail": "The challenge is still showing - it was not solved in time."})
    job = mv.start(board="Indeed")
    final = mv.wait(job["id"], timeout=5)
    assert final["state"] == "blocked"
    assert final["listings"] == []


def test_status_of_an_unknown_job_is_none():
    assert mv.status("nope") is None


def test_solve_inline_delegates_to_solve_sync(monkeypatch):
    seen = {}

    def fake(url, board, query, on_open, on_tick):
        seen["args"] = (url, board, query)
        return {"ok": True, "detail": "", "listings": []}

    monkeypatch.setattr(mv, "_solve_sync", fake)
    mv.solve_inline("LinkedIn", "designer")
    assert seen["args"] == ("", "LinkedIn", "designer")


# ---------------------------------------------------------------------------
# The scraper's inline hook
# ---------------------------------------------------------------------------

def test_try_manual_solve_is_a_noop_when_disabled(monkeypatch):
    import mcp_server_indeed_scraper as s
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "off")
    assert s._try_manual_solve("Indeed", "python") == []


def test_try_manual_solve_turns_listings_into_jobs(monkeypatch):
    import mcp_server_indeed_scraper as s
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "on")
    monkeypatch.setattr(s._manual, "solve_inline", lambda board, query: {
        "ok": True, "detail": "",
        "listings": [{"title": "Dev", "company": "Acme", "link": "https://x/1"}]})
    jobs = s._try_manual_solve("Indeed", "python")
    assert [j.title for j in jobs] == ["Dev"]
    assert jobs[0].source == "Indeed"


def test_try_manual_solve_stays_quiet_on_a_failed_solve(monkeypatch):
    import mcp_server_indeed_scraper as s
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "on")
    monkeypatch.setattr(s._manual, "solve_inline",
                        lambda board, query: {"ok": False, "listings": [], "detail": "no"})
    assert s._try_manual_solve("Indeed", "python") == []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    monkeypatch.setattr(d, "PROFILES_FILE", str(tmp_path / "no-profiles.json"))

    class _Bundle:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tools = []

    async def _fake_create_agent(cfg=None, runtime=None):
        return _Bundle(cfg)

    async def _noop_runtime():
        return object()

    monkeypatch.setattr(d, "create_agent", _fake_create_agent)
    monkeypatch.setattr(d, "create_runtime", _noop_runtime)
    monkeypatch.setattr(d.app.state, "runtime", object(), raising=False)
    d.app.state.bundles = {}
    with TestClient(d.app) as c:
        c.headers["host"] = "jobs.test"
        yield c


def _sign_in(client):
    r = client.post("/api/auth/register",
                    json={"email": "owner@jobs.test", "password": "a-good-password"})
    assert r.status_code in (200, 202), r.text
    body = r.json()
    if body.get("needs_verification"):
        token = body["dev_link"].split("verify=")[-1]
        client.get("/api/auth/verify?token=" + token)
    r = client.post("/api/auth/login",
                    json={"email": "owner@jobs.test", "password": "a-good-password"})
    assert r.status_code == 200, r.text


def test_verify_board_requires_a_sign_in(client):
    r = client.post("/api/verify-board", json={"board": "Indeed"})
    assert r.status_code == 401


def test_verify_board_400_when_no_browser_can_be_shown(client, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "off")
    _sign_in(client)
    r = client.post("/api/verify-board", json={"board": "Indeed"})
    assert r.status_code == 400
    assert "visible browser" in r.json()["detail"]


def test_verify_board_404_for_an_unknown_board(client, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "on")
    _sign_in(client)
    r = client.post("/api/verify-board", json={"board": "Nope"})
    assert r.status_code == 404


def test_verify_board_400_without_a_board_or_url(client, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "on")
    _sign_in(client)
    r = client.post("/api/verify-board", json={})
    assert r.status_code == 400


def test_verify_board_starts_a_job_and_reports_it_done(client, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_MANUAL_WALL", "on")
    monkeypatch.setattr(mv, "_solve_sync",
                        lambda url, board, query, on_open, on_tick: {
                            "ok": True, "detail": "",
                            "listings": [{"title": "Dev", "source": "Indeed"}]})
    _sign_in(client)
    r = client.post("/api/verify-board", json={"board": "Indeed"})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    state = {}
    for _ in range(200):
        state = client.get("/api/verify-board/" + job_id).json()
        if state["state"] in ("done", "blocked", "failed"):
            break
        time.sleep(0.02)
    assert state["state"] == "done"
    assert state["count"] == 1
    assert state["listings"][0]["title"] == "Dev"


def test_verify_board_status_404_for_unknown_job(client):
    _sign_in(client)
    r = client.get("/api/verify-board/deadbeef")
    assert r.status_code == 404


def test_the_ui_wires_the_verify_button_and_polls(monkeypatch):
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, "static", "index.html"), encoding="utf-8").read()
    assert "btnverify" in html
    assert "startVerify" in html
    assert "/api/verify-board" in html
    assert "pollVerify" in html
