"""manual_verify.py - let a human clear a bot wall in a real browser window.

Most walls fall to the two automated tiers in ``boards.py``: TLS impersonation
for a plain 403, and Scrapling's ``solve_cloudflare`` for a Cloudflare
interstitial. What neither can do is tick a reCAPTCHA/hCaptcha checkbox that is
deliberately unsolvable by a machine. The only honest answer there is a person:
open a *visible* browser, let them complete the challenge, and take the page
once it is through.

That is all this module does. The browser keeps a persistent profile
(``.browser_profile`` by default), so a board that also wants an account stays
signed in across runs instead of asking for the login every time. Point
``JOBSCOPE_MANUAL_PROFILE`` elsewhere to relocate it, or set it empty to opt
back out of persistence.

Shape of a job
--------------
``start()`` returns immediately with a job id and spawns a daemon thread. The
thread opens the browser and blocks inside a ``page_action`` callback until the
challenge is gone (or the deadline passes), then parses the page with the board's
normal parser and marks the job ``done``. Callers poll ``status()``.

    job = start(board="Indeed", query="python developer")
    while status(job["id"])["state"] in ("starting", "waiting"):
        ...

The module is import-safe and browser-free at import time: Scrapling and the
browser binary are touched only inside the worker thread. Tests patch the
``_fetch_page`` / ``_wait_for_human`` seams so no window ever opens.

Configuration:
    JOBSCOPE_MANUAL_WALL     auto (default) | on | off
    JOBSCOPE_MANUAL_SECONDS  how long the window stays open (default 240)
    JOBSCOPE_MANUAL_PROFILE  browser profile dir (default .browser_profile)
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid

__all__ = ["enabled", "inline_enabled", "resolve", "solve_inline",
           "start", "status", "wait", "reset"]

# The human needs time to notice the window, read the challenge and solve it.
DEFAULT_SECONDS = 240
MIN_SECONDS = 30
MAX_SECONDS = 1800

# Below this much visible text a page is still a shell (challenge, redirect in
# flight, or an empty render), not the listing behind the wall.
MIN_BODY_CHARS = 400

# Every job ever started in this process, by id. The dashboard polls it; nothing
# else reads it. Bounded because a person only ever runs a handful of these.
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MAX_JOBS = 50

# Terminal states: a poller stops on either.
_TERMINAL = ("done", "blocked", "failed")


def _seconds() -> int:
    try:
        value = int(os.environ.get("JOBSCOPE_MANUAL_SECONDS", "") or DEFAULT_SECONDS)
    except (TypeError, ValueError):
        return DEFAULT_SECONDS
    return max(MIN_SECONDS, min(MAX_SECONDS, value))


def _display_available() -> bool:
    """Whether a visible browser can actually be shown here.

    On Windows and macOS there is always a window server for the logged-in user.
    On Linux a headed Chromium needs an X/Wayland display; without one it either
    crashes or silently renders off-screen, which would leave the user staring at
    a job that says "solve the challenge" with no challenge to solve.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def enabled() -> bool:
    """Whether manual verification may run in this process.

    ``auto`` (the default) is on exactly when there is somewhere to draw the
    window; ``on`` forces it; ``off`` disables it. Anything unrecognised is
    treated as auto, so a typo can never leave a request thread waiting forever
    on a browser that was never going to open.
    """
    mode = (os.environ.get("JOBSCOPE_MANUAL_WALL") or "auto").strip().lower()
    if mode in ("off", "0", "false", "no"):
        return False
    if mode in ("on", "1", "true", "yes"):
        return True
    return _display_available()


def inline_enabled() -> bool:
    """Whether a *running search* should block on a browser instead of failing.

    Distinct from ``enabled()``, which only governs the on-demand endpoint. This
    is deliberately stricter: blocking a live search on a human is a big change
    in behaviour, so it happens only when the operator asked for it explicitly
    with ``JOBSCOPE_MANUAL_WALL=on`` - never on ``auto``, which is the default a
    normal desktop run gets (there, the blocked board is surfaced with a
    "Verify in browser" button instead).
    """
    return (os.environ.get("JOBSCOPE_MANUAL_WALL") or "").strip().lower() in (
        "on", "1", "true", "yes")


def resolve(board: str, query: str = "") -> tuple[str, dict] | None:
    """Map a board name to its entry URL and query params, or None.

    Reuses the registry in ``boards.py`` rather than duplicating URLs, so a board
    that moves its endpoint in one place moves it here too. Imported lazily: this
    module is imported by the dashboard, which must start even when Scrapling is
    not installed.
    """
    board = (board or "").strip()
    if not board:
        return None
    try:
        import boards as _boards
    except Exception:
        return None
    spec = _boards.SPECS.get(board)
    if spec is None:
        return None
    return spec.url, spec.params(query or "")


def _looks_walled(page) -> bool:
    """Delegate to boards.py so the two never disagree on what a wall is."""
    try:
        import boards as _boards
        return _boards._looks_walled(page)
    except Exception:
        return False


def _visible_text(page) -> str:
    try:
        body = page.css("body")
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return " ".join(body[0].get_all_text().split())
    except Exception:
        return ""


def _cleared(page, board: str) -> bool:
    """True once the page presents real results rather than a challenge.

    A challenge page is never cleared, however much text it carries. For a board
    we can parse, "cleared" means the board's own parser returned rows - NOT a
    long body, because a sign-in wall ("Sign in to LinkedIn") is a long, marker-
    free page with no job cards, and treating it as cleared is exactly how a
    verification run ended with "no cards" the moment a login was required. Only
    a raw URL with no known parser falls back to a substantial visible body.
    """
    if _looks_walled(page):
        return False
    if board:
        try:
            import boards as _boards
            parser = _boards.PARSERS.get(board)
        except Exception:
            parser = None
        if parser is not None:
            try:
                return bool(parser(page))
            except Exception:
                return False
        # Unknown board name but a raw URL was given: nothing to parse with.
    return len(_visible_text(page)) >= MIN_BODY_CHARS


def _wait_for_human(page, board: str, deadline_s: int, on_tick) -> bool:
    """Block until the challenge is cleared or the deadline passes.

    The window stays open for the whole wait because Scrapling holds the page
    open for the duration of ``page_action``. Returns whether it cleared; a page
    the user closed mid-wait is re-checked once and reported as it then stands
    rather than raising.
    """
    end = time.monotonic() + deadline_s
    while True:
        if _cleared(page, board):
            return True
        remaining = int(end - time.monotonic())
        if remaining <= 0:
            return _cleared(page, board)
        try:
            on_tick(remaining)
            page.wait_for_timeout(1500)
        except Exception:
            return _cleared(page, board)


def _extract(page, board: str) -> list[dict]:
    """Parse the solved page into plain listing dicts.

    Tries the named board's parser first; if the caller gave no board (a raw URL)
    or that parser finds nothing, the remaining parsers get one shot each. A
    parser that raises is skipped, not fatal - a new board's markup should not
    turn a successful solve into a failed job.
    """
    try:
        import boards as _boards
    except Exception:
        return []
    candidates: list = []
    if board and board in _boards.PARSERS:
        candidates.append(_boards.PARSERS[board])
    candidates.extend(fn for fn in _boards.PARSERS.values() if fn not in candidates)
    for parser in candidates:
        try:
            rows = parser(page)
        except Exception:
            continue
        if rows:
            return [r.to_dict() for r in rows]
    return []


def _profile_dir() -> str:
    """A persistent Chromium profile so a sign-in survives to the next run.

    Boards that wall with a captcha usually want an account too. A throwaway
    profile means the person signs in again on every single verification, which
    is the complaint that produced this: the login "was not saved". Persisting
    cookies/localStorage here keeps them signed in. Callers can point it
    elsewhere with JOBSCOPE_MANUAL_PROFILE, or blank it to opt back out.
    """
    override = os.environ.get("JOBSCOPE_MANUAL_PROFILE")
    if override is not None and not override.strip():
        return ""
    base = override.strip() if override else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".browser_profile")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        return ""
    return base


def _fetch_page(url: str, params, page_action, seconds: int):
    """Open a *visible* browser and run ``page_action`` on the loaded page.

    Imported and constructed here so importing this module never touches
    Scrapling or a browser. Unlike the automated path, resources and ads are left
    on: a captcha widget is an image/iframe, and dropping those is exactly how a
    solvable challenge turns unsolvable.
    """
    from scrapling.fetchers import StealthyFetcher

    kwargs = dict(
        params=params or None,
        headless=False,
        network_idle=False,
        disable_resources=False,
        block_ads=False,
        solve_cloudflare=True,
        page_action=page_action,
        timeout=int(seconds * 1000) + 15000,
    )
    profile = _profile_dir()
    if profile:
        kwargs["user_data_dir"] = profile
    return StealthyFetcher.fetch(url, **kwargs)


def _solve_sync(url: str, board: str, query: str, on_open, on_tick) -> dict:
    """Run one manual solve to completion. Blocking; called from the worker.

    Returns a plain result dict: ``{"ok": bool, "detail": str, "listings": [...]}``.
    Never raises - a browser failure and a wall that is still up are the same
    thing to the caller, which is "tell the user it did not work".
    """
    params = None
    if board:
        resolved = resolve(board, query)
        if resolved:
            url, params = resolved
    deadline = _seconds()
    state = {"opened": False}

    def page_action(page):
        # Runs after navigation and after Cloudflare's own solver, so this is the
        # moment the window is actually on screen with whatever challenge remains.
        if not state["opened"]:
            state["opened"] = True
            try:
                on_open()
            except Exception:
                pass
        _wait_for_human(page, board, deadline, on_tick)

    try:
        page = _fetch_page(url, params, page_action, deadline)
    except Exception as exc:
        return {"ok": False, "listings": [],
                "detail": f"The browser could not open that page ({type(exc).__name__})."}

    if _looks_walled(page):
        return {"ok": False, "listings": [],
                "detail": "The challenge is still showing - it was not solved in time."}

    listings = _extract(page, board)
    if not listings:
        return {"ok": False, "listings": [],
                "detail": ("The page opened but no job cards could be read from it. "
                           "The board may have served a different layout.")}
    return {"ok": True, "listings": listings, "detail": ""}


def _set(job_id: str, **fields) -> None:
    with _LOCK:
        rec = _JOBS.get(job_id)
        if rec is not None:
            rec.update(fields)
            rec["updated"] = time.time()


def _run(job_id: str, url: str, board: str, query: str) -> None:
    started = time.time()
    _set(job_id, state="starting",
         message="Opening a browser window for you to solve the challenge...")

    def on_open():
        _set(job_id, state="waiting", opened_at=time.time(),
             message="Solve the challenge in the browser window; this page updates "
                     "on its own once it is through.")

    def on_tick(remaining: int):
        _set(job_id, message=f"Waiting for you to solve the challenge "
                             f"({remaining}s left before it gives up)...")

    try:
        result = _solve_sync(url, board, query, on_open, on_tick)
    except Exception as exc:  # defensive: the worker must never die silently
        result = {"ok": False, "listings": [],
                  "detail": f"That verification failed ({type(exc).__name__})."}

    if result["ok"]:
        _set(job_id, state="done", listings=result["listings"],
             detail="", count=len(result["listings"]),
             seconds=round(time.time() - started, 1),
             message=f"Got {len(result['listings'])} listing(s).")
    else:
        state = "blocked" if "still showing" in result["detail"] else "failed"
        _set(job_id, state=state, listings=[], detail=result["detail"],
             seconds=round(time.time() - started, 1), message=result["detail"])


def solve_inline(board: str, query: str = "") -> dict:
    """Solve one board synchronously, for the blocked-search inline path.

    No job record and no callbacks: the caller is the search itself, which is
    already blocked waiting, and it just needs the result. Returns the same
    ``{"ok", "detail", "listings"}`` shape as ``_solve_sync``.
    """
    return _solve_sync("", board, query, lambda: None, lambda r: None)


def start(url: str = "", board: str = "", query: str = "") -> dict:
    """Begin a manual verification and return its initial job record.

    Either a board name (resolved to its URL) or an explicit URL is required. The
    returned dict is a snapshot; poll ``status(job_id)`` for progress.
    """
    board = (board or "").strip()
    url = (url or "").strip()
    if not url and board:
        resolved = resolve(board, query or "")
        if resolved:
            url = resolved[0]
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    rec = {
        "id": job_id, "url": url, "board": board, "query": query or "",
        "state": "starting", "message": "Starting...", "detail": "",
        "listings": [], "count": 0, "started": now, "updated": now,
        "seconds": 0.0,
    }
    with _LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            # Drop the oldest finished job so the map cannot grow without bound.
            for old in sorted(_JOBS, key=lambda k: _JOBS[k]["started"]):
                if _JOBS[old]["state"] in _TERMINAL:
                    _JOBS.pop(old, None)
                    break
        _JOBS[job_id] = rec
    threading.Thread(target=_run, args=(job_id, url, board, rec["query"]),
                     name=f"manual-verify-{job_id}", daemon=True).start()
    return dict(rec)


def status(job_id: str) -> dict | None:
    """A copy of the job record, or None when the id is unknown."""
    with _LOCK:
        rec = _JOBS.get(job_id)
        return dict(rec) if rec else None


def wait(job_id: str, timeout: float | None = None) -> dict | None:
    """Block until the job reaches a terminal state (or the timeout). For tests."""
    end = None if timeout is None else time.monotonic() + timeout
    while True:
        rec = status(job_id)
        if rec is None:
            return None
        if rec["state"] in _TERMINAL:
            return rec
        if end is not None and time.monotonic() >= end:
            return rec
        time.sleep(0.05)


def reset() -> None:
    """Forget every job. Tests only; a live process never needs this."""
    with _LOCK:
        _JOBS.clear()
