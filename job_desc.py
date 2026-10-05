"""Full job descriptions, fetched in the background and cached by URL.

The problem this solves is ordering. A resume should be tailored to the posting,
but collecting the posting costs 8-130 seconds behind a bot wall, and the user is
sitting there watching a spinner. So the two are split:

    render now  ->  tailor from whatever text already exists (pasted, or the
                   description the search already collected, or nothing)
    fetch after ->  get the full text in the background, cache it, and let the
                   user regenerate with it

Everything here is synchronous and blocking because it drives a browser. Callers
on the event loop go through `ensure()`, which hands the work to a thread and
returns immediately with whatever the cache already knows.

Why a cache at all: the fetch is the expensive part, a posting's text does not
change, and every resume for the same posting wants the same text. `job_desc_cache`
is keyed by URL and shared across accounts - the text is public and contains
nothing about the candidate, so one row serves everyone.

Why failures are cached too: a row with empty text means "we already asked this
posting and there was nothing there". Without it, every click on a walled page
would launch another Chromium, which is how a scraper gets itself blocked
faster rather than slower.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import auth
from mcp_server_indeed_scraper import _clean_desc

# A posting's text is effectively frozen once it is live, so a day is generous.
# It also bounds the damage of a posting that gets closed and reposted with
# different text: we would serve the old copy for at most this long.
DESC_TTL_SECONDS = 24 * 3600

# Enough to rank skills and drive BM25 over the candidate's documents. A full
# posting body is often 8k characters; past this point the extra text is
# boilerplate about benefits and legal notices that only adds retrieval noise.
DESC_MAX_CHARS = 6000

# Below this, what came back is a title, a "Sorry, no longer accepting
# applications" banner, or the site chrome - not a posting. Worth a browser.
DESC_MIN_CHARS = 400

# Two at a time. Each one is a Chromium instance, and more than that on a shared
# box is a way to make the whole app feel slow rather than to go faster.
_WORKERS = 2

# Query parameters that identify the campaign, not the posting. Stripping them
# means the URL from a search result and the URL a user pasted collapse onto the
# same cache row instead of fetching the same posting twice.
_TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "yclid", "igshid", "mc_cid",
    "mc_eid", "ref", "refsrc", "referrer", "source", "src", "trk", "trkinfo",
    "trackingid", "originalsubdomain", "lipi", "licu", "eBP", "original_referer",
}

# Tried in order. The first selector that yields a plausible posting wins, and
# the name is recorded in the cache so the UI can say where the text came from.
#
# Ordered most-specific first: `show-more-less-html__markup` is LinkedIn's
# "show more" body, `jobDescriptionText` is Indeed's dedicated element. The
# generic rules at the end exist for the boards with no class worth naming - a
# generic match is fine as long as it is filtered by length afterwards.
_DESC_SELECTORS = [
    ("linkedin", 'div.show-more-less-html__markup'),
    ("linkedin", 'div.description__text'),
    ("indeed", '#jobDescriptionText'),
    ("indeed", 'div.jobsearch-JobComponent-description'),
    ("greenhouse", '#app_job_description'),
    ("lever", '.content'),
    ("ashby", '.job-posting-description'),
    ("workable", '.job__description'),
    ("generic", '#job-description'),
    ("generic", '#jobDescription'),
    ("generic", '[class*="job-description"]'),
    ("generic", '[class*="jobDescription"]'),
    ("generic", '[id*="description"]'),
]

# Bodies that are not descriptions. Checked before length, because a wall page
# is often longer than a real posting.
_WALL_MARKERS = (
    "captcha", "are you a robot", "unusual traffic", "enable javascript",
    "checking your browser", "access denied", "security check",
    "please verify you are human", "sign in to continue", "cf-browser-verification",
)

# One worker per posting URL at most, however many times it is requested. Two
# callers asking for the same URL must not start two browsers.
_inflight: set[str] = set()
_inflight_lock = threading.Lock()

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_WORKERS,
                                       thread_name_prefix="jobdesc")
        return _pool


def normalize_url(url: str) -> str:
    """Canonical cache key for a posting.

    Drops the fragment and campaign parameters and lowercases the host, so the
    link a board handed us and the link a user pasted land on the same row.
    """
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if not parts.netloc:
        return ""
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS]
    # Some boards sign the URL; dropping the signature would break nothing for
    # us because we only ever re-fetch the canonical form.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path, urlencode(sorted(kept)), ""))


def looks_like_wall(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _WALL_MARKERS)


def _selector_text(node) -> str:
    """Plain text of one matched node, capped."""
    try:
        blob = node.html_content or ""
    except Exception:
        return ""
    return _clean_desc(blob, DESC_MAX_CHARS)


def extract(page) -> tuple[str, str]:
    """Pull the posting body out of a fetched page.

    Returns (text, source). An empty string means this page had no description
    we could read - which is a normal outcome for a walled or JS-only page, not
    an error, and the caller records it so we stop asking.
    """
    for name, sel in _DESC_SELECTORS:
        try:
            nodes = page.css(sel)
        except Exception:
            continue
        for node in nodes:
            text = _selector_text(node)
            if len(text) < DESC_MIN_CHARS:
                continue
            # A match inside a block page is worse than no match at all: it would
            # cache the wall's text as the job description and tailor to it.
            if looks_like_wall(text):
                continue
            # _clean_desc caps by word count, so its result can overshoot the
            # limit by a few characters. Truncate here rather than only on the
            # way into the cache: callers get this value directly, and the cap
            # is what bounds the BM25 query built from it.
            return text[:DESC_MAX_CHARS], name
    return "", "none"


def cached(url: str) -> dict | None:
    """The cached row for a posting, fresh or not. None if never fetched."""
    key = normalize_url(url)
    if not key:
        return None
    try:
        with auth.connect() as conn:
            row = conn.execute(
                "SELECT url, text, source, fetched_at FROM job_desc_cache "
                "WHERE url = ?", (key,)).fetchone()
    except Exception:
        # A missing table means init_db has not run; treat as "nothing cached"
        # rather than failing the resume that triggered the lookup.
        return None
    return dict(row) if row else None


def store(url: str, text: str, source: str) -> None:
    key = normalize_url(url)
    if not key:
        return
    try:
        with auth.connect() as conn:
            conn.execute(
                "INSERT INTO job_desc_cache (url, text, source, fetched_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(url) DO UPDATE SET "
                "text = excluded.text, source = excluded.source, "
                "fetched_at = excluded.fetched_at",
                (key, (text or "")[:DESC_MAX_CHARS], source or "none", time.time()))
    except Exception:
        pass


def is_fresh(row: dict | None) -> bool:
    if not row:
        return False
    return (time.time() - float(row.get("fetched_at") or 0)) < DESC_TTL_SECONDS


def status_of(url: str) -> dict:
    """What the client polls for: is there text, is one coming, or was there none.

    `ready` and `empty` are both terminal - the client stops polling on either.
    Only `pending` means come back, so a posting we cannot read costs one poll
    cycle rather than an open-ended retry loop.
    """
    key = normalize_url(url)
    if not key:
        return {"status": "empty", "chars": 0, "source": "none"}
    row = cached(key)
    if row and is_fresh(row):
        if row.get("text"):
            return {"status": "ready", "chars": len(row["text"]),
                    "source": row.get("source") or "none",
                    "text": row["text"]}
        return {"status": "empty", "chars": 0, "source": "none"}
    return {"status": "pending", "chars": 0, "source": "none"}


def fetch_sync(url: str, *, timeout: int = 35) -> tuple[str, str]:
    """Fetch and store one posting's description. Blocking; ~1s to ~130s.

    Two tiers, cheapest first. The impersonated HTTP path clears most walls in
    about a second. Only when it comes back with nothing usable do we spend a
    browser on it, and even then we stop waiting for the network to go quiet -
    a posting page has trackers polling long after its text is readable, and
    waiting for them cost 131s against 8s for the same text.

    Returns (text, source). Never raises: a wall, a timeout and a parse failure
    are all the same outcome to the caller, which is "no text, do not retry
    until the TTL expires".
    """
    key = normalize_url(url)
    if not key:
        return "", "none"
    import boards
    try:
        page = boards.fetch(key, timeout=timeout, browser=False)
        text, source = extract(page)
        if text:
            store(key, text, source)
            return text, source
    except Exception:
        pass
    try:
        page = boards.fetch(key, timeout=timeout, browser=True,
                            network_idle=False)
        text, source = extract(page)
        if text:
            text, source = text, source + "+browser"
        store(key, text, source)
        return text, source
    except Exception:
        # Recorded as a miss so the next click reads the cache instead of
        # launching another browser at the same wall.
        store(key, "", "none")
        return "", "none"


def _run(url: str) -> None:
    try:
        fetch_sync(url)
    finally:
        with _inflight_lock:
            _inflight.discard(url)


def ensure(url: str) -> dict:
    """Kick off a background fetch if the cache cannot answer, and return status.

    Cheap to call on every resume generation: a fresh row short-circuits without
    touching the executor, and a URL already being fetched is not queued twice.
    """
    key = normalize_url(url)
    if not key:
        return {"status": "empty", "chars": 0, "source": "none"}
    row = cached(key)
    if is_fresh(row):
        return status_of(key)
    with _inflight_lock:
        if key in _inflight:
            return {"status": "pending", "chars": 0, "source": "none"}
        _inflight.add(key)
    try:
        _executor().submit(_run, key)
    except Exception:
        with _inflight_lock:
            _inflight.discard(key)
        return {"status": "empty", "chars": 0, "source": "none"}
    # A stale row that still has text is reported as pending rather than ready:
    # the user keeps the old copy while the refresh runs.
    return {"status": "pending", "chars": 0, "source": "none"}


def shutdown() -> None:
    """Stop accepting work. Used by tests so a background browser cannot outlive them."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None
    with _inflight_lock:
        _inflight.clear()
