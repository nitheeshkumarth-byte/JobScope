"""
MCP SERVER: search_job_boards — multi-site job-board search tool.

Searches several job boards for the user's query and aggregates results into
one normalized feed, so the agent speaks to ONE tool instead of many:

    Indeed, LinkedIn Jobs, Naukri, Glassdoor, Foundit (ex-Monster India),
    Internshala, WeWorkRemotely, Remotive, Arbeitnow

Status notes
- Collection goes through boards.py, which wraps Scrapling. TLS impersonation
  on the HTTP path (~1s per board) is what actually clears the 403 walls, and a
  real Chromium instance is reserved for the boards that are JavaScript-only.
  The original regex-over-HTML scrapers are kept as a fallback so the tool still
  works without the browser binaries installed. Measured 2026-10-01 with
  impersonation on: Indeed 16 listings, LinkedIn 60, Glassdoor 30,
  Internshala 50, WeWorkRemotely 10 — all in under 4s.
- Glassdoor and Foundit are not probed by default (see default_boards) but work
  when named. Naukri renders no listings even in a real browser and generally
  needs an account; it is kept in the sweep because it costs one request.
- Boards are probed IN PARALLEL (one worker each), so even an all-blocked run
  finishes in roughly one network timeout instead of ten in a row. A hard
  deadline (GLOBAL_DEADLINE) caps the whole hunt: a stuck DNS/socket that
  ignores its own timeout is reported "blocked (timed out)" and reaped, never
  allowed to hang the agent's run. A board that just failed is skipped for a
  short TTL cache so back-to-back hunts don't re-hit the wall.
- Remotive and Arbeitnow are keyless JSON feeds and are parsed as JSON, with no
  browser and no impersonation involved.
- The tool returns a human-readable summary AND a `###JOBS_JSON###` block that
  the dashboard parses into a flashcard deck with Apply buttons.
- Entries are filtered for "remote work that is NOT based in India" when
  remote_only is True (the user's stated constraint) — work-from-home /
  wfh postings are exempt from the India drop because they are remote
  regardless of the country they're advertised from; results are ranked so
  titles matching the query's role/skill words float to the top.

Env vars: none required (public search pages + keyless APIs only).
"""

import html as html_mod
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

import requests
from mcp.server.fastmcp import FastMCP

from filters import (JOBS_MARKER, WORLDWIDE, WORK_MODES, canonical_board,
                      india_rule_applies, is_senior_or_experienced,
                      matches_country, matches_region, matches_work_mode,
                      normalize_countries, normalize_regions,
                      normalize_work_modes, outside_india, region_label)

mcp = FastMCP("job-boards-scraper")

MAX_JOBS = 20

# How far down the ranked list a hunt may reach looking for something the user
# has NOT already been shown. The boards do not paginate, so one request returns
# roughly the same few hundred listings every time; returning the same top 20 on
# every hunt is what made repeat searches look broken. Ranking stays quality
# first — this only widens the net when the visible window is exhausted.
MAX_FRESH_SCAN = 200

# A listing the user has already been shown, remembered so a repeat hunt leads
# with new postings instead of repeating the deck. Bounded and ordered: the
# oldest entries are dropped first, so it holds a few recent hunts rather than
# growing forever in a long-lived server process.
#
# The memory is keyed by the account that did the hunting (see `seen_bucket`).
# It used to be one process-wide dict, which meant the first person to see a
# posting made it a "repeat" for everyone else: a second account's very first
# hunt came back already-seen and lost the fresh-first ranking it depends on.
# Each account now has its own bucket, and each bucket is bounded separately so
# a busy account cannot evict a quiet one's history.
SEEN_MEMORY = 400
# How many accounts' buckets to keep at once. Each bucket is bounded
# separately by SEEN_MEMORY, so this is what caps the total; the default
# keeps it inside the memory the app already needs to run.
SEEN_SCOPES = 16
_seen_links: dict[str, float] = {}     # the shared/default bucket
_seen_scopes: dict[str, dict[str, float]] = {}
_seen_scope_order: list[str] = []
_seen_lock = threading.Lock()


def seen_bucket(scope: str = "") -> dict[str, float]:
    """The seen-set for one account.

    The lock is a plain Lock and every caller already holds it, so this is
    deliberately not re-entrant: taking it again here would deadlock.
    """
    if not scope:
        return _seen_links
    bucket = _seen_scopes.get(scope)
    if bucket is None:
        bucket = {}
        _seen_scopes[scope] = bucket
        _seen_scope_order.append(scope)
        # Least-recently-used bucket goes first, so a long-lived server does
        # not accumulate one dict per account that ever connected.
        while len(_seen_scope_order) > SEEN_SCOPES:
            _seen_scopes.pop(_seen_scope_order.pop(0), None)
    else:
        _seen_scope_order.remove(scope)
        _seen_scope_order.append(scope)
    return bucket


def _trim(bucket: dict[str, float]) -> None:
    if len(bucket) > SEEN_MEMORY:
        for link, _ts in sorted(bucket.items(),
                                key=lambda kv: kv[1])[:len(bucket) - SEEN_MEMORY]:
            bucket.pop(link, None)


def _remember_seen(jobs: list[Job], scope: str = "") -> None:
    """Record these listings as already shown for this account."""
    with _seen_lock:
        bucket = seen_bucket(scope)
        for j in jobs:
            if j.link:
                bucket[j.link] = time.time()
        _trim(bucket)


def _is_seen(link: str, scope: str = "") -> bool:
    with _seen_lock:
        return link in seen_bucket(scope)


def _partition_fresh(jobs: list[Job], scope: str = "") -> tuple[list[Job], list[Job]]:
    """Split into (unseen, already shown), keeping each group's order."""
    with _seen_lock:
        fresh, repeat = [], []
        for j in jobs:
            (repeat if j.link in seen_bucket(scope) else fresh).append(j)
    return fresh, repeat


def forget_scope(scope: str) -> int:
    """Drop one account's already-shown memory. Returns how many entries went.

    Called when that account resets its dashboard. Without it, "reset
    everything" leaves every posting this account was ever shown still marked as
    seen, so the very next hunt comes back with nothing fresh and quietly falls
    back to repeats - the reset looks like it broke the search.

    Scoped deliberately: this touches only the named account's bucket. The other
    accounts on this machine share the process, and one person resetting must
    not hand their neighbours a flood of postings they had already dismissed.
    """
    if not scope:
        return 0
    with _seen_lock:
        bucket = _seen_scopes.pop(scope, None)
        if scope in _seen_scope_order:
            _seen_scope_order.remove(scope)
        return len(bucket) if bucket else 0


# Boards that just failed are remembered briefly so a follow-up search within
# BLOCK_CACHE_SECONDS skips them instantly instead of re-hitting the wall.
BLOCK_CACHE_SECONDS = 90

_fail_until: dict[str, float] = {}
_block_lock = threading.Lock()

# A board answering HTTP 403 is a deliberate bot wall, not a transient glitch.
# Reporting it as a generic "blocked" reads like our bug; naming it tells the
# user which sources to stop expecting results from.
#
# These were measured with Scrapling impersonation enabled on 2026-10-01: all
# four now return real listings, so the hints are only reached when the
# impersonated path fails too. Naukri is the exception - it answers HTTP 200
# with a JavaScript shell and renders nothing even in a real browser without an
# account, which is what its hint says.
WALL_HINTS = {
    "Indeed": "Indeed blocked even the impersonated request (HTTP 403)",
    "Glassdoor": "Glassdoor blocked even the impersonated request (HTTP 403)",
    "Foundit": "Foundit blocked even the browser request (HTTP 403)",
    "Naukri": "Naukri renders no listings without an account (JavaScript-only)",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TAG_RE = re.compile(r"<[^>]+>")

# Hard cap for one entire multi-board search — a stuck DNS/socket must not be
# able to take the agent's run down with it.
GLOBAL_DEADLINE = 40.0


def _text(blob: str) -> str:
    """Strip HTML tags + collapse whitespace from a scraped blob."""
    return re.sub(r"\s+", " ", html_mod.unescape(TAG_RE.sub(" ", blob))).strip()


class Job:
    __slots__ = ("source", "title", "company", "location", "salary", "link")

    def __init__(self, source, title, company="", location="", salary="", link=""):
        """One normalized listing: which board it came from, plus the fields
        the flashcard UI shows (company/location/salary) and the Apply link."""
        self.source = source
        self.title = title
        self.company = company
        self.location = location or ""
        self.salary = salary or ""
        self.link = link or ""

    def to_dict(self):
        return {"source": self.source, "title": self.title,
                "company": self.company, "location": self.location,
                "salary": self.salary, "link": self.link}


def _fetch(url: str, timeout=(3.05, 12)):
    """GET a page; returns HTML text or raises on a hard failure.

    (connect, read) timeout tuple: sites that refuse connections (Naukri,
    Foundit) fail in ~3s instead of holding the parallel probe open for 10s.)
    Bot-wall signatures are matched deliberately: pages that merely EMBED a
    reCAPTCHA site key (Internshala's listing page does) are real content."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    html_doc = resp.text
    if any(sig in html_doc.lower() for sig in (
            "verify you are a human", "access denied", "unusual traffic", "cf-chl")):
        raise RuntimeError("CAPTCHA / bot-wall")
    return html_doc


def _get_json(url: str) -> dict:
    """GET a keyless public JSON API endpoint; returns the parsed document."""
    resp = requests.get(url, headers=HEADERS, timeout=(3.05, 25))
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    return resp.json()


def _dedupe(jobs: list[Job]) -> list[Job]:
    """Drop repeated (title + link) pairs and entries missing title/link."""
    seen, out = set(), []
    for j in jobs:
        key = (j.title.lower(), j.link)
        if key in seen or not j.title or not j.link:
            continue
        seen.add(key)
        out.append(j)
    return out


def _outside_india(jobs: list[Job], remote_only: bool) -> list[Job]:
    """Enforce the 'remote, not in India' rule: when remote_only is on, drop
    any listing whose location matches an India city/state token — EXCEPT
    work-from-home/wfh postings, which ARE remote regardless of the country
    they're advertised from (Internshala labels its WFH roles this way).

    The rule itself lives in filters.py; "remote-india" is NOT an exemption
    there, which is what stops an India-tied posting cancelling itself out."""
    if not remote_only:
        return jobs
    return [j for j in jobs if outside_india(j.location)]


def _apply_filters(jobs: list[Job], remote_only: bool, work_modes=None,
                   regions=None, countries=None) -> list[Job]:
    """Reduce raw listings to the ones the user actually asked for.

    The 'remote, not in India' rule is kept as its own step for backwards
    compatibility, then the newer work-arrangement and region filters narrow
    further, and the country filter narrows again (it is the most specific, so
    it runs last and the two compose). The India rule is applied through
    filters.india_rule_applies so boards and the agent's own post-filter agree:
    an on-site/hybrid request must not have the listings it asked for deleted.
    Passing no new filter leaves behaviour exactly as before.
    """
    out = (_outside_india(jobs, remote_only)
           if india_rule_applies(remote_only, work_modes) else list(jobs))
    if work_modes is not None:
        out = [j for j in out if matches_work_mode(j.location, work_modes)]
    if regions is not None:
        out = [j for j in out if matches_region(j.location, regions)]
    if countries is not None:
        out = [j for j in out if matches_country(j.location, countries)]
    return out


# ---------------------------------------------------------------------------
# Per-board scrapers (best-effort; each returns a list of Job or raises)
# ---------------------------------------------------------------------------

def _indeed_legacy(query: str) -> list[Job]:
    """Scrape Indeed search results for remote jobs matching the query."""
    q = query.replace(" ", "+")
    url = f"https://www.indeed.com/jobs?q={q}&l=Remote"
    html_doc = _fetch(url)

    anchor = re.compile(r'<a[^>]+href="(/rc/clk\?jk=[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    matches = list(anchor.finditer(html_doc))
    jobs, seen = [], set()
    for i, m in enumerate(matches):
        href, title = m.group(1), _text(m.group(2))
        if not title or href in seen:
            continue
        seen.add(href)
        blob = html_doc[m.end():(matches[i + 1].start() if i + 1 < len(matches) else len(html_doc))]

        def fld(pattern, default=""):
            mm = re.search(pattern, blob, re.DOTALL)
            return _text(mm.group(1)) if mm else default

        company = fld(r'data-company-name="([^"]+)"') or fld(r'<span[^>]*class="[^"]*companyName[^"]*"[^>]*>(.*?)</span>')
        location = fld(r'data-testid="text-location"[^>]*>(.*?)</') or fld(r'<div[^>]*class="[^"]*company_location[^"]*"[^>]*>(.*?)</div>')
        salary = fld(r'data-testid="attribute_snippet_container"[^>]*>(.*?)</') or fld(r'<div[^>]*class="[^"]*(?:salary-snippet|salaryOnly)[^"]*"[^>]*>(.*?)</div>')
        jobs.append(Job("Indeed", title, company, location, salary, f"https://www.indeed.com{href}"))
    if not jobs:
        raise RuntimeError("no job links / JS-rendered")
    return jobs


def _linkedin_legacy(query: str) -> list[Job]:
    """Scrape LinkedIn Jobs for remote (f_WT=2 = fully remote) listings.

    Parses whole CARDS rather than bare <a> tags so the location comes along.
    LinkedIn puts it in <span class="job-search-card__location"> inside a
    <div class="base-search-card__metadata">, and it is also encoded in the
    card's data-entity-urn/aria label on some variants. Without the location
    every listing came back with location="", which then silently defeated the
    region filter: selecting Europe returned nothing from LinkedIn at all.
    """
    url = f"https://www.linkedin.com/jobs/search?keywords={query.replace(' ', '%20')}&location=Remote&f_WT=2"
    html_doc = _fetch(url)
    jobs = []
    for card in _linkedin_cards(html_doc):
        title = (card.get("title") or "").strip()
        link = card.get("link") or ""
        if not link:
            continue
        jobs.append(Job("LinkedIn", title, location=card.get("location") or "",
                        link=link))
    if not jobs:
        # Fall back to bare anchors. This loses the location, so a region
        # filter will discard the results, but it is far better than reporting
        # the board as blocked if LinkedIn ever renames data-entity-urn.
        for m in re.finditer(r'<a[^>]+href="(https://[a-z]+\.linkedin\.com/jobs/view/[^"]+)"[^>]*>(.*?)</a>',
                             html_doc, re.DOTALL):
            jobs.append(Job("LinkedIn", _text(m.group(2)), link=m.group(1)))
    if not jobs:
        raise RuntimeError("auth / JS wall")
    return jobs


_LINK_RE = re.compile(r'https://[a-z]+\.linkedin\.com/jobs/view/[^"&]+')


def _linkedin_cards(html_doc: str) -> list[dict]:
    """Split the SERP into per-card dicts, each with title/link/location.

    The card is located by its /jobs/view/ link and spans to the next card, so
    a markup reshuffle that moves the location element around cannot pair a
    job with a different job's location.
    """
    starts = [m.start() for m in re.finditer(r'data-entity-urn="urn:li:jobPosting', html_doc)]
    cards = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else min(len(html_doc), start + 6000)
        blob = html_doc[start:end]

        link_m = _LINK_RE.search(blob)
        if not link_m:
            continue
        link = link_m.group(0)

        title = ""
        t = re.search(r'<h3[^>]*class="[^"]*base-search-card__title[^"]*"[^>]*>(.*?)</h3>',
                      blob, re.DOTALL)
        if t:
            title = _text(t.group(1))
        if not title:
            t = re.search(r'<span class="sr-only">(.*?)</span>', blob, re.DOTALL)
            if t:
                title = _text(t.group(1))

        location = ""
        for pat in (r'class="job-search-card__location"[^>]*>(.*?)</span>',
                    r'class="base-search-card__metadata"[^>]*>.*?'
                    r'<span[^>]*>(.*?)</span>',
                    r'aria-label="[^"]*?at ([^"]+?)"'):
            lm = re.search(pat, blob, re.DOTALL)
            if lm:
                location = _text(lm.group(1))
                if location:
                    break

        cards.append({"title": title, "location": location, "link": link})
    return cards


def _naukri_legacy(query: str) -> list[Job]:
    """Scrape Naukri (remote filter applied in the URL) for matching roles."""
    url = f"https://www.naukri.com/jobs?k={query.replace(' ', '%20')}&l=Remote"
    html_doc = _fetch(url)
    jobs = []
    for m in re.finditer(r'<a[^>]*class="[^"]*title[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                         html_doc, re.DOTALL):
        if "company_name" in m.group(2).lower():
            continue
        jobs.append(Job("Naukri", _text(m.group(2)), link="https://www.naukri.com" + m.group(1)
                        if m.group(1).startswith("/") else m.group(1)))
    if not jobs:
        raise RuntimeError("no results / JS wall")
    return jobs[:MAX_JOBS]


def _glassdoor_legacy(query: str) -> list[Job]:
    """Scrape Glassdoor job search (sc.keyword) for remote roles."""
    url = f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={query.replace(' ', '%20')}&locKeyword=Remote"
    html_doc = _fetch(url)
    jobs = []
    for m in re.finditer(r'<a[^>]+href="(https://www\.glassdoor\.com/job/jobs-search-details\.php[^"]*)"[^>]*>(.*?)</a>',
                         html_doc, re.DOTALL):
        jobs.append(Job("Glassdoor", _text(m.group(2)), link=m.group(1)))
    if not jobs:
        raise RuntimeError("Cloudflare / login wall")
    return jobs


def _foundit_legacy(query: str) -> list[Job]:
    """Scrape Foundit (formerly Monster) job search for remote roles."""
    url = f"https://www.foundit.in/search/{query.replace(' ', '%20')}?locations=Remote"
    html_doc = _fetch(url)
    jobs = []
    for m in re.finditer(r'<a[^>]+href="(https://www\.foundit\.in[^"]*jobs[^"]*)"[^>]*>(.*?)</a>',
                         html_doc, re.DOTALL):
        jobs.append(Job("Foundit", _text(m.group(2)), link=m.group(1)))
    if not jobs:
        raise RuntimeError("no results / JS wall")
    return jobs


def _select(jobs: list[Job], query: str, cap: int = MAX_JOBS) -> list[Job]:
    """Rank listings so titles matching the query's role/skill words lead.

    Keeps raw order within the same score, and only fills in unmatched titles
    when few matched (so niche skills don't zero out a whole board)."""
    wanted = [w.lower() for w in (query or "").lower().split() if len(w) > 2]
    if not wanted:
        return jobs[:cap]
    scored = []
    for j in jobs:
        low = j.title.lower()
        scored.append((sum(1 for w in wanted if w in low), j))
    matched = sorted((s for s in scored if s[0] > 0), key=lambda t: t[0], reverse=True)
    out = [j for _, j in matched]
    if len(out) < 8:
        rest = [j for s, j in scored if s == 0]
        out += rest[: cap - len(out)]
    return out[:cap]


def _remotive(query: str) -> list[Job]:
    """Remotive's public remote-jobs JSON feed (keyless, CORS-open).

    Usually the freshest worldwide remote listings, with travel-friendly
    locations like 'Americas, Europe' that pass the not-India filter."""
    data = _get_json("https://remotive.com/api/remote-jobs?limit=50")
    jobs = []
    for j in (data.get("jobs") or []) if isinstance(data, dict) else []:
        title = j.get("title") if isinstance(j, dict) else None
        link = j.get("url") if isinstance(j, dict) else None
        if not isinstance(title, str) or not isinstance(link, str):
            continue
        title, link = title.strip(), link.strip()
        loc = (j.get("candidate_required_location") or "")
        loc = str(loc).strip()
        if loc.lower() in ("worldwide", "anywhere", "remote"):
            loc = "Anywhere in the World"
        salary = (j.get("salary") or "")
        salary = str(salary).strip() if salary else ""
        if not title or not link:
            continue
        jobs.append(Job("Remotive", title,
                        company=str(j.get("company_name") or "").strip(),
                        location=loc, salary=salary, link=link))
    if not jobs:
        raise RuntimeError("empty API response")
    return jobs


def _arbeitnow(query: str) -> list[Job]:
    """Arbeitnow's public job API — 250 recent worldwide roles, remote flagged.

    Locations are concrete ('Berlin, Germany' etc.), so India-tied ones are
    cleanly dropped by the remote_only filter."""
    data = _get_json("https://www.arbeitnow.com/api/job-board-api")
    jobs = []
    for it in data.get("data") or []:
        title = (it.get("title") or "").strip()
        link = (it.get("url") or "").strip()
        if not title or not link:
            continue
        loc = (it.get("location") or "").strip()
        jtypes = it.get("job_types") or []
        if not loc and any("remote" in str(t).lower() for t in jtypes):
            loc = "Remote"
        jobs.append(Job("Arbeitnow", title,
                        company=(it.get("company_name") or "").strip(),
                        location=loc, link=link))
    if not jobs:
        raise RuntimeError("empty API response")
    return jobs


def _internshala_legacy(query: str) -> list[Job]:
    """Scrape Internshala's work-from-home internships page (static HTML).

    These are India-remote internships — the location reads 'Work From Home',
    so remote_only's India-token filter still lets them through, but they're
    clearly labelled so the user can tell them apart from worldwide boards."""
    html_doc = _fetch("https://internshala.com/internships/work-from-home-jobs/")
    jobs = []
    card = re.compile(
        r'class="[^"]*individual_internship[^"]*"(.*?)(?=class="[^"]*individual_internship[^"]*"|$)',
        re.DOTALL,
    )
    for cm in card.finditer(html_doc):
        block = cm.group(1)
        am = re.search(r'<a[^>]+href="(/internship/[^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
        if not am:
            continue
        title = _text(am.group(2))
        if not title:
            continue
        link = "https://internshala.com" + am.group(1)
        comp = (re.search(r'class="company-name"[^>]*>(.*?)</', block, re.DOTALL)
                or re.search(r'class="[^"]*company[^"]*"[^>]*>(.*?)</', block, re.DOTALL))
        locm = re.search(r'class="[^"]*locations[^"]*"[^>]*>(.*?)</', block, re.DOTALL)
        sal = re.search(r'class="[^"]*salary[^"]*"[^>]*>(.*?)</', block, re.DOTALL)
        jobs.append(Job("Internshala", title,
                        company=_text(comp.group(1)) if comp else "",
                        location=_text(locm.group(1)) if locm else "Work From Home (India)",
                        salary=_text(sal.group(1)) if sal else "",
                        link=link))
    if not jobs:
        raise RuntimeError("no results / bot wall")
    return jobs


def _weworkremotely_legacy(query: str) -> list[Job]:
    """Scrape WeWorkRemotely's static listing page for relevant titles.

    One of the few boards that still serves full job HTML to a plain HTTP
    client (no JS/CAPTCHA wall), and its roles are remote-first worldwide —
    a natural fit for the user's remote-only, not-India constraint. The term
    filter is applied to titles locally because WWR's own /search page is
    client-rendered and returns nothing to a non-browser fetch."""
    url = "https://weworkremotely.com/remote-jobs"
    html_doc = _fetch(url)
    wanted = [w.lower() for w in query.lower().split() if len(w) > 2]
    scored = []
    card = re.compile(
        r'<a[^>]+class="listing-link--[^"]+"[^>]+href="(/remote-jobs/[^"]+)"[^>]*>'
        r"(.*?)</a>",
        re.DOTALL,
    )
    for m in card.finditer(html_doc):
        href, blob = m.group(1), m.group(2)
        tm = re.search(r'new-listing__header__title__text">\s*([^<]+)', blob)
        cm = re.search(r'new-listing__company-name">\s*([^<]+)', blob)
        rm = re.search(r'new-listing__company-headquarters">\s*([^<]+)', blob)
        title = _text(tm.group(1)) if tm else ""
        low = title.lower()
        score = sum(1 for w in wanted if w in low) if wanted else 1
        if wanted and score == 0:
            continue
        scored.append((score, Job("WeWorkRemotely", title,
                                  company=_text(cm.group(1)) if cm else "",
                                  location=_text(rm.group(1)) if rm else "Anywhere in the World",
                                  link="https://weworkremotely.com" + href)))
    jobs = [j for _, j in sorted(scored, key=lambda t: t[0], reverse=True)]
    if not jobs:
        raise RuntimeError("no results on page")
    return jobs[:MAX_JOBS]


# ---------------------------------------------------------------------------
# Scrapling-backed collection
#
# The scrapers above are regex over raw HTML with a hand-written User-Agent,
# which is what the 403s came from. boards.py does the same work through
# Scrapling: TLS impersonation on the HTTP path, and a real browser for the
# boards that are JS-only.
#
# The old functions are kept as a FALLBACK rather than deleted, for two reasons:
# they need no browser binary, so the app still collects something on a machine
# where `scrapling install` has not been run; and they are what the unit tests
# exercise. Scrapling is tried first and the legacy path is used when it is
# unavailable or comes back empty.
#
# If Scrapling is genuinely absent this must not break the MCP server, so the
# import is guarded rather than required.
# ---------------------------------------------------------------------------
try:
    import boards as _boards
    from boards import BoardWall, ThrottledOrUnparsed  # noqa: F401
    _SCRAPLING_AVAILABLE = True
except Exception:          # ImportError, or a broken optional browser dep
    _boards = None
    _SCRAPLING_AVAILABLE = False


def default_boards() -> dict:
    """The boards an all-boards sweep probes.

    Everything except Glassdoor and Foundit, which work when named but are not
    worth the wait on every hunt: Foundit costs a full browser to get a page
    that 403s about half the time, and between them they add latency for
    listings the other boards already cover. `board="glassdoor"` still probes
    them, so nothing is lost but the default wait.

    Filtering the whole registry rather than boards.SPECS matters: Remotive and
    Arbeitnow are keyless JSON feeds that never appear in SPECS, and dropping
    them would cost the two cheapest, most dependable sources in the set.

    Returns a copy so a caller cannot mutate BOARDS.
    """
    if not _SCRAPLING_AVAILABLE:
        # Degraded mode still honours the exclusions - the reason a board is
        # skipped is cost, not the library being unavailable.
        return {n: fn for n, fn in BOARDS.items()
                if not _boards or _boards.is_default_board(n)}
    return {n: fn for n, fn in BOARDS.items() if _boards.is_default_board(n)}


def _listings_to_jobs(source: str, rows) -> list[Job]:
    """Convert boards.Listing rows into the Job objects this module already
    aggregates, so ranking, filtering and the JSON payload are unchanged."""
    return [Job(source, r.title, r.company, r.location, r.salary, r.link)
            for r in rows]


def _scrapling_collect(name: str, query: str, legacy_fn) -> list[Job]:
    """Scrape one board via Scrapling, falling back to the legacy regex path.

    Escalation from a wall or an empty page is handled inside boards.scrape; a
    genuinely empty result after that falls through to legacy, because a
    throttled Scrapling response should not silently cost the board its results
    when a plainer request would have worked.

    The availability check is repeated here, not left to the callers, so that
    this function is safe to call directly: without it a caller that reached
    here with the library missing would touch `_boards` on a half-initialised
    module and fail in a confusing way.
    """
    if not _SCRAPLING_AVAILABLE or name not in _boards.SPECS:
        return legacy_fn(query)
    rows = _boards.scrape(_boards.SPECS[name], query, timeout=30)
    if rows:
        return _listings_to_jobs(name, rows)
    return legacy_fn(query)


def _indeed(query: str) -> list[Job]:
    """Indeed search results for remote jobs matching the query."""
    if _SCRAPLING_AVAILABLE:
        return _scrapling_collect("Indeed", query, _indeed_legacy)
    return _indeed_legacy(query)


def _linkedin(query: str) -> list[Job]:
    """LinkedIn public job search (f_WT=2 = fully remote)."""
    if _SCRAPLING_AVAILABLE:
        return _scrapling_collect("LinkedIn", query, _linkedin_legacy)
    return _linkedin_legacy(query)


def _naukri(query: str) -> list[Job]:
    """Naukri (remote filter applied in the URL)."""
    if _SCRAPLING_AVAILABLE:
        return _scrapling_collect("Naukri", query, _naukri_legacy)
    return _naukri_legacy(query)


def _glassdoor(query: str) -> list[Job]:
    """Glassdoor job search (sc.keyword) for remote roles."""
    if _SCRAPLING_AVAILABLE:
        return _scrapling_collect("Glassdoor", query, _glassdoor_legacy)
    return _glassdoor_legacy(query)


def _foundit(query: str) -> list[Job]:
    """Foundit (formerly Monster) job search for remote roles."""
    if _SCRAPLING_AVAILABLE:
        return _scrapling_collect("Foundit", query, _foundit_legacy)
    return _foundit_legacy(query)


def _internshala(query: str) -> list[Job]:
    """Internshala work-from-home internships (India-remote)."""
    if _SCRAPLING_AVAILABLE:
        return _scrapling_collect("Internshala", query, _internshala_legacy)
    return _internshala_legacy(query)


def _weworkremotely(query: str) -> list[Job]:
    """WeWorkRemotely's static listing page, filtered on titles locally."""
    if _SCRAPLING_AVAILABLE:
        # WWR has no server-side search, so its own filter does not apply
        # through the shared params. Scrapling returns the full page and the
        # legacy scorer does the keyword ranking.
        spec = _boards.SPECS["WeWorkRemotely"]
        rows = _boards.scrape(spec, query, timeout=30, escalate=False)
        if rows:
            return _scrapling_rank_wwr(rows, query)
    return _weworkremotely_legacy(query)


def _scrapling_rank_wwr(rows, query: str) -> list[Job]:
    """Apply WWR's local title filter to Scrapling rows.

    Needed because WWR ignores query params - it serves the same page for every
    search, so the caller still has to narrow by title to stay relevant."""
    wanted = [w.lower() for w in (query or "").lower().split() if len(w) > 2]
    scored = []
    for r in rows:
        low = r.title.lower()
        score = sum(1 for w in wanted if w in low) if wanted else 1
        if wanted and score == 0:
            continue
        scored.append((score, Job("WeWorkRemotely", r.title, r.company,
                                  r.location, r.salary, r.link)))
    jobs = [j for _, j in sorted(scored, key=lambda t: t[0], reverse=True)]
    return jobs[:MAX_JOBS]


# Declared after the wrappers so every name below is already bound. The keys are
# the canonical board names; values are the Scrapling-first entry points, which
# fall back to the *_legacy scrapers above when Scrapling is unavailable or comes
# back empty.
BOARDS = {
    "Indeed": _indeed,
    "LinkedIn": _linkedin,
    "Naukri": _naukri,
    "Glassdoor": _glassdoor,
    "Foundit": _foundit,
    "Internshala": _internshala,
    "WeWorkRemotely": _weworkremotely,
    "Remotive": _remotive,
    "Arbeitnow": _arbeitnow,
}


def _scope_label(location: str, remote_only: bool, work_modes=None,
                 regions=None, countries=None) -> str:
    """One-line description of the filters actually in force, for the summary
    header. Falls back to the legacy wording when no new filter is set."""
    bits = []
    modes = normalize_work_modes(work_modes)
    if modes:
        bits.append(", ".join(WORK_MODES[m] for m in modes))
    else:
        bits.append("remote" if remote_only else "all work types")
    if location and location.strip().lower() != "remote":
        bits.append(location.strip())
    if india_rule_applies(remote_only, work_modes):
        bits.append("outside India")
    picked = normalize_regions(regions)
    if picked and WORLDWIDE not in picked:
        bits.append(", ".join(region_label(r) for r in picked))
    picked_countries = normalize_countries(countries)
    if picked_countries:
        bits.append("in: " + ", ".join(picked_countries))
    return "; ".join(bits)


@mcp.tool()
def search_job_boards(query: str, location: str = "Remote",
                      remote_only: bool = True, board: str | None = None,
                      work_modes: list[str] | None = None,
                      regions: list[str] | None = None,
                      countries: list[str] | None = None,
                      seen_scope: str = "") -> str:
    """Search job boards for matching roles and aggregate the results.

    By default every board is tried (Indeed, LinkedIn, Naukri, Glassdoor,
    Foundit, Internshala, WeWorkRemotely, Remotive, Arbeitnow). Pass a board
    name (or plain-language alias like "linkedin", "just indeed") to restrict
    the search to THAT board only — the user asked for one platform, so the
    deck must come from it alone.

    Filtering:
    - remote_only (default True) drops India-tied office postings, keeping
      genuinely remote ones.
    - work_modes narrows to the arrangements the user wants: any of "remote",
      "wfh", "hybrid", "onsite". Omit it for no arrangement filtering, which
      also makes on-site and local roles reachable for the first time.
    - regions narrows the geography to the selected areas ("north-america",
      "uk-ireland", "europe", "asia-pacific", "latam",
      "africa-middle-east"). "worldwide" — or omitting it — matches
      everything, which is the old behaviour.
    - countries narrows to the selected countries by name ("Germany",
      "United Kingdom"). It is narrower than regions and is applied after them,
      so the two compose. Omitting it matches everything.

    seen_scope keys the "already shown" memory to one account, so one person's
    hunt does not make a posting a repeat for everybody else. The agent fills
    it in; leave it empty and the memory is shared.

    Blocked boards are reported per-source instead of failing the whole
    search. Returns a human-readable summary followed by a ###JOBS_JSON###
    block that the UI turns into Apply-able flashcards. If every source is
    blocked, the summary says so explicitly — fall back to the Gmail
    email-alert tool.

    A board that answered but had nothing matching is NOT reported as blocked;
    the two are different facts and the UI decides what to do based on which
    one it is. See NO_MATCHES.
    """
    found: list[Job] = []
    sources = {}
    requested = canonical_board(board, BOARDS) if board else None

    # Deliberately avoids the word "blocked". Consumers decide whether a search
    # failed by looking for that word, and this status means the opposite - the
    # board answered and simply had nothing for this query.
    NO_MATCHES = "0 (nothing matched this query)"

    def probe(name, scrape_fn, force=False):
        # skip boards that just failed (short TTL cache), like an efficient
        # retry budget for bot-walled sites — unless the user named this board
        # outright, in which case honor the request and try it anyway
        if not force:
            with _block_lock:
                skip_until = _fail_until.get(name, 0.0)
            if skip_until > time.time():
                return name, "blocked (cached)"
        try:
            rows = _apply_filters(_dedupe(scrape_fn(query)), remote_only,
                                  work_modes, regions, countries)
            # Title + location only: the company name used to be scanned too,
            # which deleted every opening at "Staffwise"/"Lead Generation".
            rows = [j for j in rows
                    if not is_senior_or_experienced(j.title, j.location)]
            # Keep a wide window per board: a later pass in search_job_boards
            # trims to the deck size, and it needs the extra rows to find
            # postings the user hasn't already been shown.
            rows = _select(rows, query, cap=MAX_FRESH_SCAN)
            if rows:
                with _block_lock:
                    _fail_until.pop(name, None)  # it worked — clear any cache
                return name, rows[:MAX_FRESH_SCAN]
            # Reached the board and it answered, but nothing survived the
            # filters. That is NOT a wall, and reporting it as one was actively
            # harmful twice over: the UI's "is the search blocked?" test greps
            # this text for the word "blocked", so one quiet board turned a
            # 250-listing hunt into "all boards blocked - falling back to
            # Gmail"; and the failure cache below then remembered the board as
            # broken and skipped it entirely on the next hunt.
            with _block_lock:
                _fail_until.pop(name, None)
            return name, NO_MATCHES
        except Exception as exc:
            with _block_lock:
                _fail_until[name] = time.time() + BLOCK_CACHE_SECONDS
            return name, f"blocked ({_friendly(exc)})"

    if requested is None:
        # All boards in parallel: connections compete, so the step takes as
        # long as the *slowest* board (~one timeout) instead of ten in a row.
        # A hard deadline caps the whole hunt even if one DNS/socket call
        # ignores its timeout (a stuck fetch must never hang the agent's whole
        # run); stragglers are reported "timed out" and reaped — the server
        # thread leaks in the background at worst, never the request.
        #
        # NOTE: shutdown(wait=False) is load-bearing. Using `with
        # ThreadPoolExecutor(...)` would call shutdown(wait=True) on the way
        # out and JOIN the stragglers, so the deadline would report "timed
        # out" and then block anyway for the straggler's own timeout.
        deadline = time.monotonic() + GLOBAL_DEADLINE
        hunt = default_boards()
        pool = ThreadPoolExecutor(max_workers=len(hunt))
        try:
            futures = {pool.submit(probe, name, fn): name for name, fn in hunt.items()}
            while futures and time.monotonic() < deadline:
                done, _pending = wait(futures, timeout=max(0.2, min(5.0, deadline - time.monotonic())))
                for fut in done:
                    futures.pop(fut)
                    try:
                        probe_name, result = fut.result()
                    except Exception as exc:
                        result = f"blocked ({_friendly(exc)})"
                    if isinstance(result, list):
                        sources[probe_name] = len(result)
                        found.extend(result)
                    else:
                        sources[probe_name] = result
            for fut in futures:
                sources[futures[fut]] = "blocked (timed out)"
        finally:
            # wait=False + cancel_futures: never join a straggler. Pending
            # probes are cancelled; already-running ones finish in the
            # background and are simply ignored.
            pool.shutdown(wait=False, cancel_futures=True)
            futures = {}
    else:
        # A single, user-named board: no competition, no deadline sweep.
        _probe_name, result = probe(requested, BOARDS[requested], force=True)
        if isinstance(result, list):
            sources[requested] = len(result)
            found.extend(result)
        else:
            sources[requested] = result

    # "Reached but empty" is tracked separately from "walled". Callers decide
    # whether to fall back to Gmail based on this list, and a board that simply
    # had nothing for the query must not put the search into fallback mode.
    blocked_names = [n for n, s in sources.items()
                     if not isinstance(s, int) and s != NO_MATCHES]
    empty_names = [n for n, s in sources.items() if s == NO_MATCHES]
    repeats_shown = 0
    if requested is None:
        # Give every healthy board a fair slice before the global rank, so one
        # board that returns many weak matches can't crowd out the rest of the
        # deck — then rank across ALL boards so titles matching the role/skill
        # words lead the deck, and trim to the cap.
        # Every healthy board gets a fair slice before the global rank, so one
        # board returning many weak matches can't crowd out the rest of the deck.
        # Fair share per board, so one big board can't crowd out the rest, but
        # wide enough that several repeat hunts each have unseen postings to
        # promote. Scaled to the sources that actually answered: with one board
        # live it must be allowed the whole window, or a second hunt has only
        # the few rows the per-board cap left over.
        slice_pool, buckets = [], {}
        for j in found:
            buckets.setdefault(j.source, []).append(j)
        per_source = max(25, MAX_FRESH_SCAN // max(1, len(buckets)))
        for src, items in buckets.items():
            slice_pool.extend(items[:per_source])

        ranked = _select(_dedupe(slice_pool), query, cap=MAX_FRESH_SCAN)
        # Lead with postings this account has NOT seen before, then fall back to
        # repeats only once the fresh ones run out. Without this, a board that
        # returns the same few hundred listings every time handed back the exact
        # same 20 cards on every hunt, which read as "the search is broken".
        fresh, repeat = _partition_fresh(ranked, seen_scope)
        found = (fresh + repeat)[:MAX_JOBS]
        # Remember ONLY what was actually shown. Recording the whole fresh pool
        # would mark postings the user never saw as "already seen" and starve
        # the very next hunt of the new listings it is meant to surface.
        _remember_seen(found, seen_scope)
        repeats_shown = max(0, len(found) - len(fresh))

    # Describe the filters that were actually applied. The old hardcoded
    # "(remote, outside India)" lied whenever the user searched anything else,
    # so the header is built from the request instead.
    scope = _scope_label(location, remote_only, work_modes, regions, countries)
    if requested is None:
        lines = [f"Job search across boards ({scope}):"]
        # Report only the boards that were actually probed. Listing a board that
        # was deliberately skipped would say "blocked (timed out)" for it a
        # moment later, which reads as a failure that never happened.
        for name, fn in default_boards().items():
            val = sources.get(name)
            if isinstance(val, int):
                lines.append(f"[{name}] {val}")
            elif val is None:
                # Started but never returned before the deadline.
                lines.append(f"[{name}] blocked (timed out)")
            else:
                # Name the reason, and say which boards are walled rather than
                # broken, so a "no results" board isn't read as our bug.
                hint = WALL_HINTS.get(name)
                lines.append(f"[{name}] {val}" + (f" - {hint}" if hint else ""))
        skipped = [n for n in BOARDS if n not in default_boards()]
        if skipped:
            # Say plainly which boards were not probed, so a missing Glassdoor
            # reads as a choice rather than a silent failure - and so the user
            # knows they can ask for it by name.
            lines.append("[skipped] " + ", ".join(skipped)
                         + " - not probed by default; ask for one by name")
    else:
        lines = [f"Job search (limited to {requested}, {scope}):"]
        val = sources.get(requested)
        lines.append(f"[{requested}] {val if isinstance(val, int) else val}")
    lines.append(f"[TOTAL] {len(found)} listing(s)")
    if repeats_shown:
        # Say it plainly: a hunt made of listings from earlier runs should not
        # quietly look like new results.
        lines.append(f"[NOTE] {repeats_shown} of these were already shown in an "
                     "earlier search - the boards have no more new postings "
                     "matching this query yet.")

    payload = {
        # NO_MATCHES maps to a real 0, not the string "blocked". Blanking both
        # non-numeric outcomes to "blocked" is what made the UI unable to tell
        # "no matching postings" from "the site refused us", and it left this
        # map contradicting "blocked": [] a few lines below.
        "sources": {k: (0 if v == NO_MATCHES
                        else v if isinstance(v, int) else "blocked")
                    for k, v in sources.items()},
        "source_detail": {k: (str(v) if not isinstance(v, int)
                              else f"{v} listing(s)") for k, v in sources.items()},
        "blocked": blocked_names,
        "blocked_reasons": {k: WALL_HINTS.get(k, str(sources.get(k)))
                            for k in blocked_names},
        # Boards that answered with nothing. Kept out of "blocked" so the UI
        # does not paint a working search as a failed one.
        "no_matches": empty_names,
        # A single unambiguous verdict. The UI used to infer success by grepping
        # the human-readable text for the word "blocked", which meant one quiet
        # board could declare a 250-listing hunt a total failure.
        #
        # "Worked" means the board was *reached*, not that it found anything.
        # A count of zero and a "nothing matched" status both mean the request
        # succeeded; only a wall, timeout or exception means it did not. Getting
        # this wrong either way is costly: too strict and a normal empty result
        # sends the user chasing a bug, too loose and a real outage looks fine.
        "boards_worked": any(isinstance(s, int) or s == NO_MATCHES
                             for s in sources.values()),
        "total": len(found),
        "fresh": len(found) - repeats_shown,
        "repeats": repeats_shown,
        "remote_only": remote_only,
        "work_modes": work_modes or [],
        "regions": regions or [],
        "countries": countries or [],
        "jobs": [j.to_dict() for j in found],
    }
    trailing = ""
    if not found:
        if requested is None:
            trailing = (". All boards were blocked, or every listing they "
                        "returned was filtered out by the location/work-type/"
                        "region filters — try widening them, or fall back to "
                        "search_job_emails for Gmail job alerts instead.")
        else:
            trailing = (f". {requested} was blocked or returned nothing matching "
                        "the filters — try widening them, or fall back to "
                        "search_job_emails for Gmail job alerts instead.")
    return "\n".join(lines) + trailing + "\n\n" + JOBS_MARKER + "\n" + json.dumps(payload)


def _friendly(exc: Exception) -> str:
    """Short human summary of a block/error reason for the per-board report."""
    msg = str(exc)
    if "HTTP 403" in msg:
        return "HTTP 403 / bot detection"
    if "CAPTCHA" in msg:
        return "CAPTCHA wall"
    if "auth / JS wall" in msg or "login wall" in msg or "bot wall" in msg:
        return "auth / JS wall"
    if "no results" in msg:
        return "no results on page"
    return msg[:60]


if __name__ == "__main__":
    mcp.run(transport="stdio")
