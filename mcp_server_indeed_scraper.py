"""
MCP SERVER: search_job_boards — multi-site job-board search tool.

Searches several job boards for the user's query and aggregates results into
one normalized feed, so the agent speaks to ONE tool instead of five:

    Indeed, LinkedIn Jobs, Naukri, Glassdoor, Foundit (ex-Monster India),
    Internshala

Status notes
- Boards are probed IN PARALLEL (one worker each), so even an all-blocked run
  finishes in roughly one network timeout instead of six in a row. A board
  that just failed is skipped for a short TTL cache so back-to-back hunts don't
  re-hit the wall.
- Every board is fetched with a plain browser User-Agent. LinkedIn, Glassdoor,
  Naukri, Foundit and Internshala are heavily JS/anti-bot protected (HTTP 403 /
  CAPTCHA, or network-level refusal) and are *expected* to fail from a plain
  HTTP client. WeWorkRemotely serves static HTML, so it usually returns real
  listings; blocked sources are reported per-source rather than crashing the
  run, and the Gmail fallback node still covers the agent when every source
  is blocked.
- The tool returns a human-readable summary AND a `###JOBS_JSON###` block that
  the dashboard parses into a flashcard deck with Apply buttons.
- Entries are filtered for "remote work that is NOT based in India" when
  remote_only is True (the user's stated constraint).

Env vars: none required (public search pages only).
"""

import html as html_mod
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("job-boards-scraper")

JOBS_MARKER = "###JOBS_JSON###"
MAX_JOBS = 20

# Boards that just failed are remembered briefly so a follow-up search within
# BLOCK_CACHE_SECONDS skips them instantly instead of re-hitting the wall.
BLOCK_CACHE_SECONDS = 90
_fail_until: dict[str, float] = {}
_block_lock = threading.Lock()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TAG_RE = re.compile(r"<[^>]+>")

# Locations that mean the role is tied to India — dropped when remote_only.
INDIA_TOKENS = re.compile(
    r"\b(india|indian|hyderabad|bangalore|bengaluru|mumbai|pune|delhi|noida|"
    r"gurugram|gurgaon|chennai|kolkata|ahmedabad|coimbatore|indore|jaipur|"
    r"lucknow|kerala|karnataka|maharashtra|telangana|tamil ?nadu|uttar ?pradesh|"
    r"rajasthan|gujarat|bihar|punjab|west ?bengal|andhra|remote-india|"
    r"work from home india)\b",
    re.IGNORECASE,
)

# The user targets entry-level (0-2 years) roles: drop senior/lead/principal
# titles and any listing demanding 5+ years, mirroring agent.py's fallback
# filters so board results are held to the same bar.
SENIORITY_FILTER = re.compile(
    r"senior|lead\b|manager|principal|staff|architect|director|head of",
    re.IGNORECASE,
)
EXPERIENCE_FILTER = re.compile(r"\b(?:[5-9]|\d{2,})\+\s*(?:years|yrs)\b", re.IGNORECASE)


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
    Foundit) fail in ~3s instead of holding the parallel probe open for 10s."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    html_doc = resp.text
    if any(sig in html_doc.lower() for sig in ("captcha", "verify you are a human",
                                                "access denied", "unusual traffic")):
        raise RuntimeError("CAPTCHA / bot-wall")
    return html_doc


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
    any listing whose location matches an India city/state token."""
    if not remote_only:
        return jobs
    kept = []
    for j in jobs:
        loc = j.location
        if loc and INDIA_TOKENS.search(loc):
            continue  # role is India-tied — exclude
        kept.append(j)
    return kept


# ---------------------------------------------------------------------------
# Per-board scrapers (best-effort; each returns a list of Job or raises)
# ---------------------------------------------------------------------------

def _indeed(query: str) -> list[Job]:
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


def _linkedin(query: str) -> list[Job]:
    """Scrape LinkedIn Jobs for remote (f_WT=2 = fully remote) listings."""
    url = f"https://www.linkedin.com/jobs/search?keywords={query.replace(' ', '%20')}&location=Remote&f_WT=2"
    html_doc = _fetch(url)
    jobs = []
    for m in re.finditer(r'<a[^>]+href="(https://[a-z]+\.linkedin\.com/jobs/view/[^"]+)"[^>]*>(.*?)</a>',
                         html_doc, re.DOTALL):
        jobs.append(Job("LinkedIn", _text(m.group(2)), link=m.group(1)))
    # fallback: current base-card markup
    if not jobs:
        for m in re.finditer(r'<a[^>]*class="[^"]*base-card__full-link[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                             html_doc, re.DOTALL):
            jobs.append(Job("LinkedIn", _text(m.group(2)), link=m.group(1)))
    if not jobs:
        raise RuntimeError("auth / JS wall")
    return jobs


def _naukri(query: str) -> list[Job]:
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


def _glassdoor(query: str) -> list[Job]:
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


def _foundit(query: str) -> list[Job]:
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


def _internshala(query: str) -> list[Job]:
    """Scrape Internshala's keyword jobs page for matching roles."""
    slug = "-".join(query.lower().split())
    url = f"https://internshala.com/jobs/keyword-{slug}-jobs/"
    html_doc = _fetch(url)
    jobs = []
    for m in re.finditer(r'<a[^>]+href="(/job/[^"]+)"[^>]*>(.*?)</a>', html_doc, re.DOTALL):
        jobs.append(Job("Internshala", _text(m.group(2)),
                        link="https://internshala.com" + m.group(1)))
    if not jobs:
        raise RuntimeError("no results / bot wall")
    return jobs


def _weworkremotely(query: str) -> list[Job]:
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


BOARDS = {
    "Indeed": _indeed,
    "LinkedIn": _linkedin,
    "Naukri": _naukri,
    "Glassdoor": _glassdoor,
    "Foundit": _foundit,
    "Internshala": _internshala,
    "WeWorkRemotely": _weworkremotely,
}


@mcp.tool()
def search_job_boards(query: str, location: str = "Remote",
                      remote_only: bool = True) -> str:
    """Search multiple job boards (Indeed, LinkedIn, Naukri, Glassdoor,
    Foundit, Internshala) for matching roles and aggregate the results.

    Every board is tried independently; blocked boards are reported per-source
    instead of failing the whole search. When remote_only is True (default),
    India-tied postings are excluded — the user lives in India and only wants
    remote roles that are NOT based there.

    Returns a human-readable summary followed by a ###JOBS_JSON### block that
    the UI turns into Apply-able flashcards. If every source is blocked, the
    summary says so explicitly — fall back to the Gmail email-alert tool."""
    found: list[Job] = []
    sources = {}

    def probe(name, scrape_fn):
        # skip boards that just failed (short TTL cache), like an efficient
        # retry budget for bot-walled sites
        with _block_lock:
            skip_until = _fail_until.get(name, 0.0)
        if skip_until > time.time():
            return name, "blocked (cached)"
        try:
            rows = _outside_india(_dedupe(scrape_fn(query)), remote_only)
            rows = [j for j in rows
                    if not SENIORITY_FILTER.search(f"{j.title} {j.company} {j.location}")
                    and not EXPERIENCE_FILTER.search(f"{j.title} {j.company} {j.location} {j.salary}")]
            if rows:
                with _block_lock:
                    _fail_until.pop(name, None)  # it worked — clear any cache
                return name, rows[:MAX_JOBS]
            raise RuntimeError("all results were India-tied and filtered out")
        except Exception as exc:
            with _block_lock:
                _fail_until[name] = time.time() + BLOCK_CACHE_SECONDS
            return name, f"blocked ({_friendly(exc)})"

    # All boards in parallel: the six connections compete, so the step takes
    # as long as the *slowest* board (~one timeout) instead of six sequential.
    with ThreadPoolExecutor(max_workers=len(BOARDS)) as pool:
        futures = {pool.submit(probe, name, fn): name for name, fn in BOARDS.items()}
        for fut in as_completed(futures):
            name, result = fut.result()
            if isinstance(result, list):
                sources[name] = len(result)
                found.extend(result)
            else:
                sources[name] = result

    blocked_names = [n for n, s in sources.items() if not isinstance(s, int)]
    found = _dedupe(found)[:MAX_JOBS]

    lines = ["Job search across boards (remote, outside India):"]
    for name in BOARDS:
        val = sources.get(name)
        lines.append(f"[{name}] {val if isinstance(val, int) else val}")
    lines.append(f"[TOTAL] {len(found)} listing(s)")

    payload = {
        "sources": {k: (v if isinstance(v, int) else "blocked") for k, v in sources.items()},
        "blocked": blocked_names,
        "total": len(found),
        "remote_only": remote_only,
        "jobs": [j.to_dict() for j in found],
    }
    trailing = ""
    if len(blocked_names) == len(BOARDS) or not found:
        trailing = (". All boards were blocked — fall back to search_job_emails "
                    "for Gmail job alerts instead.")
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