"""
Board fetching and parsing, built on Scrapling.

Why this exists
---------------
The original scraper used `requests` with a hand-written User-Agent. That is
indistinguishable from a script to any bot wall, so Indeed / Glassdoor / Foundit
answered HTTP 403 and Naukri answered a JavaScript-only shell: five of nine boards
returned nothing. Scrapling fixes that at two very different layers, and which
layer is needed is an empirical question, so both are wired here:

1. `Fetcher` (curl_cffi) does TLS/JA3 impersonation - the request is
   byte-for-byte shaped like a real browser's. This clears the 403 walls in
   ~1s with no browser at all.
2. `StealthyFetcher` (patchright + Chromium) renders JavaScript and carries a
   real browser fingerprint. Slow (~15-45s) but it is the only thing that gets
   JS-only boards at all.

Measured on 2026-10-01, python developer / remote:
    board             cheap path              browser path
    Indeed            200, 32 cards, 1.2s     200, 75 cards, 44s
    LinkedIn          200, 60 cards, 0.9s     -
    Glassdoor         200, 30 links, 1.4s      -
    Internshala       200, 50 cards, 0.8s      -
    WeWorkRemotely    200, 10 cards, 1.6s      -
    Naukri            200, JS shell, 0 cards   needed
    Foundit           403                     needed

So the browser is the fallback, not the default: it is ~30x the cost for worse
or equal results on the boards that do answer HTTP.

Throttling
----------
The cheap path is rate-limited. Hammering it from one IP makes boards thin out
within a minute (Indeed went 32 cards -> 16 -> 0 across three consecutive
probes in one session). Two consequences are built into this module rather than
left as folklore:

- Every card lookup uses a *chain* of selectors, so a markup shuffle or a
  throttled response degrades to the next candidate instead of raising.
- "Fetched a page but found no cards" is reported as its own failure
  (`ThrottledOrUnparsed`) and is eligible for browser escalation, which is
  deliberately a different fingerprint.

Encoding
--------
LinkedIn serves `D\\xe9veloppeur` style titles, and decoding the body as latin-1
turns that into mojibake ("D?veloppeur") which then fails every downstream
match. Scrapling exposes the encoding the server actually declared, so text is
decoded through that and never through a hardcoded guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scrapling.fetchers import Fetcher, StealthyFetcher

__all__ = [
    "Listing", "ThrottledOrUnparsed", "BoardWall",
    "fetch", "parse_indeed", "parse_linkedin", "parse_naukri",
    "parse_glassdoor", "parse_foundit", "parse_internshala",
    "parse_weworkremotely", "PARSERS",
]


class BoardWall(Exception):
    """The board refused the request outright (403 / challenge page)."""


class ThrottledOrUnparsed(Exception):
    """The page came back but yielded no cards.

    Distinct from BoardWall on purpose: a 200 with nothing parseable is usually
    rate limiting rather than a hard wall, and it is the case where escalating to
    a real browser actually helps.
    """


@dataclass
class Listing:
    """One normalized posting. Mirrors the scraper's Job shape on purpose so
    the aggregation and ranking code downstream needs no changes."""
    source: str
    title: str
    company: str = ""
    location: str = ""
    salary: str = ""
    link: str = ""

    def to_dict(self) -> dict:
        return {"source": self.source, "title": self.title,
                "company": self.company, "location": self.location,
                "salary": self.salary, "link": self.link}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

# chrome is the impersonation profile. Pinned rather than "latest" so a curl_cffi
# upgrade cannot silently change which browser we claim to be.
IMPERSONATE = "chrome"

_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Challenge *pages*, not challenge *assets*. The original scraper matched the
# bare substring "captcha" against the whole body and flagged Internshala as
# blocked, because that page embeds a reCAPTCHA site key in a script tag while
# serving 50 real internship cards. These markers are matched against the visible
# text of the document instead, and are phrases that only appear on a real
# challenge or block page.
_WALL_MARKERS = (
    "verify you are a human",
    "are you a robot",
    "unusual traffic from your computer",
    "access denied",
    "enable javascript and cookies to continue",
    "checking your browser before accessing",
    "cf-browser-verification",
    "request unsuccessful. incapsula",
    "pardon the interruption",
)


def _looks_walled(page) -> bool:
    """True only for a genuine challenge/block page.

    Deliberately checks the *rendered text* of the document, so an embedded
    reCAPTCHA site key or a captcha widget script on an otherwise healthy
    listing page is not mistaken for a wall.
    """
    try:
        visible = " ".join(page.css("body")[0].get_all_text().split()).lower() \
            if page.css("body") else ""
    except Exception:
        visible = ""
    if not visible:
        # No readable body at all (JS shell). Not a wall - the caller escalates.
        return False
    return any(marker in visible for marker in _WALL_MARKERS)


def fetch(url: str, *, params: dict | None = None, timeout: int = 25,
          browser: bool = False, headless: bool = True,
          solve_cloudflare: bool = True) -> object:
    """GET a board page and return it as a parsed Scrapling Selector.

    `browser=False` (default) is the impersonated HTTP path: ~1s, no browser.
    `browser=True` is the Chromium path, for JS-only boards and as escalation.

    Raises BoardWall for a real block page, ThrottledOrUnparsed when the request
    succeeded but nothing on the page can be read yet.
    """
    if browser:
        page = StealthyFetcher.fetch(
            url,
            params=params or None,
            headless=headless,
            network_idle=True,
            disable_resources=True,
            block_ads=True,
            solve_cloudflare=solve_cloudflare,
            timeout=timeout * 1000,
        )
    else:
        page = Fetcher.get(
            url,
            params=params or None,
            impersonate=IMPERSONATE,
            stealthy_headers=True,
            headers={"User-Agent": _CHROME_UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=timeout,
            retries=1,
        )

    status = getattr(page, "status", 0)
    if status in (403, 429):
        raise BoardWall(f"HTTP {status}")
    if status and status >= 400:
        raise BoardWall(f"HTTP {status}")
    if _looks_walled(page):
        raise BoardWall("challenge page")
    return page


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _clean(node) -> str:
    """Collapsed text of one node, or '' when absent."""
    if node is None:
        return ""
    try:
        return " ".join(node.get_all_text().split())
    except Exception:
        return ""


def _pick(card, selectors: tuple[str, ...]):
    """First non-empty match from a chain of selectors.

    The chain is the point: these boards reshuffle their markup often enough
    that a single selector is a standing outage, and a board that returns an
    empty result is indistinguishable from a blocked one in the UI.
    """
    for sel in selectors:
        try:
            hits = card.css(sel)
        except Exception:
            continue
        if not hits:
            continue
        if _clean(hits[0]):
            return hits[0]
    return None


def _text_of(card, selectors: tuple[str, ...]) -> str:
    return _clean(_pick(card, selectors))


def _attr_of(card, selectors: tuple[str, ...], attr: str = "href") -> str:
    node = _pick(card, selectors)
    if node is None:
        return ""
    try:
        return (node.attrib.get(attr) or "").strip()
    except Exception:
        return ""


_ABS = re.compile(r"^https?://", re.I)


def _absolutise(href: str, base: str) -> str:
    """Board links come back root-relative on some boards, absolute on others."""
    href = (href or "").strip()
    if not href:
        return ""
    if _ABS.match(href):
        return href
    if href.startswith("//"):
        return "https:" + href
    return base.rstrip("/") + "/" + href.lstrip("/")


# ---------------------------------------------------------------------------
# Per-board parsers
#
# Each takes the fetched page and returns a list[Listing]. They never raise for
# an empty result - they return [] and let the caller decide whether that means
# "throttled" (escalate) or "genuinely nothing".
# ---------------------------------------------------------------------------

def parse_indeed(page) -> list[Listing]:
    """Indeed search results.

    Two live markup generations are in play: the older `div.job_seen_beacon`
    cards and the newer mosaic-provider `div[data-testid=slider_item]` ones.
    Both are listed, newest first, and the field selectors chain through both
    naming schemes (`h2.jobTitle` vs `h3.jobTitle`, `companyName` vs
    `company-name`) so neither generation has to be picked as the winner.
    """
    cards = (page.css('div[data-testid="slider_item"]')
             or page.css("div.job_seen_beacon")
             or page.css("div.tapItem"))
    base = "https://www.indeed.com"
    out: list[Listing] = []
    for card in cards:
        href = _attr_of(card, ('a[data-jk]', 'a.jcs-JobTitle', 'a[href*="/rc/clk"]'))
        title = _text_of(card, ("h3.jobTitle", "h2.jobTitle", "h2",
                                'a.jcs-JobTitle', '[data-testid="job-title"]'))
        # The anchor carries aria-label="full details of <title>", which is the
        # most stable title source across both generations when the heading is
        # empty or clipped.
        if not title:
            label = _attr_of(card, ("a[data-jk]", "a.jcs-JobTitle"), "aria-label")
            title = re.sub(r"^full details of\s+", "", label or "", flags=re.I)
        if not title or not href:
            continue
        out.append(Listing(
            source="Indeed",
            title=title,
            company=_text_of(card, ('[data-testid="company-name"]',
                                    "span.companyName", "div.companyName")),
            location=_text_of(card, ('[data-testid="text-location"]',
                                     "div.company_location")),
            salary=_text_of(card, ('[data-testid="attribute_snippet_container"]',
                                   "div.salary-snippet-container",
                                   "div.salaryOnly", "span.salary-snippet")),
            link=_absolutise(href, base),
        ))
    return out


def parse_linkedin(page) -> list[Listing]:
    """LinkedIn public job search (f_WT=2 = fully remote).

    LinkedIn serves per-country subdomains (in./au./fr.linkedin.com) from one
    search, so an absolute href is left exactly as given rather than rebuilt -
    rewriting it to a single www host loses the geo the results were filtered
    to. _absolutise does that: it returns an absolute href unchanged and only
    supplies the host for the relative ones some cards carry. Passing the raw
    href instead left those listings with no host at all.
    """
    base = "https://www.linkedin.com"
    cards = (page.css('li[data-entity-urn^="urn:li:jobPosting"]')
             or page.css("div.base-card")
             or page.css("li.jobs-search-result"))
    out: list[Listing] = []
    for card in cards:
        href = _attr_of(card, ('a[href*="/jobs/view/"]',
                               "a.base-card__full-link"))
        title = _text_of(card, ("h3.base-search-card__title",
                                "h3.base-search-card__full-link",
                                "h3", ".artdeco-entity-image"))
        if not title:
            title = _clean(_pick(card, ("span.sr-only",)))
        if not title or not href:
            continue
        out.append(Listing(
            source="LinkedIn",
            title=title,
            company=_text_of(card, ("h4.base-search-card__subtitle",
                                    "h4", ".base-search-card__subtitle")),
            location=_text_of(card, (".job-search-card__location",
                                     ".base-search-card__metadata",
                                     "span.base-search-card__metadata")),
            link=_absolutise(href, base),
        ))
    return out


def parse_naukri(page) -> list[Listing]:
    """Naukri. This board is JS-only, so it needs browser=True."""
    cards = (page.css("div.srpList")
             or page.css("div.job-card-wrapper")
             or page.css("div.srp-list > div"))
    base = "https://www.naukri.com"
    out: list[Listing] = []
    for card in cards:
        href = _attr_of(card, ("a.title", "a[href*='/job-detail/']"))
        title = _text_of(card, ("a.title", "h3", ".job-title"))
        if not title or not href:
            continue
        out.append(Listing(
            source="Naukri",
            title=title,
            company=_text_of(card, (".company-name", ".companyName",
                                    "span.company-info")),
            location=_text_of(card, (".loc", ".job-location",
                                     "span.loc")),
            salary=_text_of(card, (".salary", ".job-salary")),
            link=_absolutise(href, base),
        ))
    return out


def parse_glassdoor(page) -> list[Listing]:
    """Glassdoor.

    Glassdoor ships CSS-modules class names - `JobCard_jobTitle__hL8Kd` - where
    the trailing `__hash` is regenerated on every deploy. Matching the class
    exactly means a routine Glassdoor release silently returns zero listings,
    which is indistinguishable from being blocked. So every selector here is an
    attribute-contains match on the unhashed prefix (`[class*="JobCard_"]`),
    which survives the hash changing.

    Also note Glassdoor geo-redirects: www.glassdoor.com answers 302 to
    glassdoor.co.in for this search, so links keep whatever host it chose.
    """
    # Absolute, like every other parser here. Glassdoor in particular serves
    # root-relative /job-listing/ hrefs, and it is the board most likely to
    # answer on a host other than the one we asked for, so a raw href here
    # yields either a link with no host or one pointing at the pre-redirect
    # domain - which is a dead link in the email the candidate sends.
    base = "https://www.glassdoor.com"
    cards = (page.css('[class*="JobsList_jobListItem"]')
             or page.css('li[data-testid="jobs-list-item-cel"]')
             or page.css("div.jobCardWrapper")
             or page.css("div[data-testid='job-card']"))
    out: list[Listing] = []
    for card in cards:
        href = _attr_of(card, ('a[href*="/job-listing/"]',
                               'a[data-testid="job-link"]',
                               "a[href*='jobs-search-details.php']"))
        title = _text_of(card, ('[class*="JobCard_jobTitle"]',
                                '[data-testid="job-title"]',
                                "a.jobTitle", ".jobTitle", "h3", "h2"))
        if not title or not href:
            continue
        out.append(Listing(
            source="Glassdoor",
            title=title,
            company=_text_of(card, ('[class*="EmployerProfile_compactEmployerName"]',
                                    '[class*="EmployerProfile_employerName"]',
                                    '[data-testid="employer-name"]',
                                    ".jobInfoItem")),
            location=_text_of(card, ('[class*="JobCard_location"]',
                                     '[data-testid="text-location"]',
                                     ".loc", "span.loc")),
            salary=_text_of(card, ('[class*="JobCard_salaryEstimate"]',
                                   '[data-testid="attribute_snippet_container"]',
                                   ".salary", ".jobSalary")),
            link=_absolutise(href, base),
        ))
    return out


def parse_foundit(page) -> list[Listing]:
    """Foundit (ex-Monster India). Needs browser=True: it answers 403 to the
    impersonated HTTP path."""
    cards = (page.css("div.srp-list > div")
             or page.css("div.card")
             or page.css("div[data-cy*='card']"))
    base = "https://www.foundit.in"
    out: list[Listing] = []
    for card in cards:
        href = _attr_of(card, ('a[href*="/listing/"]', "a[href*='/jobs/']"))
        title = _text_of(card, ("h2", "h3", ".title", "a.title"))
        if not title or not href:
            continue
        out.append(Listing(
            source="Foundit",
            title=title,
            company=_text_of(card, (".company-name", ".companyName", ".company")),
            location=_text_of(card, (".location", ".loc", "span.location")),
            salary=_text_of(card, (".ctc", ".salary")),
            link=_absolutise(href, base),
        ))
    return out


def parse_internshala(page) -> list[Listing]:
    """Internshala work-from-home internships.

    Note this page EMBEDS a reCAPTCHA site key while serving real cards, which
    is why wall detection reads visible text rather than the raw body.
    """
    cards = page.css("div.individual_internship")
    base = "https://internshala.com"
    out: list[Listing] = []
    for card in cards:
        href = _attr_of(card, ('a[href*="/internship/"]', 'a.internship_header_title'))
        title = _text_of(card, (".job-internship-name", ".internship_meta",
                                "h3", "h4", ".individual_internship_header"))
        if not title or not href:
            continue
        out.append(Listing(
            source="Internshala",
            title=title,
            company=_text_of(card, (".company-name", ".company_name", ".company")),
            location=_text_of(card, (".locations", ".location", ".individual_internship_meta")),
            salary=_text_of(card, (".stipend", ".salary", ". stipend")),
            link=_absolutise(href, base),
        ))
    return out


def parse_weworkremotely(page) -> list[Listing]:
    """WeWorkRemotely. No server-side search, so the caller filters titles."""
    cards = (page.css("li.wr_listing")
             or page.css("div.wr_listing")
             or page.css("article")
             or page.css("div.listing"))
    base = "https://weworkremotely.com"
    out: list[Listing] = []
    for card in cards:
        href = _attr_of(card, ('a[href*="/remote-jobs/"]', "a"))
        title = _text_of(card, (".new-listing__header__title__text",
                                "h2", "h3", ".wr_listing__title",
                                ".listing__title"))
        if not title or not href:
            continue
        out.append(Listing(
            source="WeWorkRemotely",
            title=title,
            company=_text_of(card, (".new-listing__company-name",
                                    ".wr_listing__company",
                                    ".company", "span.company")),
            location=_text_of(card, (".new-listing__company-headquarters",
                                     ".wr_listing__location",
                                     ".location", ".region")),
            link=_absolutise(href, base),
        ))
    return out


PARSERS = {
    "Indeed": parse_indeed,
    "LinkedIn": parse_linkedin,
    "Naukri": parse_naukri,
    "Glassdoor": parse_glassdoor,
    "Foundit": parse_foundit,
    "Internshala": parse_internshala,
    "WeWorkRemotely": parse_weworkremotely,
}


@dataclass
class BoardSpec:
    """Everything needed to fetch and parse one board.

    `browser` is not a preference, it is what the board answers to: Naukri
    serves a JavaScript shell to any HTTP client and Foundit 403s it, while the
    other five serve complete HTML in about a second.
    """
    name: str
    url: str
    parser: object
    browser: bool = False
    #: Keyless JSON feeds are not in PARSERS - they are parsed by the caller's
    #: existing JSON handling, which is cheaper and needs no browser.
    json_feed: bool = False
    #: Boards excluded from the all-boards sweep. They work when asked for by
    #: name, but probing them by default costs a browser and mostly returns
    #: throttled pages, so they are opt-in.
    default_hunt: bool = True

    def params(self, query: str) -> dict:
        return _PARAMS[self.name](query)


def _indeed_params(query: str) -> dict:
    return {"q": query, "l": "Remote"}


def _linkedin_params(query: str) -> dict:
    return {"keywords": query, "location": "Remote", "f_WT": 2}


def _naukri_params(query: str) -> dict:
    # /job-search 404s; /jobs is the real entry point and redirects to
    # /jobs-in-india, which is fine - the parser reads whatever we land on.
    return {"k": query, "l": "Remote"}


def _glassdoor_params(query: str) -> dict:
    return {"sc.keyword": query, "locKeyword": "Remote"}


def _foundit_params(query: str) -> dict:
    return {"locations": "Remote"}


def _empty_params(query: str) -> dict:
    return {}


_PARAMS = {
    "Indeed": _indeed_params,
    "LinkedIn": _linkedin_params,
    "Naukri": _naukri_params,
    "Glassdoor": _glassdoor_params,
    "Foundit": _foundit_params,
    "Internshala": _empty_params,
    "WeWorkRemotely": _empty_params,
}

SPECS = {
    "Indeed": BoardSpec("Indeed", "https://www.indeed.com/jobs", parse_indeed),
    "LinkedIn": BoardSpec("LinkedIn", "https://www.linkedin.com/jobs/search",
                          parse_linkedin),
    "Naukri": BoardSpec("Naukri", "https://www.naukri.com/jobs", parse_naukri,
                        browser=True),
    "Glassdoor": BoardSpec("Glassdoor", "https://www.glassdoor.com/Job/jobs.htm",
                           parse_glassdoor, default_hunt=False),
    "Foundit": BoardSpec("Foundit",
                         "https://www.foundit.in/search/python-developer-jobs",
                         parse_foundit, browser=True, default_hunt=False),
    "Internshala": BoardSpec("Internshala",
                             "https://internshala.com/internships/work-from-home-jobs/",
                             parse_internshala),
    "WeWorkRemotely": BoardSpec("WeWorkRemotely",
                                "https://weworkremotely.com/remote-jobs",
                                parse_weworkremotely),
}

#: Boards that work when asked for by name but are not probed by a default sweep.
#:
#: This is an exclusion list rather than a flag on BoardSpec, because a default
#: sweep spans more than SPECS: Remotive and Arbeitnow are keyless JSON feeds
#: handled outside this module, and deriving the sweep from SPECS silently
#: dropped them - the two cheapest, most reliable sources in the set.
#:
#: Glassdoor is largely redundant with the other boards and adds latency to
#: every hunt; Foundit needs a full browser and 403s about half the time.
#:
#: Naukri is here for a measured reason, not a guess. It answers the cheap HTTP
#: request with a JavaScript shell, so it escalates to a real browser - which
#: measured 21s and still returned nothing, because it wants a signed-in
#: session. That is a fifth of the whole hunt spent on a guaranteed zero, on
#: every single search. It is the one board where not probing by default is a
#: straight win; `board="Naukri"` still reaches it.
#:
#: Note on the double-encoded Glassdoor URL. Asking www.glassdoor.com 302s to
#: glassdoor.co.in with a keyword-in-path URL, and the Location header echoes
#: our parameter back as sc.keyword=python%2520developer - the space encoded
#: twice. This looks like the classic double-encoding bug and it is not: the
#: board honours the keyword regardless. Measured 2026-10-03, same code both
#: ways: "rust engineer" returned "Rust Backend Engineer" and 7 of 30 titles
#: containing rust, despite %2520 in the final URL. A separate query for
#: "python developer" returned 0 python titles - that is Glassdoor's inventory
#: for this geography and filters, not a request we malformed. Do not add
#: encoding workarounds here on the strength of the URL alone.
NOT_DEFAULT = ("Glassdoor", "Foundit", "Naukri")


def is_default_board(name: str) -> bool:
    return name not in NOT_DEFAULT


def scrape(spec: BoardSpec, query: str, *, timeout: int = 30,
           escalate: bool = True) -> list[Listing]:
    """Fetch and parse one board.

    Escalation: a board that is hard-walled (BoardWall) or that answered but
    yielded nothing (ThrottledOrUnparsed) is retried through the browser, which
    has a different fingerprint. Escalation is skipped for a board already
    configured to use the browser, and can be turned off for a caller that is
    already paying for one.
    """
    params = spec.params(query)
    try:
        page = fetch(spec.url, params=params, browser=spec.browser,
                     timeout=timeout)
    except (BoardWall, ThrottledOrUnparsed):
        if spec.browser or not escalate:
            raise
        page = fetch(spec.url, params=params, browser=True, timeout=timeout)
    rows = spec.parser(page)
    if not rows and not spec.browser and escalate:
        # A 200 with no cards is usually throttling. One browser retry is
        # worth it, but not a loop: a board that genuinely has no results for
        # this query would otherwise cost a browser on every hunt.
        page = fetch(spec.url, params=params, browser=True, timeout=timeout)
        rows = spec.parser(page)
    return rows