"""
filters.py — the job-listing rules, defined exactly once.

The agent (agent.py) and the board scraper (mcp_server_indeed_scraper.py) both
apply the same "is this role worth showing?" rules, so the rules live here
instead of being copy-pasted into both files. That duplication had already
drifted once: `remote-india` was listed as a DROP token in INDIA_TOKENS and
simultaneously as a KEEP exemption in the work-from-home regex, so a
"Remote-India" posting cancelled itself out and survived. Sharing the rule set
is what prevents the next drift.

Everything here is pure and side-effect free — see tests/test_filters.py.

Rules
- SENIORITY_FILTER / EXPERIENCE_FILTER are matched against the listing's TITLE
  and LOCATION only. They used to be matched against the company name too,
  which silently deleted every opening at companies called "Staffwise",
  "Lead Generation" or "Senior Living Partners".
- EXPERIENCE_FILTER reads the usual ways of writing a requirement — "5+ years",
  "5 years", "5-8 yrs", "min 5 years" — while ignoring the phrasings that do
  NOT mean 5+ years: "0-2 years", "less than 5 years", "up to 5 years",
  "10 years ago".
- WORK_FROM_HOME exempts a listing from the India drop: an internship posted
  from Internshala as "Work From Home (India)" is remote regardless of the
  country it is advertised from. It deliberately does NOT exempt
  "remote-india", which is genuinely India-tied.

Searching
- WORK_MODES is the work-arrangement filter (remote / work-from-home / hybrid
  / on-site). It replaced the single remote_only toggle: previously on-site and
  local roles were unreachable, so a local search could not be expressed at all.
- REGIONS is the geography filter. Without it the boards return roles from
  everywhere on earth, which is rarely what someone targeting Europe or the UK
  wants.
- Both are optional everywhere they are accepted, and omitting them preserves
  the pre-existing behaviour exactly.
"""

import os
import re

# Machine-readable JSON block of parsed listings that tool servers append to
# their output; the agent and the dashboard parse it into the flashcard deck.
JOBS_MARKER = "###JOBS_JSON###"

# Job-alert senders the fallback path queries (see mcp_server_gmail.py).
#
# INDEED_ALERT_SENDER is kept even though it is no longer queried by default:
# a single-account .env may still have alerts from the older
# donotreply@match.indeed.com sender, and dropping it would silently return
# nothing for users who never re-ran their Indeed alert preferences.
GOOGLE_CAREERS_SENDER = "careers-noreply@google.com"
INDEED_ALERT_SENDER = "donotreply@match.indeed.com"
INDEED_JOBALERT_SENDER = "donotreply@jobalert.indeed.com"
# The LinkedIn address that carries job alerts. Measured across both mailboxes:
# 16 + 1 messages, ~8 of them genuine job alerts.
LINKEDIN_ALERT_SENDER = "jobs-noreply@linkedin.com"
FALLBACK_SENDERS = (
    INDEED_JOBALERT_SENDER,
    INDEED_ALERT_SENDER,
    LINKEDIN_ALERT_SENDER,
    GOOGLE_CAREERS_SENDER,
)
# Every sender we know how to read. Used only to pick a default when the user
# has not named a source.
#
# Deduplicated, and that is a fix rather than tidiness: `INDEED_ALERT_SENDER`
# is already in FALLBACK_SENDERS, so concatenating it again put the same address
# in the list twice. A source picker built on this then offered "Indeed" once
# and searched it twice, and a sender-count limit was spent twice over on one
# address.
ALL_ALERT_SENDERS = tuple(dict.fromkeys(FALLBACK_SENDERS))

# LinkedIn also sends from these addresses, and they look like job mail but are
# not: measured, `messages-noreply` is 378 + 103 messages of "8 people viewed
# your profile" / Streak Freeze / newsletter digests that yielded 11 listings,
# while `jobs-noreply` is the sender that actually holds the alerts. Syncing
# them would bury the real jobs under four hundred notifications per mailbox,
# so they are named here only to record why they are absent.
LINKEDIN_NOTIFICATION_SENDERS = (
    "messages-noreply@linkedin.com",   # 481 msgs, ~11 listings: all noise
    "updates-noreply@linkedin.com",    # 201 msgs: LinkedIn newsletter/news
    "notifications-noreply@linkedin.com",  # 269 msgs: marketing digests
)

# Gmail accounts, read the same way mcp_server_gmail._accounts() does so the
# agent and the tool can never disagree about how many mailboxes exist.
#
# The primary pair is GMAIL_ADDRESS / GMAIL_APP_PASSWORD (unchanged), then
# _2 and _3. A pair with only one half set is ignored rather than half-used.
GMAIL_ACCOUNT_VARS = (
    ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "GMAIL_SENDERS"),
    ("GMAIL_ADDRESS_2", "GMAIL_APP_PASSWORD_2", "GMAIL_SENDERS_2"),
    ("GMAIL_ADDRESS_3", "GMAIL_APP_PASSWORD_3", "GMAIL_SENDERS_3"),
)

# Cap on searches per single Gmail run. Each search is a separate IMAP login,
# so this is a latency and rate-limit budget, not a correctness limit.
GMAIL_SEARCH_BUDGET = 3


def gmail_accounts() -> list[tuple[str, str]]:
    """Configured (address, app password) pairs, in order. Blank-safe."""
    out: list[tuple[str, str]] = []
    for addr_key, pw_key, _senders_key in GMAIL_ACCOUNT_VARS:
        addr = (os.environ.get(addr_key) or "").strip()
        pw = (os.environ.get(pw_key) or "").replace(" ", "").strip()
        if addr and pw:
            out.append((addr, pw))
    return out


def _senders_for(index: int) -> list[str]:
    """Alert senders to search in account `index` (0-based).

    GMAIL_SENDERS / _2 / _3 name the senders for that specific mailbox,
    comma-separated. Per-account configuration is what makes two accounts
    actually useful: alert volume differs per mailbox, so a single global
    sender list either misses one account's senders or searches a sender the
    other account never receives.

    Falls back to the first two of FALLBACK_SENDERS when unset, which is the
    single-account behaviour from before.
    """
    key = GMAIL_ACCOUNT_VARS[index][2] if index < len(GMAIL_ACCOUNT_VARS) else None
    raw = (os.environ.get(key) or "") if key else ""
    senders = [s.strip().lower() for s in raw.split(",") if s.strip()]
    # Falls back to all three defaults, not the first two: an unset
    # GMAIL_SENDERS previously meant Google Careers + Indeed, which skipped the
    # one sender that actually carries most of the LinkedIn alerts.
    return senders or list(FALLBACK_SENDERS)


def gmail_senders_for_account(account: int) -> list[str]:
    """`gmail_senders_for(account)` with 1-based numbering, as callers use it."""
    if account < 1 or account > len(GMAIL_ACCOUNT_VARS):
        return list(FALLBACK_SENDERS)
    return _senders_for(account - 1)


def gmail_search_plan() -> list[tuple[int, str]]:
    """The (account, sender) pairs one Gmail run should search.

    Ordered account-by-account so every mailbox is read before any mailbox is
    read twice: with two accounts and one sender each, the second account is
    reached on the second search rather than after all of the first account's
    senders, which is the ordering that matters when one login is slow.
    """
    plan: list[tuple[int, str]] = []
    for i in range(len(gmail_accounts())):
        for sender in _senders_for(i):
            plan.append((i + 1, sender))
            if len(plan) >= GMAIL_SEARCH_BUDGET:
                return plan
    return plan

# Plain-language board names a user might type ("only linkedin", "just
# indeed") mapped onto the scraper's canonical board keys.
BOARD_ALIASES = {
    "Indeed": r"\bindeed\b",
    "LinkedIn": r"\blinked\s*in\b|\blinkedin\b",
    "Naukri": r"\bnaukri\b",
    "Glassdoor": r"\bglassdoor\b",
    "Foundit": r"\bfoundit\b|\bmonster\b",
    "Internshala": r"\binternshala\b",
    "WeWorkRemotely": r"\bwework(?:remotely)?\b|\bwe\s+work\s+remote(?:ly)?\b|\bwwr\b",
    "Remotive": r"\bremotive\b",
    "Arbeitnow": r"\barbeitnow\b",
}

# Roles that hire for the same work under a different title, and the titles a
# board is likely to be using for them. A DevOps Engineer opening is frequently
# posted as "Cloud Engineer", "Platform Engineer", "SRE" or "AWS Engineer", and
# a search for one spelling never sees the others.
#
# Keys are matched as whole words against the query, so "devops" matches
# "DevOps Engineer" but never "develops" or "DevOpsOps". Each value is ordered
# best-first: related_roles() returns the head of the list, so the ordering is
# the priority.
#
# Deliberately role-shaped, not skill-shaped. "aws" alone is not a key: it is a
# technology that appears inside half the titles here already, and treating it
# as a role on its own would drag AWS-shaped results into every search.
ROLE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "devops": ("cloud engineer", "platform engineer", "aws engineer",
               "site reliability engineer", "sre", "infrastructure engineer",
               "cloud architect", "devops engineer", "kubernetes engineer"),
    "devops engineer": ("cloud engineer", "platform engineer", "aws engineer",
                        "site reliability engineer", "sre",
                        "infrastructure engineer"),
    "site reliability": ("devops engineer", "cloud engineer", "sre",
                         "platform engineer"),
    "sre": ("devops engineer", "site reliability engineer",
            "cloud engineer", "platform engineer"),
    "cloud": ("devops engineer", "platform engineer", "cloud engineer",
              "aws engineer", "infrastructure engineer"),
    "aws": ("cloud engineer", "devops engineer", "aws engineer",
            "cloud architect", "solutions architect"),
    "azure": ("cloud engineer", "cloud architect", "devops engineer",
              "azure engineer"),
    "gcp": ("cloud engineer", "cloud architect", "devops engineer",
            "google cloud engineer"),
    "kubernetes": ("devops engineer", "platform engineer", "cloud engineer",
                   "site reliability engineer", "sre"),
    "platform": ("devops engineer", "platform engineer", "site reliability",
                 "infrastructure engineer", "cloud engineer"),
    "infrastructure": ("devops engineer", "infrastructure engineer",
                       "platform engineer", "cloud engineer"),
    "frontend": ("front end developer", "frontend developer", "ui developer",
                 "web developer", "frontend engineer"),
    "front end": ("frontend developer", "front end developer", "ui developer",
                  "web developer"),
    "ui": ("frontend developer", "ui developer", "front end developer",
           "web developer"),
    "backend": ("back end developer", "backend developer", "api engineer",
                "software engineer", "web developer"),
    "back end": ("backend developer", "back end developer", "api engineer",
                 "software engineer"),
    "fullstack": ("full stack developer", "fullstack developer",
                  "software engineer", "web developer"),
    "full stack": ("full stack developer", "fullstack developer",
                   "software engineer", "web developer"),
    "software": ("software engineer", "software developer", "application engineer",
                 "systems engineer"),
    "data analyst": ("business intelligence analyst", "data analyst",
                     "analytics engineer", "reporting analyst"),
    "business intelligence": ("data analyst", "business intelligence analyst",
                              "analytics engineer"),
    "data engineer": ("data engineer", "big data engineer", "etl developer",
                      "data platform engineer"),
    "data scientist": ("data scientist", "machine learning engineer",
                       "applied scientist", "research scientist"),
    "machine learning": ("machine learning engineer", "ml engineer",
                         "data scientist", "ai engineer"),
    "ml engineer": ("machine learning engineer", "ai engineer",
                    "data scientist"),
    "qa": ("qa engineer", "quality assurance engineer", "test engineer",
           "sdETest"),
    "test": ("qa engineer", "test engineer", "quality assurance engineer",
             "sdetest"),
    "security": ("security engineer", "application security engineer",
                 "devsecops engineer", "cybersecurity analyst"),
    "mobile": ("android engineer", "ios engineer", "mobile engineer",
               "mobile developer"),
    "android": ("android engineer", "mobile developer", "mobile engineer"),
    "ios": ("ios engineer", "mobile developer", "mobile engineer"),
    "devsecops": ("security engineer", "devsecops engineer",
                  "application security engineer", "cloud security engineer"),
}

# One related-role lookup per search: enough to cover a genuinely thin result
# set without turning a single hunt back into the parallel fan-out this project
# deliberately stopped doing.
RELATED_ROLE_LIMIT = 1

_ROLE_SYNONYM_KEYS = tuple(ROLE_SYNONYMS)


def role_synonyms_for(query: str) -> tuple[str, ...]:
    """Titles that are the same job as `query`, best match first.

    Returns an empty tuple when nothing is recognised, and never returns the
    query's own words - the caller uses this to widen a search, so echoing the
    original spelling back would waste the one follow-up attempt.
    """
    text = (query or "").lower()
    if not text.strip():
        return ()
    hits: list[tuple[int, int, tuple[str, ...]]] = []
    for key in _ROLE_SYNONYM_KEYS:
        m = re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(key), text)
        if m:
            # Longer key = more specific match = wins when two both appear.
            hits.append((m.start(), -len(key), ROLE_SYNONYMS[key]))
    if not hits:
        return ()
    best = min(hits)[2]
    # Drop anything already in the query: "DevOps Engineer cloud" gains nothing
    # from being told to also look for "cloud engineer".
    return tuple(t for t in best if t not in text)


def related_roles(query: str, limit: int = RELATED_ROLE_LIMIT) -> tuple[str, ...]:
    """The single best related-role query for `query`, or () if none applies."""
    return role_synonyms_for(query)[:max(0, limit)]


def synonym_words(query: str) -> set[str]:
    """Every word of every related title - used to RANK, never to filter.

    A listing titled "Cloud Engineer" should outrank an unrelated one for a
    DevOps search, but it must never be dropped for failing to say "devops".
    """
    words: set[str] = set()
    for phrase in role_synonyms_for(query):
        words.update(w for w in phrase.split() if len(w) > 2)
    return words

# Locations that mean the role is tied to India.
INDIA_TOKENS = re.compile(
    r"\b(india|indian|hyderabad|bangalore|bengaluru|mumbai|pune|delhi|noida|"
    r"gurugram|gurgaon|chennai|kolkata|ahmedabad|coimbatore|indore|jaipur|"
    r"lucknow|kerala|karnataka|maharashtra|telangana|tamil ?nadu|uttar ?pradesh|"
    r"rajasthan|gujarat|bihar|punjab|west ?bengal|andhra|remote-india|"
    r"work from home india)\b",
    re.IGNORECASE,
)

# The user targets entry-level (0-2 years) roles. Every alternative is
# word-bounded: unbounded "lead" matched "Lead Generation" and "senior"
# matched "Seniority" as substrings of longer words.
SENIORITY_FILTER = re.compile(
    r"\b(?:senior|lead|principal|staff|architect|director|manager|"
    r"vp|vice\s+president|chief|head\s+of)\b",
    re.IGNORECASE,
)

# "5+ years" AND "5 years" AND "5-8 yrs" AND "min 5 years" all count as a
# senior requirement. The leading lookbehinds reject the phrasings that do NOT
# mean 5+ ("less than 5", "up to 5", "0 to 5") and the digit/hyphen guard
# rejects the far end of a sub-5 range ("0-2 years", "2-5 years") without
# breaking a genuine one ("10-12 years" matches on its "10").
EXPERIENCE_FILTER = re.compile(
    r"(?<!less than )(?<!fewer than )(?<!under )(?<!up to )(?<!max )(?<!to )"
    r"(?<![\d–—-])"
    r"\b(?:[5-9]|[1-9]\d)\s*(?:\+|plus)?\s*(?:[-–—~]|\bto\b)?\s*"
    r"(?:[0-9]{1,2}\s*)?(?:years?|yrs?)\b(?!\s+ago)",
    re.IGNORECASE,
)

# A genuinely remote posting. NOT "remote-india" — that is India-tied.
WORK_FROM_HOME = re.compile(r"\b(?:work[\s-]*from[\s-]*home|wfh)\b", re.IGNORECASE)

# Remote phrasings that are explicitly scoped TO India. These contain a
# work-from-home token but are still India-tied, so they are checked BEFORE
# the exemption — otherwise "Work From Home India" would be rescued by
# WORK_FROM_HOME and survive.
INDIA_SCOPED_REMOTE = re.compile(
    r"\b(?:remote[\s-]*india|work[\s-]*from[\s-]*home[\s-]*india|"
    r"wfh[\s-]*india|india[\s-]*remote|india[\s-]*wfh)\b",
    re.IGNORECASE,
)


def detect_board(text: str) -> str | None:
    """First board name mentioned in free-form user text, canonicalized to a
    key of BOARD_ALIASES (None when no board was explicitly requested).

    "First" means first by POSITION IN THE TEXT, not first in BOARD_ALIASES.
    Iterating the alias table made the answer depend on dict order, so
    "naukri and linkedin" resolved to LinkedIn just because that key is
    declared above Naukri - the opposite of what the user typed. Collect every
    match with its offset and return the earliest one. Longest-alternation
    breaks ties within a single spot ("we work remotely" must not be read as a
    bare "wework").
    """
    hits: list[tuple[int, int, str]] = []
    for canonical, pattern in BOARD_ALIASES.items():
        for m in re.finditer(pattern, text or "", re.IGNORECASE):
            hits.append((m.start(), -(m.end() - m.start()), canonical))
    if not hits:
        return None
    return min(hits)[2]


def canonical_board(name: str, known: tuple | list | None = None) -> str | None:
    """Resolve a user-supplied board name/alias onto a real board key.

    `known` is the scraper's BOARDS mapping (or its keys); passing it lets the
    scraper stay the single owner of which boards actually exist. Returns None
    for a typo so the caller can degrade to the normal multi-board run.
    """
    if not name:
        return None
    lowered = name.strip().lower()
    hit = detect_board(lowered)
    if hit and (known is None or hit in known):
        return hit
    if known:
        for canonical in known:
            if canonical.lower() == lowered:
                return canonical
    return None


def is_senior_or_experienced(title: str, location: str = "") -> bool:
    """True when a listing's title/location marks it as too senior.

    The company name is deliberately NOT an input — that was the source of the
    false positives described in the module docstring.
    """
    hay = f"{title or ''} {location or ''}"
    return bool(SENIORITY_FILTER.search(hay) or EXPERIENCE_FILTER.search(hay))


def is_india_tied(location: str) -> bool:
    """True when a location is India-based, i.e. the role is not usable."""
    return bool(INDIA_TOKENS.search(location or ""))


def outside_india(location: str) -> bool:
    """The remote-only rule for one location: keep it unless it is India-tied,
    with work-from-home postings exempt (they are remote wherever they are
    advertised from — see the module docstring). An explicitly India-scoped
    "Work From Home India" is NOT exempt."""
    loc = location or ""
    if INDIA_SCOPED_REMOTE.search(loc):
        return False
    return WORK_FROM_HOME.search(loc) is not None or not INDIA_TOKENS.search(loc)


def india_rule_applies(remote_only: bool, work_modes=None) -> bool:
    """Should the legacy 'not based in India' rule still bite?

    The rule exists to keep an India-tied office posting out of a *remote*
    hunt. Once the user ticks On-site or Hybrid they are deliberately asking
    for roles in a physical location, and applying the rule on top would drop
    the exact listings they selected (an on-site search in Bengaluru would
    return nothing). So the chips win over the legacy remote_only flag; with no
    chips chosen the flag decides, as before.
    """
    if not remote_only:
        return False
    wanted = normalize_work_modes(work_modes)
    if not wanted:
        return True
    return bool(set(wanted) & set(REMOTE_ONLY_MODES))


def keep_job(job: dict, remote_only: bool = False, work_modes=None,
             regions=None, countries=None) -> bool:
    """Should this structured listing reach the user?

    Drops senior / 5+ years roles, India-tied roles (only when the retired
    remote_only flag is explicitly on), and anything outside the requested work
    arrangement, region or country.
    Listings without a usable http(s) Apply link are dropped too — the link is
    what the flashcard deck's Apply button needs, so a listing without one is
    useless.

    `work_modes` / `regions` / `countries` are optional; omitting them preserves
    the pre-existing behaviour exactly.
    """
    if not isinstance(job, dict):
        return False
    title = job.get("title") or ""
    location = job.get("location") or ""
    if is_senior_or_experienced(title, location):
        return False
    if india_rule_applies(remote_only, work_modes) and not outside_india(location):
        return False
    if work_modes is not None and not matches_work_mode(location, work_modes):
        return False
    if regions is not None and not matches_region(location, regions):
        return False
    if countries is not None and not matches_country(location, countries):
        return False
    link = job.get("link") or ""
    return isinstance(link, str) and link.startswith(("http://", "https://"))


# ---------------------------------------------------------------------------
# Work arrangement: remote / work-from-home / hybrid / on-site
# ---------------------------------------------------------------------------

# The chips offered in Settings. `remote` and `wfh` are kept separate on
# purpose: an internship advertised as "Work From Home (India)" is a
# different animal from a "Remote - US" role, and users ask for one or the
# other explicitly.
WORK_MODES = {
    "remote": "Remote",
    "wfh": "Work From Home",
    "hybrid": "Hybrid",
    "onsite": "On-site / Office",
}

DEFAULT_WORK_MODES = ["remote"]

REMOTE_TOKENS = re.compile(
    r"\b(remote|virtual|anywhere|worldwide|world-wide|globally|distributed|"
    r"telecommute\w*|online)\b",
    re.IGNORECASE,
)
HYBRID_TOKENS = re.compile(r"\bhybrid\b", re.IGNORECASE)

REMOTE_ONLY_MODES = ("remote", "wfh")


def normalize_work_modes(modes) -> list[str]:
    """Sanitize user/config-supplied work modes into known keys, in the
    canonical chip order, without duplicates. Unknown values are dropped;
    an empty/None result means "don't filter by arrangement"."""
    wanted = {str(m).strip().lower() for m in (modes or [])}
    return [k for k in WORK_MODES if k in wanted]


def listing_work_modes(location: str) -> set[str]:
    """Which arrangements does this listing's location string EXPLICITLY state?

    Returns an EMPTY set when the location names no arrangement. That is the
    important case: "Paris, France" says where the job is, not how you work on
    it. A remote listing is tagged 'Paris, France' precisely because the worker
    is remote, so reading a bare city as on-site would delete every remote role
    outside the user's own country — a Europe + Remote search returned nothing
    at all before this was fixed.

    A hybrid posting is tagged 'hybrid' only — not 'remote' and not 'onsite' —
    so each chip stays meaningful instead of everything matching everything.
    """
    loc = (location or "").strip()
    if not loc:
        return set()
    # A hybrid posting is tagged 'hybrid' only — not 'remote' and not 'onsite' —
    # so each chip stays meaningful instead of everything matching everything.
    #
    # 'wfh' and 'remote' are one family, not two. "Work From Home" IS working
    # remotely; separating them meant matches_work_mode() computed an empty
    # intersection and silently deleted every WFH listing from a remote search.
    # DEFAULT_WORK_MODES is ["remote"], so that was the DEFAULT configuration,
    # and boards label the same role either way depending on the posting.
    if WORK_FROM_HOME.search(loc):
        return {"wfh", "remote"}
    if HYBRID_TOKENS.search(loc):
        return {"hybrid"}
    if REMOTE_TOKENS.search(loc):
        return {"remote", "wfh"}
    return set()


def matches_work_mode(location: str, modes) -> bool:
    """Is this listing's arrangement compatible with what the user asked for?

    Only listings whose location EXPLICITLY contradicts the selection are
    dropped ("Hybrid - London" when you asked for remote). A location that
    states no arrangement is kept for every selection, because the work type is
    enforced at SEARCH time instead: the boards are queried with the remote
    filter (LinkedIn f_WT=2) or with the chosen city, and the per-listing
    string is only geography. Inferring on-site from a city name would empty
    every remote search.
    """
    if not modes:
        return True                      # no arrangement selected -> no filter
    wanted = set(normalize_work_modes(modes))
    if not wanted:
        return True
    stated = listing_work_modes(location)
    if not stated:
        return True                      # nothing claimed -> don't contradict
    return bool(stated & wanted)


def boards_location(work_modes=None, cities=None, remote_only: bool = False,
                    countries=None) -> str:
    """The `location` argument to hand the boards.

    The legacy remote_only flag used to default to True, which made every
    unqualified search read "Remote" and then dropped India-tied postings. It
    is off by default now: with no work type, city or country chosen the
    location stays empty, which is the boards' own worldwide search.

    Once chips ARE set, remote/WFH still search "Remote", but an on-site or
    hybrid request needs a real place, so the first chosen city becomes the
    location — reusing the Cities field rather than adding a new one.

    Countries come last: a city is more specific than a country, so a chosen
    city wins. When no city is set and the search is NOT remote, the first
    chosen country is the place, which is what makes the country picker useful
    on its own.
    """
    place = next(iter(normalize_cities(cities)), "")
    if not place and not remote_only:
        place = next(iter(normalize_countries(countries)), "")
    wanted = normalize_work_modes(work_modes)
    if not wanted:
        return place if (place and not remote_only) else "Remote"
    if set(wanted) & {"onsite", "hybrid"} and place:
        return place
    return "Remote"


# ---------------------------------------------------------------------------
# City catalog
#
# A SUGGESTION list for the Cities picker, in exactly the same sense as
# JOB_ROLE_OPTIONS in dashboard.py: putting a city here makes it selectable,
# it does not put it in anybody's search. It deliberately does NOT gate what a
# user may type — a small town with no listing is still a legitimate place to
# live, and silently discarding the city would make an on-site search return
# nothing while the filter looked like it had worked. Unknown cities pass
# through untouched.
#
# The catalog exists for two reasons. A dropdown needs something to list, and
# `boards_location()` hands the FIRST chosen city to the boards as the search
# location, so a picker that always spells a city the way the board does is
# worth more than free text. Every entry is keyed by a REGIONS key so the
# picker can label a city with its region the way it labels a country.
# ---------------------------------------------------------------------------

CITY_OPTIONS: dict[str, str] = {
    # India first: every city the CV reader can infer (dashboard.INDIAN_CITIES)
    # has to be here, or a city seeded from a CV would look unrecognised.
    "Hyderabad": "asia-pacific",
    "Bangalore": "asia-pacific",
    "Bengaluru": "asia-pacific",
    "Mumbai": "asia-pacific",
    "Pune": "asia-pacific",
    "Delhi": "asia-pacific",
    "Noida": "asia-pacific",
    "Gurugram": "asia-pacific",
    "Gurgaon": "asia-pacific",
    "Chennai": "asia-pacific",
    "Kolkata": "asia-pacific",
    "Ahmedabad": "asia-pacific",
    "Coimbatore": "asia-pacific",
    "Indore": "asia-pacific",
    "Jaipur": "asia-pacific",
    "Lucknow": "asia-pacific",
    "Kochi": "asia-pacific",
    "Trivandrum": "asia-pacific",
    "Vizag": "asia-pacific",
    "Bhopal": "asia-pacific",
    "Nagpur": "asia-pacific",
    "Chandigarh": "asia-pacific",
    # Asia-Pacific
    "Singapore": "asia-pacific",
    "Tokyo": "asia-pacific",
    "Osaka": "asia-pacific",
    "Seoul": "asia-pacific",
    "Taipei": "asia-pacific",
    "Hong Kong": "asia-pacific",
    "Beijing": "asia-pacific",
    "Shanghai": "asia-pacific",
    "Shenzhen": "asia-pacific",
    "Sydney": "asia-pacific",
    "Melbourne": "asia-pacific",
    "Brisbane": "asia-pacific",
    "Perth": "asia-pacific",
    "Auckland": "asia-pacific",
    "Kuala Lumpur": "asia-pacific",
    "Bangkok": "asia-pacific",
    "Jakarta": "asia-pacific",
    "Manila": "asia-pacific",
    "Hanoi": "asia-pacific",
    "Ho Chi Minh City": "asia-pacific",
    "Karachi": "asia-pacific",
    "Lahore": "asia-pacific",
    "Dhaka": "asia-pacific",
    "Colombo": "asia-pacific",
    "Kathmandu": "asia-pacific",
    # UK & Ireland
    "London": "uk-ireland",
    "Manchester": "uk-ireland",
    "Birmingham": "uk-ireland",
    "Bristol": "uk-ireland",
    "Leeds": "uk-ireland",
    "Glasgow": "uk-ireland",
    "Edinburgh": "uk-ireland",
    "Cardiff": "uk-ireland",
    "Belfast": "uk-ireland",
    "Oxford": "uk-ireland",
    "Cambridge": "uk-ireland",
    "Dublin": "uk-ireland",
    "Cork": "uk-ireland",
    # Europe
    "Amsterdam": "europe",
    "Rotterdam": "europe",
    "Berlin": "europe",
    "Munich": "europe",
    "Hamburg": "europe",
    "Frankfurt": "europe",
    "Cologne": "europe",
    "Paris": "europe",
    "Lyon": "europe",
    "Marseille": "europe",
    "Toulouse": "europe",
    "Madrid": "europe",
    "Barcelona": "europe",
    "Valencia": "europe",
    "Lisbon": "europe",
    "Porto": "europe",
    "Milan": "europe",
    "Rome": "europe",
    "Florence": "europe",
    "Turin": "europe",
    "Zurich": "europe",
    "Geneva": "europe",
    "Basel": "europe",
    "Vienna": "europe",
    "Salzburg": "europe",
    "Brussels": "europe",
    "Antwerp": "europe",
    "Stockholm": "europe",
    "Gothenburg": "europe",
    "Oslo": "europe",
    "Copenhagen": "europe",
    "Helsinki": "europe",
    "Tallinn": "europe",
    "Riga": "europe",
    "Vilnius": "europe",
    "Warsaw": "europe",
    "Krakow": "europe",
    "Wroclaw": "europe",
    "Prague": "europe",
    "Brno": "europe",
    "Budapest": "europe",
    "Bucharest": "europe",
    "Cluj-Napoca": "europe",
    "Sofia": "europe",
    "Belgrade": "europe",
    "Zagreb": "europe",
    "Ljubljana": "europe",
    "Athens": "europe",
    "Thessaloniki": "europe",
    "Kyiv": "europe",
    "Reykjavik": "europe",
    "Luxembourg": "europe",
    # North America
    "New York": "north-america",
    "San Francisco": "north-america",
    "San Jose": "north-america",
    "Oakland": "north-america",
    "Los Angeles": "north-america",
    "San Diego": "north-america",
    "Seattle": "north-america",
    "Portland": "north-america",
    "Denver": "north-america",
    "Austin": "north-america",
    "Dallas": "north-america",
    "Houston": "north-america",
    "Atlanta": "north-america",
    "Miami": "north-america",
    "Orlando": "north-america",
    "Tampa": "north-america",
    "Boston": "north-america",
    "Cambridge, MA": "north-america",
    "Chicago": "north-america",
    "Minneapolis": "north-america",
    "Nashville": "north-america",
    "Detroit": "north-america",
    "Pittsburgh": "north-america",
    "Philadelphia": "north-america",
    "Washington": "north-america",
    "Phoenix": "north-america",
    "Salt Lake City": "north-america",
    "Toronto": "north-america",
    "Vancouver": "north-america",
    "Montreal": "north-america",
    "Ottawa": "north-america",
    "Calgary": "north-america",
    "Edmonton": "north-america",
    "Mexico City": "north-america",
    "Guadalajara": "north-america",
    "Monterrey": "north-america",
    # Latin America
    "Sao Paulo": "latam",
    "Rio de Janeiro": "latam",
    "Buenos Aires": "latam",
    "Cordoba": "latam",
    "Santiago": "latam",
    "Bogota": "latam",
    "Medellin": "latam",
    "Lima": "latam",
    "Montevideo": "latam",
    "Asuncion": "latam",
    "Quito": "latam",
    # Africa & Middle East
    "Johannesburg": "africa-middle-east",
    "Cape Town": "africa-middle-east",
    "Durban": "africa-middle-east",
    "Pretoria": "africa-middle-east",
    "Lagos": "africa-middle-east",
    "Abuja": "africa-middle-east",
    "Accra": "africa-middle-east",
    "Nairobi": "africa-middle-east",
    "Kampala": "africa-middle-east",
    "Dar es Salaam": "africa-middle-east",
    "Cairo": "africa-middle-east",
    "Casablanca": "africa-middle-east",
    "Dubai": "africa-middle-east",
    "Abu Dhabi": "africa-middle-east",
    "Doha": "africa-middle-east",
    "Riyadh": "africa-middle-east",
    "Jeddah": "africa-middle-east",
    "Tel Aviv": "africa-middle-east",
    "Jerusalem": "africa-middle-east",
    "Amman": "africa-middle-east",
    "Beirut": "africa-middle-east",
    "Istanbul": "africa-middle-east",
    "Ankara": "africa-middle-east",
}


def city_region(name: str) -> str:
    """The region key a catalogued city belongs to ('' for a city we do not
    know, which is not an error — unknown cities are kept as typed)."""
    return CITY_OPTIONS.get(str(name).strip(), "")


def known_cities() -> list[str]:
    """Every catalogued city, alphabetical — the order the picker lists them."""
    return sorted(CITY_OPTIONS, key=lambda c: c.lower())


def normalize_cities(cities) -> list[str]:
    """Sanitize user/config city lists: trimmed, de-duplicated case-insensitively,
    order preserved.

    Unlike normalize_countries this does NOT drop unknown values. Countries are
    matched against a token table, so a typo has to be dropped or it silently
    matches nothing; cities are handed to the boards as a search string, so an
    unrecognised-but-real place is still the right answer. Only exact
    case-insensitive duplicates are collapsed, so "Bengaluru" and "Bangalore"
    both survive as the two different searches they are.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in (cities or []):
        key = str(raw).strip()
        if not key:
            continue
        low = key.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(key)
    return out


# ---------------------------------------------------------------------------
# Region: retired from the UI. The table is kept because the country picker
# still labels each country with its region, and so old saved profiles load.
# Nothing forwards regions to the scraper any more, and DEFAULT_REGIONS is
# empty so a fresh session has no geography narrowing at all.
# ---------------------------------------------------------------------------

REGIONS = {
    "worldwide": "Worldwide",
    "north-america": "North America",
    "uk-ireland": "UK & Ireland",
    "europe": "Europe",
    "asia-pacific": "Asia-Pacific",
    "latam": "Latin America",
    "africa-middle-east": "Africa & Middle East",
}

DEFAULT_REGIONS: list[str] = []

REGION_TOKENS = {
    "north-america": r"\b(usa|u\.s\.|u\.s\.a\.|united states|america|north america|"
                     r"canada|toronto|vancouver|montreal|ottawa|calgary|new york|"
                     r"nyc|brooklyn|san francisco|sf bay|austin|seattle|boston|"
                     r"chicago|los angeles|la,? ca|denver|atlanta|dallas|houston|"
                     r"philadelphia|miami|washington|portland|phoenix|san diego|"
                     r"minneapolis|nashville|detroit|pittsburgh)\b",
    "uk-ireland": r"\b(uk|u\.k\.|united kingdom|great britain|england|scotland|"
                  r"wales|northern ireland|ireland|republic of ireland|dublin|"
                  r"london|manchester|birmingham|bristol|edinburgh|glasgow|"
                  r"leeds|belfast|cardiff|oxford|cambridge)\b",
    # UK places are NOT repeated here. They resolve to uk-ireland and
    # matches_region() promotes that to europe when Europe is what was asked
    # for, so a Europe search still keeps UK roles without the two lists
    # disagreeing about which one a place belongs to.
    "europe": r"\b(europe|european union|\beu\b|germany|german|berlin|munich|"
              r"hamburg|france|french|paris|lille|"
              r"spain|spanish|madrid|barcelona|"
              r"italy|italian|milan|rome|florence|netherlands|dutch|amsterdam|"
              r"rotterdam|poland|polish|warsaw|krakow|portugal|portuguese|lisbon|"
              r"switzerland|swiss|zurich|geneva|vienna|austria|salzburg|belgium|"
              r"brussels|sweden|swedish|stockholm|norway|norwegian|oslo|denmark|"
              r"danish|copenhagen|finland|helsinki|estonia|latvia|lithuania|"
              r"czech|prague|hungary|budapest|romania|bucharest|greece|athens|"
              r"ukraine|kyiv|kiev|iceland|reykjavik|luxembourg)\b",
    "asia-pacific": r"\b(asia|asiatic|apac|japan|japanese|tokyo|osaka|singapore|"
                    r"australia|australian|sydney|melbourne|brisbane|perth|"
                    r"new zealand|auckland|wellington|south korea|korean|seoul|"
                    r"taiwan|taipei|hong kong|china|beijing|shanghai|shenzhen|"
                    r"malaysia|kuala lumpur|thailand|bangkok|vietnam|indonesia|"
                    r"jakarta|philippines|manila|india|indian|bengaluru|bangalore|"
                    r"hyderabad|mumbai|pune|delhi|noida|gurugram|chennai|kolkata|"
                    r"ahmedabad|coimbatore|pakistan|karachi|bangladesh|dhaka|"
                    r"sri lanka)\b",
    "latam": r"\b(latin america|latam|south america|central america|brazil|"
             r"brazilian|sao paulo|rio de janeiro|mexico|mexican|argentina|"
             r"buenos aires|colombia|bogota|chile|santiago|peru|lima|uruguay|"
             r"montevideo|paraguay|bolivia|ecuador|venezuela|costa rica|panama)\b",
    "africa-middle-east": r"\b(africa|african|south africa|johannesburg|cape town|"
                          r"nigeria|lagos|kenya|nairobi|egypt|cairo|morocco|"
                          r"ghana|accra|tanzania|uganda|ethiopia|senegal|"
                          r"middle east|middle eastern|gulf|uae|dubai|abu dhabi|"
                          r"saudi|riyadh|jeddah|israel|tel aviv|jerusalem|"
                          r"turkey|turkish|istanbul|ankara|qatar|doha|kuwait|"
                          r"bahrain|oman|lebanon|beirut|iran|tehran|iraq|baghdad|"
                          r"jordan|amman)\b",
}

# Bare two-letter country codes are checked WITHOUT re.IGNORECASE, separately
# from the token lists. "Remote (US)" is one of the most common location
# strings the boards return, and boards are inconsistent about the case they
# write it in, so both "US" and "U.S." have to resolve. The lookahead/behind
# keeps it to a standalone code, so "Trust" and "Guzzle" never match.
#
# Note this still reads a capitalised "Us" inside prose ("Join Us") as the US;
# the alternative — requiring a delimiter like "Remote (US)" — would drop the
# most common location format there is, so the looser rule wins.
REGION_ABBREV = {
    "north-america": r"(?<![A-Za-z])(?:U\.?\s?S\.?A?\.?)(?![A-Za-z])",
    "uk-ireland": r"(?<![A-Za-z])(?:U\.?\s?K\.?|U\.?\s?K)(?![A-Za-z])",
    "europe": r"(?<![A-Za-z])(?:EU)(?![A-Za-z])",
}

WORLDWIDE = "worldwide"

# A listing that is open to everyone is in every region. Boards write this as
# "Remote, Worldwide" / "Anywhere" / "Global", and dropping those from a
# Europe search would hide exactly the roles that are easiest to apply to.
_GLOBAL_TOKENS = re.compile(r"\b(worldwide|world-wide|anywhere|global|globally|"
                            r"international|any country|multiple locations)\b",
                            re.IGNORECASE)

# Country names, checked FIRST and given priority over the city lists above.
# Bare place names are ambiguous across the world and the boards do hand us
# real collisions: "Paris, Ontario, Canada" is Canada but reads as Paris,
# France; "Sydney Olympic Park, New South Wales" is Australia but contains
# "Wales"; "New Mexico" is the US but reads as Mexico. Whenever the location
# names a country, that country decides — a city name can no longer overrule it.
REGION_COUNTRIES = {
    "north-america": r"\b(united states(?: of america)?|usa|u\.s\.|america|"
                     r"canada)\b",
    # "New South Wales" is guarded by the FULL phrase. Guarding on "New " alone
    # does not help: there "Wales" is preceded by "South ", and since the
    # country lists are scanned in order that stray "Wales" decides the region
    # before the "Australia" later in the same string is ever reached.
    "uk-ireland": r"\b(united kingdom|great britain|england|scotland|"
                  r"(?<!New South )wales|northern ireland|ireland|"
                  r"republic of ireland)\b",
    "europe": r"\b(germany|france|spain|italy|netherlands|poland|portugal|"
              r"switzerland|austria|belgium|sweden|norway|denmark|finland|"
              r"estonia|latvia|lithuania|czech(?: republic)?|hungary|romania|"
              r"greece|ukraine|iceland|luxembourg|slovenia|slovakia|croatia|"
              r"serbia|bulgaria|european union)\b",
    "asia-pacific": r"\b(japan|singapore|australia|new zealand|south korea|"
                    r"north korea|taiwan|hong kong|china|malaysia|thailand|"
                    r"vietnam|indonesia|philippines|india|pakistan|bangladesh|"
                    r"sri lanka|nepal|cambodia|myanmar)\b",
    "latam": r"\b(brazil|mexico|argentina|colombia|chile|peru|uruguay|paraguay|"
             r"bolivia|ecuador|venezuela|costa rica|panama|guatemala|"
             r"dominican republic|puerto rico)\b",
    "africa-middle-east": r"\b(south africa|nigeria|kenya|egypt|morocco|ghana|"
                          r"tanzania|uganda|ethiopia|senegal|zambia|zimbabwe|"
                          r"united arab emirates|uae|saudi arabia|israel|turkey|"
                          r"qatar|kuwait|bahrain|oman|lebanon|iran|iraq|jordan|"
                          r"tunisia|algeria)\b",
}


def identify_region(location: str) -> str:
    """Best single region for a location string, or '' when undecidable.

    Countries win over city names; bare two-letter codes are read
    case-sensitively. Returns '' for a string that identifies no country and
    matches no city, so callers can tell "unknown" from a confident answer.
    """
    loc = location or ""
    if not loc.strip():
        return ""
    for region, pattern in REGION_COUNTRIES.items():
        if re.search(pattern, loc, re.IGNORECASE):
            return region
    for region, abbrev in REGION_ABBREV.items():
        if re.search(abbrev, loc):          # case-sensitive on purpose
            return region
    for region, pattern in REGION_TOKENS.items():
        if re.search(pattern, loc, re.IGNORECASE):
            return region
    return ""


def normalize_regions(regions) -> list[str]:
    """Sanitize user/config-supplied regions into known keys, canonical order,
    without duplicates. Unknown values are dropped."""
    wanted = {str(r).strip().lower() for r in (regions or [])}
    return [k for k in REGIONS if k in wanted]


def matches_region(location: str, regions) -> bool:
    """Is this listing's location inside one of the requested regions?

    'worldwide' (or nothing selected) matches everything. Otherwise the
    location is attributed to a region — preferring an explicitly named
    country over a city name, see REGION_COUNTRIES — and kept only if that
    region was selected.

    A blank location is DROPPED when a specific region is asked for: the
    boards return a large share of listings with no location at all (LinkedIn
    especially), and passing them through would make the region filter look
    like it worked while every result was actually unattributed. It is kept for
    'worldwide', where no attribution is being claimed.
    """
    wanted = normalize_regions(regions)
    if not wanted or WORLDWIDE in wanted:
        return True
    loc = location or ""
    if not loc.strip():
        return False        # can't be attributed to a region — don't claim it is
    if _GLOBAL_TOKENS.search(loc):
        return True         # open to everyone -> belongs to every region
    found = identify_region(loc)
    if not found:
        return False
    if found == "uk-ireland" and "europe" in wanted:
        # Europe deliberately includes the UK: the boards file UK roles under
        # Europe, so a Europe-only search that hid them would look broken. The
        # dedicated uk-ireland chip is for wanting the UK *exclusively*.
        return True
    return found in wanted


def region_label(key: str) -> str:
    """Display name for a region key ('' for an unknown one)."""
    return REGIONS.get(str(key).strip().lower(), "")


# ---------------------------------------------------------------------------
# Country filter
#
# The region chips above answer "where in the world"; the country picker
# answers "which countries exactly". A listing that names a country decides it
# outright (same precedence rule as identify_region), so "Paris, Ontario,
# Canada" is Canada and not Paris. Canonical name -> (region, alias pattern).
# ---------------------------------------------------------------------------
COUNTRIES: dict[str, tuple[str, str]] = {
    "United States": ("north-america", r"united states(?: of america)?|usa|"
                      r"u\.s\.|america"),
    "Canada": ("north-america", r"canada"),
    # "New Mexico" is the US state, not Mexico. Guarding on the FULL "New "
    # phrase is what makes the country outrank the city/word, matching how
    # identify_region relies on scanning north-america first.
    "Mexico": ("latam", r"(?<!New )\bmexico\b"),
    "United Kingdom": ("uk-ireland", r"united kingdom|great britain|england|"
                       r"scotland|(?<!New South )wales|northern ireland"),
    "Ireland": ("uk-ireland", r"(?<!Northern )\bireland\b|republic of ireland"),
    "Germany": ("europe", r"germany"),
    "France": ("europe", r"france"),
    "Spain": ("europe", r"spain"),
    "Italy": ("europe", r"italy"),
    "Netherlands": ("europe", r"netherlands"),
    "Poland": ("europe", r"poland"),
    "Portugal": ("europe", r"portugal"),
    "Switzerland": ("europe", r"switzerland"),
    "Austria": ("europe", r"austria"),
    "Belgium": ("europe", r"belgium"),
    "Sweden": ("europe", r"sweden"),
    "Norway": ("europe", r"norway"),
    "Denmark": ("europe", r"denmark"),
    "Finland": ("europe", r"finland"),
    "Estonia": ("europe", r"estonia"),
    "Latvia": ("europe", r"latvia"),
    "Lithuania": ("europe", r"lithuania"),
    "Czechia": ("europe", r"czechia|czech(?: republic)?"),
    "Hungary": ("europe", r"hungary"),
    "Romania": ("europe", r"romania"),
    "Greece": ("europe", r"greece"),
    "Ukraine": ("europe", r"ukraine"),
    "Iceland": ("europe", r"iceland"),
    "Luxembourg": ("europe", r"luxembourg"),
    "Slovenia": ("europe", r"slovenia"),
    "Slovakia": ("europe", r"slovakia"),
    "Croatia": ("europe", r"croatia"),
    "Serbia": ("europe", r"serbia"),
    "Bulgaria": ("europe", r"bulgaria"),
    "European Union": ("europe", r"european union"),
    "Japan": ("asia-pacific", r"japan"),
    "Singapore": ("asia-pacific", r"singapore"),
    "Australia": ("asia-pacific", r"australia"),
    "New Zealand": ("asia-pacific", r"new zealand"),
    "South Korea": ("asia-pacific", r"south korea"),
    "North Korea": ("asia-pacific", r"north korea"),
    "Taiwan": ("asia-pacific", r"taiwan"),
    "Hong Kong": ("asia-pacific", r"hong kong"),
    "China": ("asia-pacific", r"china"),
    "Malaysia": ("asia-pacific", r"malaysia"),
    "Thailand": ("asia-pacific", r"thailand"),
    "Vietnam": ("asia-pacific", r"vietnam"),
    "Indonesia": ("asia-pacific", r"indonesia"),
    "Philippines": ("asia-pacific", r"philippines"),
    "India": ("asia-pacific", r"india"),
    "Pakistan": ("asia-pacific", r"pakistan"),
    "Bangladesh": ("asia-pacific", r"bangladesh"),
    "Sri Lanka": ("asia-pacific", r"sri lanka"),
    "Nepal": ("asia-pacific", r"nepal"),
    "Cambodia": ("asia-pacific", r"cambodia"),
    "Myanmar": ("asia-pacific", r"myanmar"),
    "Brazil": ("latam", r"brazil"),
    "Argentina": ("latam", r"argentina"),
    "Colombia": ("latam", r"colombia"),
    "Chile": ("latam", r"chile"),
    "Peru": ("latam", r"peru"),
    "Uruguay": ("latam", r"uruguay"),
    "Paraguay": ("latam", r"paraguay"),
    "Bolivia": ("latam", r"bolivia"),
    "Ecuador": ("latam", r"ecuador"),
    "Venezuela": ("latam", r"venezuela"),
    "Costa Rica": ("latam", r"costa rica"),
    "Panama": ("latam", r"panama"),
    "Guatemala": ("latam", r"guatemala"),
    "Dominican Republic": ("latam", r"dominican republic"),
    "Puerto Rico": ("latam", r"puerto rico"),
    "South Africa": ("africa-middle-east", r"south africa"),
    "Nigeria": ("africa-middle-east", r"nigeria"),
    "Kenya": ("africa-middle-east", r"kenya"),
    "Egypt": ("africa-middle-east", r"egypt"),
    "Morocco": ("africa-middle-east", r"morocco"),
    "Ghana": ("africa-middle-east", r"ghana"),
    "Tanzania": ("africa-middle-east", r"tanzania"),
    "Uganda": ("africa-middle-east", r"uganda"),
    "Ethiopia": ("africa-middle-east", r"ethiopia"),
    "Senegal": ("africa-middle-east", r"senegal"),
    "Zambia": ("africa-middle-east", r"zambia"),
    "Zimbabwe": ("africa-middle-east", r"zimbabwe"),
    "United Arab Emirates": ("africa-middle-east", r"united arab emirates|\buae\b"),
    "Saudi Arabia": ("africa-middle-east", r"saudi arabia"),
    "Israel": ("africa-middle-east", r"israel"),
    "Turkey": ("africa-middle-east", r"turkey"),
    "Qatar": ("africa-middle-east", r"qatar"),
    "Kuwait": ("africa-middle-east", r"kuwait"),
    "Bahrain": ("africa-middle-east", r"bahrain"),
    "Oman": ("africa-middle-east", r"oman"),
    "Lebanon": ("africa-middle-east", r"lebanon"),
    "Iran": ("africa-middle-east", r"iran"),
    "Iraq": ("africa-middle-east", r"iraq"),
    "Jordan": ("africa-middle-east", r"jordan"),
    "Tunisia": ("africa-middle-east", r"tunisia"),
    "Algeria": ("africa-middle-east", r"algeria"),
}

_COUNTRY_RE = {name: re.compile(r"\b(?:%s)\b" % pat, re.IGNORECASE)
               for name, (_region, pat) in COUNTRIES.items()}

# Alias -> canonical, for anything the caller types that isn't a full name.
_COUNTRY_ALIASES = {
    "usa": "United States", "us": "United States",
    "u.s.": "United States", "u.s.a.": "United States",
    "america": "United States", "united states of america": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom",
    "great britain": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom",
    "uae": "United Arab Emirates",
    "czech republic": "Czechia",
    "south korea": "South Korea", "republic of korea": "South Korea",
    "korea": "South Korea",
    "holland": "Netherlands",
    "burma": "Myanmar",
}


def normalize_countries(countries) -> list[str]:
    """Sanitize a user/config country list into canonical names, alphabetical,
    de-duplicated. Aliases resolve ('uk' -> 'United Kingdom'); unknown values are
    dropped rather than passed through to the boards."""
    wanted: set[str] = set()
    for raw in (countries or []):
        key = str(raw).strip().lower()
        if not key:
            continue
        if key in _COUNTRY_ALIASES:
            wanted.add(_COUNTRY_ALIASES[key])
            continue
        for name in COUNTRIES:
            if key == name.lower():
                wanted.add(name)
                break
    return sorted(wanted)


def identify_countries(location: str) -> list[str]:
    """Every country this location string names, canonical order.

    An explicitly named country always wins over a city name, matching
    identify_region: 'Paris, Ontario, Canada' reports Canada only.
    """
    loc = location or ""
    if not loc.strip():
        return []
    return sorted(name for name, rx in _COUNTRY_RE.items() if rx.search(loc))


def matches_country(location: str, countries) -> bool:
    """Is this listing in one of the requested countries?

    Nothing selected matches everything. An open-to-everyone listing
    ('Remote, Worldwide') matches any country. A listing naming one of the
    wanted countries matches. A blank location is DROPPED, for the same reason
    as matches_region: the boards return many unlocated listings and letting
    them through would make the filter look like it worked while every result
    was unattributed.
    """
    wanted = normalize_countries(countries)
    if not wanted:
        return True
    loc = location or ""
    if not loc.strip():
        return False
    if _GLOBAL_TOKENS.search(loc):
        return True
    return bool(set(identify_countries(loc)) & set(wanted))


def country_region(name: str) -> str:
    """The region key a country belongs to ('' for an unknown country)."""
    entry = COUNTRIES.get(str(name).strip().title())
    if entry:
        return entry[0]
    canon = normalize_countries([name])
    return COUNTRIES[canon[0]][0] if canon else ""

