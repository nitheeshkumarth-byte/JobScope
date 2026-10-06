"""
MCP SERVER: searches Gmail via IMAP using an app password.

Why IMAP instead of the Gmail API/OAuth flow: an app password lets you
authenticate with plain username+password over IMAP (a standard email
protocol Python supports natively via `imaplib`). No Google Cloud
project, no OAuth consent screen, no token refresh logic. Trade-off:
IMAP's search syntax is more limited than Gmail's web search (e.g. no
"newer_than:60d" shorthand — we compute the date ourselves below).

Runs over stdio, same as mcp_server_github.py — the LangGraph client
launches this as a subprocess automatically.

Env vars required (put these in your .env, never in chat):
  GMAIL_ADDRESS=youraddress@gmail.com
  GMAIL_APP_PASSWORD=your16charapppassword   (no spaces needed either way)
"""

import os
import re
import json
import imaplib
import hashlib
import email
from email.header import decode_header
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP

import gmail_index
# The alert senders live in filters so the agent and this tool can never drift
# apart on which addresses are job alerts. filters imports only os/re, so there
# is no import cycle.
from filters import FALLBACK_SENDERS, gmail_senders_for_account

# Load .env from the project directory, not the CWD.
#
# This module runs as a subprocess launched by the agent, so its CWD is wherever
# the app was started. agent.py forwards GMAIL_* into the subprocess env, but a
# CWD-dependent dotenv lookup silently found nothing there, leaving every
# variable unset and every search empty. An absolute path makes the file work
# whether it is run by the app, by hand, or by a test from any directory.
try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_ENV_PATH):
        # override=False: an explicitly forwarded subprocess env still wins,
        # so the agent can override a sender list for one run without editing
        # .env.
        load_dotenv(_ENV_PATH, override=False)
except Exception:
    pass  # python-dotenv is optional; a real environment still works

mcp = FastMCP("gmail-tools")

IMAP_HOST = "imap.gmail.com"

# Marks a structured JSON block of parsed job listings. The agent's fallback
# node and the dashboard's SSE parser both understand this marker, so Gmail
# fallback listings can also be rendered as flashcards with Apply buttons.
JOBS_MARKER = "###JOBS_JSON###"

# Regexes used to turn each email into a structured listing {title, link, ...}
_LINK_RE = re.compile(r"https?://[^\s\"<>]+", re.IGNORECASE)
_TITLE_PREFIX_RE = re.compile(
    r"(?i)^(?:job alert|indeed job alert|new jobs|new job|job matches|"
    r"you've got new bounties|recommended jobs)[:\s\-]*"
)
_COMPANY_RE = re.compile(r"(?:at|@)\s+([A-Z][A-Za-z0-9&.\- ]{1,28})")
# Indeed/Google alert bodies append the subscriber's name ("at Portcast
# Hi NITHEESH,"); drop any trailing greeting so the company field stays clean.
_GREETING_RE = re.compile(r"\s+hi\s+[a-z0-9_.\-]+$", re.IGNORECASE)
_LOCATION_RE = re.compile(r"\b(?:located in|location:?)\s*([^,\n]{2,40})", re.IGNORECASE)

# URLs that appear in alert bodies but are not the job. Left unfiltered they
# became the Apply button: a scrape of the real mailbox produced links like
# `https://in.indeed.com/legal?hl=en#tos` and `https://careers.google.com/
# jobs/dist/img/email...`, i.e. every card pointed at Indeed's ToS or at a
# spacer GIF. A link that is not recognisably a job posting is not actionable,
# so it is discarded rather than shown.
_JUNK_LINK_RE = re.compile(
    r"(?://(?:www\.)?(?:legal|privacy|terms|tos|imprint|help|about|contact)"
    r"|/(?:legal|privacy|terms|tos|imprint|unsubscribe|preferences"
    r"|account|signin|login|signup)(?:[/?#]|$)"
    r"|/dist/img/|\.(?:png|gif|jpe?g|svg|webp|css|js)(?:$|[?#])"
    r"|support\.indeed\.com|/hc/|/about/careers/applications/?$"
    r"|/jobs/view/(?:$|[?#]))",
    re.IGNORECASE)
# Indeed's click-tracking host. Every URL on it is a 30x redirector to the real
# posting, so it is not junk: it is the *worst* link, not a missing one. Ranking
# puts a direct `in.indeed.com/viewjob` above a tracker, and falls back to the
# tracker rather than dropping the job — discarding these lost 133 of 144
# listings, because that is the only form the URL takes in the email.
_CTS_HOSTS = ("cts.indeed.com", "cts.is.com", "t.click.linkedin.com")

# Hosts whose URLs are jobs when they contain a job-ish path. Ordered by
# specificity so a LinkedIn posting beats a generic pattern match.
_JOB_PATH_RE = re.compile(
    r"(?:/jobs?/view/|/jobs/view/|/job/|/viewjob|/jobsearch|/jobs?/"
    r"|/careers?/jobs?/|/jobs?/results/|/job/landing|/position/)",
    re.IGNORECASE)


def _is_junk_link(url: str) -> bool:
    low = url.lower()
    return bool(_JUNK_LINK_RE.search(low))


# A search-results page rather than one posting. It matched the generic
# `/jobs?` path pattern and won, so cards pointed at "all engineering intern
# jobs in India" instead of the posting that was in the email.
_SEARCH_PAGE_RE = re.compile(
    r"[?&](?:q|keywords|query|search|what|location|l)=\S"
    r"|/jobs\?|/jobs/search|/searchjobs|/jobsearch\?",
    re.IGNORECASE)


def _is_tracking_link(url: str) -> bool:
    return any(h in url.lower() for h in _CTS_HOSTS)


def _best_job_link(candidates: list[str]) -> str:
    """The most job-like URL in a list, or "" when none qualifies.

    Ranked rather than first-match, because the first URL in an alert body is
    usually the logo or a preferences link, and taking it produced Apply
    buttons that went nowhere. Preference order is: direct posting URL > a
    click-tracking redirect that resolves to one > any other job-ish URL. The
    tracker is kept and a search page is last, because both halves of that
    trade were measured: dropping trackers lost 133 real jobs, while preferring
    a search page sends the user to a results list instead of the posting.
    """
    clean = [u.rstrip(".,);\"'") for u in candidates
             if u and not _is_junk_link(u)]

    def _pick(pred, need_direct: bool = True):
        for u in clean:
            if need_direct and _is_tracking_link(u):
                continue
            if _SEARCH_PAGE_RE.search(u):
                continue
            if pred(u):
                return u
        return ""

    got = _pick(lambda u: re.search(
        r"linkedin\.com/jobs?/view/\d+|indeed\.com/viewjob\?|/jobs/results/\d+",
        u, re.IGNORECASE))
    if got:
        return got
    got = _pick(lambda u: bool(_JOB_PATH_RE.search(u)))
    if got:
        return got
    # Direct but job-y on a known employer host, e.g. /careers/job/1234.
    got = _pick(lambda u: bool(re.search(
        r"(?:indeed|linkedin|google|amazon|ibm|accenture|cognizant|phoenix|"
        r"hexaware|wipro|itc|infosys|tcs|hcl)\.", u, re.IGNORECASE)))
    if got:
        return got
    # A tracking redirect resolves to the posting; a search page does not.
    for u in clean:
        if _is_tracking_link(u) and _JOB_PATH_RE.search(u):
            return u
    for u in clean:
        if _is_tracking_link(u):
            return u
    for u in clean:
        if _SEARCH_PAGE_RE.search(u) and _JOB_PATH_RE.search(u):
            return u
    return ""

# Job-alert emails (esp. Indeed) pad every message with 8-10 huge
# tracking URLs (View job, Apply now, Yes/No/Maybe, Edit profile,
# Recent work experience) that can be 300-600+ chars EACH. Sent to a
# local LLM with a small context window, this noise crowds out the
# actual job info and causes hallucination. We truncate at the first
# footer marker, keeping the real listing + its one "View job" link.
_FOOTER_MARKERS = [
    "Do you want to get more jobs like this",
    "Keep your Indeed profile up to date",
]


def _clean_body(text: str) -> str:
    # cut everything from the first footer marker onward
    cut_at = len(text)
    for marker in _FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    text = text[:cut_at]

# Long URLs are de-parametrised rather than dropped.
    #
    # This used to discard any line containing a URL of 80+ characters, on the
    # theory that those were tracking noise. Every real job link is longer than
    # that: the Google Careers digest lost all 12 of its jobs and the whole
    # sender produced zero listings, because a job URL with campaign parameters
    # is routinely 150+ characters. Tracking noise is now removed by asking
    # whether the URL points at a job, not by how long it is.
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        if "http" not in line:
            out_lines.append(line)
            continue
        urls = _LINK_RE.findall(line)
        kept = [u for u in urls if not _is_junk_link(u)]
        if not kept:
            continue  # pure chrome: logos, spacers, unsubscribe
        label = _LINK_RE.sub("", line).strip()
        out_lines.append((label + " " if label else "")
                         + " ".join(_shorten_url(u) for u in kept))
    return "\n".join(out_lines).strip()


_ANCHOR_RE = re.compile(
    r"<a\b[^>]*?href\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+))"
    r"[^>]*>(.*?)</a\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_END_RE = re.compile(
    r"</(?:p|div|tr|li|h[1-6]|table|br)\s*>", re.IGNORECASE)


def _html_to_text(html: str) -> str:
    """Turn an HTML alert body into text, keeping anchors as `label: url`.

    Some alert senders (Google Careers) send only text/html. Reading the raw
    markup meant the job links, which live in `href` attributes, were lost:
    a scrape of the Google Careers digest found exactly one URL in the whole
    message and it was a `search-white.png` spacer. Converting each anchor into
    a `label: url` line lets the normal text parser — which looks for a job URL
    on a line mentioning view/apply — see them, and keeps the label as the
    nearby title text.
    """
    if not html or "<a" not in html.lower():
        return html

    def repl(m: re.Match) -> str:
        href = m.group(1) or m.group(2) or m.group(3) or ""
        label = _TAG_RE.sub(" ", m.group(4) or "")
        label = re.sub(r"\s+", " ", label).strip()
        if not href.lower().startswith(("http://", "https://", "mailto:")):
            return label
        return f"{label or 'link'}: {href}" if label else href

    text = _ANCHOR_RE.sub(repl, html)
    # Structural tags become newlines so chunks stay separable; remaining inline
    # tags are dropped.
    text = _BLOCK_END_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    for ent, ch in (("&amp;", "&"), ("&nbsp;", " "), ("&lt;", "<"),
                    ("&gt;", ">"), ("&#39;", "'"), ("&quot;", '"')):
        text = text.replace(ent, ch)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_TRACKING_PARAMS = ("utm_", "src=", "ref=", "mid=", "trk=", "trkEmail",
                   "trackingId", "lipi", "licu", "eBP", "tracking_id",
                   "campaign", "source=")


def _shorten_url(url: str) -> str:
    """Drop campaign parameters so the URL stays readable but still resolves.

    Alerts wrap links in 150-character campaign tags; keeping them is harmless
    for the Apply button but they crowd out the job text. Only recognised
    tracking keys are removed, so `?jk=abc123` (the Indeed job key) and
    `/jobs/view/4422451392` (the LinkedIn id) always survive.
    """
    url = url.rstrip(".,);\"'")
    if "?" not in url:
        return url
    base, _, query = url.partition("?")
    kept = []
    for part in query.split("&"):
        if not part:
            continue
        if any(part.startswith(p) or p in part for p in _TRACKING_PARAMS):
            continue
        kept.append(part)
    return base + ("?" + "&".join(kept) if kept else "")


def _accounts() -> list[tuple[str, str]]:
    """Every configured (address, app password) pair, in order.

    A second pair is optional and enables searching two separate Gmail
    accounts - which is the only way to read alerts for two different
    identities, since one IMAP connection is bound to one mailbox.

    Pair 2 is read from GMAIL_ADDRESS_2 / GMAIL_APP_PASSWORD_2 rather than by
    looping over numbered suffixes: a half-filled pair (address without
    password, or the reverse) is silently skipped instead of raising a
    KeyError deep inside a search, which is what made a mistyped variable
    look like "no emails found".
    """
    out: list[tuple[str, str]] = []
    for addr_key, pw_key in (("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"),
                             ("GMAIL_ADDRESS_2", "GMAIL_APP_PASSWORD_2"),
                             ("GMAIL_ADDRESS_3", "GMAIL_APP_PASSWORD_3")):
        addr = (os.environ.get(addr_key) or "").strip()
        pw = (os.environ.get(pw_key) or "").replace(" ", "").strip()
        if addr and pw:
            out.append((addr, pw))
    return out


def _connect(account: int = 1):
    """Open an IMAP connection to the Nth configured Gmail account (1-based).

    The primary account is still read from GMAIL_ADDRESS /
    GMAIL_APP_PASSWORD, so an existing single-account .env keeps working with
    no change at all.
    """
    accounts = _accounts()
    if not accounts:
        raise RuntimeError(
            "No Gmail account configured. Set GMAIL_ADDRESS and "
            "GMAIL_APP_PASSWORD (2-Step Verification + an app password at "
            "https://myaccount.google.com/apppasswords).")
    try:
        address, app_password = accounts[account - 1]
    except IndexError:
        raise RuntimeError(
            f"Gmail account {account} is not configured. "
            f"{len(accounts)} account(s) available - set "
            f"GMAIL_ADDRESS_{account} and GMAIL_APP_PASSWORD_{account} to "
            f"add another.") from None
    # timeout=15: imaplib defaults to NO timeout, so a blocked network (or
    # wrong firewall) would hang the caller forever. Fail fast instead.
    conn = imaplib.IMAP4_SSL(IMAP_HOST, timeout=15)
    conn.login(address, app_password)
    return conn


def _decode(value) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    out = ""
    for text, enc in parts:
        out += text.decode(enc or "utf-8", errors="ignore") if isinstance(text, bytes) else text
    return out


def _get_body(msg: email.message.Message) -> str:
    """Extract text content from an email, preferring plain text,
    falling back to HTML (Google Careers digests are HTML even when
    a plain-text part exists but is empty)."""
    if msg.is_multipart():
        html_fallback = ""
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
            if ctype == "text/plain" and text.strip():
                return text
            if ctype == "text/html":
                html_fallback = text
        return html_fallback
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
        return ""


_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})")


def _sent_at(msg: email.message.Message) -> str:
    """The Date header as an ISO string, for sorting and for age filtering.

    Uses parsedate_to_datetime rather than a regex: alert mail carries RFC 2822
    dates ("Wed, 15 Oct 2025 10:22:33 +0000"), so an ISO-pattern regex matched
    nothing and every listing came back with an empty date, which then sorted
    them as the oldest rows and made `days_back` filtering wrong.

    Falls back to an empty string rather than to "now": a message with an
    unparseable date must sort as the OLDEST possible row, so that an age
    filter eventually drops it. Defaulting to the current time would pin an
    unparseable message at the top of every result set forever.
    """
    raw = msg.get("Date", "") or ""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass
    m = _ISO_DATE.search(raw)
    return f"{m.group(1)}T{m.group(2)}" if m else ""


def _extract_listings(subject: str, body: str) -> list[dict]:
    """Every job in one alert email, not just the first.

    _extract_listing() returned a single dict, which silently threw away the
    rest of the digest: an Indeed alert typically lists 15-25 roles and a Google
    Careers digest several, so one alert produced exactly one flashcard and the
    other twenty were invisible even though they were right there in the
    message we had already downloaded.

    Listings are split on the job-link pattern rather than on blank lines,
    because alert HTML varies too much to trust a line-based boundary. Each
    chunk gets the subject-derived title only when the subject is itself the
    job (a one-job alert); for a multi-job digest the subject is "40 new jobs"
    and belongs to no single posting, so those titles come from the chunk text.
    """
    chunks: list[tuple[str, str]] = []  # (link, surrounding text)
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if "view job" not in line.lower() and "apply" not in line.lower():
            continue
        m = _LINK_RE.search(line)
        if not m:
            continue
        link = _best_job_link([m.group(0)])
        if not link:
            continue
        # A few lines around the link carry the title and the location.
        window = " ".join(lines[max(0, i - 3):i + 1])
        chunks.append((link, window))
    if not chunks:
        single = _extract_listing(subject, body)
        return [single] if single else []

    multi = _looks_like_digest(subject)
    out: list[dict] = []
    seen: set[str] = set()
    for link, window in chunks:
        if link in seen:
            continue
        seen.add(link)
        title = ""
        if not multi:
            title = _TITLE_PREFIX_RE.sub("", subject or "").strip()
        if not title:
            title = _title_from_window(window)
        if not title:
            # Never fall back to a digest subject. "30 new jobs for you" is not
            # the title of any posting, and putting it on every card is how a
            # 20-job alert becomes twenty identical cards.
            title = "Job alert" if multi else (
                _TITLE_PREFIX_RE.sub("", subject or "").strip() or "Job alert")
        title = title[:90]
        company = ""
        m = _COMPANY_RE.search(window)
        if m:
            company = m.group(1).strip().rstrip(".,;")
            company = _GREETING_RE.sub("", company).strip()
        location = ""
        m = _LOCATION_RE.search(window)
        if m:
            location = m.group(1).strip()
        out.append({"source": "Gmail", "title": title, "company": company,
                    "location": location, "salary": "", "link": link})
    return out


def _looks_like_digest(subject: str) -> bool:
    """True when the subject describes a batch rather than one job.

    "40 new jobs for devops" and "Your Indeed job alerts" are batches;
    "Software Engineer at Acme" is a single posting whose subject is its title.
    """
    s = (subject or "").lower()
    if re.search(r"\b\d+\s+(?:new\s+)?(?:jobs?|matches|results?|roles?)\b", s):
        return True
    return bool(re.search(r"\b(?:job alerts?|new jobs|job matches|"
                          r"recommended jobs|your .* alerts)\b", s))


def _title_from_window(window: str) -> str:
    """Best title guess from the lines surrounding a job link.

    Job titles in these digests are the line immediately above the link, and
    they are usually the only sentence-like fragment in the window. Everything
    else in the window ("Apply now", "Save", the location) is chrome, so the
    longest fragment that is not one of those is preferred.
    """
    chrome = {"view job", "apply", "apply now", "save", "save job", "see job",
              "view details", "job details", "read more", "remove", "hide",
              "report", "not interested", "click here", "learn more"}
    best = ""
    for raw in window.split("|"):
        frag = raw.strip().strip("-–—•*· \t")
        if not frag or len(frag) < 4 or len(frag) > 110:
            continue
        low = frag.lower().rstrip(".:")
        if low in chrome or low.startswith(("apply", "save job", "view job")):
            continue
        if re.fullmatch(r"https?://\S+", frag):
            continue
        # Prefer the longest fragment: the title is the descriptive one.
        if len(frag) > len(best):
            best = frag
    return best.strip()


def parse_message(raw: bytes, account: int = 1) -> dict | None:
    """One RFC822 message -> the row shape gmail_index stores. None if junk."""
    try:
        msg = email.message_from_bytes(raw)
    except Exception:
        return None
    subject = _decode(msg.get("Subject"))
    # Normalise HTML before cleaning: the cleaner and the listing parser both
    # work on text, and HTML-only senders (Google Careers) would otherwise lose
    # every job link to a href attribute.
    body = _clean_body(_html_to_text(_get_body(msg)))
    if not body.strip() and not subject.strip():
        return None
    msg_id = (msg.get("Message-ID") or "").strip()
    if not msg_id:
        # A digest with no Message-ID cannot be de-duplicated across re-syncs,
        # so key it on something stable instead of skipping it.
        msg_id = "sha:" + hashlib.sha1(
            (subject + body[:400] + str(msg.get("Date", ""))).encode(
                "utf-8", "replace")).hexdigest()
    return {
        "msg_id": msg_id,
        "account": account,
        "sender": _sender_address(msg),
        "subject": subject,
        "sent_at": _sent_at(msg),
        "body": body[:1200],
        "listings": _extract_listings(subject, body),
    }


_SENDER_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _sender_address(msg: email.message.Message) -> str:
    """The bare lowercased From address, for filtering by sender.

    A real From header is `"indeed" <donotreply@jobalert.indeed.com>`, so the
    display name has to be stripped before the address can be compared to the
    configured sender list. Matching on the whole header is what made a
    sender-filtered search return nothing on a mailbox full of matching mail.
    """
    m = _SENDER_RE.search(msg.get("From", "") or "")
    return m.group(0).lower() if m else ""


def sync_alerts(account: int = 1, days_back: int = 3650,
                senders: list[str] | None = None,
                max_per_sender: int = 400) -> dict:
    """Fetch this mailbox's alert mail ONCE into the local cache.

    Returns {messages, listings, senders, error}. Never raises: a failed sync is
    reported in `error` so a search can still run against whatever the cache
    already holds instead of taking the whole run down.
    """
    try:
        conn = _connect(account)
    except Exception as e:
        gmail_index.record_sync(account, 0, str(e))
        return {"messages": 0, "listings": 0, "senders": [], "error": str(e)}
    gmail_index.init()
    parsed: list[dict] = []
    hit: list[str] = []
    try:
        conn.select("INBOX")
        # Per-mailbox senders, because the alert mail is not distributed evenly:
        # one mailbox receives the Indeed alerts and the other the Google
        # Careers digest. Searching a global list in the wrong mailbox returns
        # nothing and looks like "no jobs in my inbox".
        for sender in (senders or gmail_senders_for_account(account)):
            st, data = conn.search(
                None,
                f'(FROM "{sender}" SINCE "{(datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")}")')
            if st != "OK":
                continue
            ids = data[0].split()[-max_per_sender:]
            if not ids:
                continue
            hit.append(sender)
            for mid in ids:
                st, d = conn.fetch(mid, "(RFC822)")
                if st != "OK" or not d or not isinstance(d[0], tuple):
                    continue
                row = parse_message(d[0][1], account)
                if row:
                    parsed.append(row)
        gmail_index.upsert_messages(parsed)
        n_listings = gmail_index.count_listings(account)
        gmail_index.record_sync(account, len(parsed), "")
        return {"messages": len(parsed), "listings": n_listings,
                "senders": hit, "error": ""}
    except Exception as e:
        # Whatever was parsed before the failure is still worth keeping.
        gmail_index.upsert_messages(parsed)
        gmail_index.record_sync(account, len(parsed), str(e))
        return {"messages": len(parsed),
                "listings": gmail_index.count_listings(account),
                "senders": hit, "error": str(e)}
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def sync_all_accounts(days_back: int = 3650) -> dict:
    """Sync every configured mailbox. Safe to call from the dashboard boot."""
    out = {}
    accounts = gmail_index_fallback_accounts()
    for account in accounts:
        out[account] = sync_alerts(account, days_back)
    return out


def gmail_index_fallback_accounts() -> list[int]:
    """1-based account numbers available, without importing the agent.

    Kept local so this module has no dependency on agent.py (which imports the
    scraper and starts a LangGraph client); the Gmail tool must stay importable
    on its own as an MCP subprocess.
    """
    pairs = 0
    for addr_key, pw_key in (("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"),
                             ("GMAIL_ADDRESS_2", "GMAIL_APP_PASSWORD_2"),
                             ("GMAIL_ADDRESS_3", "GMAIL_APP_PASSWORD_3")):
        if (os.environ.get(addr_key) or "").strip() and \
           (os.environ.get(pw_key) or "").replace(" ", "").strip():
            pairs += 1
    return list(range(1, pairs + 1)) or [1]


def _extract_listing(subject: str, body: str) -> dict | None:
    """Best-effort parse of one job-alert email into a structured listing.

    Keeps the single "View job:" URL (tracking URLs are already stripped by
    _clean_body) so the dashboard can offer a working Apply button. Returns
    None when there is no usable link."""
    link = ""
    for line in body.splitlines():
        if "view job" in line.lower() or "apply" in line.lower():
            m = _LINK_RE.search(line)
            if m:
                link = _best_job_link([m.group(0)])
                if link:
                    break
    if not link:
        link = _best_job_link(_LINK_RE.findall(body))
    if not link:
        return None  # an alert without a link isn't actionable

    title = _TITLE_PREFIX_RE.sub("", subject or "").strip()
    if not title:
        title = next((ln.strip() for ln in (body or "").splitlines() if ln.strip()), "Job alert")
    title = title[:90]

    company = ""
    m = _COMPANY_RE.search((subject or "") + " " + (body or "")[:300])
    if m:
        company = m.group(1).strip().rstrip(".,;")
        company = _GREETING_RE.sub("", company).strip()
    location = ""
    m = _LOCATION_RE.search((body or "")[:400])
    if m:
        location = m.group(1).strip()

    return {"source": "Gmail", "title": title, "company": company,
            "location": location, "salary": "",
            "link": link.split("&")[0] if "indeed.com" in link else link}


@mcp.tool()
def sync_gmail_cache(days_back: int = 3650, account: int = 0) -> str:
    """Download this project's job-alert mail into the local cache. Call once.

    The cache is what every search reads, so this is the expensive step: one
    IMAP login per mailbox, fetching each alert once and parsing out every job
    in it. Afterwards, searches are local SQLite queries and need no network.

    account: 0 = every configured mailbox (default), or a single 1-based
    account number to sync just that one.

    Returns a short summary: messages and listings cached per account, which
    senders actually matched, and any error (a failed mailbox does not stop the
    others, and does not delete what is already cached)."""
    accounts = gmail_index_fallback_accounts() if account == 0 else [account]
    out = {}
    total_msgs = total_listings = 0
    for acct in accounts:
        res = sync_alerts(acct, days_back)
        out[str(acct)] = res
        total_msgs += res["messages"]
        total_listings += res["listings"]
    lines = [f"Cached {total_msgs} alert message(s), {total_listings} listing(s)."]
    for acct, res in out.items():
        bits = []
        if res["senders"]:
            bits.append("senders: " + ", ".join(res["senders"]))
        if res["messages"]:
            bits.append(f"{res['messages']} msg")
        if res["listings"]:
            bits.append(f"{res['listings']} listings")
        if res["error"]:
            bits.append("ERROR " + res["error"])
        lines.append(f"  mailbox {acct}: " + (", ".join(bits) or "nothing matched"))
    return "\n".join(lines)


@mcp.tool()
def search_job_emails(sender: str = "", days_back: int = 60, max_results: int = 10,
                      account: int = 0, query: str = "",
                      resync_if_stale: bool = True) -> str:
    """Find cached job-alert listings. Reads the local cache, not Gmail.

    A stale cache (older than an hour, or never synced) is refreshed once,
    automatically, so callers do not have to remember to sync.

    sender: restrict to one alert sender address. Empty = every sender.
    query: free-text filter over title/company/location, e.g. "devops" or
    "cloud aws". Every whitespace-separated term must match.
    account: 0 = search every configured mailbox (default), or one 1-based
    account number.
    days_back / max_results: age and cap.

    Returns a summary of what matched, plus a ###JOBS_JSON### block of the
    listings, which the dashboard turns into flashcards with Apply buttons."""
    accounts = gmail_index_fallback_accounts() if account == 0 else [account]

    # Refresh a stale cache, but only once, and never fail the search on it: the
    # point of the cache is that a query still answers while IMAP is down.
    if resync_if_stale:
        for acct in accounts:
            if gmail_index.is_stale(acct):
                sync_alerts(acct)

    rows: list[dict] = []
    for acct in accounts:
        rows += gmail_index.search_listings(
            sender=sender, account=acct, query=query,
            days_back=days_back, limit=max_results)
    # Cross-account merge, newest first. Re-sorted here because concatenating
    # per-account pages would group the results by mailbox instead of by date.
    rows.sort(key=lambda r: r.get("sent_at", ""), reverse=True)
    rows = rows[:max_results]

    by_sender: dict[str, int] = {}
    for r in rows:
        k = r.get("sender") or "unknown"
        by_sender[k] = by_sender.get(k, 0) + 1

    if not rows:
        cached = gmail_index.count_listings()
        return (f"No cached listings matched"
                f"{f' sender {sender}' if sender else ''}"
                f"{f' {query!r}' if query else ''}"
                f" in the last {days_back} days."
                f" ({cached} listing(s) cached in total — run sync_gmail_cache "
                f"if the mailbox has not been indexed yet.)")

    lines = [f"{len(rows)} cached listing(s) in the last {days_back} days"]
    lines += [f"  {k}: {v}" for k, v in sorted(by_sender.items(), key=lambda kv: -kv[1])]
    if len(rows) == max_results:
        lines.append(f"  (capped at {max_results} — raise max_results for more)")
    return "\n".join(lines) + "\n\n" + JOBS_MARKER + "\n" + json.dumps(
        {"sources": by_sender, "total": len(rows), "jobs": rows},
        ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
