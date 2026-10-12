"""MCP SERVER: Gmail access via OAuth 2.0 (Gmail API).

Transport is OAuth (Gmail API). The parsing helpers below turn each alert
message into structured listings and are transport-agnostic; they were restored
because the OAuth rewrite had dropped them, which broke job-link ranking and
digest parsing (every card pointed at a search page, a logo or Indeed's ToS,
and multi-job alerts collapsed to a single listing).
"""
import os
import re
import json
import email
import hashlib
from datetime import datetime, timedelta, timezone
from email.header import decode_header
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

import gmail_index
from filters import gmail_senders_for_account

try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_ENV_PATH):
        load_dotenv(_ENV_PATH, override=False)
except Exception:
    pass

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    try:
        from google.auth.transport.requests import Request
    except Exception:
        Request = None
except Exception as e:
    raise ImportError(f"Missing Google API deps: {e}") from e

mcp = FastMCP("gmail-tools")
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_FILE = os.environ.get(
    "GMAIL_OAUTH_CREDENTIALS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "client_secret_711408450947-oepdiicj2ja7daeji525ib6vkgunkrbp.apps.googleusercontent.com.json"),
)
TOKEN_FILE = os.environ.get(
    "GMAIL_OAUTH_TOKEN_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "gmail_oauth_token.json"),
)

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
# tracker rather than dropping the job - discarding these lost 133 of 144
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


# Job-alert emails (esp. Indeed) pad every message with 8-10 huge tracking URLs
# (View job, Apply now, Yes/No/Maybe, Edit profile, Recent work experience) that
# can be 300-600+ chars EACH. Sent to a local LLM with a small context window,
# this noise crowds out the actual job info and causes hallucination. We
# truncate at the first footer marker, keeping the real listing + its one
# "View job" link.
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

    # Long URLs are de-parametrised rather than dropped. This used to discard
    # any line containing a URL of 80+ characters, on the theory that those were
    # tracking noise. Every real job link is longer than that: the Google
    # Careers digest lost all 12 of its jobs and the whole sender produced zero
    # listings, because a job URL with campaign parameters is routinely 150+
    # characters. Tracking noise is now removed by asking whether the URL points
    # at a job, not by how long it is.
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
    a `label: url` line lets the normal text parser - which looks for a job URL
    on a line mentioning view/apply - see them, and keeps the label as the
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


def _decode(value) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    out = ""
    for text, enc in parts:
        out += text.decode(enc or "utf-8", errors="ignore") if isinstance(text, bytes) else text
    return out


def _get_body(msg) -> str:
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


def _sent_at(msg) -> str:
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
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass
    m = _ISO_DATE.search(raw)
    return f"{m.group(1)}T{m.group(2)}" if m else ""


def _extract_listings(subject: str, body: str) -> list[dict]:
    """Every job in one alert email, not just the first.

    An Indeed alert typically lists 15-25 roles and a Google Careers digest
    several, so returning one listing per email left the other twenty invisible
    even though they were right there in the message we had already downloaded.

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
        frag = raw.strip().strip("-\u2013\u2014\u2022*\u00b7 \t")
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


def _sender_address(msg) -> str:
    """The bare lowercased From address, for filtering by sender.

    A real From header is `"indeed" <donotreply@jobalert.indeed.com>`, so the
    display name has to be stripped before the address can be compared to the
    configured sender list. Matching on the whole header is what made a
    sender-filtered search return nothing on a mailbox full of matching mail.
    """
    m = _SENDER_RE.search(msg.get("From", "") or "")
    return m.group(0).lower() if m else ""


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


def _get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token and Request:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(f"OAuth client secret not found: {CREDENTIALS_FILE}")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        try:
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
        except Exception:
            pass
    return creds


def _api_call(request, tries: int = 4):
    """Run a Gmail API request, retrying transient rate limits.

    A busy sweep of several senders can trip Gmail's per-user rate limit and
    answer 403 rateLimitExceeded / 429 instead of the message. Without a retry
    those messages were silently dropped (the per-message except swallowed
    them), so the cache came back short and looked like "no jobs in the inbox".
    """
    import time
    delay = 1.5
    for attempt in range(tries):
        try:
            return request.execute()
        except HttpError as e:
            status = getattr(getattr(e, "resp", None), "status", None)
            if status in (403, 429) and attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return request.execute()


@mcp.tool()
def sync_gmail_cache(days_back: int = 3650, account: int = 0) -> str:
    """Download job-alert mail into the local cache (OAuth)."""
    accounts = [1] if account != 0 else [1, 2]
    total_msgs = total_listings = 0
    out_lines = []
    try:
        creds = _get_credentials()
        service = build("gmail", "v1", credentials=creds)
    except Exception as e:
        return f"OAuth init failed: {e}"
    for acct in accounts:
        try:
            senders = gmail_senders_for_account(acct)
        except Exception:
            senders = []
        parsed = []
        hit = []
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y/%m/%d")
            for sender in senders:
                q = f'from:"{sender}" after:{cutoff}'
                res = _api_call(service.users().messages().list(
                    userId="me", q=q, maxResults=500))
                msgs = res.get("messages", [])
                for m in msgs:
                    try:
                        raw = _api_call(service.users().messages().get(
                            userId="me", id=m["id"], format="raw"))
                        data = raw.get("raw")
                        if data:
                            import base64
                            row = parse_message(base64.urlsafe_b64decode(data + '==='), account=acct)
                            if row:
                                parsed.append(row)
                                if sender not in hit:
                                    hit.append(sender)
                    except Exception:
                        continue
            gmail_index.upsert_messages(parsed)
            n_listings = gmail_index.count_listings(acct)
            gmail_index.record_sync(acct, len(parsed), "")
            total_msgs += len(parsed)
            total_listings += n_listings
            out_lines.append(f"mailbox {acct}: {len(parsed)} msgs, {n_listings} listings, senders: {hit}")
        except Exception as e:
            gmail_index.upsert_messages(parsed)
            gmail_index.record_sync(acct, len(parsed), str(e))
            out_lines.append(f"mailbox {acct}: ERROR {e}")
    return "\n".join(["Cached " + str(total_msgs) + " messages, " + str(total_listings) + " listings."] + out_lines)


@mcp.tool()
def search_job_emails(sender: str = "", days_back: int = 60, max_results: int = 10,
                      account: int = 0, query: str = "",
                      resync_if_stale: bool = True) -> str:
    accounts = [1] if account != 0 else [1, 2]
    if resync_if_stale:
        for acct in accounts:
            try:
                if gmail_index.is_stale(acct):
                    sync_gmail_cache(days_back=days_back, account=acct)
            except Exception:
                pass
    pool: list[dict] = []
    for acct in accounts:
        try:
            # Over-fetch per account, then interleave across senders. Sorting a
            # per-account top-N by date let one prolific sender (LinkedIn's
            # 18-listing digests) fill the whole result and push Indeed out.
            pool += gmail_index.search_listings(sender=sender, account=acct, query=query,
                                                days_back=days_back,
                                                limit=max(max_results * 6, 200))
        except Exception:
            pass
    rows = gmail_index.diverse_listings(pool, max_results)
    by_sender: Dict[str, int] = {}
    for r in rows:
        k = r.get("sender") or "unknown"
        by_sender[k] = by_sender.get(k, 0) + 1
    if not rows:
        cached = gmail_index.count_listings()
        return (f"No cached listings matched" +
                f"{f' sender {sender}' if sender else ''}" +
                f"{f' {query!r}' if query else ''}" +
                f" in the last {days_back} days. ({cached} listing(s) cached in total)")
    lines = [f"{len(rows)} cached listing(s) in the last {days_back} days"]
    lines += [f"  {k}: {v}" for k, v in sorted(by_sender.items(), key=lambda kv: -kv[1])]
    if len(rows) == max_results:
        lines.append(f"  (capped at {max_results})")
    return "\n".join(lines) + "\n\n" + JOBS_MARKER + "\n" + json.dumps(
        {"sources": by_sender, "total": len(rows), "jobs": rows}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
