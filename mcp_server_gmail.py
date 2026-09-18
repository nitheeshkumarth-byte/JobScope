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
import email
from email.header import decode_header
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP

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

    # collapse the remaining long tracking-style URLs (still present in
    # "Apply now:" links etc.) down to a short placeholder, but keep the
    # FIRST url on a "View job:" line since that's the one worth keeping
    lines = text.splitlines()
    kept_view_job_link = False
    out_lines = []
    for line in lines:
        if line.strip().lower().startswith("view job:") and not kept_view_job_link:
            out_lines.append(line.strip())
            kept_view_job_link = True
        elif re.search(r"https?://\S{80,}", line):
            # very long url on this line and it's not the one we want — drop it
            continue
        else:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def _connect():
    address = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")
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
                link = m.group(0).rstrip(".,);")
                break
    if not link:
        m = _LINK_RE.search(body)
        if m:
            link = m.group(0).rstrip(".,);")
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
def search_job_emails(sender: str, days_back: int = 60, max_results: int = 10) -> str:
    """Search Gmail for emails from a specific sender within the last N days.
    Example sender values: 'careers-noreply@google.com' (Google Careers
    digest) or 'donotreply@match.indeed.com' (Indeed job alerts).
    Returns subject, date, and raw body content for each matching email,
    followed by a ###JOBS_JSON### block of structured listings parsed from
    those emails (the UI turns those into flashcards with Apply buttons)."""
    try:
        conn = _connect()
        conn.select("INBOX")
    except Exception as e:
        return f"Gmail connection failed: {e}"

    try:
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
        search_query = f'(FROM "{sender}" SINCE "{since_date}")'
        status, data = conn.search(None, search_query)
        if status != "OK":
            return f"IMAP search failed: {status}"

        msg_ids = data[0].split()
        msg_ids = msg_ids[-max_results:]  # most recent N

        if not msg_ids:
            return f"No emails found from '{sender}' in the last {days_back} days."

        results = []
        listings = []
        for msg_id in reversed(msg_ids):  # newest first
            status, msg_data = conn.fetch(msg_id, "(RFC822)")
            if status != "OK":
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            subject = _decode(msg.get("Subject"))
            date = msg.get("Date", "")
            body = _clean_body(_get_body(msg))[:1200]  # cleaned first, then capped
            results.append(f"--- Subject: {subject}\nDate: {date}\nBody:\n{body}\n")
            listing = _extract_listing(subject, body)
            if listing and listing["link"]:
                listings.append(listing)

        text = "\n".join(results)
        if listings:
            text += "\n\n" + JOBS_MARKER + "\n" + json.dumps(
                {"sources": {sender: len(listings)}, "total": len(listings),
                 "jobs": listings}, ensure_ascii=False)
        return text
    except Exception as e:
        return f"Gmail search failed: {e}"
    finally:
        try:
            conn.logout()
        except Exception:
            pass


if __name__ == "__main__":
    mcp.run(transport="stdio")
