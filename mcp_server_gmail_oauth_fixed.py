"""MCP SERVER: Gmail access via OAuth 2.0 (Gmail API)."""
import os
import re
import json
import hashlib
from datetime import datetime, timedelta, timezone
from email.header import decode_header

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
JOBS_MARKER = "###JOBS_JSON###"

_LINK_RE = re.compile(r"https?://[^\s<>()\[\]'\";]+", re.IGNORECASE)
_TITLE_PREFIX_RE = re.compile(
    r"^(?:Your |New |Top |Daily |Weekly |Latest )?(?:Job )?(?:Alert|Matches|Recommendations|Vacancies|Opening|Postings)[:\s\-]*",
    re.IGNORECASE,
)
_COMPANY_RE = re.compile(r"(?:at|with|by|@|from)\s+([A-Z][A-Za-z0-9&\.\- ]{2,40})", re.IGNORECASE)
_LOCATION_RE = re.compile(r"(?:in|location|remote|:)\s+([A-Za-z0-9,\.\-+/ ]{2,60})$", re.IGNORECASE)
_GREETING_RE = re.compile(r"^(?:Hi|Hello|Hey)\s+[A-Za-z]+[,;:\s]*", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_SPACE_RE = re.compile(r"\n{3,}")
def _decode(value) -> str:
    if not value:
        return ""
    try:
        parts = []
        for chunk, enc in decode_header(value):
            if isinstance(chunk, bytes):
                parts.append(chunk.decode(enc or "utf-8", "replace"))
            else:
                parts.append(str(chunk))
        return "".join(parts)
    except Exception:
        return str(value)
def _html_to_text(s: str) -> str:
    if not s:
        return ""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(s, "html.parser").get_text("\n")
    except Exception:
        return _HTML_TAG_RE.sub(" ", s)
def _clean_body(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()
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

def _parse_message_from_raw(raw: bytes, account: int = 1) -> Dict[str, Any] | None:
    try:
        import email
        msg = email.message_from_bytes(raw)
    except Exception:
        return None
    subject = _decode(msg.get("Subject"))
    body = _clean_body(_html_to_text(_get_body(msg)))
    if not body.strip() and not subject.strip():
        return None
    msg_id = (msg.get("Message-ID") or "").strip()
    if not msg_id:
        msg_id = "sha:" + hashlib.sha1(
            (subject + body[:400] + str(msg.get("Date", ""))).encode("utf-8", "replace")).hexdigest()
    return {
        "msg_id": msg_id,
        "account": account,
        "sender": _sender_address(msg),
        "subject": subject,
        "sent_at": _sent_at(msg),
        "body": body[:1200],
        "listings": _extract_listings(subject, body),
    }
def _get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True) or b""
            if part.get_content_type() == "text/html" and not part.is_attachment():
                return part.get_payload(decode=True) or b""
        return b""
    return msg.get_payload(decode=True) or b""

def _sender_address(msg) -> str:
    import re
    s = msg.get("From", "") or ""
    m = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", s)
    return m.group(0).lower() if m else ""

def _sent_at(msg) -> str:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(msg.get("Date") or "")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

def _best_job_link(cands):
    best = ""
    for u in cands:
        frag = u.split("?", 1)[0]
        if not frag:
            continue
        if re.fullmatch(r"https?://\S+", frag):
            pass
        if len(frag) > len(best):
            best = frag
    return best.strip()

def _extract_listings(subject: str, body: str):
    listings = []
    seen = set()
    for u in _LINK_RE.findall(body + " " + subject):
        link = _best_job_link([u])
        if not link or link in seen:
            continue
        seen.add(link)
        title = _TITLE_PREFIX_RE.sub("", subject or "").strip()
        if not title:
            title = next((ln.strip() for ln in body.splitlines() if ln.strip()), "Job alert")[:90]
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
        listings.append({
            "source": "Gmail",
            "title": title,
            "company": company,
            "location": location,
            "salary": "",
            "link": link.split("&")[0] if "indeed.com" in link else link,
        })
    return listings
@mcp.tool()
def sync_gmail_cache(days_back: int = 3650, account: int = 0) -> str:
    """Download job-alert mail into local cache (OAuth)."""
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
                res = service.users().messages().list(userId="me", q=q, maxResults=500).execute()
                msgs = res.get("messages", [])
                for m in msgs:
                    try:
                        raw = service.users().messages().get(userId="me", id=m["id"], format="raw").execute()
                        data = raw.get("raw")
                        if data:
                            import base64
                            row = _parse_message_from_raw(base64.urlsafe_b64decode(data + '==='), account=acct)
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
                if gmail_index.is_stale(acct):\n                    sync_gmail_cache(days_back=days_back, account=acct)
            except Exception:
                pass
    rows: List[Dict[str, Any]] = []
    for acct in accounts:
        try:
            rows += gmail_index.search_listings(sender=sender, account=acct, query=query,
                                               days_back=days_back, limit=max_results)
        except Exception:
            pass
    rows.sort(key=lambda r: r.get("sent_at", ""), reverse=True)
    rows = rows[:max_results]
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
def _connect(account: int = 1):
    class _Dummy:
        def select(self, *a, **k):
            return "OK", []
        def search(self, *a, **k):
            return "OK", [b""]
        def fetch(self, *a, **k):
            return "OK", []
        def logout(self):
            pass
    return _Dummy()

def sync_alerts(account: int = 1, days_back: int = 3650, senders=None, max_per_sender=400):
    return {"messages": 0, "listings": 0, "senders": [], "error": ""}


