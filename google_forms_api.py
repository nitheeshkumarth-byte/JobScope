"""Recruiter Applications intake from Google Forms.

Each recruiter connects their own Google account once (Forms read-only), links
a form to one of their jobs, and syncs responses. A response becomes a normal
`applications` row against an unverified candidate account keyed by the
respondent's email, so it shows up in Applications, Resume Screening and the
pipeline exactly like an application made in the portal.

The refresh token is stored per user, never globally, so revoking one
recruiter's access cannot affect another's. The OAuth client is the existing
installed-app `client_secret.json` (see `GOOGLE_OAUTH_CREDENTIALS`); the only
extra setup is enabling the Google Forms API and allowing the callback URL in
the OAuth consent screen.

Answers are kept as a Q&A transcript in `applications.cover_letter`. A CV the
respondent uploaded to Drive is downloaded into the same candidate resume store
the portal upload uses, so it shows up in Resume Screening with no extra click.

Creating a job while Google is connected also creates and links a published
`Application — <job title>` form (forms are created in the recruiter's own Drive).
shortcut: the Forms API cannot create file-upload questions, so that CV question
is added once by hand in the Google Form UI; upgrade when Google exposes it.
"""
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import auth
import candidates_api
import google_accounts

router = APIRouter()
log = logging.getLogger("google_forms")

HERE = os.path.dirname(os.path.abspath(__file__))

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.readonly",
]

MAX_COVER_LETTER = 4000
STATE_MINUTES = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS google_oauth (
    user_id       INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    email         TEXT,
    refresh_token TEXT NOT NULL,
    scopes        TEXT,
    connected_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS google_oauth_states (
    state         TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    code_verifier TEXT,
    redirect_uri  TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS google_form_links (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id     INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    form_id        TEXT NOT NULL,
    form_title     TEXT,
    job_id         INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    last_synced_at TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE(user_id, form_id)
);
CREATE INDEX IF NOT EXISTS idx_gform_company ON google_form_links(company_id);
"""
_ready: set = set()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _db():
    """Connection with this module's tables guaranteed (the DB path changes under tests)."""
    conn = auth.connect()
    if auth.DB_FILE not in _ready:
        conn.executescript(_SCHEMA)
        # Older databases predate the newer columns; add them in place.
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(google_oauth_states)")}
        for column in ("code_verifier", "redirect_uri"):
            if column not in columns:
                conn.execute(f"ALTER TABLE google_oauth_states ADD COLUMN {column} TEXT")
        _ready.add(auth.DB_FILE)
    return conn


# ---- who is calling -------------------------------------------------------
def _user(request: Request) -> dict:
    user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE) or "")
    if not user:
        raise HTTPException(status_code=401, detail="sign in to continue")
    if not user.get("verified_at"):
        raise HTTPException(status_code=403, detail="confirm your email address")
    return user


def _recruiter(request: Request) -> dict:
    user = _user(request)
    if user.get("role") not in ("recruiter", "super_admin"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user


def _cid(user: dict) -> int | None:
    p = auth.get_recruiter_profile(user["id"])
    return p.get("company_id") if p else None


def _need_company(user: dict) -> int:
    cid = _cid(user)
    if not cid:
        raise HTTPException(status_code=400, detail="Set up your company first")
    return cid


# ---- Google plumbing ------------------------------------------------------
def _credentials_file() -> str:
    return os.environ.get("GOOGLE_OAUTH_CREDENTIALS", os.path.join(HERE, "client_secret.json"))


def _redirect_uri(request: Request | None = None) -> str:
    """Explicit override wins; otherwise mirror the host:port the portal is served on,
    so a non-default port (or 127.0.0.1 vs localhost) still completes the round trip."""
    override = os.environ.get("GOOGLE_FORMS_REDIRECT_URI")
    if override:
        return override
    base = (str(request.base_url).rstrip("/") if request
            else "http://localhost:8000")
    return base + "/api/recruiter/integrations/google-forms/callback"


def _client_config() -> tuple[str, str]:
    """client_id/client_secret from the installed-app file dashboard already ships."""
    try:
        with open(_credentials_file(), encoding="utf-8") as fh:
            cfg = json.load(fh)
        block = cfg.get("installed") or cfg.get("web") or {}
        return block["client_id"], block["client_secret"]
    except (OSError, ValueError, KeyError):
        raise HTTPException(status_code=400,
                            detail="Google OAuth client is not configured on this server")


def _flow(redirect_uri: str, state: str | None = None):
    from google_auth_oauthlib.flow import Flow
    return Flow.from_client_secrets_file(_credentials_file(), SCOPES,
                                         redirect_uri=redirect_uri, state=state)


def _authorize_url(state: str, redirect_uri: str) -> tuple[str, str]:
    """(consent URL, PKCE verifier).

    google_auth_oauthlib sends only the SHA-256 challenge to Google and keeps the
    verifier on the Flow object, so the verifier must be stored with the state and
    replayed on the callback — a fresh Flow cannot finish the exchange.
    """
    flow = _flow(redirect_uri, state=state)
    url, _ = flow.authorization_url(access_type="offline", prompt="consent",
                                    include_granted_scopes="true")
    return url, flow.code_verifier or ""


def _exchange(code: str, redirect_uri: str, code_verifier: str = "") -> tuple[str, str]:
    """(refresh_token, account email) for a consent code.

    The token endpoint is called directly instead of via oauthlib: a rejected
    exchange must show Google's own error code, not a generic message, and the
    `redirect_uri` must be byte-identical to the one the consent URL carried.
    """
    client_id, client_secret = _client_config()
    payload = {"code": code, "client_id": client_id, "client_secret": client_secret,
               "redirect_uri": redirect_uri, "grant_type": "authorization_code"}
    if code_verifier:
        payload["code_verifier"] = code_verifier
    try:
        res = requests.post("https://oauth2.googleapis.com/token", data=payload, timeout=30)
    except requests.RequestException as exc:
        log.warning("google token exchange unreachable: %s", exc)
        raise HTTPException(status_code=400,
                            detail="Could not reach Google, check the network and try again")
    body = res.json() if res.content else {}
    if res.status_code != 200:
        reason = body.get("error") or f"HTTP {res.status_code}"
        log.warning("google token exchange failed: %s %s", reason, res.text[:300])
        raise HTTPException(status_code=400,
                            detail=f"Google rejected that sign-in ({reason}), try again")
    if not body.get("refresh_token"):
        raise HTTPException(status_code=400,
                            detail="Google did not return a refresh token; remove the app's "
                                   "access in your Google account and connect again")
    email = ""
    try:
        info = requests.get("https://www.googleapis.com/oauth2/v2/userinfo", timeout=30,
                            headers={"Authorization": "Bearer " + body["access_token"]})
        email = (info.json() or {}).get("email", "") if info.status_code == 200 else ""
    except (requests.RequestException, ValueError, KeyError):
        pass
    return body["refresh_token"], email


def _need_connection(user_id: int) -> None:
    with _db() as conn:
        row = conn.execute("SELECT 1 FROM google_oauth WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="Connect a Google account first")


def _credentials(refresh_token: str):
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2.credentials import Credentials
    client_id, client_secret = _client_config()
    if refresh_token.startswith("fernet:"):
        refresh_token = google_accounts.decrypt_refresh_token(refresh_token.removeprefix("fernet:"))
    creds = Credentials(token=None, refresh_token=refresh_token,
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=client_id, client_secret=client_secret, scopes=SCOPES)
    creds.refresh(GoogleRequest())
    return creds


def _service(user_id: int, api: str = "forms", version: str = "v1"):
    with _db() as conn:
        row = conn.execute("SELECT refresh_token FROM google_oauth WHERE user_id = ?",
                           (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="Connect a Google account first")
    from googleapiclient.discovery import build
    try:
        return build(api, version, credentials=_credentials(row["refresh_token"]),
                     cache_discovery=False)
    except Exception:
        raise HTTPException(status_code=400,
                            detail="Google sign-in has expired; reconnect your Google account")


def _form_questions(form: dict) -> dict[str, str]:
    questions = {}
    for item in form.get("items") or []:
        question = (item.get("questionItem") or {}).get("question") or {}
        if question.get("questionId"):
            questions[question["questionId"]] = (item.get("title") or "").strip()
    return questions


def _pull(user_id: int, form_id: str, responses: bool):
    """(form title, {question id: title}, response rows).

    The single place that talks to Google, so tests monkeypatch one function.
    """
    service = _service(user_id)
    from googleapiclient.errors import HttpError
    try:
        form = service.forms().get(formId=form_id).execute()
        questions = _form_questions(form)
        rows: list[dict] = []
        if responses:
            token = None
            while True:
                page = service.forms().responses().list(
                    formId=form_id, pageSize=200,
                    **({"pageToken": token} if token else {})).execute()
                rows.extend(page.get("responses") or [])
                token = page.get("nextPageToken")
                if not token:
                    break
    except HttpError as exc:
        status = getattr(exc.resp, "status", 0)
        if status in (401, 403):
            raise HTTPException(status_code=400,
                                detail="That form is not readable with the connected Google "
                                       "account, or access was revoked")
        if status == 404:
            raise HTTPException(status_code=404, detail="Google Form not found")
        raise HTTPException(status_code=502, detail="Google Forms did not answer, try again")
    return (form.get("title") or "").strip(), questions, rows


def _form_id(value: str) -> str:
    """Accept a Forms edit URL or a bare form id.

    A responder link (`/forms/d/e/<id>/viewform` or `forms.gle/...`) carries the
    published id, which the Forms API cannot resolve, so say that instead of
    failing later with a 404 from Google.
    """
    value = (value or "").strip()
    if re.search(r"/forms/d/e/|forms\.gle/", value):
        raise HTTPException(status_code=400,
                            detail="That is the responder link; paste the edit link "
                                   "(open the form and choose Edit) or the form id")
    match = re.search(r"/forms/d/([A-Za-z0-9_-]{10,})", value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
        return value
    raise HTTPException(status_code=400, detail="Paste a Google Form link or its form id")


def _download(user_id: int, file_id: str) -> bytes:
    """Bytes of a file a respondent uploaded to Drive through the form."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{10,}", file_id or ""):
        raise HTTPException(status_code=400, detail="Invalid uploaded file")
    from googleapiclient.errors import HttpError
    try:
        data = _service(user_id, "drive", "v3").files().get_media(fileId=file_id).execute()
    except HttpError:
        raise HTTPException(status_code=502, detail="Could not download the uploaded resume")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise HTTPException(status_code=502, detail="Could not download the uploaded resume")
    return bytes(data)


FORM_QUESTIONS = (  # (label, paragraph, required)
    ("Full name", False, True),
    ("Email", False, True),
    ("Phone", False, False),
    ("Why do you want this role?", True, False),
)


def _create_form(user_id: int, title: str) -> tuple[str, str, str]:
    """(form id, title, responder uri) for a new published application form.

    The single place that writes to Google, so tests monkeypatch one function.
    """
    service = _service(user_id)
    from googleapiclient.errors import HttpError
    try:
        form = service.forms().create(body={"info": {"title": title}}).execute()
        form_id = form["formId"]
        requests = [{"createItem": {"item": {
            "title": label,
            "questionItem": {"question": {
                "required": required,
                "textQuestion": {"paragraph": paragraph}}}}}}
            for label, paragraph, required in FORM_QUESTIONS]
        service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()
        service.forms().setPublishSettings(
            formId=form_id, updateMask="publish_state",
            body={"publishSettings": {"publishState": {
                "isPublished": True, "isAcceptingResponses": True}}}).execute()
    except HttpError as exc:
        status = getattr(exc.resp, "status", 0)
        if status in (401, 403):
            raise HTTPException(status_code=400,
                                detail="Reconnect Google to allow creating forms")
        raise HTTPException(status_code=502, detail="Google Forms did not create the form")
    return form_id, title, form.get("responderUri") or ""


def create_form_for_job(user: dict, job: dict) -> dict | None:
    """Create + link a Google Form for a just-created job.

    Returns None when this recruiter has not connected Google, or when Google
    fails: creating the job itself must never depend on the optional form.
    """
    with _db() as conn:
        connected = conn.execute("SELECT 1 FROM google_oauth WHERE user_id = ?",
                                 (user["id"],)).fetchone()
    if not connected:
        return None
    cid = _need_company(user)
    title = f"Application — {job['title']}"[:200]
    try:
        form_id, form_title, responder_uri = _create_form(user["id"], title)
    except Exception as exc:
        log.warning("job %s: Google Form not created: %s", job["id"], exc)
        return None
    with _db() as conn:
        conn.execute(
            "INSERT INTO google_form_links (company_id, user_id, form_id, form_title, job_id, "
            "created_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(user_id, form_id) DO UPDATE SET form_title = excluded.form_title, "
            "job_id = excluded.job_id",
            (cid, user["id"], form_id, form_title, job["id"], _now()))
    return {"form_id": form_id, "form_title": form_title, "responder_uri": responder_uri}


# ---- responses -> applications -------------------------------------------
def _answer_values(response: dict, questions: dict[str, str]) -> dict[str, str]:
    values = {}
    answers = response.get("answers") or {}
    for qid in questions:
        texts = [a.get("value", "") for a in
                 ((answers.get(qid) or {}).get("textAnswers") or {}).get("answers") or []]
        text = " ".join(t.strip() for t in texts if t and t.strip())
        if text:
            values[questions[qid] or qid] = text
    return values


def _respondent(values: dict[str, str], response: dict) -> tuple[str, str]:
    """(email, name) for one response, or ("", "") when there is no usable email."""
    email = (response.get("respondentEmail") or "").strip()
    if not email:
        for text in values.values():
            match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
            if match:
                email = match.group(0)
                break
    email = auth.normalize_email(email)
    if not auth._looks_like_email(email):
        return "", ""
    name = ""
    for title, text in values.items():
        low = title.lower()
        if "name" in low and "company" not in low:
            name = text.strip()
            break
    return email, name[:120] or email.split("@")[0]


def _submitted(response: dict) -> str:
    value = (response.get("lastSubmittedTime") or "").strip()
    if value.endswith("Z"):
        return value
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return _now()


def _file_answers(response: dict) -> list[dict]:
    """Files a respondent uploaded to Drive, as {file_id, file_name}."""
    files = []
    for answer in (response.get("answers") or {}).values():
        for f in ((answer or {}).get("fileUploadAnswers") or {}).get("answers") or []:
            if f.get("fileId") and f.get("fileName"):
                files.append({"file_id": f["fileId"], "file_name": f["fileName"]})
    return files


def _apply_response(conn, link, response: dict, questions: dict[str, str]) -> tuple[str, str | None]:
    """Insert the response as an application.

    Returns ("added", crm candidate id), ("duplicate", None) when that candidate
    already has this application, or ("no_email", None) when the response carries
    no address — a form that never asks for one silently produces nothing.
    """
    values = _answer_values(response, questions)
    email, name = _respondent(values, response)
    if not email:
        return "no_email", None
    now = _now()
    # An unverified applicant account; the hash is unrunnable so only a later
    # password reset / sign-up can ever claim it.
    conn.execute(
        "INSERT OR IGNORE INTO users (email, password_hash, created_at, nickname, role) "
        "VALUES (?, ?, ?, ?, 'candidate')",
        (email, "!" + secrets.token_hex(32), now, name))
    row = conn.execute("SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email,)).fetchone()
    # One CRM candidate per respondent, so the recruiter can open them, attach a
    # resume and have it show up in Resume Screening.
    known = conn.execute("SELECT id FROM candidates WHERE user_id = ? AND email = ? COLLATE NOCASE",
                         (link["user_id"], email)).fetchone()
    candidate_id = known["id"] if known else secrets.token_hex(6)
    if not known:
        conn.execute(
            "INSERT INTO candidates (id, user_id, name, email, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'recruiter_entry', ?, ?)",
            (candidate_id, link["user_id"], name, email, now, now))
    cover = "\n".join(f"{title}: {text}" for title, text in values.items())[:MAX_COVER_LETTER]
    cur = conn.execute(
        "INSERT INTO applications (job_id, candidate_user_id, status, cover_letter, applied_at, "
        "updated_at, source) VALUES (?, ?, 'applied', ?, ?, ?, 'google_form') "
        "ON CONFLICT(job_id, candidate_user_id) DO NOTHING",
        (link["job_id"], row["id"], cover, _submitted(response), now))
    return ("added", candidate_id) if cur.rowcount > 0 else ("duplicate", None)


def _sync(user: dict, link_id: int) -> dict:
    cid = _need_company(user)
    with _db() as conn:
        link = conn.execute(
            "SELECT * FROM google_form_links WHERE id = ? AND company_id = ?",
            (link_id, cid)).fetchone()
    if not link:
        raise HTTPException(status_code=404, detail="Linked form not found")
    _need_connection(user["id"])
    _, questions, responses = _pull(user["id"], link["form_id"], responses=True)
    added = duplicates = no_email = 0
    pending: list[tuple[str, dict]] = []
    with _db() as conn:
        for response in responses:
            status, candidate_id = _apply_response(conn, link, response, questions)
            if status == "added":
                added += 1
                pending.extend((candidate_id, f) for f in _file_answers(response))
            elif status == "no_email":
                no_email += 1
            else:
                duplicates += 1
        conn.execute("UPDATE google_form_links SET last_synced_at = ? WHERE id = ?",
                     (_now(), link_id))
    # The link owner owns the CRM candidate rows, even when a colleague syncs.
    owner = {"id": link["user_id"]}
    resumes = 0
    for candidate_id, upload in pending:
        try:
            data = _download(user["id"], upload["file_id"])
            candidates_api.save_resume(owner, candidate_id, data, upload["file_name"])
            resumes += 1
        except Exception:
            continue  # application is already saved; a bad upload must not undo it
    return {"ok": True, "added": added, "seen": len(responses),
            "skipped": duplicates + no_email, "duplicates": duplicates,
            "no_email": no_email, "resumes": resumes}


# ---- routes ---------------------------------------------------------------
class LinkIn(BaseModel):
    form: str = Field(default="", max_length=500)
    job_id: int


@router.get("/api/recruiter/integrations/google-forms/start")
def google_forms_start(request: Request, user: dict = Depends(_recruiter)):
    """Consent URL for this recruiter's own Google account."""
    _need_company(user)
    state = secrets.token_urlsafe(24)
    redirect_uri = _redirect_uri(request)
    try:
        url, code_verifier = _authorize_url(state, redirect_uri)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400,
                            detail="Google OAuth client is not configured on this server")
    with _db() as conn:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=STATE_MINUTES)) \
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute("DELETE FROM google_oauth_states WHERE created_at < ?", (cutoff,))
        conn.execute("INSERT INTO google_oauth_states "
                     "(state, user_id, code_verifier, redirect_uri, created_at) "
                     "VALUES (?,?,?,?,?)",
                     (state, user["id"], code_verifier, redirect_uri, _now()))
    return {"ok": True, "url": url}


@router.get("/api/recruiter/integrations/google-forms/callback")
def google_forms_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Google's redirect back into the portal.

    Identity comes from the single-use `state`, not the portal session cookie:
    the browser may reach this host on a different name/port than the one that
    set the cookie (localhost vs 127.0.0.1), which would otherwise 401 here.
    """
    if error:
        raise HTTPException(status_code=400, detail="Google sign-in was cancelled")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Google sign-in did not finish")
    with _db() as conn:
        row = conn.execute(
            "SELECT user_id, code_verifier, redirect_uri, created_at "
            "FROM google_oauth_states WHERE state = ?",
            (state,)).fetchone()
        conn.execute("DELETE FROM google_oauth_states WHERE state = ?", (state,))
    if not row:
        raise HTTPException(status_code=403, detail="That sign-in link has expired, try again")
    try:
        fresh = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")) >= \
            datetime.now(timezone.utc) - timedelta(minutes=STATE_MINUTES)
    except (TypeError, ValueError):
        fresh = False
    if not fresh:
        raise HTTPException(status_code=403, detail="That sign-in link has expired, try again")
    # Defence in depth: a cookie, when the browser still has this host's, must
    # belong to the same recruiter. Its absence is fine (see the docstring).
    caller = auth.session_user(request.cookies.get(auth.SESSION_COOKIE) or "")
    if caller and caller["id"] != row["user_id"]:
        raise HTTPException(status_code=403, detail="That sign-in link has expired, try again")
    refresh_token, email = _exchange(code, row["redirect_uri"] or _redirect_uri(request),
                                     row["code_verifier"] or "")
    if os.environ.get("JOBSCOPE_GOOGLE_TOKEN_KEY"):
        refresh_token = "fernet:" + google_accounts.encrypt_refresh_token(refresh_token)
    with _db() as conn:
        conn.execute(
            "INSERT INTO google_oauth (user_id, email, refresh_token, scopes, connected_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET email = excluded.email, "
            "refresh_token = excluded.refresh_token, scopes = excluded.scopes, "
            "connected_at = excluded.connected_at",
            (row["user_id"], email, refresh_token, " ".join(SCOPES), _now()))
    return RedirectResponse(url="/recruiter#google-forms", status_code=303)


@router.get("/api/recruiter/integrations/google-forms")
def google_forms_status(user: dict = Depends(_recruiter)):
    cid = _cid(user)
    with _db() as conn:
        token = conn.execute("SELECT email FROM google_oauth WHERE user_id = ?",
                             (user["id"],)).fetchone()
        links = []
        if cid:
            links = [dict(r) for r in conn.execute(
                "SELECT l.id, l.form_id, l.form_title, l.job_id, j.title AS job_title, "
                "l.last_synced_at, l.created_at FROM google_form_links l "
                "JOIN jobs j ON j.id = l.job_id WHERE l.company_id = ? ORDER BY l.id DESC",
                (cid,))]
    return {"ok": True, "connected": bool(token), "email": (token["email"] if token else None),
            "links": links}


@router.delete("/api/recruiter/integrations/google-forms")
def google_forms_disconnect(user: dict = Depends(_recruiter)):
    with _db() as conn:
        conn.execute("DELETE FROM google_oauth WHERE user_id = ?", (user["id"],))
    return {"ok": True}


@router.post("/api/recruiter/integrations/google-forms/links")
def google_forms_link(req: LinkIn, user: dict = Depends(_recruiter)):
    cid = _need_company(user)
    form_id = _form_id(req.form)
    with _db() as conn:
        job = conn.execute("SELECT 1 FROM jobs WHERE id = ? AND company_id = ?",
                           (req.job_id, cid)).fetchone()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    _need_connection(user["id"])
    title, _, _ = _pull(user["id"], form_id, responses=False)
    with _db() as conn:
        conn.execute(
            "INSERT INTO google_form_links (company_id, user_id, form_id, form_title, job_id, "
            "created_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(user_id, form_id) DO UPDATE SET form_title = excluded.form_title, "
            "job_id = excluded.job_id",
            (cid, user["id"], form_id, title or form_id, req.job_id, _now()))
        link = conn.execute(
            "SELECT l.id, l.form_id, l.form_title, l.job_id, j.title AS job_title, "
            "l.last_synced_at, l.created_at FROM google_form_links l "
            "JOIN jobs j ON j.id = l.job_id WHERE l.user_id = ? AND l.form_id = ?",
            (user["id"], form_id)).fetchone()
    return {"ok": True, "link": dict(link)}


@router.delete("/api/recruiter/integrations/google-forms/links/{link_id}")
def google_forms_unlink(link_id: int, user: dict = Depends(_recruiter)):
    cid = _need_company(user)
    with _db() as conn:
        n = conn.execute("DELETE FROM google_form_links WHERE id = ? AND company_id = ?",
                         (link_id, cid)).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Linked form not found")
    return {"ok": True}


@router.post("/api/recruiter/integrations/google-forms/links/{link_id}/sync")
def google_forms_sync(link_id: int, user: dict = Depends(_recruiter)):
    return _sync(user, link_id)
