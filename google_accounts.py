"""Google account sign-in and separately consented Gmail connections."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import requests
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from google.auth import exceptions as google_exceptions
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import credentials, id_token
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel

import auth

router = APIRouter()
CALLBACK = "/api/google/callback"
STATE_COOKIE = "js_google_flow"
STATE_TTL = 600
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
LOGIN_SCOPES = ["openid", "email"]
GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def _client() -> dict:
    file = os.environ.get("GOOGLE_WEB_OAUTH_CREDENTIALS", "").strip()
    if file:
        try:
            data = json.loads((Path(auth.HERE) / file).read_text(encoding="utf-8"))["web"]
            if data["client_id"] and data["client_secret"]:
                return {key: data[key] for key in ("client_id", "client_secret")} | {
                    "redirect_uris": data.get("redirect_uris") or [],
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    raise HTTPException(503, "Configure a Google Web application OAuth client to enable this feature.")


def _cipher() -> Fernet:
    try:
        return Fernet(os.environ.get("JOBSCOPE_GOOGLE_TOKEN_KEY", "").encode())
    except ValueError as exc:
        raise HTTPException(503, "Configure a Google token encryption key to enable this connection.") from exc


def encrypt_refresh_token(token: str) -> str:
    return _cipher().encrypt(token.encode()).decode()


def decrypt_refresh_token(token: str) -> str:
    try:
        return _cipher().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise HTTPException(502, "Reconnect Google. The saved connection could not be opened.") from exc


def enabled() -> bool:
    try:
        _client()
    except HTTPException:
        return False
    return True


def _origin(request: Request) -> str:
    base = os.environ.get("APP_URL", "").strip() or str(request.base_url)
    parsed = urlsplit(base)
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and loopback)) or not parsed.hostname:
        raise HTTPException(503, "Configure APP_URL to the dashboard's HTTPS origin.")
    if parsed.username or parsed.password or parsed.path.rstrip("/") or parsed.query or parsed.fragment:
        raise HTTPException(503, "APP_URL must be the dashboard origin without a path.")
    return f"{parsed.scheme}://{parsed.netloc}"


def same_origin(request: Request) -> str:
    origin = _origin(request)
    if request.headers.get("origin") != origin:
        raise HTTPException(403, "Open JobScope using its configured URL and try again.")
    return origin


def _allowed_user(user: dict | None) -> bool:
    return bool(user and user.get("verified_at") and not user.get("is_owner")
                and user["role"] in {"candidate", "recruiter"})


def _return_path(user: dict | None) -> str:
    if user and user["role"] == "recruiter":
        return "/recruiter"
    return "/candidate" if user else "/"


def _finish(path: str, result: str) -> RedirectResponse:
    response = RedirectResponse(f"{path}?google={result}", status_code=303)
    response.delete_cookie(STATE_COOKIE, path=CALLBACK)
    return response


class StartRequest(BaseModel):
    intent: Literal["signin", "link", "gmail"]


@router.post("/api/google/start")
def start(req: StartRequest, request: Request):
    origin = same_origin(request)
    client = _client()
    session = request.cookies.get(auth.SESSION_COOKIE, "")
    user = auth.session_user(session)
    if req.intent != "signin" and not _allowed_user(user):
        raise HTTPException(403, "Sign in to your verified candidate or recruiter account first.")
    if req.intent == "signin" and user:
        raise HTTPException(409, "Use Link Google from your signed-in account.")
    if req.intent == "gmail":
        if user["role"] != "candidate":
            raise HTTPException(403, "Gmail job alerts require a candidate account.")
        _cipher()
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(48)
    redirect = origin + CALLBACK
    registered = client.get("redirect_uris", [])
    if redirect not in registered and redirect + "/" in registered:
        redirect += "/"
    scopes = LOGIN_SCOPES + ([GMAIL_SCOPE] if req.intent == "gmail" else [])
    flow = Flow.from_client_config(
        {"web": {**client, "auth_uri": AUTH_URI, "token_uri": TOKEN_URI}},
        scopes, state=state, code_verifier=verifier, autogenerate_code_verifier=False,
        redirect_uri=redirect,
    )
    options = {"prompt": "select_account", "nonce": nonce, "access_type": "online"}
    if req.intent == "gmail":
        options.update(access_type="offline", prompt="consent select_account")
    url, _ = flow.authorization_url(**options)
    now = int(time.time())
    with auth.connect() as conn:
        conn.execute("DELETE FROM google_account_states WHERE expires_at <= ?", (now,))
        conn.execute("INSERT INTO google_account_states VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (auth._token_hash(state), req.intent, user["id"] if user else None,
                      auth._token_hash(session) if user else None, nonce, verifier, redirect, now + STATE_TTL))
    response = JSONResponse({"url": url})
    response.set_cookie(STATE_COOKIE, state, max_age=STATE_TTL, path=CALLBACK,
                        httponly=True, samesite="lax", secure=origin.startswith("https:"))
    return response


def _exchange(code: str, attempt, client: dict) -> tuple[dict, dict]:
    response = requests.post(TOKEN_URI, data={
        "client_id": client["client_id"], "client_secret": client["client_secret"],
        "code": code, "code_verifier": attempt["verifier"],
        "redirect_uri": attempt["redirect_uri"], "grant_type": "authorization_code",
    }, timeout=15)
    response.raise_for_status()
    tokens = response.json()
    claims = id_token.verify_oauth2_token(tokens["id_token"], GoogleRequest(), client["client_id"])
    if claims.get("nonce") != attempt["nonce"] or not claims.get("sub") or claims.get("email_verified") is not True:
        raise ValueError("Google identity did not match the request")
    claims["email"] = auth.normalize_email(claims.get("email"))
    if not auth._looks_like_email(claims["email"]):
        raise ValueError("Google did not provide a verified email")
    return claims, tokens


def _user_for_signin(claims: dict) -> dict:
    def linked_user():
        with auth.connect() as conn:
            row = conn.execute("SELECT u.* FROM google_identities g JOIN users u ON u.id = g.user_id "
                               "WHERE g.subject = ?", (claims["sub"],)).fetchone()
            return dict(row) if row else None

    user = linked_user()
    if not user:
        email = claims["email"]
        # Google is authoritative for Gmail and hosted Workspace addresses.
        if not (email.endswith("@gmail.com") or claims.get("hd")):
            raise HTTPException(409, "Create a verified JobScope account before linking Google.")
        try:
            user = auth.create_user(email, secrets.token_urlsafe(32), verified=True,
                                    google_subject=claims["sub"])
        except (ValueError, sqlite3.IntegrityError):
            # A concurrent callback for the same Google account may have created it.
            user = linked_user()
            if not user:
                raise HTTPException(409, "Sign in to the existing JobScope account and link Google first.")
    if not _allowed_user(user):
        raise HTTPException(409, "Sign in with your existing JobScope account.")
    return user


def _link(user: dict, claims: dict) -> None:
    if claims["email"] != auth.normalize_email(user["email"]):
        raise HTTPException(403, "Choose the Google account with your JobScope email address.")
    try:
        with auth.connect() as conn:
            conn.execute("INSERT INTO google_identities (subject, user_id) VALUES (?, ?) "
                         "ON CONFLICT(subject) DO NOTHING", (claims["sub"], user["id"]))
            row = conn.execute("SELECT user_id FROM google_identities WHERE subject = ?",
                               (claims["sub"],)).fetchone()
            if row["user_id"] != user["id"]:
                raise HTTPException(409, "That Google account is already linked.")
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "This JobScope account already has a Google sign-in.") from exc


def gmail_connection(uid: int):
    with auth.connect() as conn:
        return conn.execute("SELECT * FROM google_gmail_connections WHERE user_id = ?", (uid,)).fetchone()


def _connect_gmail(user: dict, claims: dict, tokens: dict, client: dict) -> None:
    refresh_token = tokens.get("refresh_token")
    if not refresh_token or GMAIL_SCOPE not in (tokens.get("scope") or "").split():
        raise ValueError("Google did not grant Gmail read access")
    encrypted = encrypt_refresh_token(refresh_token)
    with auth.connect() as conn:
        # Reconnecting gets a new connection id. Old results cascade away, even
        # when the account changes or a previous sync finishes after the switch.
        conn.execute("DELETE FROM google_gmail_connections WHERE user_id = ?", (user["id"],))
        conn.execute("INSERT INTO google_gmail_connections (user_id, client_id, subject, email, refresh_token) "
                     "VALUES (?, ?, ?, ?, ?)",
                     (user["id"], client["client_id"], claims["sub"], claims["email"], encrypted))


@router.get(CALLBACK + "/", include_in_schema=False)
@router.get(CALLBACK)
def callback(request: Request, state: str = "", code: str = "", error: str = ""):
    if not state or not secrets.compare_digest(state.encode(), request.cookies.get(STATE_COOKIE, "").encode()):
        return _finish("/", "expired")
    with auth.connect() as conn:
        attempt = conn.execute("DELETE FROM google_account_states WHERE state_hash = ? AND expires_at > ? "
                               "RETURNING *", (auth._token_hash(state), int(time.time()))).fetchone()
    if not attempt:
        return _finish("/", "expired")
    session = request.cookies.get(auth.SESSION_COOKIE, "")
    user = auth.session_user(session)
    path = _return_path(user)
    if not code or error:
        return _finish(path, "cancelled")
    if attempt["user_id"] is not None and (not _allowed_user(user) or user["id"] != attempt["user_id"]
                                           or auth._token_hash(session) != attempt["session_hash"]):
        return _finish(path, "expired")
    if attempt["intent"] == "signin" and user:
        return _finish(path, "link_first")
    try:
        client = _client()
        claims, tokens = _exchange(code, attempt, client)
        if attempt["user_id"] is not None:
            user = auth.session_user(session)
            if not _allowed_user(user) or user["id"] != attempt["user_id"]:
                return _finish(path, "expired")
        if attempt["intent"] == "link":
            _link(user, claims)
            return _finish(path, "linked")
        if attempt["intent"] == "gmail":
            if user["role"] != "candidate":
                return _finish(path, "failed")
            _connect_gmail(user, claims, tokens, client)
            return _finish(path, "gmail_connected")
        user = _user_for_signin(claims)
    except HTTPException as exc:
        result = "link_first" if attempt["intent"] == "signin" and exc.status_code == 409 else "failed"
        return _finish(path, result)
    except (requests.RequestException, google_exceptions.GoogleAuthError, ValueError, KeyError):
        return _finish(path, "failed")
    response = _finish(_return_path(user), "signed_in")
    token, expires = auth.create_session(user["id"])
    auth.set_session_profile(token, "default")
    response.set_cookie(auth.SESSION_COOKIE, token, httponly=True, samesite="lax",
                        secure=attempt["redirect_uri"].startswith("https:"), expires=expires, path="/")
    return response


def gmail_credentials(connection):
    client = _client()
    if client["client_id"] != connection["client_id"]:
        raise HTTPException(502, "Reconnect Gmail after changing the Google OAuth client.")
    refresh = decrypt_refresh_token(connection["refresh_token"])
    return credentials.Credentials(None, refresh_token=refresh, token_uri=TOKEN_URI,
                                   client_id=client["client_id"], client_secret=client["client_secret"],
                                   scopes=[GMAIL_SCOPE])


@router.get("/api/integrations/gmail")
def gmail_status(request: Request):
    user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE, ""))
    if not _allowed_user(user) or user["role"] != "candidate":
        raise HTTPException(403, "Sign in to your verified candidate account first.")
    connection = gmail_connection(user["id"])
    try:
        gmail_enabled = enabled() and bool(_cipher())
    except HTTPException:
        gmail_enabled = False
    return {"enabled": gmail_enabled, "connected": connection is not None,
            "email": connection["email"] if connection else ""}


@router.delete("/api/integrations/gmail")
def gmail_disconnect(request: Request):
    same_origin(request)
    user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE, ""))
    if not _allowed_user(user) or user["role"] != "candidate":
        raise HTTPException(403, "Sign in to your verified candidate account first.")
    with auth.connect() as conn:
        conn.execute("DELETE FROM google_gmail_connections WHERE user_id = ?", (user["id"],))
    return {"ok": True}
