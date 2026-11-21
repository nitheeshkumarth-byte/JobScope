from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import auth
import google_accounts
import google_gmail
import mcp_server_gmail

ORIGIN = "http://localhost"
CLIENT = {"client_id": "test.apps.googleusercontent.com", "client_secret": "unused-in-tests"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(google_accounts, "_client", lambda: CLIENT)
    monkeypatch.setenv("JOBSCOPE_GOOGLE_TOKEN_KEY", Fernet.generate_key().decode())
    app = FastAPI()
    app.include_router(google_accounts.router)
    with TestClient(app, base_url=ORIGIN) as browser:
        yield browser


def google_exchange(monkeypatch, email, subject, gmail=False):
    claims = {"sub": subject, "email": email, "email_verified": True}
    tokens = {"refresh_token": "refresh-for-" + subject,
              "scope": google_accounts.GMAIL_SCOPE} if gmail else {}
    monkeypatch.setattr(google_accounts, "_exchange", lambda *args: (claims, tokens))


def sign_in(client, uid):
    session, _ = auth.create_session(uid)
    client.cookies.set(auth.SESSION_COOKIE, session)
    return session


def start(client, intent):
    response = client.post("/api/google/start", json={"intent": intent}, headers={"Origin": ORIGIN})
    assert response.status_code == 200, response.text
    params = parse_qs(urlsplit(response.json()["url"]).query)
    assert params["code_challenge_method"] == ["S256"]
    assert params["nonce"] and params["redirect_uri"] == [ORIGIN + google_accounts.CALLBACK]
    if intent == "gmail":
        assert google_accounts.GMAIL_SCOPE in params["scope"][0].split()
        assert params["access_type"] == ["offline"]
    else:
        assert google_accounts.GMAIL_SCOPE not in params["scope"][0].split()
    return params["state"][0]


def finish(client, state):
    return client.get(google_accounts.CALLBACK, params={"state": state, "code": "oauth-code"},
                      follow_redirects=False)


def test_google_signup_creates_a_candidate_and_does_not_reuse_state(client, monkeypatch):
    google_exchange(monkeypatch, "candidate@gmail.com", "google-candidate")
    state = start(client, "signin")
    response = finish(client, state)
    assert response.headers["location"] == "/candidate?google=signed_in"
    user = auth.session_user(client.cookies.get(auth.SESSION_COOKIE))
    assert user["role"] == "candidate" and user["verified_at"] and not user["is_owner"]
    client.cookies.set(google_accounts.STATE_COOKIE, state, path=google_accounts.CALLBACK)
    assert finish(client, state).headers["location"] == "/?google=expired"


def test_existing_recruiter_must_explicitly_link_before_google_sign_in(client, monkeypatch):
    user = auth.create_user("recruiter@gmail.com", "a-good-password", role="recruiter", verified=True)
    google_exchange(monkeypatch, user["email"], "recruiter")
    assert finish(client, start(client, "signin")).headers["location"] == "/?google=link_first"
    assert not client.cookies.get(auth.SESSION_COOKIE)
    sign_in(client, user["id"])
    assert finish(client, start(client, "link")).headers["location"] == "/recruiter?google=linked"
    client.cookies.clear()
    assert finish(client, start(client, "signin")).headers["location"] == "/recruiter?google=signed_in"
    assert auth.session_user(client.cookies.get(auth.SESSION_COOKIE))["id"] == user["id"]


def test_callback_is_bound_to_the_original_jobscope_session(client, monkeypatch):
    user = auth.create_user("first@gmail.com", "a-good-password", verified=True)
    other = auth.create_user("other@gmail.com", "a-good-password", verified=True)
    google_exchange(monkeypatch, user["email"], "first", gmail=True)
    sign_in(client, user["id"])
    state = start(client, "gmail")
    sign_in(client, other["id"])
    assert finish(client, state).headers["location"] == "/candidate?google=expired"
    assert google_accounts.gmail_connection(other["id"]) is None
    assert google_accounts.gmail_connection(user["id"]) is None


def test_oauth_start_rejects_a_foreign_origin_and_anonymous_gmail(client):
    response = client.post("/api/google/start", json={"intent": "signin"},
                           headers={"Origin": "https://unrelated.example"})
    assert response.status_code == 403
    response = client.post("/api/google/start", json={"intent": "gmail"}, headers={"Origin": ORIGIN})
    assert response.status_code == 403


def test_oauth_uses_the_registered_callback_with_trailing_slash(client, monkeypatch):
    path = google_accounts.CALLBACK + "/"
    monkeypatch.setattr(google_accounts, "_client", lambda: {**CLIENT, "redirect_uris": [ORIGIN + path]})
    google_exchange(monkeypatch, "candidate@gmail.com", "google-candidate")
    response = client.post("/api/google/start", json={"intent": "signin"}, headers={"Origin": ORIGIN})
    params = parse_qs(urlsplit(response.json()["url"]).query)
    assert params["redirect_uri"] == [ORIGIN + path]
    result = client.get(path, params={"state": params["state"][0], "code": "oauth-code"},
                        follow_redirects=False)
    assert result.headers["location"] == "/candidate?google=signed_in"


def test_candidates_have_separate_gmail_tokens_and_alerts(client, monkeypatch):
    users = [auth.create_user(f"mailbox{number}@gmail.com", "a-good-password", verified=True)
             for number in (1, 2)]
    monkeypatch.setattr(google_gmail, "_new_service", lambda connection: connection["subject"])

    def fetch(subject, senders, days_back):
        return ([{"msg_id": "shared-message-id", "sender": "job-alerts@example.com",
                  "sent_at": auth._now(), "body": "private email body",
                  "listings": [{"title": subject, "company": "Example", "location": "Remote",
                                "link": "https://example.com/job"}]}], [])

    monkeypatch.setattr(mcp_server_gmail, "fetch_job_alerts", fetch)
    connections = []
    for user in users:
        sign_in(client, user["id"])
        google_exchange(monkeypatch, user["email"], user["email"], gmail=True)
        assert finish(client, start(client, "gmail")).headers["location"] == "/candidate?google=gmail_connected"
        connection = google_accounts.gmail_connection(user["id"])
        assert user["email"] not in connection["refresh_token"]
        assert google_accounts.gmail_credentials(connection).refresh_token == "refresh-for-" + user["email"]
        google_gmail.sync(connection, 30)
        connections.append(connection)

    for user, connection in zip(users, connections):
        assert [r["title"] for r in google_gmail.listings(connection, 30, "", 20)["jobs"]] == [user["email"]]
    with auth.connect() as conn:
        assert all("private email body" not in row[0] for row in conn.execute("SELECT listings FROM google_gmail_alerts"))

    sign_in(client, users[0]["id"])
    assert client.delete("/api/integrations/gmail", headers={"Origin": ORIGIN}).status_code == 200
    assert not google_gmail.listings(connections[0], 30, "", 20)["jobs"]
    assert google_gmail.listings(connections[1], 30, "", 20)["jobs"]
    with pytest.raises(HTTPException) as err:
        google_gmail.sync(connections[0], 30)
    assert err.value.status_code == 409
