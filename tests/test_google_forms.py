"""Google Forms intake: OAuth connect, form linking, sync into applications,
company scoping and idempotent re-sync."""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import auth
import candidates_api
import google_forms_api
import mailer
import recruiter_api

FORM_TITLE = "Apply here"
QUESTIONS = {"q1": "Email", "q2": "Full name", "q3": "Why do you want this job?",
             "q4": "Resume"}
FORM_URL = "https://docs.google.com/forms/d/abcDEF123456/edit"


def _user(email: str, role: str = "recruiter", company: str | None = None) -> dict:
    user = auth.create_user(email, "pw12345678", role=role)
    auth.mark_verified(user["id"])
    if role == "recruiter":
        owner = auth.create_company(company or email, created_by=user["id"])
        auth.create_recruiter_profile(user["id"], owner["id"])
        user["company_id"] = owner["id"]
    return user


def _client(user: dict | None) -> TestClient:
    app = FastAPI()
    app.include_router(google_forms_api.router)
    app.include_router(recruiter_api.router)
    client = TestClient(app)
    if user:
        client.cookies.set(auth.SESSION_COOKIE, auth.create_session(user["id"])[0])
    return client


def _answer(value: str) -> dict:
    return {"textAnswers": {"answers": [{"value": value}]}}


RESPONSES = [
    {"respondentEmail": "Form@X.com", "lastSubmittedTime": "2026-11-10T09:00:00.000Z",
     "answers": {"q1": _answer("form@x.com"), "q2": _answer("Ada Lovelace"),
                 "q3": _answer("I like APIs")}},
    {"lastSubmittedTime": "2026-11-10T10:00:00.000Z",
     "answers": {"q1": _answer("two@x.com"), "q2": _answer("Grace Hopper")}},
    {"lastSubmittedTime": "2026-11-10T11:00:00.000Z",
     "answers": {"q2": _answer("No Email")}},
]

FILE_RESPONSE = {
    "respondentEmail": "cv@x.com", "lastSubmittedTime": "2026-11-10T12:00:00.000Z",
    "answers": {"q1": _answer("cv@x.com"), "q2": _answer("Ada CV"),
                "q4": {"fileUploadAnswers": {"answers": [
                    {"fileId": "driveFILE123456", "fileName": "ada-cv.pdf",
                     "mimeType": "application/pdf"}]}}},
}


def _pull_stub(rows):
    def pull(user_id, form_id, responses):
        return FORM_TITLE, QUESTIONS, list(rows) if responses else []
    return pull


def _connect(user_id: int, email: str = "rec@gmail.com") -> None:
    with google_forms_api._db() as conn:
        conn.execute(
            "INSERT INTO google_oauth (user_id, email, refresh_token, scopes, connected_at) "
            "VALUES (?,?,?,?,?)",
            (user_id, email, "refresh-token", " ".join(google_forms_api.SCOPES),
             google_forms_api._now()))


@pytest.fixture(autouse=True)
def no_real_mail(monkeypatch):
    """Scheduling an interview emails the candidate; a test must never reach SMTP."""
    monkeypatch.setattr(recruiter_api.mailer, "send",
                        lambda *a, **k: recruiter_api.mailer.MailResult(True, "test"))


@pytest.fixture
def data(monkeypatch):
    """Two companies, a job each, and clients for both (plus anon/candidate)."""
    rec = _user("rec@x.com", company="Acme")
    other = _user("other@x.com", company="Other Co")
    job = auth.create_job(rec["company_id"], rec["id"],
                          {"title": "Backend engineer", "description": "Python APIs",
                           "status": "published"})
    other_job = auth.create_job(other["company_id"], other["id"],
                                {"title": "Elsewhere", "description": "Nothing",
                                 "status": "published"})
    return {"rec": rec, "other": other, "job": job, "other_job": other_job,
            "rec_client": _client(rec), "other_client": _client(other),
            "anon": _client(None),
            "candidate_client": _client(_user("person@x.com", role="candidate"))}


def _link(client, job_id: int, form: str = FORM_URL) -> dict:
    res = client.post("/api/recruiter/integrations/google-forms/links",
                      json={"form": form, "job_id": job_id})
    assert res.status_code == 200, res.text
    return res.json()["link"]


def test_start_returns_a_consent_url_and_stores_one_state(data, monkeypatch):
    seen = []

    def fake_url(state, redirect_uri):
        seen.append(state)
        return "https://accounts.google.com/o/oauth2/auth?state=" + state, "verifier-" + state

    monkeypatch.setattr(google_forms_api, "_authorize_url", fake_url)
    res = data["rec_client"].get("/api/recruiter/integrations/google-forms/start")
    assert res.status_code == 200, res.text
    assert res.json()["url"].startswith("https://accounts.google.com/")
    assert len(seen) == 1
    with google_forms_api._db() as conn:
        rows = [tuple(r) for r in conn.execute(
            "SELECT state, code_verifier FROM google_oauth_states")]
    assert rows == [(seen[0], "verifier-" + seen[0])]


def test_callback_stores_the_refresh_token_and_returns_to_the_portal(data, monkeypatch):
    states, verifiers = [], []
    monkeypatch.setattr(google_forms_api, "_authorize_url",
                        lambda state, redirect_uri: (states.append(state), ("https://x/" + state, "verifier"))[1])
    monkeypatch.setattr(google_forms_api, "_exchange",
                        lambda code, redirect_uri, code_verifier: (
                            verifiers.append(code_verifier), ("refresh-token", "rec@gmail.com"))[1])
    data["rec_client"].get("/api/recruiter/integrations/google-forms/start")

    res = data["rec_client"].get(
        "/api/recruiter/integrations/google-forms/callback",
        params={"code": "c", "state": states[0]}, follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/recruiter#google-forms"
    # PKCE: the verifier generated with the consent URL must reach the exchange.
    assert verifiers == ["verifier"]

    status = data["rec_client"].get("/api/recruiter/integrations/google-forms").json()
    assert status["connected"] is True and status["email"] == "rec@gmail.com"
    # The state is single use.
    again = data["rec_client"].get(
        "/api/recruiter/integrations/google-forms/callback",
        params={"code": "c", "state": states[0]}, follow_redirects=False)
    assert again.status_code == 403


def test_an_older_state_table_gets_the_pkce_column(data):
    """A database created before PKCE support must gain the column in place."""
    with auth.connect() as conn:
        conn.execute("DROP TABLE IF EXISTS google_oauth_states")
        conn.execute("CREATE TABLE google_oauth_states (state TEXT PRIMARY KEY, "
                     "user_id INTEGER NOT NULL, created_at TEXT NOT NULL)")
    google_forms_api._ready.discard(auth.DB_FILE)
    with google_forms_api._db() as conn:
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(google_oauth_states)")}
    assert "code_verifier" in columns


def test_callback_works_without_the_portal_session_cookie(data, monkeypatch):
    """Google's redirect can land on a host/port that never saw the session cookie."""
    states = []
    monkeypatch.setattr(google_forms_api, "_authorize_url",
                        lambda state, redirect_uri: (states.append(state), ("https://x/" + state, "verifier"))[1])
    monkeypatch.setattr(google_forms_api, "_exchange",
                        lambda code, redirect_uri, code_verifier: ("refresh-token", "rec@gmail.com"))
    data["rec_client"].get("/api/recruiter/integrations/google-forms/start")

    res = data["anon"].get(
        "/api/recruiter/integrations/google-forms/callback",
        params={"code": "c", "state": states[0]}, follow_redirects=False)
    assert res.status_code == 303, res.text
    assert data["rec_client"].get(
        "/api/recruiter/integrations/google-forms").json()["connected"] is True


def test_callback_reuses_the_redirect_uri_the_consent_url_carried(data, monkeypatch):
    """The token exchange must repeat the exact redirect_uri Google saw at consent."""
    states, seen = [], []
    monkeypatch.setenv("GOOGLE_FORMS_REDIRECT_URI", "http://localhost:8000/cb")
    monkeypatch.setattr(google_forms_api, "_authorize_url",
                        lambda state, redirect_uri: (states.append(state), ("https://x/" + state, "v"))[1])
    monkeypatch.setattr(google_forms_api, "_exchange",
                        lambda code, redirect_uri, code_verifier: (
                            seen.append(redirect_uri), ("rt", "rec@gmail.com"))[1])
    data["rec_client"].get("/api/recruiter/integrations/google-forms/start")

    # The browser comes back on another host/port; the stored URI must win.
    data["anon"].get("/api/recruiter/integrations/google-forms/callback",
                     params={"code": "c", "state": states[0]}, follow_redirects=False)
    assert seen == ["http://localhost:8000/cb"]


def test_a_rejected_exchange_names_googles_error(data, monkeypatch):
    """'try again' is useless on its own: the reason must be visible."""
    states = []
    monkeypatch.setattr(google_forms_api, "_authorize_url",
                        lambda state, redirect_uri: (states.append(state), ("https://x/" + state, "v"))[1])

    class Rejected:
        status_code = 400
        content = b'{"error": "invalid_grant"}'
        text = '{"error": "invalid_grant", "error_description": "Bad Request"}'

        def json(self):
            return {"error": "invalid_grant", "error_description": "Bad Request"}

    monkeypatch.setattr(google_forms_api.requests, "post", lambda *a, **k: Rejected())
    data["rec_client"].get("/api/recruiter/integrations/google-forms/start")
    res = data["rec_client"].get(
        "/api/recruiter/integrations/google-forms/callback",
        params={"code": "c", "state": states[0]}, follow_redirects=False)
    assert res.status_code == 400
    assert "invalid_grant" in res.json()["detail"]
    assert data["rec_client"].get(
        "/api/recruiter/integrations/google-forms").json()["connected"] is False


def test_another_recruiter_cannot_replay_a_state(data, monkeypatch):
    states = []
    monkeypatch.setattr(google_forms_api, "_authorize_url",
                        lambda state, redirect_uri: (states.append(state), ("https://x/" + state, "verifier"))[1])
    monkeypatch.setattr(google_forms_api, "_exchange",
                        lambda code, redirect_uri, code_verifier: ("refresh-token", "rec@gmail.com"))
    data["rec_client"].get("/api/recruiter/integrations/google-forms/start")
    res = data["other_client"].get(
        "/api/recruiter/integrations/google-forms/callback",
        params={"code": "c", "state": states[0]}, follow_redirects=False)
    assert res.status_code == 403


def test_sync_turns_responses_into_company_scoped_applications(data, monkeypatch):
    _connect(data["rec"]["id"])
    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub(RESPONSES))
    link = _link(data["rec_client"], data["job"]["id"])
    assert link["form_title"] == FORM_TITLE

    res = data["rec_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync")
    assert res.status_code == 200, res.text
    assert res.json() == {"ok": True, "added": 2, "seen": 3, "skipped": 1,
                          "duplicates": 0, "no_email": 1, "resumes": 0}

    with auth.connect() as conn:
        apps = [dict(r) for r in conn.execute(
            "SELECT a.status, a.source, a.applied_at, a.cover_letter, u.email, u.verified_at "
            "FROM applications a JOIN users u ON u.id = a.candidate_user_id "
            "WHERE a.job_id = ? ORDER BY u.email", (data["job"]["id"],))]
        # Each respondent also becomes one CRM candidate the recruiter can open.
        assert conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE user_id = ? AND email IN "
            "('form@x.com', 'two@x.com')", (data["rec"]["id"],)).fetchone()[0] == 2
    assert [a["email"] for a in apps] == ["form@x.com", "two@x.com"]
    assert all(a["source"] == "google_form" and a["status"] == "applied" for a in apps)
    assert all(a["verified_at"] is None for a in apps)
    assert "Full name: Ada Lovelace" in apps[0]["cover_letter"]
    assert apps[0]["applied_at"].startswith("2026-11-10T09:00:00")

    # A recruiter's own status change survives the next sync, and nothing duplicates.
    with auth.connect() as conn:
        conn.execute("UPDATE applications SET status = 'shortlisted' WHERE job_id = ?",
                     (data["job"]["id"],))
    again = data["rec_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync").json()
    assert again["added"] == 0 and again["seen"] == 3 and again["duplicates"] == 2
    with auth.connect() as conn:
        assert [r[0] for r in conn.execute(
            "SELECT status FROM applications WHERE job_id = ?", (data["job"]["id"],))] \
            == ["shortlisted", "shortlisted"]
        assert conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE user_id = ?", (data["rec"]["id"],)).fetchone()[0] == 2


def test_a_response_without_an_email_is_skipped_not_invented(data, monkeypatch):
    _connect(data["rec"]["id"])
    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub([RESPONSES[2]]))
    link = _link(data["rec_client"], data["job"]["id"])
    res = data["rec_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync").json()
    assert res["added"] == 0 and res["skipped"] == 1 and res["no_email"] == 1
    with auth.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM users WHERE verified_at IS NULL").fetchone()[0] == 0


def test_sync_downloads_an_uploaded_resume_into_screening(data, monkeypatch, tmp_path):
    _connect(data["rec"]["id"])
    monkeypatch.setattr(candidates_api, "RESUME_DIR", str(tmp_path / "resumes"))
    monkeypatch.setattr(candidates_api, "_resume_kind", lambda blob: "pdf")
    monkeypatch.setattr(candidates_api, "_resume_text", lambda blob, kind: "Python and SQL")
    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub([FILE_RESPONSE]))
    monkeypatch.setattr(google_forms_api, "_download",
                        lambda user_id, file_id: b"%PDF-1.4 uploaded cv")
    link = _link(data["rec_client"], data["job"]["id"])

    res = data["rec_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync").json()
    assert res["added"] == 1 and res["resumes"] == 1

    with auth.connect() as conn:
        row = conn.execute("SELECT filename, parsed_text FROM candidate_resumes").fetchone()
    assert row["filename"] == "ada-cv.pdf"
    assert "Python" in row["parsed_text"]


def test_a_broken_resume_download_keeps_the_application(data, monkeypatch):
    _connect(data["rec"]["id"])

    def boom(user_id, file_id):
        raise RuntimeError("Drive said no")

    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub([FILE_RESPONSE]))
    monkeypatch.setattr(google_forms_api, "_download", boom)
    link = _link(data["rec_client"], data["job"]["id"])
    res = data["rec_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync").json()
    assert res["added"] == 1 and res["resumes"] == 0
    with auth.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 1


def test_scheduling_an_interview_emails_the_form_respondent(data, monkeypatch):
    """The address and the name the form collected are what the interview email uses."""
    _connect(data["rec"]["id"])
    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub(RESPONSES))
    link = _link(data["rec_client"], data["job"]["id"])
    data["rec_client"].post(f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync")
    with auth.connect() as conn:
        app_id = conn.execute(
            "SELECT a.id FROM applications a JOIN users u ON u.id = a.candidate_user_id "
            "WHERE a.job_id = ? AND u.email = 'form@x.com'", (data["job"]["id"],)).fetchone()[0]

    sent = []
    monkeypatch.setattr(recruiter_api.mailer, "send",
                        lambda to, subject, body, **kw: (
                            sent.append((to, subject, body)), mailer.MailResult(True))[1])
    res = data["rec_client"].post("/api/recruiter/interviews", json={
        "application_id": app_id, "scheduled_at": "2026-12-01T10:00:00Z",
        "duration_minutes": 45, "meeting_link": "https://meet.example/abc",
        "instructions": "Bring your ID"})
    assert res.status_code == 200, res.text
    assert res.json()["mail"] == {"sent": True, "error": ""}

    to, subject, body = sent[0]
    assert to == "form@x.com"
    assert "Backend engineer" in subject
    assert "Ada Lovelace" in body                       # the name from the form answer
    assert "01 Dec 2026, 10:00 UTC" in body and "45 minutes" in body
    assert "https://meet.example/abc" in body and "Bring your ID" in body
    assert "Acme" in body


def test_a_failed_email_still_books_the_interview(data, monkeypatch):
    """SMTP trouble must be reported, not turned into a failed booking."""
    candidate = auth.create_user("plain@x.com", "pw12345678")
    app_id = auth.apply_to_job(data["job"]["id"], candidate["id"])["id"]
    monkeypatch.setattr(recruiter_api.mailer, "send",
                        lambda *a, **k: mailer.MailResult(False, "SMTP is not configured"))
    res = data["rec_client"].post("/api/recruiter/interviews", json={
        "application_id": app_id, "scheduled_at": "2026-12-02T09:00:00Z"})
    assert res.status_code == 200, res.text
    assert res.json()["interview"]["status"] == "scheduled"
    assert res.json()["mail"] == {"sent": False, "error": "SMTP is not configured"}


def test_creating_a_job_creates_and_links_a_form(data, monkeypatch):
    _connect(data["rec"]["id"])
    monkeypatch.setattr(
        google_forms_api, "_create_form",
        lambda user_id, title: ("FORMnew123456", title,
                                "https://docs.google.com/forms/d/FORMnew123456/viewform"))
    res = data["rec_client"].post(
        "/api/recruiter/jobs", json={"title": "Data engineer", "description": "ETL pipelines"})
    assert res.status_code == 200, res.text
    made = res.json()
    assert made["form"]["form_id"] == "FORMnew123456"
    assert made["form"]["responder_uri"].endswith("/viewform")
    links = data["rec_client"].get(
        "/api/recruiter/integrations/google-forms").json()["links"]
    assert [(l["form_id"], l["job_id"]) for l in links] == \
        [("FORMnew123456", made["job"]["id"])]


def test_creating_a_job_without_google_just_skips_the_form(data):
    res = data["other_client"].post(
        "/api/recruiter/jobs", json={"title": "Ops", "description": "Run things"})
    assert res.status_code == 200 and res.json()["form"] is None


def test_links_are_scoped_to_the_recruiters_company(data, monkeypatch):
    _connect(data["rec"]["id"])
    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub(RESPONSES))
    link = _link(data["rec_client"], data["job"]["id"])

    assert data["other_client"].get(
        "/api/recruiter/integrations/google-forms").json()["links"] == []
    assert data["other_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync").status_code == 404
    assert data["other_client"].delete(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}").status_code == 404
    # ...and a form cannot be pointed at another company's job.
    assert data["rec_client"].post(
        "/api/recruiter/integrations/google-forms/links",
        json={"form": FORM_URL, "job_id": data["other_job"]["id"]}).status_code == 404


def test_disconnect_stops_sync_until_reconnected(data, monkeypatch):
    _connect(data["rec"]["id"])
    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub(RESPONSES))
    link = _link(data["rec_client"], data["job"]["id"])
    assert data["rec_client"].delete(
        "/api/recruiter/integrations/google-forms").status_code == 200
    res = data["rec_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync")
    assert res.status_code == 400
    assert "Connect a Google account" in res.json()["detail"]


def test_a_form_respondent_can_later_claim_the_account(data, monkeypatch):
    """Signing up with a form respondent's address takes over the created row."""
    _connect(data["rec"]["id"])
    monkeypatch.setattr(google_forms_api, "_pull", _pull_stub(RESPONSES))
    link = _link(data["rec_client"], data["job"]["id"])
    data["rec_client"].post(
        f"/api/recruiter/integrations/google-forms/links/{link['id']}/sync")

    claimed = auth.create_user("form@x.com", "brandnewpass")
    with auth.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications WHERE candidate_user_id IN "
                            "(SELECT id FROM users WHERE email = 'form@x.com')").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM profiles WHERE user_id = ?",
                            (claimed["id"],)).fetchone()[0] == 1
    assert auth.authenticate("form@x.com", "brandnewpass")["id"] == claimed["id"]


def test_only_recruiters_can_use_the_integration(data):
    for client in (data["anon"], data["candidate_client"]):
        assert client.get(
            "/api/recruiter/integrations/google-forms").status_code in (401, 403)


@pytest.mark.parametrize("value,expected", [
    ("https://docs.google.com/forms/d/abcDEF123456/edit", "abcDEF123456"),
    ("  1FAIpQLSabc_-123  ", "1FAIpQLSabc_-123"),
])
def test_form_id_accepts_edit_links_and_bare_ids(value, expected):
    assert google_forms_api._form_id(value) == expected


@pytest.mark.parametrize("value", [
    "https://docs.google.com/forms/d/e/1FAIpQLSabc_-123/viewform",
    "https://forms.gle/shortlink",
])
def test_a_responder_link_is_refused_with_a_usable_hint(value):
    """The published id in a responder link is not the Forms API form id."""
    with pytest.raises(HTTPException) as err:
        google_forms_api._form_id(value)
    assert "edit link" in err.value.detail
