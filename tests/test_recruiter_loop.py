"""Recruiter portal API end to end: company scoping, candidates, interviews, tasks, and the
candidate-slot -> calendar -> confirm flow. Uses the router alone (no agent needed)."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import auth
import candidates_api
import recruiter_api


def _client(email, role):
    u = auth.create_user(email, "pw12345678", role=role)
    auth.mark_verified(u["id"])
    app = FastAPI()
    app.include_router(candidates_api.router)
    app.include_router(recruiter_api.router)
    c = TestClient(app)
    c.cookies.set(auth.SESSION_COOKIE, auth.create_session(u["id"])[0])
    return c, u


def _future(days=3):
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture(autouse=True)
def no_real_mail(monkeypatch):
    """Scheduling an interview emails the candidate; a test must never reach SMTP."""
    monkeypatch.setattr(recruiter_api.mailer, "send",
                        lambda *a, **k: recruiter_api.mailer.MailResult(True, "test"))


def test_hiring_loop_and_calendar():
    rec, _ = _client("rec@x.com", "recruiter")
    other, _ = _client("rec2@x.com", "recruiter")
    cand, cu = _client("cand@x.com", "candidate")

    assert rec.get("/api/recruiter/jobs").json()["jobs"] == []
    assert rec.post("/api/recruiter/jobs", json={"title": "Dev", "description": "d"}).status_code == 400  # no company
    rec.post("/api/recruiter/company", json={"name": "Acme"})
    rec.post("/api/recruiter/company", json={"name": "Acme Inc", "industry": "SaaS"})  # edits, no second company
    assert rec.get("/api/recruiter/company").json()["company"]["name"] == "Acme Inc"

    # profile: full name becomes the display name, company link survives the save
    rec.post("/api/recruiter/profile", json={"full_name": "Rita Rec", "title": "HR"})
    prof = rec.get("/api/recruiter/profile").json()
    assert prof["profile"]["full_name"] == "Rita Rec" and prof["identity"]["display_name"] == "Rita Rec"
    assert rec.get("/api/recruiter/company").json()["company"] is not None

    job = rec.post("/api/recruiter/jobs", json={"title": "Dev", "description": "d", "status": "published"}).json()["job"]
    rec.post(f"/api/recruiter/jobs/{job['id']}/update", json={"title": "Dev II", "required_skills": ["py"]})
    assert other.post(f"/api/recruiter/jobs/{job['id']}/status", json={"status": "closed"}).status_code == 404
    assert rec.post(f"/api/recruiter/jobs/{job['id']}/status", json={"status": "bogus"}).status_code == 400

    app_id = auth.apply_to_job(job["id"], cu["id"])["id"]
    apps = rec.get("/api/recruiter/applications").json()["applications"]
    assert apps[0]["candidate_email"] == "cand@x.com" and apps[0]["job_title"] == "Dev II"
    assert other.post(f"/api/recruiter/applications/{app_id}/status", json={"status": "hired"}).status_code in (400, 404)

    # candidates form feeds the dashboard numbers
    assert rec.post("/api/candidates", json={"name": ""}).status_code == 400
    rec.post("/api/candidates", json={"name": "Zed", "status": "ready_for_interview"})
    cid = rec.post("/api/candidates", json={"name": "Yan", "status": "working"}).json()["id"]
    rec.post("/api/candidates", json={"id": cid, "name": "Yan", "status": "ready_for_interview", "email": "y@y.com"})
    assert len(rec.get("/api/candidates").json()["candidates"]) == 2
    st = rec.get("/api/recruiter/dashboard/stats").json()["stats"]
    assert st["candidates"] == 2 and st["candidates_ready"] == 2 and st["total_applications"] == 1
    assert rec.delete(f"/api/candidates/{cid}").json()["ok"]

    # candidate proposes slots -> shows on recruiter calendar -> recruiter confirms
    mine = cand.get("/api/candidate/applications").json()["applications"]
    assert mine[0]["job_title"] == "Dev II"
    assert cand.post(f"/api/candidate/applications/{app_id}/slots", json={"slots": [{"start_at": "2020-01-01T10:00:00Z"}]}).status_code == 400
    assert cand.post(f"/api/candidate/applications/{app_id}/slots",
                     json={"slots": [{"start_at": _future(3)}, {"start_at": _future(4), "duration_minutes": 30}]}).json()["ok"]
    assert rec.get("/api/recruiter/dashboard/stats").json()["stats"]["pending_slots"] == 2
    assert rec.get("/api/recruiter/calendar").json()["slots"][0]["candidate_email"] == "cand@x.com"
    assert other.get("/api/recruiter/calendar").json()["slots"] == []
    slot = rec.get("/api/recruiter/calendar").json()["slots"][0]
    assert other.post(f"/api/recruiter/slots/{slot['id']}/confirm", json={}).status_code in (400, 404)
    iv = rec.post(f"/api/recruiter/slots/{slot['id']}/confirm", json={"interview_type": "technical_interview"}).json()["interview"]
    assert iv["scheduled_at"] == slot["start_at"]
    cal = rec.get("/api/recruiter/calendar").json()
    assert cal["slots"] == [] and len(cal["interviews"]) == 1  # the other slot was released
    assert rec.get("/api/recruiter/applications").json()["applications"][0]["status"] == "interview_scheduled"
    assert cand.get("/api/candidate/applications").json()["applications"][0]["interview"]["scheduled_at"] == slot["start_at"]

    # own events + tasks
    ev = rec.post("/api/recruiter/events", json={"title": "Call Zed", "start_at": _future(1)}).json()["event"]
    assert rec.get("/api/recruiter/calendar").json()["events"][0]["title"] == "Call Zed"
    assert other.delete(f"/api/recruiter/events/{ev['id']}").status_code == 404
    assert rec.delete(f"/api/recruiter/events/{ev['id']}").json()["ok"]
    t = rec.post("/api/recruiter/tasks", json={"title": "Call"}).json()["task"]
    assert other.post(f"/api/recruiter/tasks/{t['id']}/update", json={"status": "completed"}).status_code == 404
    assert rec.post(f"/api/recruiter/tasks/{t['id']}/update", json={"status": "completed"}).json()["task"]["status"] == "completed"

    # role guards
    assert cand.get("/api/recruiter/jobs").status_code == 403
    assert rec.get("/api/candidate/applications").status_code == 403


def test_application_status_emails_and_candidate_rejection_reason(monkeypatch):
    recruiter, _ = _client("recruiter@example.com", "recruiter")
    candidate, person = _client("candidate@example.com", "candidate")
    other, _ = _client("other@example.com", "candidate")
    recruiter.post("/api/recruiter/company", json={"name": "Example Hiring"})
    job = recruiter.post("/api/recruiter/jobs", json={"title": "Data Analyst", "description": "A role"}).json()["job"]
    app_id = auth.apply_to_job(job["id"], person["id"])["id"]
    route = f"/api/recruiter/applications/{app_id}/status"
    sent = []

    def capture(to, subject, body):
        sent.append((to, subject, body))
        return recruiter_api.mailer.MailResult(True)

    monkeypatch.setattr(recruiter_api.mailer, "send", capture)
    assert recruiter.post(route, json={"status": "recruiter_review"}).json()["email_state"] == "skipped"
    assert recruiter.post(route, json={"status": "shortlisted"}).json()["email_state"] == "sent"
    assert recruiter.post(route, json={"status": "shortlisted"}).json()["changed"] is False
    assert recruiter.post(route, json={"status": "offer_released"}).json()["email_state"] == "sent"
    assert recruiter.post(route, json={"status": "rejected", "rejection_reason": "  "}).status_code == 400
    assert candidate.get("/api/candidate/applications").json()["applications"][0]["status"] == "offer_released"

    reason = "We need more SQL experience for this role."
    result = recruiter.post(route, json={"status": "rejected", "rejection_reason": " " + reason + " "})
    assert result.json()["email_state"] == "sent"
    assert len(sent) == 3
    assert all(to == "candidate@example.com" and "Data Analyst" in body for to, _, body in sent)
    assert "Current status: Shortlisted" in sent[0][2]
    assert "offer letter will be released soon" in sent[1][2]
    assert reason in sent[2][2] and "future opportunities" in sent[2][2]
    mine = candidate.get("/api/candidate/applications").json()["applications"][0]
    assert mine["status"] == "rejected" and mine["rejection_reason"] == reason
    assert other.get("/api/candidate/applications").json()["applications"] == []
    recruiter.post(route, json={"status": "on_hold"})
    assert candidate.get("/api/candidate/applications").json()["applications"][0]["rejection_reason"] is None


def test_failed_application_email_can_be_retried_without_changing_status(monkeypatch):
    recruiter, _ = _client("recruiter@example.com", "recruiter")
    other, _ = _client("other@example.com", "recruiter")
    candidate, person = _client("candidate@example.com", "candidate")
    recruiter.post("/api/recruiter/company", json={"name": "Hiring"})
    other.post("/api/recruiter/company", json={"name": "Other company"})
    job = recruiter.post("/api/recruiter/jobs", json={"title": "Developer", "description": "A role"}).json()["job"]
    app_id = auth.apply_to_job(job["id"], person["id"])["id"]
    route = f"/api/recruiter/applications/{app_id}"
    attempts = []

    def flaky_send(*args):
        attempts.append(args)
        return recruiter_api.mailer.MailResult(len(attempts) == 2)

    monkeypatch.setattr(recruiter_api.mailer, "send", flaky_send)
    result = recruiter.post(route + "/status", json={"status": "shortlisted"}).json()
    assert result["ok"] and result["email_state"] == "failed"
    assert candidate.get("/api/candidate/applications").json()["applications"][0]["status"] == "shortlisted"
    assert other.post(route + "/status-email/retry").status_code == 404
    assert recruiter.post(route + "/status-email/retry").json()["email_state"] == "sent"
    assert recruiter.post(route + "/status-email/retry").json()["email_state"] == "sent"
    assert len(attempts) == 2
