"""HTTP tests for POST /api/screen - the machine-to-machine screening endpoint
the n8n resume-screening workflow calls.

The route has no session by design, so these tests are about the shared-secret
gate (present, wrong, unset) and about the screening answer itself: the score
has to come from ats_score, keywords from the JD, and nothing may be invented.
"""

import io

import pytest
from fastapi.testclient import TestClient

import auth
import dashboard as d


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    monkeypatch.setattr(d, "PROFILES_FILE", str(tmp_path / "no-profiles.json"))

    class _Bundle:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tools = []

    async def _fake_create_agent(cfg=None, runtime=None):
        return _Bundle(cfg)

    async def _noop_runtime():
        return object()

    monkeypatch.setattr(d, "create_agent", _fake_create_agent)
    monkeypatch.setattr(d, "create_runtime", _noop_runtime)
    monkeypatch.setattr(d.app.state, "bundles", {}, raising=False)
    d.app.state.bundles = {}
    monkeypatch.setattr(d.app.state, "runtime", object(), raising=False)
    monkeypatch.setenv("SCREEN_TOKEN", "test-screen-token")

    with TestClient(d.app) as c:
        yield c


def _upload(client, resume=b"Jane Doe\njane@example.com\n", jd="",
            token="test-screen-token", filename="cv.txt"):
    headers = {"X-JobScope-Token": token} if token is not None else {}
    return client.post("/api/screen",
                       files={"resume": (filename, resume, "text/plain")},
                       data={"jd_text": jd}, headers=headers)


def _screen_json(client, resume=b"Jane Doe\njane@example.com\n", jd="",
                 token="test-screen-token", **extra):
    """The n8n shape: application/json with the file as base64."""
    import base64 as b64
    payload = {"resume_base64": b64.b64encode(resume).decode(),
               "filename": "cv.txt", "jd_text": jd}
    payload.update(extra)
    headers = {"X-JobScope-Token": token} if token is not None else {}
    return client.post("/api/screen", json=payload, headers=headers)


def test_the_endpoint_refuses_a_call_without_the_header(client):
    r = _upload(client, token=None)
    assert r.status_code == 401
    assert "X-JobScope-Token" in r.json()["detail"]


def test_the_endpoint_refuses_a_wrong_token(client):
    r = _upload(client, token="not-the-token")
    assert r.status_code == 401


def test_the_endpoint_is_dead_when_no_token_is_configured(client, monkeypatch):
    monkeypatch.delenv("SCREEN_TOKEN")
    r = _upload(client)
    assert r.status_code == 503
    assert "SCREEN_TOKEN" in r.json()["detail"]


def test_a_screen_returns_the_full_explainable_score(client):
    resume = ("Jane Doe\njane@example.com\n\n"
              "Skills\nPython, SQL, Docker\n\n"
              "Experience\nAcme Corp - Data Analyst, 2021 to present\n"
              "- Built dashboards\n")
    jd = ("We need a Python and Kubernetes engineer. Kubernetes is required."
          )
    r = _upload(client, resume=resume.encode(), jd=jd)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    ats = body["ats"]
    assert isinstance(ats["total"], (int, float))
    assert ats["band"] in ("strong", "good", "fair", "weak")
    # The JD's own keywords drive the match: Python is in the resume,
    # Kubernetes is not, and the endpoint must say so rather than guess.
    assert "python" in [k.lower() for k in ats["hits"]]
    assert "kubernetes" in [k.lower() for k in ats["missing"]]
    assert body["jd_origin"] == "pasted"
    assert body["resume_chars"] > 0


def test_an_empty_file_is_rejected_before_scoring(client):
    r = _upload(client, resume=b"", jd="anything")
    assert r.status_code == 422


def test_an_oversize_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(d, "MAX_UPLOAD_BYTES", 32)
    r = _upload(client, resume=b"x" * 64, jd="anything")
    assert r.status_code == 413


def test_a_screen_needs_no_browser_session(client):
    """n8n holds no cookie: the header alone must be enough, so the response
    must never be a 401/403 that a session would have prevented."""
    r = _upload(client, resume=b"Jane Doe\njane@example.com\n", jd="")
    assert r.status_code == 200
    # With no JD there are no posting keywords to miss, but the structural
    # score still comes back and says the JD is absent.
    body = r.json()
    assert body["jd_origin"] == "none"
    assert body["ats"]["keyword_total"] == 0


def test_the_json_shape_scores_exactly_like_the_upload(client):
    """The workflow sends base64 JSON; it must reach the same scorer."""
    resume = ("Jane Doe\njane@example.com\n\n"
              "Skills\nPython, SQL, Docker\n\n"
              "Experience\nAcme Corp - Data Analyst, 2021 to present\n")
    jd = "We need a Python and Kubernetes engineer."
    r = _screen_json(client, resume=resume.encode(), jd=jd)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    ats = body["ats"]
    assert "python" in [k.lower() for k in ats["hits"]]
    assert "kubernetes" in [k.lower() for k in ats["missing"]]
    assert body["jd_origin"] == "pasted"


def test_the_json_shape_uses_the_same_shared_secret(client):
    r = _screen_json(client, token="not-the-token")
    assert r.status_code == 401


def test_the_json_shape_rejects_a_missing_or_broken_payload(client):
    import base64 as b64
    r = client.post("/api/screen", json={"jd_text": "x"},
                    headers={"X-JobScope-Token": "test-screen-token"})
    assert r.status_code == 400
    assert "resume_base64" in r.json()["detail"]
    r = client.post("/api/screen",
                    json={"resume_base64": "not base64!",
                          "jd_text": "x"},
                    headers={"X-JobScope-Token": "test-screen-token"})
    assert r.status_code == 400
    # ...but a well-formed body with no text in it still fails downstream.
    r = client.post("/api/screen",
                    json={"resume_base64": b64.b64encode(b"").decode(),
                          "jd_text": "x"},
                    headers={"X-JobScope-Token": "test-screen-token"})
    assert r.status_code == 422
