"""HTTP tests for the document library and RAG-driven resume generation.

The library endpoints are the only way content enters the retrieval corpus, so
these cover the account scoping as much as the happy path: a document id from
one account must be invisible, undeletable, and un-retrievable from another.
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

    with TestClient(d.app) as c:
        c.headers["host"] = "jobs.test"
        yield c


@pytest.fixture(autouse=True)
def no_smtp(monkeypatch):
    """Default to 'no mail server' so registration self-confirms via the dev
    link. Without this the real .env's SMTP_HOST is read and the account waits
    on a mail server the test does not run."""
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                 "SMTP_FROM", "AUTH_DEV_LINKS"):
        monkeypatch.delenv(name, raising=False)


def register(client, email):
    """Create a confirmed, signed-in account. Registration sets the cookie."""
    r = client.post("/api/auth/register",
                    json={"email": email, "password": "a-good-password"})
    assert r.status_code in (200, 202), r.text
    body = r.json()
    if body.get("needs_verification"):
        token = body["dev_link"].split("verify=")[-1]
        assert client.get("/api/auth/verify?token=" + token).status_code == 200
    return body


def sign_in(client, email):
    r = client.post("/api/auth/login",
                    json={"email": email, "password": "a-good-password"})
    assert r.status_code == 200, r.text
    return r


DOC = """EXPERIENCE
Acme - Data Analyst
- Built a Spark ETL pipeline that cut batch time from 6h to 40m
"""

OTHER_CV = """EXPERIENCE
Globex - Support Engineer
- Answered 60 tickets a week for a retail banking portal
"""


def set_cv(email, text=DOC):
    """Give an account an open profile that has a CV of its own."""
    uid = auth.get_user_by_email(email)["id"]
    prof = auth.get_profile(uid, "default") or {}
    prof["resume_text"] = text
    auth.save_profile(uid, "default", prof)


def upload(client, name="My CV.txt", text=DOC):
    return client.post("/api/rag/docs",
                       files={"file": (name, io.BytesIO(text.encode()),
                                       "text/plain")})


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------

def test_the_library_starts_empty(client):
    register(client, "a@jobs.test")
    body = client.get("/api/rag/docs").json()
    assert body["ok"] is True
    assert body["docs"] == []


def test_an_uploaded_document_is_listed(client):
    register(client, "a@jobs.test")
    body = upload(client).json()
    assert body["ok"] is True
    assert body["doc"]["chars"] == len(DOC.strip())
    assert [x["name"] for x in body["docs"]] == ["My CV.txt"]


def test_reuploading_the_same_filename_updates_in_place(client):
    register(client, "a@jobs.test")
    upload(client, "My CV.txt", DOC)
    upload(client, "My CV.txt", "PROJECTS\n- A different writeup")
    docs = client.get("/api/rag/docs").json()["docs"]
    assert len(docs) == 1


def test_two_different_files_are_both_kept(client):
    register(client, "a@jobs.test")
    upload(client, "cv.txt", DOC)
    upload(client, "projects.txt", "PROJECTS\n- Kubernetes autoscaler")
    assert len(client.get("/api/rag/docs").json()["docs"]) == 2


def test_an_unreadable_upload_is_rejected(client):
    register(client, "a@jobs.test")
    body = client.post("/api/rag/docs",
                       files={"file": ("empty.txt", io.BytesIO(b"   "),
                                       "text/plain")}).json()
    assert body["ok"] is False
    assert "text" in body["error"].lower()


def test_a_document_can_be_deleted(client):
    register(client, "a@jobs.test")
    upload(client, "cv.txt", DOC)
    assert client.delete("/api/rag/docs/cv-txt").json()["ok"] is True
    assert client.get("/api/rag/docs").json()["docs"] == []


def test_deleting_a_missing_document_is_a_clean_miss(client):
    register(client, "a@jobs.test")
    body = client.delete("/api/rag/docs/nope").json()
    assert body["ok"] is False


def test_the_library_is_scoped_to_the_account(client):
    register(client, "a@jobs.test")
    upload(client, "secret.txt", "PROJECTS\n- Classified ledger rewrite")
    client.cookies.clear()
    register(client, "b@jobs.test")
    assert client.get("/api/rag/docs").json()["docs"] == []
    # ...and the other account cannot delete it by guessing the id.
    assert client.delete("/api/rag/docs/secret-txt").json()["ok"] is False
    client.cookies.clear()
    sign_in(client, "a@jobs.test")
    assert len(client.get("/api/rag/docs").json()["docs"]) == 1


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def _gen(client, **extra):
    payload = {"title": "Data Engineer", "company": "Acme",
               "location": "Remote", "link": "", "source": "Indeed"}
    payload.update(extra)
    return client.post("/api/resume/gen", json=payload).json()


def test_the_description_collected_at_search_time_is_used(client):
    """The listing already carried the posting body, so generation must use it
    rather than re-fetching a page that will usually lose to a bot wall."""
    register(client, "a@jobs.test")
    body = _gen(client, job_desc="Kafka streaming for the ingestion tier")
    assert body["ok"] is True
    assert body["jd_origin"] == "listing"
    assert body["desc_fetched"] is True


def test_a_pasted_jd_still_beats_the_collected_description(client):
    register(client, "a@jobs.test")
    body = _gen(client, jd_text="my own words",
                job_desc="the board's words")
    assert body["jd_origin"] == "pasted"


def test_the_collected_description_drives_retrieval(client):
    register(client, "a@jobs.test")
    upload(client, "cv.txt", DOC)
    body = _gen(client, job_desc="Spark ETL pipeline")
    assert body["rag_hits"] > 0
    assert any("cv.txt" in ref for ref in body["rag_sources"])


def test_a_pasted_jd_is_used_in_preference_to_the_link(client):
    register(client, "a@jobs.test")
    body = _gen(client, jd_text="Kubernetes autoscaling for the API tier")
    assert body["ok"] is True
    assert body["jd_origin"] == "pasted"
    assert body["desc_fetched"] is True


def test_with_no_pasted_jd_the_origin_reports_that_it_had_none(client):
    # No link to scrape either, so tailoring had nothing to go on. Reported
    # honestly rather than implied to have worked.
    register(client, "a@jobs.test")
    body = _gen(client)
    assert body["jd_origin"] == "none"
    assert body["desc_fetched"] is False


def test_generation_reports_the_documents_it_matched(client):
    register(client, "a@jobs.test")
    upload(client, "cv.txt", DOC)
    body = _gen(client, jd_text="Spark ETL pipeline")
    assert body["rag_hits"] > 0
    assert any("cv.txt" in ref for ref in body["rag_sources"])


def test_generation_still_works_with_an_empty_library(client):
    register(client, "a@jobs.test")
    body = _gen(client, jd_text="Spark ETL pipeline")
    assert body["ok"] is True
    assert body["rag_hits"] == 0
    assert body["rag_sources"] == []


def test_the_tailored_resume_reports_its_provenance(client):
    register(client, "a@jobs.test")
    set_cv("a@jobs.test", DOC)
    upload(client, "cv.txt", DOC)
    body = _gen(client, jd_text="Spark ETL pipeline")
    assert "Tailored from" in body["tex"]


def test_a_library_document_becomes_the_source_when_there_is_no_cv(client):
    # The account uploaded a CV to the library but never set one on the open
    # profile. Before this, the resume described resume_data's placeholder
    # person and the uploaded document was ignored entirely.
    register(client, "a@jobs.test")
    upload(client, "cv.txt", DOC)
    body = _gen(client, jd_text="Spark ETL pipeline")
    assert body["ok"] is True
    assert "Built from your uploaded document" in body["tex"]
    assert "Spark" in body["tex"]
    # The placeholder person's details must not appear.
    assert "Telugu" not in body["tex"]


def test_an_existing_cv_is_still_preferred_over_the_library(client):
    register(client, "a@jobs.test")
    set_cv("a@jobs.test", OTHER_CV)
    upload(client, "other.txt", DOC)
    body = _gen(client, jd_text="Spark ETL pipeline")
    # Tailored, not rebuilt: the profile's own CV stays the content source.
    assert "Tailored from" in body["tex"]
    assert "Spark" not in body["tex"]


def test_no_cv_and_no_library_still_falls_back_to_canonical_data(client):
    register(client, "a@jobs.test")
    body = _gen(client, jd_text="Spark ETL pipeline")
    assert body["ok"] is True
    assert body["rag_hits"] == 0
    # Nothing to source from, so the canonical template stands - and it is not
    # dressed up with a provenance line it does not deserve.
    assert "Tailored from" not in body["tex"]
    assert "Built from" not in body["tex"]


def test_generation_is_scoped_to_the_signed_in_account(client):
    register(client, "a@jobs.test")
    upload(client, "secret.txt", "PROJECTS\n- Classified ledger rewrite")
    client.cookies.clear()
    register(client, "b@jobs.test")
    # b's retrieval must not see a's documents, however well they match.
    body = _gen(client, jd_text="classified ledger rewrite")
    assert body["ok"] is True
    assert body["rag_hits"] == 0
