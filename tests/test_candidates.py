"""Phase 1 candidate records: resume parsing/download, notes, tags, tags search,
sourced pipeline entries, scoping and the cascading delete."""
import io
import os
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ats_score
import auth
import candidates_api


def _pdf(text: str) -> bytes:
    """A tiny valid PDF with one text line, so pypdf can exercise extraction."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    ]
    stream = ("BT /F1 12 Tf 72 720 Td (" + text + ") Tj ET").encode()
    objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _docx(text: str) -> bytes:
    """Minimal DOCX (a zip with word/document.xml), enough for docx2txt."""
    body = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>' + text + '</w:t></w:r></w:p></w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", body)
    return buf.getvalue()


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
    app.include_router(candidates_api.router)
    client = TestClient(app)
    if user:
        client.cookies.set(auth.SESSION_COOKIE, auth.create_session(user["id"])[0])
    return client


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates_api, "RESUME_DIR", str(tmp_path / "resumes"))
    rec = _user("rec@x.com", company="Acme")
    other = _user("other@x.com", company="Other Co")
    job = auth.create_job(rec["company_id"], rec["id"],
                          {"title": "Backend engineer", "description": "Python APIs", "status": "published"})
    other_job = auth.create_job(other["company_id"], other["id"],
                                {"title": "Elsewhere", "description": "Nothing", "status": "published"})
    cand = auth.upsert_candidate(rec["id"], name="Ada Lovelace", email="ada@x.com",
                                 current_location="London", status="working")
    other_cand = auth.upsert_candidate(other["id"], name="Eve Other", email="eve@other.com",
                                       status="working")
    return {"rec": rec, "other": other, "job": job, "other_job": other_job,
            "cand": cand, "other_cand": other_cand,
            "rec_client": _client(rec), "other_client": _client(other),
            "anon": _client(None), "candidate_client": _client(_user("person@x.com", role="candidate"))}


def _upload(client, candidate_id, pdf=None):
    return client.post(f"/api/candidates/{candidate_id}/resume",
                       files={"resume": ("cv.pdf", pdf or _pdf("Skills: Python SQL Docker React AWS Java"),
                                         "application/pdf")})


def test_candidate_record_round_trip(data):
    rec = data["rec_client"]
    cid = data["cand"]

    listed = rec.get("/api/candidates")
    assert listed.status_code == 200
    assert [c["id"] for c in listed.json()["candidates"]] == [cid]
    assert rec.get("/api/candidates?q=Ada").json()["candidates"][0]["id"] == cid
    assert rec.get("/api/candidates?q=nobody").json()["candidates"] == []
    assert rec.get("/api/candidates?status=working").json()["candidates"][0]["id"] == cid
    assert rec.get("/api/candidates?status=not-a-status").status_code == 400
    assert rec.get("/api/candidates?tag=missing").json()["candidates"] == []

    detail = rec.get(f"/api/candidates/{cid}").json()
    assert detail["candidate"]["name"] == "Ada Lovelace"
    assert detail["notes"] == [] and detail["tags"] == [] and detail["resume"] is None
    assert detail["pipeline"] == []

    # Resume: the PDF's text is parsed and known skills are extracted.
    up = _upload(rec, cid)
    assert up.status_code == 200, up.text
    resume = up.json()["resume"]
    assert resume["filename"] == "cv.pdf" and resume["size"] > 0
    for skill in ("python", "sql", "docker", "react", "aws", "java"):
        assert skill in resume["skills"]
    assert rec.get("/api/candidates?skill=python").json()["candidates"][0]["id"] == cid
    assert rec.get("/api/candidates?skill=cobol").json()["candidates"] == []

    download = rec.get(f"/api/candidates/{cid}/resume")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"].lower()
    assert download.content.startswith(b"%PDF-")

    # Notes are newest first, with the author and a UTC timestamp.
    assert rec.post(f"/api/candidates/{cid}/notes", json={"note": "first"}).status_code == 200
    second = rec.post(f"/api/candidates/{cid}/notes", json={"note": "second"}).json()["note"]
    assert second["author"] == "rec" and second["created_at"].endswith("Z")
    notes = rec.get(f"/api/candidates/{cid}").json()["notes"]
    assert [n["note"] for n in notes] == ["second", "first"]

    # Tags can be added, filtered on and removed.
    assert rec.post(f"/api/candidates/{cid}/tags", json={"tag": "vip"}).json()["tags"] == ["vip"]
    rec.post(f"/api/candidates/{cid}/tags", json={"tag": "senior"})
    assert rec.get("/api/candidates?tag=vip").json()["candidates"][0]["id"] == cid
    assert rec.post(f"/api/candidates/{cid}/tags",
                    json={"tag": "vip", "remove": True}).json()["tags"] == ["senior"]

    # Sourced candidate on the pipeline board, then moved across stages.
    added = rec.post("/api/recruiter/pipeline",
                     json={"job_id": data["job"]["id"], "candidate_id": cid})
    assert added.status_code == 200, added.text
    entry = added.json()["entry"]
    assert entry["status"] == "applied" and entry["candidate_name"] == "Ada Lovelace"
    assert entry["job_title"] == "Backend engineer"
    board = rec.get("/api/recruiter/pipeline").json()["entries"]
    assert [e["id"] for e in board] == [entry["id"]]
    moved = rec.post(f"/api/recruiter/pipeline/{entry['id']}/status", json={"status": "shortlisted"})
    assert moved.status_code == 200 and moved.json()["entry"]["status"] == "shortlisted"
    assert rec.get(f"/api/candidates/{cid}").json()["pipeline"][0]["status"] == "shortlisted"
    assert rec.post(f"/api/recruiter/pipeline/{entry['id']}/status",
                    json={"status": "sideways"}).status_code == 400
    # Adding the same candidate to the same job updates rather than duplicates.
    rec.post("/api/recruiter/pipeline", json={"job_id": data["job"]["id"], "candidate_id": cid,
                                              "status": "recruiter_review"})
    assert len(rec.get("/api/recruiter/pipeline").json()["entries"]) == 1
    assert rec.get("/api/recruiter/pipeline").json()["entries"][0]["status"] == "recruiter_review"


def test_docx_resume_replaces_the_pdf(data):
    rec = data["rec_client"]
    cid = data["cand"]
    assert _upload(rec, cid).status_code == 200
    res = rec.post(f"/api/candidates/{cid}/resume",
                   files={"resume": ("cv.docx", _docx("Skills: Python SQL"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert res.status_code == 200, res.text
    assert res.json()["resume"]["filename"] == "cv.docx"
    assert "python" in res.json()["resume"]["skills"]
    assert len(os.listdir(candidates_api.RESUME_DIR)) == 1, "the replaced PDF should be deleted"


def test_scanned_pdf_falls_back_to_ocr(data, monkeypatch):
    """A PDF with no usable text layer is recovered through cv_ocr before parsing."""
    monkeypatch.setattr(candidates_api.cv_ocr, "needs_ocr", lambda text: True)
    monkeypatch.setattr(
        candidates_api.cv_ocr, "ocr_document",
        lambda data, filename: candidates_api.cv_ocr.OcrResult(ok=True, text="Skills: Python SQL Kubernetes"))
    res = _upload(data["rec_client"], data["cand"], _pdf(""))
    assert res.status_code == 200, res.text
    assert {"python", "sql", "kubernetes"} <= set(res.json()["resume"]["skills"])


def test_screening_reuses_the_ats_scorer_against_the_job(data):
    rec = data["rec_client"]
    cid = data["cand"]
    assert _upload(rec, cid).status_code == 200
    with candidates_api._db() as conn:
        text = conn.execute("SELECT parsed_text FROM candidate_resumes "
                            "WHERE user_id = ? AND candidate_id = ?",
                            (data["rec"]["id"], cid)).fetchone()["parsed_text"]
    # A candidate with no resume must not show up, even though they are in the list.
    auth.upsert_candidate(data["rec"]["id"], name="No Resume", email="nr@x.com",
                          status="working")
    expected = ats_score.score_resume(text, "Backend engineer\nPython APIs")

    res = rec.get(f"/api/recruiter/screening?job_id={data['job']['id']}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["job_title"] == "Backend engineer"
    assert [r["candidate_id"] for r in body["results"]] == [cid]
    row = body["results"][0]
    assert row["name"] == "Ada Lovelace" and row["email"] == "ada@x.com"
    assert row["resume"] == "cv.pdf"
    assert row["score"] == expected["total"] and row["band"] == expected["band"]
    assert row["hits"] == expected["hits"] and row["missing"] == expected["missing"]
    assert row["keyword_total"] == expected["keyword_total"]


def test_screening_bad_input(data):
    rec = data["rec_client"]
    assert rec.get("/api/recruiter/screening").status_code == 400
    assert rec.get("/api/recruiter/screening?job_id=abc").status_code == 422
    assert rec.get(f"/api/recruiter/screening?job_id={data['other_job']['id']}").status_code == 404
    assert rec.get("/api/recruiter/screening?job_id=999999").status_code == 404


def test_bad_resume_input(data):
    rec = data["rec_client"]
    cid = data["cand"]
    assert rec.post(f"/api/candidates/{cid}/resume",
                    files={"resume": ("cv.txt", b"plain text", "text/plain")}).status_code == 400
    assert rec.post(f"/api/candidates/{cid}/resume",
                    files={"resume": ("cv.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")}).status_code == 400
    big = b"%PDF-" + b"0" * (candidates_api.MAX_RESUME + 1)
    assert rec.post(f"/api/candidates/{cid}/resume",
                    files={"resume": ("cv.pdf", big, "application/pdf")}).status_code == 400
    assert rec.post(f"/api/candidates/{cid}/notes", json={"note": "   "}).status_code == 400
    assert rec.post(f"/api/candidates/{cid}/tags", json={"tag": ""}).status_code == 400
    assert rec.post("/api/recruiter/pipeline", json={"job_id": 0, "candidate_id": cid}).status_code == 422
    assert rec.post("/api/recruiter/pipeline",
                    json={"job_id": data["other_job"]["id"], "candidate_id": cid}).status_code == 404
    assert rec.post("/api/recruiter/pipeline",
                    json={"job_id": data["job"]["id"], "candidate_id": data["other_cand"]}).status_code == 404
    assert rec.get(f"/api/candidates/{cid}/resume").status_code == 404


def test_signed_out_and_wrong_role(data):
    paths = [
        ("get", "/api/candidates", {}),
        ("get", f"/api/candidates/{data['cand']}", {}),
        ("post", f"/api/candidates/{data['cand']}/notes", {"json": {"note": "x"}}),
        ("post", f"/api/candidates/{data['cand']}/tags", {"json": {"tag": "x"}}),
        ("post", f"/api/candidates/{data['cand']}/resume",
         {"files": {"resume": ("cv.pdf", _pdf("Python"), "application/pdf")}}),
        ("get", f"/api/candidates/{data['cand']}/resume", {}),
        ("delete", f"/api/candidates/{data['cand']}", {}),
        ("get", "/api/recruiter/pipeline", {}),
        ("get", f"/api/recruiter/screening?job_id={data['job']['id']}", {}),
        ("post", "/api/recruiter/pipeline", {"json": {"job_id": data["job"]["id"],
                                                      "candidate_id": data["cand"]}}),
        ("post", "/api/recruiter/pipeline/1/status", {"json": {"status": "hired"}}),
    ]
    for method, path, kwargs in paths:
        assert getattr(data["anon"], method)(path, **kwargs).status_code == 401, path
        assert getattr(data["candidate_client"], method)(path, **kwargs).status_code == 403, path


def test_another_company_gets_404_everywhere(data):
    rec, other = data["rec_client"], data["other_client"]
    cid = data["cand"]
    assert other.get(f"/api/candidates/{cid}").status_code == 404
    assert other.get(f"/api/candidates/{data['other_cand']}").status_code == 200
    assert other.post(f"/api/candidates/{cid}/notes", json={"note": "nope"}).status_code == 404
    assert other.post(f"/api/candidates/{cid}/tags", json={"tag": "nope"}).status_code == 404
    assert _upload(other, cid).status_code == 404
    assert other.get(f"/api/candidates/{cid}/resume").status_code == 404
    assert other.delete(f"/api/candidates/{cid}").status_code == 404

    entry = rec.post("/api/recruiter/pipeline",
                     json={"job_id": data["job"]["id"], "candidate_id": cid}).json()["entry"]
    assert other.post(f"/api/recruiter/pipeline/{entry['id']}/status",
                      json={"status": "hired"}).status_code == 404
    assert other.get("/api/recruiter/pipeline").json()["entries"] == []
    assert other.get(f"/api/recruiter/screening?job_id={data['job']['id']}").status_code == 404
    assert rec.get(f"/api/candidates/{data['other_cand']}").status_code == 404


def test_delete_removes_resume_notes_tags_and_pipeline(data):
    rec = data["rec_client"]
    cid = data["cand"]
    _upload(rec, cid)
    rec.post(f"/api/candidates/{cid}/notes", json={"note": "keep me"})
    rec.post(f"/api/candidates/{cid}/tags", json={"tag": "vip"})
    rec.post("/api/recruiter/pipeline", json={"job_id": data["job"]["id"], "candidate_id": cid})
    files = os.listdir(candidates_api.RESUME_DIR)
    assert len(files) == 1

    assert rec.delete(f"/api/candidates/{cid}").status_code == 200
    assert rec.get(f"/api/candidates/{cid}").status_code == 404
    assert rec.get("/api/candidates").json()["candidates"] == []
    assert rec.get("/api/recruiter/pipeline").json()["entries"] == []
    assert os.listdir(candidates_api.RESUME_DIR) == []
    with auth.connect() as conn:
        for table in ("candidate_resumes", "candidate_tags", "candidate_notes", "pipeline_entries"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
