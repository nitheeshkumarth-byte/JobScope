"""Recruiter CRM phase 1: candidate records (resume, notes, tags) and sourced
pipeline entries.

Self-contained on purpose: it depends only on `auth` plus the existing CV
helpers, not on dashboard.py, so it cannot drift out of step with that file.
dashboard.py mounts it straight after `app = FastAPI(...)`, before
recruiter_api, so the filtered `/api/candidates` list and the cascading
candidate delete win over the older same-path routes (first registered wins):

    import candidates_api
    app.include_router(candidates_api.router)

Times are stored as UTC ISO strings ending in "Z"; the browser renders them in
the viewer's own time zone.
"""
import io
import json
import os
import re
import secrets
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import ats_score
import auth
import cv_ocr
import resume_data

router = APIRouter()

CAND_STATUSES = ("working", "recruiter_entry", "available", "ready_for_interview",
                 "interviewing", "offered", "placed", "not_interested")
APP_STATUSES = ("applied", "ai_screening", "recruiter_review", "shortlisted", "interview_scheduled",
                "interview_completed", "technical_round", "hr_round", "offer_released", "offer_accepted",
                "hired", "rejected", "withdrawn", "on_hold", "offer_declined")
MAX_RESUME = 5 * 1024 * 1024
RESUME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "resumes")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    size INTEGER NOT NULL,
    parsed_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_resumes_owner ON candidate_resumes(user_id, candidate_id);

CREATE TABLE IF NOT EXISTS candidate_tags (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (user_id, candidate_id, tag)
);

CREATE TABLE IF NOT EXISTS candidate_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL,
    author TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cnotes_owner ON candidate_notes(user_id, candidate_id);

CREATE TABLE IF NOT EXISTS pipeline_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'applied',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pipeline_company ON pipeline_entries(company_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_job ON pipeline_entries(job_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_entries(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_unique
    ON pipeline_entries(company_id, job_id, candidate_id);
"""
_ready: set = set()


def _now() -> str:
    """UTC, with the Z the rest of the spec asks for (auth._now is local and bare)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _db():
    """Connection with this module's tables guaranteed (the DB path changes under tests)."""
    conn = auth.connect()
    if auth.DB_FILE not in _ready:
        conn.executescript(_SCHEMA)
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


def _own_candidate(conn, user: dict, candidate_id: str):
    row = conn.execute("SELECT * FROM candidates WHERE user_id = ? AND id = ?",
                       (user["id"], candidate_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return row


# ---- resume helpers -------------------------------------------------------
def resume_skills(text: str) -> list[str]:
    """Known skills present in the parsed CV text, deduplicated and ordered by the lexicon."""
    low = (text or "").lower()
    if not low.strip():
        return []
    found: list[str] = []
    seen: set = set()
    terms = [k.lower() for k in resume_data.flat_skill_keywords()]
    terms += [k.lower() for k in ats_score.extract_keywords(text, limit=40)]
    for term in terms:
        term = term.strip()
        if not term or term in seen:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            seen.add(term)
            found.append(term)
    return found[:60]


def _resume_kind(data: bytes) -> str:
    """PDF/DOCX by content, never by the name the browser sent."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
    raise HTTPException(status_code=400, detail="Upload a PDF or DOCX file")


def _resume_text(data: bytes, kind: str) -> str:
    try:
        if kind == "pdf":
            from pypdf import PdfReader
            text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages)
            if cv_ocr.needs_ocr(text):
                # Scanned/image-only PDF: recover the text with the same OCR tier /api/screen uses.
                ocr = cv_ocr.ocr_document(data, "resume.pdf")
                if ocr.ok and ocr.text.strip():
                    return ocr.text
            return text
        import docx2txt  # same soft dependency dashboard._extract_text uses
        return docx2txt.process(io.BytesIO(data))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that file")


def _resume_info(row) -> dict:
    return {"id": row["id"], "filename": row["filename"], "size": row["size"],
            "created_at": row["created_at"], "skills": resume_skills(row["parsed_text"])}


def _remove_resume(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ---- candidate list & detail ---------------------------------------------
@router.get("/api/candidates")
async def list_candidates(q: str | None = None, skill: str | None = None,
                          status: str | None = None, tag: str | None = None,
                          limit: int = 100, offset: int = 0,
                          user: dict = Depends(_recruiter)):
    if status and status not in CAND_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    sql = "SELECT DISTINCT c.* FROM candidates c"
    args: list = []
    if skill:
        sql += (" JOIN candidate_resumes r ON r.user_id = c.user_id AND r.candidate_id = c.id"
                " AND LOWER(r.parsed_text) LIKE ?")
        args.append("%" + skill.strip().lower() + "%")
    if tag:
        sql += " JOIN candidate_tags t ON t.user_id = c.user_id AND t.candidate_id = c.id AND t.tag = ?"
        args.append(tag.strip())
    sql += " WHERE c.user_id = ?"
    args.append(user["id"])
    if q:
        like = "%" + q.strip() + "%"
        sql += (" AND (c.name LIKE ? OR c.email LIKE ? OR c.phone LIKE ? OR c.current_location LIKE ?)")
        args.extend([like] * 4)
    if status:
        sql += " AND c.status = ?"
        args.append(status)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    sql += " ORDER BY c.created_at DESC, c.id LIMIT ? OFFSET ?"
    args.extend([limit, offset])
    with _db() as conn:
        rows = [dict(r) for r in conn.execute(sql, args)]
    return {"ok": True, "candidates": rows}


@router.get("/api/candidates/{candidate_id}")
async def get_candidate(candidate_id: str, user: dict = Depends(_recruiter)):
    cid = _cid(user)
    with _db() as conn:
        cand = dict(_own_candidate(conn, user, candidate_id))
        notes = [dict(r) for r in conn.execute(
            "SELECT id, author, note, created_at FROM candidate_notes "
            "WHERE user_id = ? AND candidate_id = ? ORDER BY id DESC", (user["id"], candidate_id))]
        tags = [r["tag"] for r in conn.execute(
            "SELECT tag FROM candidate_tags WHERE user_id = ? AND candidate_id = ? ORDER BY tag",
            (user["id"], candidate_id))]
        resume_row = conn.execute(
            "SELECT * FROM candidate_resumes WHERE user_id = ? AND candidate_id = ? "
            "ORDER BY id DESC LIMIT 1", (user["id"], candidate_id)).fetchone()
        pipeline = [dict(r) for r in conn.execute(
            "SELECT p.id, p.job_id, p.status, p.created_at, p.updated_at, j.title AS job_title "
            "FROM pipeline_entries p JOIN jobs j ON j.id = p.job_id AND j.company_id = p.company_id "
            "WHERE p.company_id = ? AND p.candidate_id = ? ORDER BY p.updated_at DESC",
            (cid, candidate_id))] if cid else []
    return {"ok": True, "candidate": cand, "notes": notes, "tags": tags,
            "resume": _resume_info(resume_row) if resume_row else None, "pipeline": pipeline}


# ---- notes & tags ---------------------------------------------------------
class NoteIn(BaseModel):
    note: str = Field(default="", max_length=5000)


@router.post("/api/candidates/{candidate_id}/notes")
async def add_note(candidate_id: str, req: NoteIn, user: dict = Depends(_recruiter)):
    note = req.note.strip()
    if not note or len(note) > 5000:
        raise HTTPException(status_code=400, detail="Note is required (max 5000 characters)")
    now = _now()
    with _db() as conn:
        _own_candidate(conn, user, candidate_id)
        cur = conn.execute(
            "INSERT INTO candidate_notes (user_id, candidate_id, author, note, created_at) VALUES (?,?,?,?,?)",
            (user["id"], candidate_id, auth.display_name(user), note, now))
        row = conn.execute("SELECT id, author, note, created_at FROM candidate_notes WHERE id = ?",
                           (cur.lastrowid,)).fetchone()
    return {"ok": True, "note": dict(row)}


class TagIn(BaseModel):
    tag: str = Field(default="", max_length=60)
    remove: bool = False


@router.post("/api/candidates/{candidate_id}/tags")
async def change_tag(candidate_id: str, req: TagIn, user: dict = Depends(_recruiter)):
    tag = req.tag.strip()
    if not tag or len(tag) > 60:
        raise HTTPException(status_code=400, detail="Tag is required (max 60 characters)")
    with _db() as conn:
        _own_candidate(conn, user, candidate_id)
        if req.remove:
            conn.execute("DELETE FROM candidate_tags WHERE user_id = ? AND candidate_id = ? AND tag = ?",
                         (user["id"], candidate_id, tag))
        else:
            conn.execute("INSERT OR IGNORE INTO candidate_tags (user_id, candidate_id, tag) VALUES (?,?,?)",
                         (user["id"], candidate_id, tag))
        tags = [r["tag"] for r in conn.execute(
            "SELECT tag FROM candidate_tags WHERE user_id = ? AND candidate_id = ? ORDER BY tag",
            (user["id"], candidate_id))]
    return {"ok": True, "tags": tags}


# ---- resume upload & authenticated download ------------------------------
def save_resume(user: dict, candidate_id: str, data: bytes, filename: str = "") -> dict:
    """Store a resume for one of the caller's candidates; replaces the previous file.

    Shared by the browser upload and the Google Forms sync (which downloads the
    file a respondent uploaded to Drive), so both reach Resume Screening the same way.
    """
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty")
    if len(data) > MAX_RESUME:
        raise HTTPException(status_code=400, detail="Resume must be 5 MB or smaller")
    kind = _resume_kind(data)
    text = _resume_text(data, kind)
    original = os.path.basename(filename or ("resume." + kind))
    stored = secrets.token_hex(16) + "." + kind
    os.makedirs(RESUME_DIR, exist_ok=True)
    path = os.path.join(RESUME_DIR, stored)
    with open(path, "wb") as fh:
        fh.write(data)
    try:
        with _db() as conn:
            _own_candidate(conn, user, candidate_id)
            old = [r["stored_name"] for r in conn.execute(
                "SELECT stored_name FROM candidate_resumes WHERE user_id = ? AND candidate_id = ?",
                (user["id"], candidate_id))]
            conn.execute("DELETE FROM candidate_resumes WHERE user_id = ? AND candidate_id = ?",
                         (user["id"], candidate_id))
            cur = conn.execute(
                "INSERT INTO candidate_resumes (candidate_id, user_id, filename, stored_name, size, parsed_text, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (candidate_id, user["id"], original, stored, len(data), text, _now()))
            row = conn.execute("SELECT * FROM candidate_resumes WHERE id = ?", (cur.lastrowid,)).fetchone()
    except Exception:
        _remove_resume(path)
        raise
    for name in old:
        _remove_resume(os.path.join(RESUME_DIR, os.path.basename(name)))
    return _resume_info(row)


@router.post("/api/candidates/{candidate_id}/resume")
async def upload_resume(candidate_id: str, resume: UploadFile = File(...),
                        user: dict = Depends(_recruiter)):
    data = await resume.read(MAX_RESUME + 1)
    return {"ok": True, "resume": save_resume(user, candidate_id, data, resume.filename or "")}


@router.get("/api/candidates/{candidate_id}/resume")
async def download_resume(candidate_id: str, user: dict = Depends(_recruiter)):
    with _db() as conn:
        _own_candidate(conn, user, candidate_id)
        row = conn.execute(
            "SELECT filename, stored_name FROM candidate_resumes WHERE user_id = ? AND candidate_id = ? "
            "ORDER BY id DESC LIMIT 1", (user["id"], candidate_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Resume not found")
    path = os.path.join(RESUME_DIR, os.path.basename(row["stored_name"]))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Resume not found")
    return FileResponse(path, filename=row["filename"])


# ---- sourced candidates on the pipeline board ----------------------------
def _company_candidate(conn, company_id: int, candidate_id: str):
    return conn.execute(
        "SELECT c.user_id, c.id, c.name, c.email FROM candidates c "
        "JOIN recruiter_profiles rp ON rp.user_id = c.user_id "
        "WHERE c.id = ? AND rp.company_id = ? LIMIT 1", (candidate_id, company_id)).fetchone()


def _pipeline_entry(conn, entry_id: int, company_id: int) -> dict:
    row = conn.execute(
        "SELECT p.*, c.name AS candidate_name, c.email AS candidate_email, "
        "c.current_location, j.title AS job_title "
        "FROM pipeline_entries p JOIN candidates c ON c.id = p.candidate_id "
        "JOIN recruiter_profiles rp ON rp.user_id = c.user_id AND rp.company_id = p.company_id "
        "JOIN jobs j ON j.id = p.job_id AND j.company_id = p.company_id "
        "WHERE p.id = ? AND p.company_id = ?", (entry_id, company_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Pipeline entry not found")
    return dict(row)


@router.get("/api/recruiter/pipeline")
async def list_pipeline(user: dict = Depends(_recruiter)):
    cid = _cid(user)
    if not cid:
        return {"ok": True, "entries": []}
    with _db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT p.id, p.job_id, p.candidate_id, p.status, p.created_at, p.updated_at, "
            "c.name AS candidate_name, c.email AS candidate_email, c.current_location, j.title AS job_title "
            "FROM pipeline_entries p JOIN candidates c ON c.id = p.candidate_id "
            "JOIN recruiter_profiles rp ON rp.user_id = c.user_id AND rp.company_id = p.company_id "
            "JOIN jobs j ON j.id = p.job_id AND j.company_id = p.company_id "
            "WHERE p.company_id = ? ORDER BY p.updated_at DESC", (cid,))]
    return {"ok": True, "entries": rows}


def _job_text(job) -> str:
    """The posting text the existing ATS helper scores against."""
    parts = [job["title"], job["description"]]
    try:
        skills = json.loads(job["required_skills"] or "[]")
        if isinstance(skills, list):
            parts.append(", ".join(str(s) for s in skills))
    except (TypeError, ValueError):
        pass
    return "\n".join(p for p in parts if p)


@router.get("/api/recruiter/screening")
async def screen_resumes(job_id: int | None = None, limit: int = 100, offset: int = 0,
                         user: dict = Depends(_recruiter)):
    if not job_id:
        raise HTTPException(status_code=400, detail="Choose a job to screen against")
    cid = _need_company(user)
    limit, offset = max(1, min(limit, 200)), max(0, offset)
    with _db() as conn:
        job = conn.execute("SELECT title, description, required_skills FROM jobs "
                           "WHERE id = ? AND company_id = ?", (job_id, cid)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        rows = conn.execute(
            "SELECT c.id AS candidate_id, c.name, c.email, r.filename, r.created_at, r.parsed_text "
            "FROM candidate_resumes r "
            "JOIN candidates c ON c.user_id = r.user_id AND c.id = r.candidate_id "
            "JOIN recruiter_profiles rp ON rp.user_id = r.user_id "
            "WHERE rp.company_id = ? ORDER BY r.id DESC LIMIT ? OFFSET ?",
            (cid, limit, offset)).fetchall()
    jd = _job_text(job)
    results = []
    for row in rows:
        scored = ats_score.score_resume(row["parsed_text"], jd)
        results.append({
            "candidate_id": row["candidate_id"], "name": row["name"], "email": row["email"],
            "resume": row["filename"], "resume_at": row["created_at"],
            "score": scored["total"], "band": scored["band"],
            "hits": scored["hits"], "missing": scored["missing"],
            "keyword_total": scored["keyword_total"],
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return {"ok": True, "job_id": job_id, "job_title": job["title"], "results": results}


class PipelineIn(BaseModel):
    job_id: int = Field(gt=0)
    candidate_id: str = Field(min_length=1, max_length=64)
    status: str = Field(default="applied", max_length=40)


@router.post("/api/recruiter/pipeline")
async def add_to_pipeline(req: PipelineIn, user: dict = Depends(_recruiter)):
    if req.status not in APP_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    cid = _need_company(user)
    now = _now()
    with _db() as conn:
        job = conn.execute("SELECT 1 FROM jobs WHERE id = ? AND company_id = ?", (req.job_id, cid)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not _company_candidate(conn, cid, req.candidate_id):
            raise HTTPException(status_code=404, detail="Candidate not found")
        conn.execute(
            "INSERT INTO pipeline_entries (company_id, job_id, candidate_id, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(company_id, job_id, candidate_id) DO UPDATE SET "
            "status = excluded.status, updated_at = excluded.updated_at",
            (cid, req.job_id, req.candidate_id, req.status, now, now))
        row = conn.execute("SELECT id FROM pipeline_entries WHERE company_id = ? AND job_id = ? AND candidate_id = ?",
                           (cid, req.job_id, req.candidate_id)).fetchone()
        entry = _pipeline_entry(conn, row["id"], cid)
    return {"ok": True, "entry": entry}


class StatusIn(BaseModel):
    status: str = Field(default="", max_length=40)


@router.post("/api/recruiter/pipeline/{entry_id}/status")
async def set_pipeline_status(entry_id: int, req: StatusIn, user: dict = Depends(_recruiter)):
    if req.status not in APP_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    cid = _need_company(user)
    with _db() as conn:
        n = conn.execute("UPDATE pipeline_entries SET status = ?, updated_at = ? WHERE id = ? AND company_id = ?",
                         (req.status, _now(), entry_id, cid)).rowcount
        if not n:
            raise HTTPException(status_code=404, detail="Pipeline entry not found")
        entry = _pipeline_entry(conn, entry_id, cid)
    return {"ok": True, "entry": entry}


# ---- delete a candidate and everything attached to it --------------------
@router.delete("/api/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str, user: dict = Depends(_recruiter)):
    cid = _cid(user)
    with _db() as conn:
        _own_candidate(conn, user, candidate_id)
        files = [r["stored_name"] for r in conn.execute(
            "SELECT stored_name FROM candidate_resumes WHERE user_id = ? AND candidate_id = ?",
            (user["id"], candidate_id))]
        conn.execute("DELETE FROM candidate_resumes WHERE user_id = ? AND candidate_id = ?",
                     (user["id"], candidate_id))
        conn.execute("DELETE FROM candidate_tags WHERE user_id = ? AND candidate_id = ?",
                     (user["id"], candidate_id))
        conn.execute("DELETE FROM candidate_notes WHERE user_id = ? AND candidate_id = ?",
                     (user["id"], candidate_id))
        if cid:
            conn.execute("DELETE FROM pipeline_entries WHERE company_id = ? AND candidate_id = ?",
                         (cid, candidate_id))
        conn.execute("DELETE FROM candidates WHERE user_id = ? AND id = ?", (user["id"], candidate_id))
    for name in files:
        _remove_resume(os.path.join(RESUME_DIR, os.path.basename(name)))
    return {"ok": True}
