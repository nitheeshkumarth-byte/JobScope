"""Recruiter CRM / ATS data layer.

Everything here works on the tables declared in ``auth.SCHEMA``. The split
between a *master candidate* (``candidate_profiles``) and a *job application*
(``applications``) is deliberate: pipeline stages live on the application, so
one person can be in Screening for one job and Interview for another.

Only the API layer (``dashboard.py``) decides who may call what; these helpers
assume an authorised caller.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import secrets
import time

from auth import DB_FILE, connect, hash_password, normalize_email, _now

# The 19 stages from the product spec, in workflow order. Kept here rather than
# spread across the API and the UI so the Kanban, the filters and the closure
# maths cannot drift apart.
STAGES = [
    ("sourced", "Sourced"),
    ("screening_pending", "Screening Pending"),
    ("screening_completed", "Screening Completed"),
    ("shortlisted", "Shortlisted"),
    ("submitted_to_client", "Submitted to Client"),
    ("client_review", "Client Review"),
    ("interview_scheduled", "Interview Scheduled"),
    ("interview_in_progress", "Interview in Progress"),
    ("interview_completed", "Interview Completed"),
    ("selected", "Selected"),
    ("offer_negotiation", "Offer Negotiation"),
    ("offer_released", "Offer Released"),
    ("offer_accepted", "Offer Accepted"),
    ("notice_period", "Notice Period"),
    ("joined", "Joined"),
    ("placed", "Successfully Placed"),
    ("rejected", "Rejected"),
    ("withdrawn", "Withdrawn"),
    ("on_hold", "On Hold"),
]
STAGE_KEYS = [k for k, _ in STAGES]
STAGE_LABELS = dict(STAGES)
ACTIVE_STAGES = [k for k in STAGE_KEYS if k not in ("rejected", "withdrawn", "placed")]

# Old `applications.status` values are still read by the candidate portal, so
# keep the column meaningful when a stage moves.
_STAGE_STATUS = {
    "sourced": "applied",
    "screening_pending": "ai_screening",
    "screening_completed": "recruiter_review",
    "shortlisted": "shortlisted",
    "submitted_to_client": "recruiter_review",
    "client_review": "recruiter_review",
    "interview_scheduled": "interview_scheduled",
    "interview_in_progress": "interview_scheduled",
    "interview_completed": "interview_completed",
    "selected": "selected",
    "offer_negotiation": "offer_released",
    "offer_released": "offer_released",
    "offer_accepted": "offer_accepted",
    "notice_period": "offer_accepted",
    "joined": "hired",
    "placed": "hired",
    "rejected": "rejected",
    "withdrawn": "withdrawn",
    "on_hold": "on_hold",
}

JOINED_STATUSES = ("joined", "successfully_placed")
PLACEMENT_STATUSES = ["selected", "offer_released", "offer_accepted", "awaiting_joining",
                      "joined", "successfully_placed", "dropped_before_joining",
                      "rejected", "replacement_required"]
JOB_STATUSES = ["draft", "open", "active", "on_hold", "partially_filled", "filled",
                "closed", "cancelled"]
PRIORITIES = ["low", "medium", "high", "urgent"]

# Uploads are stored outside the static tree so a document can never be fetched
# without going through the authenticated download route.
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(DB_FILE)), "uploads")
ALLOWED_UPLOAD_EXT = {".pdf", ".doc", ".docx", ".txt", ".rtf", ".png", ".jpg",
                      ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _rows(rows):
    return [dict(r) for r in rows]


def _row(row):
    return dict(row) if row else None


def _json(value, fallback):
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _dump(value):
    return json.dumps(value or [], ensure_ascii=False)


def _like(text: str) -> str:
    return f"%{text.strip()}%"


def _now_plus_days(days: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + days * 86400))


def audit(user_id, action, entity_type="", entity_id=None, details=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, action, entity_type, entity_id, details, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, action, entity_type, entity_id,
             json.dumps(details or {}, ensure_ascii=False), _now()))


def log_activity(user_id, company_id, kind, description, entity_type="", entity_id=None,
                 metadata=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO activities (user_id, company_id, type, description, entity_type, "
            "entity_id, metadata, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, company_id, kind, description, entity_type, entity_id,
             json.dumps(metadata or {}, ensure_ascii=False), _now()))


def notify(user_id, kind, title, message, link=""):
    if not user_id:
        return
    with connect() as conn:
        conn.execute(
            "INSERT INTO notifications (user_id, type, title, message, link, created_at) "
            "VALUES (?,?,?,?,?,?)", (user_id, kind, title, message, link, _now()))


def _notify_company_recruiters(company_id, kind, title, message, link="", exclude=None):
    if not company_id:
        return
    with connect() as conn:
        rows = conn.execute(
            "SELECT user_id FROM recruiter_profiles WHERE company_id = ?", (company_id,)).fetchall()
    for r in rows:
        if r["user_id"] != exclude:
            notify(r["user_id"], kind, title, message, link)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------

_CAND_COLS = ("full_name, professional_title, location, phone, email, alt_phone, "
              "current_company, current_designation, total_experience, relevant_experience, "
              "current_ctc, expected_ctc, notice_period, preferred_location, employment_type, "
              "professional_summary, primary_skills, secondary_skills, skills, linkedin, "
              "portfolio, resume_url, recruiter_id, source, status, availability")


def _candidate(row):
    if not row:
        return None
    d = dict(row)
    for key in ("skills", "primary_skills", "secondary_skills", "experience", "education",
                "certifications", "projects", "portfolio_links"):
        d[key] = _json(d.get(key), [])
    return d


def _candidate_select(company_id=None):
    sql = """
        SELECT cp.*, u.email AS account_email,
               rp.nickname AS recruiter_nickname
        FROM candidate_profiles cp
        LEFT JOIN users u ON u.id = cp.user_id
        LEFT JOIN users rp ON rp.id = cp.recruiter_id
        WHERE COALESCE(cp.status, 'active') != 'archived'
    """
    args = []
    if company_id:
        sql += (" AND (cp.recruiter_id IN (SELECT user_id FROM recruiter_profiles "
                "WHERE company_id = ?) OR cp.user_id IN ("
                "SELECT a.candidate_user_id FROM applications a JOIN jobs j ON j.id = a.job_id "
                "WHERE j.company_id = ?))")
        args += [company_id, company_id]
    return sql, args


def list_candidates(company_id=None, search="", status="", recruiter_id=None, skill="",
                    location="", min_experience=None, max_experience=None,
                    page=1, per_page=25, sort="created"):
    sql, args = _candidate_select(company_id)
    if search:
        sql += (" AND (cp.full_name LIKE ? OR cp.email LIKE ? OR cp.phone LIKE ? "
                "OR cp.current_company LIKE ? OR cp.professional_title LIKE ?)")
        args += [_like(search)] * 5
    if status:
        sql += " AND COALESCE(cp.status,'active') = ?"
        args.append(status)
    if recruiter_id:
        sql += " AND cp.recruiter_id = ?"
        args.append(recruiter_id)
    if skill:
        sql += " AND (cp.primary_skills LIKE ? OR cp.secondary_skills LIKE ? OR cp.skills LIKE ?)"
        args += [_like(skill)] * 3
    if location:
        sql += " AND cp.location LIKE ?"
        args.append(_like(location))
    if min_experience is not None:
        sql += " AND COALESCE(cp.total_experience,0) >= ?"
        args.append(min_experience)
    if max_experience is not None:
        sql += " AND COALESCE(cp.total_experience,999) <= ?"
        args.append(max_experience)
    order = "cp.updated_at DESC, cp.id DESC" if sort == "updated" else "COALESCE(cp.created_at,'') DESC, cp.id DESC"
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM ({sql})", args).fetchone()["n"]
        sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
        rows = conn.execute(sql, args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    return [_candidate(r) for r in rows], total


def get_candidate(cid: int):
    sql, args = _candidate_select(None)
    sql += " AND cp.id = ?"
    with connect() as conn:
        row = conn.execute(sql, args + [cid]).fetchone()
    return _candidate(row)


def _find_user_by_email(conn, email):
    if not email:
        return None
    return conn.execute("SELECT * FROM users WHERE email = ?", (normalize_email(email),)).fetchone()


def save_candidate(data: dict, candidate_id=None, created_by=None):
    """Create or update a master candidate.

    A recruiter-created candidate gets a real (login-disabled) user row so the
    existing applications/interviews/offers foreign keys keep working; the
    account can later be claimed through the normal password-reset flow because
    the email is the unique key.
    """
    email = normalize_email(data.get("email") or "")
    full_name = (data.get("full_name") or data.get("name") or "").strip()
    if not full_name:
        raise ValueError("candidate name is required")
    now = _now()
    with connect() as conn:
        if candidate_id:
            cur = conn.execute("SELECT * FROM candidate_profiles WHERE id = ?", (candidate_id,)).fetchone()
            if not cur:
                raise ValueError("candidate not found")
            uid = cur["user_id"]
            if email and email != (cur["email"] or ""):
                clash = _find_user_by_email(conn, email)
                if clash and clash["id"] != uid:
                    raise ValueError("another account already uses that email")
                conn.execute("UPDATE users SET email = ? WHERE id = ?", (email, uid))
            fields = {k: data.get(k) for k in (
                "full_name", "professional_title", "location", "phone", "email", "alt_phone",
                "current_company", "current_designation", "total_experience",
                "relevant_experience", "current_ctc", "expected_ctc", "notice_period",
                "preferred_location", "employment_type", "professional_summary", "linkedin",
                "portfolio", "resume_url", "recruiter_id", "source", "status", "availability")}
            fields["full_name"] = full_name
            sets = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(f"UPDATE candidate_profiles SET {sets}, updated_at = ? WHERE id = ?",
                         list(fields.values()) + [now, candidate_id])
            for key in ("skills", "primary_skills", "secondary_skills"):
                if key in data:
                    conn.execute(f"UPDATE candidate_profiles SET {key} = ? WHERE id = ?",
                                 (_dump(data.get(key)), candidate_id))
            cid = candidate_id
        else:
            if not email:
                raise ValueError("candidate email is required")
            user = _find_user_by_email(conn, email)
            if user:
                existing = conn.execute(
                    "SELECT id FROM candidate_profiles WHERE user_id = ?", (user["id"],)).fetchone()
                if existing:
                    raise ValueError("a candidate with that email already exists")
                uid = user["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO users (email, password_hash, created_at, verified_at, role) "
                    "VALUES (?,?,?,?, 'candidate')",
                    (email, hash_password(secrets.token_urlsafe(32)), now, now))
                uid = cur.lastrowid
            cur = conn.execute(
                """INSERT INTO candidate_profiles
                   (user_id, full_name, professional_title, location, phone, email, alt_phone,
                    current_company, current_designation, total_experience, relevant_experience,
                    current_ctc, expected_ctc, notice_period, preferred_location, employment_type,
                    professional_summary, skills, primary_skills, secondary_skills, linkedin,
                    portfolio, resume_url, recruiter_id, source, status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, full_name, data.get("professional_title"), data.get("location"),
                 data.get("phone"), email, data.get("alt_phone"), data.get("current_company"),
                 data.get("current_designation"), data.get("total_experience"),
                 data.get("relevant_experience"), data.get("current_ctc"), data.get("expected_ctc"),
                 data.get("notice_period"), data.get("preferred_location"),
                 data.get("employment_type"), data.get("professional_summary"),
                 _dump(data.get("skills")), _dump(data.get("primary_skills")),
                 _dump(data.get("secondary_skills")), data.get("linkedin"),
                 data.get("portfolio"), data.get("resume_url"), data.get("recruiter_id"),
                 data.get("source"), data.get("status") or "active", now, now))
            cid = cur.lastrowid
    log_activity(created_by, None, "candidate_added",
                 f"Candidate {full_name} added", "candidate", cid)
    audit(created_by, "candidate.create" if not candidate_id else "candidate.update",
          "candidate", cid, {"name": full_name})
    return get_candidate(cid)


def archive_candidate(cid: int, user_id=None):
    with connect() as conn:
        cur = conn.execute(
            "UPDATE candidate_profiles SET status = 'archived', updated_at = ? WHERE id = ?",
            (_now(), cid))
        ok = cur.rowcount > 0
    if ok:
        audit(user_id, "candidate.archive", "candidate", cid)
        log_activity(user_id, None, "candidate_archived", "Candidate archived", "candidate", cid)
    return ok


def candidate_portfolio(cid: int, company_id=None):
    cand = get_candidate(cid)
    if not cand:
        return None
    uid = cand["user_id"]
    with connect() as conn:
        docs = _rows(conn.execute(
            "SELECT * FROM candidate_documents WHERE candidate_id = ? ORDER BY created_at DESC",
            (cid,)).fetchall())
        experiences = _rows(conn.execute(
            "SELECT * FROM candidate_experiences WHERE candidate_id = ? "
            "ORDER BY COALESCE(end_date,'9999') DESC, id DESC", (cid,)).fetchall())
        education = _rows(conn.execute(
            "SELECT * FROM candidate_education WHERE candidate_id = ? ORDER BY id DESC",
            (cid,)).fetchall())
        skills = _rows(conn.execute(
            "SELECT * FROM candidate_skills WHERE candidate_id = ? ORDER BY skill",
            (cid,)).fetchall())
        apps = _rows(conn.execute(
            """SELECT a.*, j.title AS job_title, j.job_code, j.openings,
                      c.name AS company_name, c.id AS company_id,
                      rp.nickname AS recruiter_name
               FROM applications a
               JOIN jobs j ON j.id = a.job_id
               LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN users rp ON rp.id = a.recruiter_id
               WHERE a.candidate_user_id = ?
               ORDER BY a.applied_at DESC""", (uid,)).fetchall())
        app_ids = [a["id"] for a in apps]
        interviews = []
        feedback = []
        offers = []
        placements = []
        if app_ids:
            marks = ",".join("?" * len(app_ids))
            interviews = _rows(conn.execute(
                f"""SELECT i.*, j.title AS job_title FROM interviews i
                    LEFT JOIN jobs j ON j.id = i.job_id
                    WHERE i.application_id IN ({marks}) ORDER BY i.scheduled_at DESC""",
                app_ids).fetchall())
            feedback = _rows(conn.execute(
                f"""SELECT f.*, i.application_id, i.round, i.interview_type, i.scheduled_at
                    FROM interview_feedback f JOIN interviews i ON i.id = f.interview_id
                    WHERE i.application_id IN ({marks}) ORDER BY f.created_at DESC""",
                app_ids).fetchall())
            offers = _rows(conn.execute(
                f"""SELECT o.*, j.title AS job_title FROM offers o
                    LEFT JOIN jobs j ON j.id = o.job_id
                    WHERE o.application_id IN ({marks}) ORDER BY o.created_at DESC""",
                app_ids).fetchall())
            placements = _rows(conn.execute(
                f"""SELECT p.*, j.title AS job_title, c.name AS company_name
                    FROM placements p LEFT JOIN jobs j ON j.id = p.job_id
                    LEFT JOIN companies c ON c.id = p.company_id
                    WHERE p.application_id IN ({marks})
                    ORDER BY COALESCE(p.placement_date, p.created_at) DESC""",
                app_ids).fetchall())
        notes = _rows(conn.execute(
            """SELECT n.*, u.nickname AS recruiter_name FROM recruiter_notes n
               LEFT JOIN users u ON u.id = n.recruiter_id
               WHERE n.candidate_user_id = ? ORDER BY n.created_at DESC""", (uid,)).fetchall())
        comms = _rows(conn.execute(
            """SELECT m.*, u.nickname AS recruiter_name FROM communications m
               LEFT JOIN users u ON u.id = m.recruiter_id
               WHERE m.candidate_user_id = ? ORDER BY m.created_at DESC""", (uid,)).fetchall())
        timeline = _rows(conn.execute(
            """SELECT * FROM activities
               WHERE (entity_type = 'candidate' AND entity_id = ?)
                  OR (entity_type IN ('application','interview','offer','placement')
                      AND entity_id IN (SELECT id FROM applications WHERE candidate_user_id = ?))
               ORDER BY created_at DESC LIMIT 200""", (cid, uid)).fetchall())
        stage_history = []
        if app_ids:
            marks = ",".join("?" * len(app_ids))
            stage_history = _rows(conn.execute(
                f"""SELECT h.*, u.nickname AS changed_by_name FROM application_stage_history h
                    LEFT JOIN users u ON u.id = h.changed_by
                    WHERE h.application_id IN ({marks}) ORDER BY h.created_at DESC""",
                app_ids).fetchall())
    return {"candidate": cand, "documents": docs, "experiences": experiences,
            "education": education, "skills": skills, "applications": apps,
            "interviews": interviews, "feedback": feedback, "offers": offers,
            "placements": placements, "notes": notes, "communications": comms,
            "timeline": timeline, "stage_history": stage_history}


# -- candidate sub-records -------------------------------------------------

def _add_child(table, candidate_id, data, columns):
    now = _now()
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO {table} (candidate_id, {', '.join(columns)}, created_at) "
            f"VALUES (?, {', '.join('?' * len(columns))}, ?)",
            [candidate_id] + [data.get(c) for c in columns] + [now])
        rid = cur.lastrowid
        row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (rid,)).fetchone()
    return dict(row) if row else {"id": rid}


def _delete_child(table, row_id, candidate_id):
    with connect() as conn:
        cur = conn.execute(f"DELETE FROM {table} WHERE id = ? AND candidate_id = ?",
                           (row_id, candidate_id))
        return cur.rowcount > 0


def add_document(candidate_id, data, uploaded_by=None):
    name = (data.get("name") or "document").strip()
    ext = os.path.splitext(name)[1].lower()
    raw = data.get("data") or ""
    if not raw:
        raise ValueError("file content is required")
    if ext not in ALLOWED_UPLOAD_EXT:
        raise ValueError("that file type is not allowed")
    try:
        import base64
        blob = base64.b64decode(raw.split(",")[-1], validate=True)
    except Exception:
        raise ValueError("the uploaded file could not be decoded")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise ValueError("file is larger than 10 MB")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    stored = f"{secrets.token_hex(16)}{ext}"
    with open(os.path.join(UPLOAD_DIR, stored), "wb") as fh:
        fh.write(blob)
    with connect() as conn:
        version = conn.execute(
            "SELECT COALESCE(MAX(version),0)+1 AS v FROM candidate_documents "
            "WHERE candidate_id = ? AND kind = ?", (candidate_id, data.get("kind") or "resume")
        ).fetchone()["v"]
        cur = conn.execute(
            """INSERT INTO candidate_documents
               (candidate_id, name, kind, mime, size, storage_path, version, uploaded_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (candidate_id, name, data.get("kind") or "resume", data.get("mime"),
             len(blob), stored, version, uploaded_by, _now()))
        rid = cur.lastrowid
        row = conn.execute("SELECT * FROM candidate_documents WHERE id = ?", (rid,)).fetchone()
    audit(uploaded_by, "document.upload", "candidate", candidate_id, {"name": name})
    return dict(row)


def delete_document(doc_id, candidate_id, user_id=None):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_documents WHERE id = ? AND candidate_id = ?",
            (doc_id, candidate_id)).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM candidate_documents WHERE id = ?", (doc_id,))
    if row["storage_path"]:
        try:
            os.remove(os.path.join(UPLOAD_DIR, row["storage_path"]))
        except OSError:
            pass
    audit(user_id, "document.delete", "candidate", candidate_id, {"doc": doc_id})
    return True


def document_path(doc_id, candidate_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_documents WHERE id = ? AND candidate_id = ?",
            (doc_id, candidate_id)).fetchone()
    if not row or not row["storage_path"]:
        return None
    path = os.path.join(UPLOAD_DIR, row["storage_path"])
    return path if os.path.exists(path) else None


def add_experience(candidate_id, data):
    return _add_child("candidate_experiences", candidate_id, data,
                      ("company_name", "designation", "start_date", "end_date",
                       "responsibilities", "technologies", "reason_for_leaving"))


def delete_experience(row_id, candidate_id):
    return _delete_child("candidate_experiences", row_id, candidate_id)


def add_education(candidate_id, data):
    return _add_child("candidate_education", candidate_id, data,
                      ("qualification", "institution", "field", "graduation_year", "grade"))


def delete_education(row_id, candidate_id):
    return _delete_child("candidate_education", row_id, candidate_id)


def add_skill(candidate_id, data):
    return _add_child("candidate_skills", candidate_id, data,
                      ("skill", "level", "years", "assessment"))


def delete_skill(row_id, candidate_id):
    return _delete_child("candidate_skills", row_id, candidate_id)


def add_note(candidate_id, user_id, note, application_id=None):
    cand = get_candidate(candidate_id)
    if not cand:
        raise ValueError("candidate not found")
    now = _now()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO recruiter_notes (recruiter_id, candidate_user_id, application_id, "
            "note, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (user_id, cand["user_id"], application_id, note, now, now))
        row = conn.execute("SELECT * FROM recruiter_notes WHERE id = ?", (cur.lastrowid,)).fetchone()
    log_activity(user_id, None, "note_added", "Recruiter note added", "candidate", candidate_id)
    return dict(row)


def add_communication(candidate_id, user_id, data):
    cand = get_candidate(candidate_id)
    if not cand:
        raise ValueError("candidate not found")
    now = _now()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO communications (recruiter_id, candidate_user_id, application_id, type, "
            "subject, message, direction, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, cand["user_id"], data.get("application_id"), data.get("type") or "message",
             data.get("subject"), data.get("message") or "", data.get("direction") or "outbound",
             now))
        row = conn.execute("SELECT * FROM communications WHERE id = ?", (cur.lastrowid,)).fetchone()
    log_activity(user_id, None, "communication", f"{data.get('type') or 'message'} logged",
                 "candidate", candidate_id)
    return dict(row)


# --------------------------------------------------------------------------
# companies & contacts
# --------------------------------------------------------------------------

def _company(row):
    d = dict(row)
    return d


def list_companies(search="", page=1, per_page=25):
    where = "WHERE 1=1"
    args = []
    if search:
        where += " AND (c.name LIKE ? OR c.industry LIKE ? OR c.location LIKE ?)"
        args += [_like(search)] * 3
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM companies c {where}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT c.*,
                       (SELECT COUNT(*) FROM jobs j WHERE j.company_id = c.id) AS job_count,
                       (SELECT COUNT(*) FROM placements p JOIN jobs j ON j.id = p.job_id
                        WHERE j.company_id = c.id AND p.status IN ('joined','successfully_placed'))
                           AS placement_count,
                       (SELECT COUNT(*) FROM company_contacts cc WHERE cc.company_id = c.id)
                           AS contact_count
                FROM companies c {where}
                ORDER BY c.name LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    return [_company(r) for r in rows], total


def get_company(cid):
    with connect() as conn:
        row = conn.execute(
            """SELECT c.*,
                      (SELECT COUNT(*) FROM jobs j WHERE j.company_id = c.id
                       AND j.status NOT IN ('closed','cancelled')) AS open_requirements
               FROM companies c WHERE c.id = ?""", (cid,)).fetchone()
    return _company(row) if row else None


def save_company(data, cid=None, user_id=None):
    name = (data.get("name") or "").strip()
    if not name:
        raise ValueError("company name is required")
    now = _now()
    fields = ("name", "website", "logo_url", "industry", "description", "location",
              "address", "company_size", "contact_email", "contact_phone", "status",
              "account_manager")
    with connect() as conn:
        if cid:
            row = conn.execute("SELECT id FROM companies WHERE id = ?", (cid,)).fetchone()
            if not row:
                raise ValueError("company not found")
            sets = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(f"UPDATE companies SET {sets} WHERE id = ?",
                         [data.get(k) for k in fields] + [cid])
        else:
            cur = conn.execute(
                f"INSERT INTO companies ({', '.join(fields)}, created_at, created_by) "
                f"VALUES ({', '.join('?' * len(fields))}, ?, ?)",
                [data.get(k) for k in fields] + [now, user_id])
            cid = cur.lastrowid
    audit(user_id, "company.create" if not data.get("id") else "company.update", "company", cid)
    return get_company(cid)


def delete_company(cid, user_id=None):
    with connect() as conn:
        jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE company_id = ?", (cid,)).fetchone()["n"]
        if jobs:
            raise ValueError("move or close this client's jobs before deleting it")
        cur = conn.execute("DELETE FROM companies WHERE id = ?", (cid,))
        ok = cur.rowcount > 0
    if ok:
        audit(user_id, "company.delete", "company", cid)
    return ok


def list_contacts(company_id):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM company_contacts WHERE company_id = ? "
            "ORDER BY is_primary DESC, name", (company_id,)).fetchall()
    return _rows(rows)


def save_contact(company_id, data, contact_id=None, user_id=None):
    name = (data.get("name") or "").strip()
    if not name:
        raise ValueError("contact name is required")
    now = _now()
    fields = ("name", "designation", "email", "phone", "is_primary", "notes")
    with connect() as conn:
        if data.get("is_primary"):
            conn.execute("UPDATE company_contacts SET is_primary = 0 WHERE company_id = ?",
                         (company_id,))
        if contact_id:
            sets = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(f"UPDATE company_contacts SET {sets} WHERE id = ? AND company_id = ?",
                         [data.get(k) for k in fields] + [contact_id, company_id])
            rid = contact_id
        else:
            cur = conn.execute(
                f"INSERT INTO company_contacts (company_id, {', '.join(fields)}, created_at) "
                f"VALUES (?, {', '.join('?' * len(fields))}, ?)",
                [company_id] + [data.get(k) for k in fields] + [now])
            rid = cur.lastrowid
        row = conn.execute("SELECT * FROM company_contacts WHERE id = ?", (rid,)).fetchone()
    audit(user_id, "contact.save", "company", company_id, {"contact": rid})
    return dict(row)


def delete_contact(company_id, contact_id, user_id=None):
    with connect() as conn:
        cur = conn.execute("DELETE FROM company_contacts WHERE id = ? AND company_id = ?",
                           (contact_id, company_id))
        ok = cur.rowcount > 0
    if ok:
        audit(user_id, "contact.delete", "company", company_id, {"contact": contact_id})
    return ok


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

_JOB_SKILLS = ("required_skills", "preferred_skills", "mandatory_skills", "good_to_have_skills")


def _job(row):
    if not row:
        return None
    d = dict(row)
    for key in _JOB_SKILLS:
        d[key] = _json(d.get(key), [])
    return d


def _closure(conn, job_id, openings):
    placed = conn.execute(
        "SELECT COUNT(*) AS n FROM placements WHERE job_id = ? AND status IN ('joined','successfully_placed')",
        (job_id,)).fetchone()["n"]
    total = openings or 0
    remaining = max(0, total - placed)
    pct = round((placed / total) * 100, 1) if total else 0.0
    return placed, remaining, pct


def job_with_closure(row):
    d = _job(row)
    with connect() as conn:
        placed, remaining, pct = _closure(conn, d["id"], d.get("openings"))
        d["placed"], d["remaining_openings"], d["closure_percentage"] = placed, remaining, pct
        d["candidate_count"] = conn.execute(
            "SELECT COUNT(*) AS n FROM applications WHERE job_id = ?", (d["id"],)).fetchone()["n"]
        d["assigned_recruiters"] = _rows(conn.execute(
            """SELECT u.id, u.email, u.nickname FROM job_assignments ja
               JOIN users u ON u.id = ja.recruiter_id WHERE ja.job_id = ?""", (d["id"],)).fetchall())
    return d


def list_jobs(company_id=None, search="", status="", recruiter_id=None, page=1,
              per_page=25, sort="created"):
    where = "WHERE 1=1"
    args = []
    if company_id:
        where += " AND j.company_id = ?"
        args.append(company_id)
    if recruiter_id:
        where += (" AND (j.recruiter_id = ? OR j.id IN (SELECT job_id FROM job_assignments "
                  "WHERE recruiter_id = ?))")
        args += [recruiter_id, recruiter_id]
    if status:
        where += " AND j.status = ?"
        args.append(status)
    if search:
        where += " AND (j.title LIKE ? OR j.description LIKE ? OR j.job_code LIKE ? OR j.location LIKE ?)"
        args += [_like(search)] * 4
    order = "j.updated_at DESC" if sort == "updated" else "j.created_at DESC"
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM jobs j {where}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT j.*, c.name AS company_name,
                       u.nickname AS recruiter_name
                FROM jobs j LEFT JOIN companies c ON c.id = j.company_id
                LEFT JOIN users u ON u.id = j.recruiter_id
                {where} ORDER BY {order} LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    items = []
    for r in rows:
        d = _job(r)
        with connect() as conn:
            placed, remaining, pct = _closure(conn, d["id"], d.get("openings"))
        d["placed"], d["remaining_openings"], d["closure_percentage"] = placed, remaining, pct
        items.append(d)
    return items, total


def get_job(jid):
    with connect() as conn:
        row = conn.execute(
            """SELECT j.*, c.name AS company_name, c.industry AS company_industry,
                      c.location AS company_location, c.id AS company_id,
                      u.nickname AS recruiter_name
               FROM jobs j LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN users u ON u.id = j.recruiter_id WHERE j.id = ?""", (jid,)).fetchone()
        if not row:
            return None
        d = job_with_closure(row)
        d["pipeline"] = _rows(conn.execute(
            """SELECT COALESCE(a.stage,'sourced') AS stage, COUNT(*) AS n
               FROM applications a WHERE a.job_id = ? GROUP BY COALESCE(a.stage,'sourced')""",
            (jid,)).fetchall())
        d["interview_count"] = conn.execute(
            """SELECT COUNT(*) AS n FROM interviews i JOIN applications a ON a.id = i.application_id
               WHERE a.job_id = ?""", (jid,)).fetchone()["n"]
        d["offer_count"] = conn.execute(
            """SELECT COUNT(*) AS n FROM offers o JOIN applications a ON a.id = o.application_id
               WHERE a.job_id = ?""", (jid,)).fetchone()["n"]
        d["placements"] = _rows(conn.execute(
            """SELECT p.*, cp.full_name AS candidate_name FROM placements p
               JOIN applications a ON a.id = p.application_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
               WHERE p.job_id = ? ORDER BY p.created_at DESC""", (jid,)).fetchall())
        d["activities"] = _rows(conn.execute(
            "SELECT * FROM activities WHERE entity_type = 'job' AND entity_id = ? "
            "ORDER BY created_at DESC LIMIT 50", (jid,)).fetchall())
    return d


def save_job(data, recruiter_id=None, job_id=None):
    title = (data.get("title") or "").strip()
    if not title:
        raise ValueError("job title is required")
    now = _now()
    fields = ("company_id", "title", "department", "description", "min_experience",
              "max_experience", "education_req", "employment_type", "work_mode", "location",
              "salary_min", "salary_max", "currency", "openings", "application_deadline",
              "status", "job_code", "priority", "hiring_manager", "client_contact_id",
              "target_closure_date")
    values = [data.get(f) for f in fields]
    values = [data.get("title") if f == "title" else v for f, v in zip(fields, values)]
    values = [recruiter_id if f == "recruiter_id" else v for f, v in zip(fields, values)]
    with connect() as conn:
        if job_id:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                raise ValueError("job not found")
            # A generated code is stable; never let an edit blank it out.
            if not data.get("job_code"):
                values[fields.index("job_code")] = row["job_code"]
            sets = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(f"UPDATE jobs SET {sets}, updated_at = ? WHERE id = ?",
                         values + [now, job_id])
            jid = job_id
        else:
            if not values[fields.index("company_id")]:
                raise ValueError("company is required")
            cur = conn.execute(
                f"INSERT INTO jobs ({', '.join(fields)}, recruiter_id, created_at, updated_at) "
                f"VALUES ({', '.join('?' * len(fields))}, ?, ?, ?)",
                values + [recruiter_id, now, now])
            jid = cur.lastrowid
            conn.execute("UPDATE jobs SET job_code = ? WHERE id = ?",
                         (f"JOB-{jid:05d}", jid))
        for key in _JOB_SKILLS:
            if key in data:
                conn.execute(f"UPDATE jobs SET {key} = ? WHERE id = ?", (_dump(data.get(key)), jid))
        for rid in (data.get("assigned_recruiters") or []):
            conn.execute("INSERT OR IGNORE INTO job_assignments (job_id, recruiter_id, created_at) "
                         "VALUES (?,?,?)", (jid, rid, now))
    audit(recruiter_id, "job.create" if not job_id else "job.update", "job", jid)
    log_activity(recruiter_id, data.get("company_id"), "job_created" if not job_id else "job_updated",
                 f"Job {title} {'created' if not job_id else 'updated'}", "job", jid)
    return get_job(jid)


def delete_job(jid, user_id=None):
    """Cancel rather than delete: applications and placements are history."""
    with connect() as conn:
        cur = conn.execute("UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                           (_now(), jid))
        ok = cur.rowcount > 0
    if ok:
        audit(user_id, "job.cancel", "job", jid)
        log_activity(user_id, None, "job_cancelled", f"Job #{jid} cancelled", "job", jid)
    return ok


# --------------------------------------------------------------------------
# applications & pipeline
# --------------------------------------------------------------------------

def _application(row):
    return dict(row) if row else None


def list_applications(company_id=None, job_id=None, candidate_id=None, recruiter_id=None,
                      stage=None, search="", page=1, per_page=50):
    where = "WHERE 1=1"
    args = []
    if company_id:
        where += " AND j.company_id = ?"
        args.append(company_id)
    if job_id:
        where += " AND a.job_id = ?"
        args.append(job_id)
    if candidate_id:
        where += " AND a.candidate_user_id = ?"
        args.append(candidate_id)
    if recruiter_id:
        where += " AND a.recruiter_id = ?"
        args.append(recruiter_id)
    if stage:
        where += " AND COALESCE(a.stage,'sourced') = ?"
        args.append(stage)
    if search:
        where += " AND (cp.full_name LIKE ? OR j.title LIKE ? OR c.name LIKE ?)"
        args += [_like(search)] * 3
    base = f"""FROM applications a
               JOIN jobs j ON j.id = a.job_id
               LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
               LEFT JOIN users r ON r.id = a.recruiter_id
               {where}"""
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n {base}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT a.*, j.title AS job_title, j.job_code, j.openings, j.company_id,
                       c.name AS company_name, cp.full_name AS candidate_name,
                       cp.email AS candidate_email, cp.phone AS candidate_phone,
                       cp.professional_title AS candidate_title,
                       cp.total_experience AS candidate_experience,
                       cp.current_ctc AS candidate_current_ctc,
                       cp.expected_ctc AS candidate_expected_ctc,
                       cp.notice_period AS candidate_notice,
                       r.nickname AS recruiter_name
                {base} ORDER BY a.updated_at DESC, a.id DESC LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
        items = []
        for r in rows:
            d = _application(r)
            d["stage_label"] = STAGE_LABELS.get(d.get("stage") or "sourced", d.get("stage"))
            d["days_in_stage"] = _days_since(d.get("stage_entered_at") or d.get("applied_at"))
            items.append(d)
    return items, total


def _days_since(iso):
    if not iso:
        return 0
    try:
        from datetime import datetime
        return max(0, (datetime.now() - datetime.fromisoformat(iso)).days)
    except (TypeError, ValueError):
        return 0


def get_application(app_id):
    with connect() as conn:
        row = conn.execute(
            """SELECT a.*, j.title AS job_title, j.openings, c.name AS company_name,
                      cp.full_name AS candidate_name, cp.email AS candidate_email
               FROM applications a JOIN jobs j ON j.id = a.job_id
               LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
               WHERE a.id = ?""", (app_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["stage_label"] = STAGE_LABELS.get(d.get("stage") or "sourced")
        d["history"] = _rows(conn.execute(
            """SELECT h.*, u.nickname AS changed_by_name FROM application_stage_history h
               LEFT JOIN users u ON u.id = h.changed_by WHERE h.application_id = ?
               ORDER BY h.created_at DESC""", (app_id,)).fetchall())
    return d


def add_candidate_to_job(job_id, candidate_id=None, candidate=None, recruiter_id=None,
                         source="", created_by=None):
    """Map an existing master candidate (or a brand-new one) onto a job."""
    if not candidate_id:
        candidate = save_candidate(candidate or {}, created_by=created_by)
        candidate_id = candidate["id"]
    cand = get_candidate(candidate_id)
    if not cand:
        raise ValueError("candidate not found")
    with connect() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise ValueError("job not found")
        existing = conn.execute(
            "SELECT id FROM applications WHERE job_id = ? AND candidate_user_id = ?",
            (job_id, cand["user_id"])).fetchone()
        if existing:
            raise ValueError("this candidate is already submitted to that job")
        now = _now()
        cur = conn.execute(
            """INSERT INTO applications
               (job_id, candidate_user_id, candidate_profile_id, resume_id, status, stage,
                recruiter_id, source, stage_entered_at, applied_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (job_id, cand["user_id"], candidate_id, cand.get("resume_url"), "applied", "sourced",
             recruiter_id or job["recruiter_id"], source or cand.get("source"), now, now, now))
        app_id = cur.lastrowid
        conn.execute(
            "INSERT INTO application_stage_history (application_id, from_stage, to_stage, "
            "note, changed_by, created_at) VALUES (?,?,?,?,?,?)",
            (app_id, None, "sourced", "Added to job", created_by, now))
    log_activity(created_by, job["company_id"], "candidate_assigned",
                 f"{cand['full_name']} added to {job['title']}", "application", app_id)
    notify(recruiter_id or job["recruiter_id"], "candidate_assigned", "New candidate assigned",
           f"{cand['full_name']} was added to {job['title']}", f"/recruiter#pipeline")
    return get_application(app_id)


def move_application(app_id, stage, note="", reason="", user_id=None):
    if stage not in STAGE_KEYS:
        raise ValueError("unknown stage")
    now = _now()
    with connect() as conn:
        app = conn.execute(
            """SELECT a.*, j.company_id, j.recruiter_id AS job_recruiter, j.title AS job_title
               FROM applications a JOIN jobs j ON j.id = a.job_id WHERE a.id = ?""",
            (app_id,)).fetchone()
        if not app:
            raise ValueError("application not found")
        old = app["stage"] or "sourced"
        if old == stage and not note:
            return get_application(app_id)
        conn.execute(
            "UPDATE applications SET stage = ?, status = ?, stage_entered_at = ?, updated_at = ?, "
            "rejection_reason = COALESCE(?, rejection_reason) WHERE id = ?",
            (stage, _STAGE_STATUS.get(stage, "applied"), now, now,
             reason if stage == "rejected" else None, app_id))
        conn.execute(
            "INSERT INTO application_stage_history (application_id, from_stage, to_stage, "
            "note, reason, changed_by, created_at) VALUES (?,?,?,?,?,?,?)",
            (app_id, old, stage, note, reason, user_id, now))
        company_id = app["company_id"]
    log_activity(user_id, company_id, "stage_changed",
                 f"{app['job_title']}: {STAGE_LABELS.get(old, old)} → {STAGE_LABELS.get(stage, stage)}",
                 "application", app_id, {"from": old, "to": stage, "note": note})
    notify(app["job_recruiter"], "stage_changed", "Pipeline updated",
           f"{app['job_title']} moved to {STAGE_LABELS.get(stage, stage)}", "/recruiter#pipeline")
    if stage in ("offer_accepted", "joined", "placed"):
        notify(app["job_recruiter"], "offer_update", "Candidate progressed",
               f"{app['job_title']} is now {STAGE_LABELS.get(stage, stage)}", "/recruiter#placements")
    audit(user_id, "application.stage", "application", app_id, {"from": old, "to": stage})
    return get_application(app_id)


# --------------------------------------------------------------------------
# interviews
# --------------------------------------------------------------------------

def list_interviews(company_id=None, candidate_id=None, job_id=None, status="",
                    date_from="", date_to="", interviewer_email="", page=1, per_page=50):
    where = "WHERE 1=1"
    args = []
    if company_id:
        where += " AND j.company_id = ?"
        args.append(company_id)
    if candidate_id:
        where += " AND a.candidate_user_id = ?"
        args.append(candidate_id)
    if job_id:
        where += " AND i.job_id = ?"
        args.append(job_id)
    if status:
        where += " AND i.status = ?"
        args.append(status)
    if date_from:
        where += " AND i.scheduled_at >= ?"
        args.append(date_from)
    if date_to:
        where += " AND i.scheduled_at <= ?"
        args.append(date_to + "T23:59:59")
    if interviewer_email:
        where += " AND i.interviewer_email = ?"
        args.append(interviewer_email)
    base = f"""FROM interviews i
               JOIN applications a ON a.id = i.application_id
               JOIN jobs j ON j.id = i.job_id
               LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = i.candidate_user_id
               LEFT JOIN users r ON r.id = i.recruiter_id
               {where}"""
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n {base}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT i.*, a.stage AS application_stage, j.title AS job_title,
                       j.job_code, c.name AS company_name, cp.full_name AS candidate_name,
                       cp.phone AS candidate_phone, cp.email AS candidate_email,
                       r.nickname AS recruiter_name,
                       (SELECT COUNT(*) FROM interview_feedback f WHERE f.interview_id = i.id) AS feedback_count
                {base} ORDER BY i.scheduled_at DESC LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    return _rows(rows), total


def get_interview(interview_id):
    with connect() as conn:
        row = conn.execute(
            """SELECT i.*, a.stage AS application_stage, j.title AS job_title,
                      c.name AS company_name, cp.full_name AS candidate_name
               FROM interviews i JOIN applications a ON a.id = i.application_id
               JOIN jobs j ON j.id = i.job_id LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = i.candidate_user_id
               WHERE i.id = ?""", (interview_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["feedback"] = _rows(conn.execute(
            """SELECT f.*, u.nickname AS interviewer_name FROM interview_feedback f
               LEFT JOIN users u ON u.id = f.interviewer_id
               WHERE f.interview_id = ? ORDER BY f.created_at DESC""", (interview_id,)).fetchall())
    return d


def save_interview(data, user_id=None, interview_id=None):
    app_id = data.get("application_id")
    if not app_id:
        raise ValueError("application is required")
    now = _now()
    scheduled_at = data.get("scheduled_at")
    if not scheduled_at:
        raise ValueError("interview date and time are required")
    duration = int(data.get("duration_minutes") or 60)
    fields = ("application_id", "job_id", "candidate_user_id", "recruiter_id", "interview_type",
              "scheduled_at", "duration_minutes", "meeting_link", "location", "instructions",
              "status", "round", "mode", "timezone", "interviewer_name", "interviewer_email",
              "interviewer_contact")
    with connect() as conn:
        app = conn.execute(
            """SELECT a.*, j.id AS jid, j.company_id, j.recruiter_id AS job_recruiter,
                      j.title AS job_title FROM applications a JOIN jobs j ON j.id = a.job_id
               WHERE a.id = ?""", (app_id,)).fetchone()
        if not app:
            raise ValueError("application not found")
        values = dict(data)
        values["job_id"] = values.get("job_id") or app["jid"]
        values["candidate_user_id"] = values.get("candidate_user_id") or app["candidate_user_id"]
        values["recruiter_id"] = values.get("recruiter_id") or app["recruiter_id"] or user_id
        values.setdefault("interview_type", "technical_interview")
        values.setdefault("status", "scheduled")
        values["duration_minutes"] = duration
        if interview_id:
            sets = ", ".join(f"{f} = ?" for f in fields)
            conn.execute(f"UPDATE interviews SET {sets}, updated_at = ? WHERE id = ?",
                         [values.get(f) for f in fields] + [now, interview_id])
            iid = interview_id
        else:
            cur = conn.execute(
                f"INSERT INTO interviews ({', '.join(fields)}, created_at, updated_at) "
                f"VALUES ({', '.join('?' * len(fields))}, ?, ?)",
                [values.get(f) for f in fields] + [now, now])
            iid = cur.lastrowid
        row = conn.execute("SELECT * FROM interviews WHERE id = ?", (iid,)).fetchone()
        company_id = app["company_id"]
    if not interview_id:
        log_activity(user_id, company_id, "interview_scheduled",
                     f"Interview scheduled for {app['job_title']}", "interview", iid)
        notify(values.get("recruiter_id"), "interview_scheduled", "Interview scheduled",
               f"{app['job_title']} on {scheduled_at}", "/recruiter#interviews")
        current = app["stage"] or "sourced"
        if current in ("sourced", "screening_pending", "screening_completed", "shortlisted",
                       "submitted_to_client", "client_review"):
            move_application(app_id, "interview_scheduled", note="Interview scheduled", user_id=user_id)
    else:
        log_activity(user_id, company_id, "interview_updated",
                     f"Interview updated for {app['job_title']}", "interview", iid)
    audit(user_id, "interview.save", "interview", iid)
    return dict(row)


def save_feedback(interview_id, data, user_id=None):
    with connect() as conn:
        interview = conn.execute(
            """SELECT i.*, j.company_id, j.title AS job_title FROM interviews i
               JOIN jobs j ON j.id = i.job_id WHERE i.id = ?""", (interview_id,)).fetchone()
        if not interview:
            raise ValueError("interview not found")
        now = _now()
        cur = conn.execute(
            """INSERT INTO interview_feedback
               (interview_id, interviewer_id, rating, technical_score, communication_score,
                overall_score, strengths, improvements, comments, recommendation, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (interview_id, user_id, data.get("rating"), data.get("technical_score"),
             data.get("communication_score"), data.get("overall_score"),
             data.get("strengths"), data.get("weaknesses") or data.get("improvements"),
             data.get("comments"), data.get("recommendation"), now))
        fid = cur.lastrowid
        conn.execute("UPDATE interviews SET status = 'completed', updated_at = ? WHERE id = ?",
                     (now, interview_id))
        conn.execute(
            "INSERT INTO application_stage_history (application_id, from_stage, to_stage, note, "
            "changed_by, created_at) VALUES (?,?,?,?,?,?)",
            (interview["application_id"], interview["application_stage"] if "application_stage" in interview else None,
             "interview_completed", "Interview feedback submitted", user_id, now))
        conn.execute("UPDATE applications SET stage = 'interview_completed', "
                     "status = 'interview_completed', stage_entered_at = ?, updated_at = ? "
                     "WHERE id = ?", (now, now, interview["application_id"]))
    log_activity(user_id, interview["company_id"], "feedback_submitted",
                 f"Feedback submitted for {interview['job_title']}", "interview", interview_id)
    notify(interview["recruiter_id"], "feedback_submitted", "Interview feedback received",
           f"Feedback is in for {interview['job_title']}", "/recruiter#interviews")
    audit(user_id, "interview.feedback", "interview", interview_id)
    return get_interview(interview_id)


# --------------------------------------------------------------------------
# offers
# --------------------------------------------------------------------------

def list_offers(company_id=None, status="", candidate_id=None, job_id=None, page=1,
                per_page=50):
    where = "WHERE 1=1"
    args = []
    if company_id:
        where += " AND j.company_id = ?"
        args.append(company_id)
    if status:
        where += " AND o.status = ?"
        args.append(status)
    if candidate_id:
        where += " AND a.candidate_user_id = ?"
        args.append(candidate_id)
    if job_id:
        where += " AND o.job_id = ?"
        args.append(job_id)
    base = f"""FROM offers o JOIN applications a ON a.id = o.application_id
               JOIN jobs j ON j.id = o.job_id LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
               {where}"""
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n {base}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT o.*, j.title AS job_title, j.job_code, c.name AS company_name,
                       cp.full_name AS candidate_name, a.stage AS application_stage
                {base} ORDER BY o.created_at DESC LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    return _rows(rows), total


def get_offer(offer_id):
    with connect() as conn:
        row = conn.execute(
            """SELECT o.*, j.title AS job_title, c.name AS company_name,
                      cp.full_name AS candidate_name
               FROM offers o JOIN applications a ON a.id = o.application_id
               JOIN jobs j ON j.id = o.job_id LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
               WHERE o.id = ?""", (offer_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["history"] = _rows(conn.execute(
            "SELECT * FROM offer_history WHERE offer_id = ? ORDER BY created_at DESC",
            (offer_id,)).fetchall())
    return d


def save_offer(data, user_id=None, offer_id=None):
    app_id = data.get("application_id")
    if not app_id:
        raise ValueError("application is required")
    now = _now()
    fields = ("application_id", "job_id", "candidate_user_id", "offered_by", "salary",
              "currency", "offer_letter", "benefits", "start_date", "expiry_date", "status",
              "designation", "fixed_ctc", "variable_ctc", "joining_bonus", "notice_period",
              "expected_joining_date")
    with connect() as conn:
        app = conn.execute(
            """SELECT a.*, j.id AS jid, j.company_id, j.recruiter_id AS job_recruiter,
                      j.title AS job_title FROM applications a JOIN jobs j ON j.id = a.job_id
               WHERE a.id = ?""", (app_id,)).fetchone()
        if not app:
            raise ValueError("application not found")
        values = dict(data)
        values["job_id"] = values.get("job_id") or app["jid"]
        values["candidate_user_id"] = values.get("candidate_user_id") or app["candidate_user_id"]
        values["offered_by"] = values.get("offered_by") or user_id
        values.setdefault("status", "pending_approval")
        if offer_id:
            sets = ", ".join(f"{f} = ?" for f in fields)
            conn.execute(f"UPDATE offers SET {sets}, updated_at = ? WHERE id = ?",
                         [values.get(f) for f in fields] + [now, offer_id])
            oid = offer_id
        else:
            cur = conn.execute(
                f"INSERT INTO offers ({', '.join(fields)}, created_at, updated_at) "
                f"VALUES ({', '.join('?' * len(fields))}, ?, ?)",
                [values.get(f) for f in fields] + [now, now])
            oid = cur.lastrowid
        row = conn.execute("SELECT * FROM offers WHERE id = ?", (oid,)).fetchone()
        conn.execute("INSERT INTO offer_history (offer_id, status, note, changed_by, created_at) "
                     "VALUES (?,?,?,?,?)",
                     (oid, values.get("status"), data.get("note") or "Offer saved", user_id, now))
        company_id = app["company_id"]
    log_activity(user_id, company_id, "offer_saved", f"Offer for {app['job_title']}",
                 "offer", oid)
    _apply_offer_stage(app_id, values.get("status"), user_id, app["job_title"],
                       app["job_recruiter"], company_id)
    audit(user_id, "offer.save", "offer", oid)
    return get_offer(oid)


def set_offer_status(offer_id, status, note="", user_id=None):
    now = _now()
    with connect() as conn:
        offer = conn.execute(
            """SELECT o.*, j.company_id, j.recruiter_id AS job_recruiter, j.title AS job_title,
                      a.stage AS application_stage
               FROM offers o JOIN applications a ON a.id = o.application_id
               JOIN jobs j ON j.id = o.job_id WHERE o.id = ?""", (offer_id,)).fetchone()
        if not offer:
            raise ValueError("offer not found")
        conn.execute("UPDATE offers SET status = ?, updated_at = ? WHERE id = ?",
                     (status, now, offer_id))
        conn.execute("INSERT INTO offer_history (offer_id, status, note, changed_by, created_at) "
                     "VALUES (?,?,?,?,?)", (offer_id, status, note, user_id, now))
    _apply_offer_stage(offer["application_id"], status, user_id, offer["job_title"],
                       offer["job_recruiter"], offer["company_id"])
    log_activity(user_id, offer["company_id"], "offer_status",
                 f"Offer {status.replace('_', ' ')} for {offer['job_title']}", "offer", offer_id)
    audit(user_id, "offer.status", "offer", offer_id, {"status": status})
    return get_offer(offer_id)


def _apply_offer_stage(app_id, status, user_id, job_title, recruiter_id, company_id):
    stage = None
    if status in ("pending_approval", "approved", "negotiation"):
        stage = "offer_negotiation"
    elif status == "released":
        stage = "offer_released"
    elif status == "accepted":
        stage = "offer_accepted"
    elif status in ("declined", "expired", "withdrawn"):
        stage = "withdrawn"
    if stage:
        move_application(app_id, stage, note=f"Offer {status.replace('_', ' ')}", user_id=user_id)
    if status == "accepted":
        notify(recruiter_id, "offer_accepted", "Offer accepted",
               f"Offer accepted for {job_title}. Confirm the joining date.", "/recruiter#placements")
        # The joining follow-up is the piece people forget; create it here.
        with connect() as conn:
            app = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
            conn.execute(
                "INSERT INTO tasks (recruiter_id, company_id, title, description, due_date, "
                "status, priority, related_to_type, related_to_id, created_at, updated_at) "
                "VALUES (?,?,?,?,?,'pending','high','application',?,?,?)",
                (recruiter_id, company_id, "Confirm candidate joining",
                 f"Offer accepted for {job_title}. Confirm the joining date and create the placement.",
                 _now_plus_days(7), app_id, _now(), _now()))


# --------------------------------------------------------------------------
# placements & closure
# --------------------------------------------------------------------------

def list_placements(company_id=None, status="", recruiter_id=None, job_id=None, page=1,
                    per_page=50):
    where = "WHERE 1=1"
    args = []
    if company_id:
        where += " AND j.company_id = ?"
        args.append(company_id)
    if status:
        where += " AND p.status = ?"
        args.append(status)
    if recruiter_id:
        where += " AND p.recruiter_id = ?"
        args.append(recruiter_id)
    if job_id:
        where += " AND p.job_id = ?"
        args.append(job_id)
    base = f"""FROM placements p
               JOIN applications a ON a.id = p.application_id
               JOIN jobs j ON j.id = p.job_id LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
               LEFT JOIN users r ON r.id = p.recruiter_id
               {where}"""
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n {base}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT p.*, j.title AS job_title, j.job_code, j.openings,
                       c.name AS company_name, cp.full_name AS candidate_name,
                       r.nickname AS recruiter_name
                {base} ORDER BY p.created_at DESC LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    return _rows(rows), total


def get_placement(pid):
    with connect() as conn:
        row = conn.execute(
            """SELECT p.*, j.title AS job_title, j.job_code, j.openings,
                      c.name AS company_name, cp.full_name AS candidate_name
               FROM placements p JOIN applications a ON a.id = p.application_id
               JOIN jobs j ON j.id = p.job_id LEFT JOIN companies c ON c.id = j.company_id
               LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
               WHERE p.id = ?""", (pid,)).fetchone()
    return dict(row) if row else None


def save_placement(data, user_id=None, placement_id=None):
    app_id = data.get("application_id")
    if not app_id:
        raise ValueError("application is required")
    now = _now()
    with connect() as conn:
        app = conn.execute(
            """SELECT a.*, j.id AS jid, j.company_id, j.recruiter_id AS job_recruiter,
                      j.title AS job_title FROM applications a JOIN jobs j ON j.id = a.job_id
               WHERE a.id = ?""", (app_id,)).fetchone()
        if not app:
            raise ValueError("application not found")
        values = dict(data)
        values.setdefault("job_id", app["jid"])
        values.setdefault("candidate_user_id", app["candidate_user_id"])
        values.setdefault("company_id", app["company_id"])
        values.setdefault("recruiter_id", app["recruiter_id"] or user_id)
        values.setdefault("status", "selected")
        fields = ("application_id", "job_id", "candidate_user_id", "company_id", "recruiter_id",
                  "offered_ctc", "currency", "joining_date", "actual_joining_date",
                  "placement_date", "status", "replacement_period", "guarantee_period",
                  "client_confirmation", "billing_status", "placement_fee", "invoice_number",
                  "invoice_details")
        if placement_id:
            sets = ", ".join(f"{f} = ?" for f in fields)
            conn.execute(f"UPDATE placements SET {sets}, updated_at = ? WHERE id = ?",
                         [values.get(f) for f in fields] + [now, placement_id])
            pid = placement_id
        else:
            cur = conn.execute(
                f"INSERT INTO placements ({', '.join(fields)}, created_at, updated_at) "
                f"VALUES ({', '.join('?' * len(fields))}, ?, ?)",
                [values.get(f) for f in fields] + [now, now])
            pid = cur.lastrowid
        row = conn.execute("SELECT * FROM placements WHERE id = ?", (pid,)).fetchone()
    log_activity(user_id, app["company_id"], "placement_saved",
                 f"Placement recorded for {app['job_title']}", "placement", pid)
    audit(user_id, "placement.save", "placement", pid)
    _sync_placement(pid, user_id)
    return get_placement(pid)


def set_placement_status(pid, status, data=None, user_id=None):
    if status not in PLACEMENT_STATUSES:
        raise ValueError("unknown placement status")
    data = data or {}
    now = _now()
    with connect() as conn:
        placement = conn.execute("SELECT * FROM placements WHERE id = ?", (pid,)).fetchone()
        if not placement:
            raise ValueError("placement not found")
        conn.execute(
            "UPDATE placements SET status = ?, actual_joining_date = COALESCE(?, actual_joining_date), "
            "placement_date = COALESCE(?, placement_date), updated_at = ? WHERE id = ?",
            (status, data.get("actual_joining_date"), data.get("placement_date"), now, pid))
    _sync_placement(pid, user_id)
    audit(user_id, "placement.status", "placement", pid, {"status": status})
    return get_placement(pid)


def _sync_placement(pid, user_id=None):
    """Propagate a placement into the application stage and the job closure."""
    with connect() as conn:
        p = conn.execute(
            """SELECT p.*, j.title AS job_title, j.company_id, j.openings,
                      j.recruiter_id AS job_recruiter
               FROM placements p JOIN jobs j ON j.id = p.job_id WHERE p.id = ?""", (pid,)).fetchone()
        if not p:
            return
        stage = {
            "selected": "selected",
            "offer_released": "offer_released",
            "offer_accepted": "offer_accepted",
            "awaiting_joining": "notice_period",
            "joined": "joined",
            "successfully_placed": "placed",
            "dropped_before_joining": "withdrawn",
            "rejected": "rejected",
            "replacement_required": "on_hold",
        }.get(p["status"], "selected")
        conn.execute("UPDATE applications SET stage = ?, status = ?, updated_at = ? WHERE id = ?",
                     (stage, _STAGE_STATUS.get(stage, "applied"), _now(), p["application_id"]))
        conn.execute(
            "INSERT INTO application_stage_history (application_id, from_stage, to_stage, note, "
            "changed_by, created_at) VALUES (?,?,?,?,?,?)",
            (p["application_id"], None, stage, f"Placement {p['status'].replace('_', ' ')}",
             user_id, _now()))
        placed, remaining, pct = _closure(conn, p["job_id"], p["openings"])
        job_status = None
        if p["status"] in ("joined", "successfully_placed"):
            job_status = "filled" if remaining == 0 else "partially_filled"
        if job_status:
            conn.execute("UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                         (job_status, _now(), p["job_id"]))
    log_activity(user_id, p["company_id"], "placement_status",
                 f"Placement {p['status'].replace('_', ' ')} — {p['job_title']}",
                 "placement", pid, {"openings": p["openings"], "filled": placed,
                                    "remaining": remaining, "closure_percentage": pct})
    if p["status"] in ("joined", "successfully_placed"):
        notify(p["job_recruiter"], "placement_confirmed", "Candidate joined",
               f"{p['job_title']}: {placed} of {p['openings']} openings filled", "/recruiter#placements")


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

def list_tasks(recruiter_id=None, company_id=None, status="", overdue=False, page=1,
               per_page=50):
    where = "WHERE 1=1"
    args = []
    if recruiter_id:
        where += " AND t.recruiter_id = ?"
        args.append(recruiter_id)
    if company_id:
        where += " AND (t.company_id = ? OR t.company_id IS NULL)"
        args.append(company_id)
    if status:
        where += " AND t.status = ?"
        args.append(status)
    if overdue:
        where += " AND t.status NOT IN ('completed','cancelled') AND t.due_date < ?"
        args.append(_now())
    base = f"""FROM tasks t LEFT JOIN companies c ON c.id = t.company_id
               LEFT JOIN users u ON u.id = t.recruiter_id {where}"""
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n {base}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT t.*, c.name AS company_name, u.nickname AS recruiter_name
                {base} ORDER BY CASE WHEN t.status IN ('completed','cancelled') THEN 1 ELSE 0 END,
                COALESCE(t.due_date,'9999') ASC, t.id DESC LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    items = _rows(rows)
    now = _now()
    for t in items:
        t["overdue"] = bool(t.get("due_date") and t["due_date"] < now
                            and t["status"] not in ("completed", "cancelled"))
    return items, total


def save_task(data, user_id=None, task_id=None):
    title = (data.get("title") or "").strip()
    if not title:
        raise ValueError("task title is required")
    now = _now()
    fields = ("recruiter_id", "company_id", "title", "description", "due_date", "status",
              "priority", "related_to_type", "related_to_id", "reminder_date")
    values = dict(data)
    values["recruiter_id"] = values.get("recruiter_id") or user_id
    values["status"] = values.get("status") or "pending"
    values["priority"] = values.get("priority") or "medium"
    with connect() as conn:
        if task_id:
            sets = ", ".join(f"{f} = ?" for f in fields)
            conn.execute(f"UPDATE tasks SET {sets}, updated_at = ? WHERE id = ?",
                         [values.get(f) for f in fields] + [now, task_id])
            tid = task_id
        else:
            cur = conn.execute(
                f"INSERT INTO tasks ({', '.join(fields)}, created_at, updated_at) "
                f"VALUES ({', '.join('?' * len(fields))}, ?, ?)",
                [values.get(f) for f in fields] + [now, now])
            tid = cur.lastrowid
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    audit(user_id, "task.save", "task", tid)
    return dict(row)


def set_task_status(task_id, status, user_id=None):
    if status not in ("pending", "in_progress", "completed", "cancelled", "overdue"):
        raise ValueError("unknown task status")
    with connect() as conn:
        cur = conn.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                           (status, _now(), task_id))
        if not cur.rowcount:
            return False
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    audit(user_id, "task.status", "task", task_id, {"status": status})
    return dict(row)


def delete_task(task_id, user_id=None):
    with connect() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        ok = cur.rowcount > 0
    if ok:
        audit(user_id, "task.delete", "task", task_id)
    return ok


# --------------------------------------------------------------------------
# notifications, activities, audit
# --------------------------------------------------------------------------

def list_notifications(user_id, unread_only=False, page=1, per_page=30):
    where = "WHERE user_id = ?"
    args = [user_id]
    if unread_only:
        where += " AND is_read = 0"
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM notifications {where}", args).fetchone()["n"]
        unread = conn.execute("SELECT COUNT(*) AS n FROM notifications WHERE user_id = ? AND is_read = 0",
                              (user_id,)).fetchone()["n"]
        rows = conn.execute(
            f"SELECT * FROM notifications {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    return _rows(rows), total, unread


def mark_notification(notification_id, user_id, read=True):
    with connect() as conn:
        cur = conn.execute("UPDATE notifications SET is_read = ? WHERE id = ? AND user_id = ?",
                           (1 if read else 0, notification_id, user_id))
        return cur.rowcount > 0


def mark_all_notifications(user_id):
    with connect() as conn:
        conn.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,))
    return True


def list_activities(company_id=None, candidate_id=None, job_id=None, page=1, per_page=50):
    where = "WHERE 1=1"
    args = []
    if company_id:
        where += " AND a.company_id = ?"
        args.append(company_id)
    if candidate_id:
        where += " AND a.entity_type = 'candidate' AND a.entity_id = ?"
        args.append(candidate_id)
    if job_id:
        where += " AND a.entity_type = 'job' AND a.entity_id = ?"
        args.append(job_id)
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM activities a {where}", args).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT a.*, u.nickname AS user_name FROM activities a
                LEFT JOIN users u ON u.id = a.user_id {where}
                ORDER BY a.created_at DESC LIMIT ? OFFSET ?""",
            args + [per_page, max(0, (page - 1) * per_page)]).fetchall()
    return _rows(rows), total


def list_audit(company_id=None, page=1, per_page=50):
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM audit_logs").fetchone()["n"]
        rows = conn.execute(
            """SELECT a.*, u.email AS user_email FROM audit_logs a
               LEFT JOIN users u ON u.id = a.user_id ORDER BY a.created_at DESC
               LIMIT ? OFFSET ?""", (per_page, max(0, (page - 1) * per_page))).fetchall()
    return _rows(rows), total


# --------------------------------------------------------------------------
# dashboard & analytics
# --------------------------------------------------------------------------

def _scalar(conn, sql, args=()):
    return conn.execute(sql, args).fetchone()[0] or 0


def dashboard(company_id=None, recruiter_id=None):
    where = ""
    args = []
    if company_id:
        where += " AND j.company_id = ?"
        args.append(company_id)
    if recruiter_id:
        where += " AND (a.recruiter_id = ? OR j.recruiter_id = ?)"
        args += [recruiter_id, recruiter_id]
    with connect() as conn:
        jobs = conn.execute(f"SELECT j.* FROM jobs j WHERE 1=1 {where}", args).fetchall()
        active_jobs = [j for j in jobs if j["status"] not in ("closed", "cancelled", "filled")]
        opens = sum((j["openings"] or 0) for j in jobs)
        placed = _scalar(conn, f"""SELECT COUNT(*) FROM placements p JOIN jobs j ON j.id = p.job_id
                                  WHERE p.status IN ('joined','successfully_placed') {where}""", args)
        app_rows = conn.execute(
            f"""SELECT a.stage, COUNT(*) AS n FROM applications a JOIN jobs j ON j.id = a.job_id
                WHERE 1=1 {where} GROUP BY a.stage""", args).fetchall()
        stages = {r["stage"] or "sourced": r["n"] for r in app_rows}
        interviews_scheduled = _scalar(
            conn, f"""SELECT COUNT(*) FROM interviews i JOIN jobs j ON j.id = i.job_id
                      WHERE i.status = 'scheduled' {where}""", args)
        interviews_completed = _scalar(
            conn, f"""SELECT COUNT(*) FROM interviews i JOIN jobs j ON j.id = i.job_id
                      WHERE i.status = 'completed' {where}""", args)
        offers = conn.execute(
            f"""SELECT o.status, COUNT(*) AS n FROM offers o JOIN jobs j ON j.id = o.job_id
                WHERE 1=1 {where} GROUP BY o.status""", args).fetchall()
        offer_map = {r["status"]: r["n"] for r in offers}
        revenue = _scalar(
            conn, f"""SELECT COALESCE(SUM(p.placement_fee),0) FROM placements p
                      JOIN jobs j ON j.id = p.job_id
                      WHERE p.status IN ('joined','successfully_placed') {where}""", args)
        month = time.strftime("%Y-%m")
        monthly = _rows(conn.execute(
            f"""SELECT substr(COALESCE(p.actual_joining_date, p.placement_date, p.created_at),1,7)
                       AS month, COUNT(*) AS placements
                FROM placements p JOIN jobs j ON j.id = p.job_id
                WHERE p.status IN ('joined','successfully_placed') {where}
                GROUP BY month ORDER BY month""", args).fetchall())
        upcoming = _rows(conn.execute(
            f"""SELECT i.id, i.scheduled_at, i.interview_type, i.round, i.status,
                       i.interviewer_name, j.title AS job_title, cp.full_name AS candidate_name
                FROM interviews i JOIN jobs j ON j.id = i.job_id
                LEFT JOIN candidate_profiles cp ON cp.user_id = i.candidate_user_id
                WHERE i.status = 'scheduled' AND i.scheduled_at >= ? {where}
                ORDER BY i.scheduled_at LIMIT 8""",
            [time.strftime("%Y-%m-%dT00:00:00")] + args).fetchall())
        activities = _rows(conn.execute(
            f"""SELECT ac.*, u.nickname AS user_name FROM activities ac
                LEFT JOIN users u ON u.id = ac.user_id
                WHERE 1=1 {('AND ac.company_id = ?' if company_id else '')}
                ORDER BY ac.created_at DESC LIMIT 12""",
            ([company_id] if company_id else [])).fetchall())
        tasks, _ = list_tasks(recruiter_id=recruiter_id, company_id=company_id,
                              status="", overdue=False, page=1, per_page=6)
        pending_tasks = [t for t in tasks if t["status"] not in ("completed", "cancelled")]
        overdue = [t for t in tasks if t["overdue"]]
        funnel = [(k, stages.get(k, 0)) for k in ACTIVE_STAGES]
    return {
        "stats": {
            "active_jobs": len(active_jobs),
            "total_jobs": len(jobs),
            "total_candidates": _scalar(conn, "SELECT COUNT(*) FROM candidate_profiles WHERE COALESCE(status,'active')!='archived'"),
            "in_pipeline": sum(v for k, v in stages.items() if k in ACTIVE_STAGES),
            "interviews_scheduled": interviews_scheduled,
            "interviews_completed": interviews_completed,
            "offers_released": offer_map.get("released", 0) + offer_map.get("accepted", 0)
                               + offer_map.get("declined", 0),
            "offers_accepted": offer_map.get("accepted", 0),
            "placements": placed,
            "open_positions": max(0, opens - placed),
            "closed_positions": placed,
            "pending_followups": len(pending_tasks),
            "overdue_tasks": len(overdue),
            "revenue": revenue,
        },
        "funnel": funnel,
        "stage_distribution": stages,
        "monthly_placements": monthly,
        "interview_success": _interview_success(conn, where, args),
        "job_progress": _job_progress(conn, where, args),
        "upcoming_interviews": upcoming,
        "recent_activities": activities,
        "pending_tasks": pending_tasks,
    }


def _interview_success(conn, where, args):
    total = _scalar(conn, f"""SELECT COUNT(*) FROM interviews i JOIN jobs j ON j.id = i.job_id
                              WHERE i.status IN ('completed','cancelled','no_show') {where}""", args)
    positive = _scalar(conn, f"""SELECT COUNT(*) FROM interview_feedback f
                                 JOIN interviews i ON i.id = f.interview_id
                                 JOIN jobs j ON j.id = i.job_id
                                 WHERE f.recommendation IN ('strong_hire','hire') {where}""", args)
    return {"completed": total, "positive": positive,
            "rate": round((positive / total) * 100, 1) if total else 0.0}


def _job_progress(conn, where, args):
    rows = conn.execute(
        f"""SELECT j.id, j.title, j.openings, j.status, c.name AS company_name,
                   (SELECT COUNT(*) FROM applications a WHERE a.job_id = j.id) AS applications,
                   (SELECT COUNT(*) FROM placements p WHERE p.job_id = j.id
                    AND p.status IN ('joined','successfully_placed')) AS placed
            FROM jobs j LEFT JOIN companies c ON c.id = j.company_id
            WHERE j.status NOT IN ('cancelled') {where}
            ORDER BY j.created_at DESC LIMIT 20""", args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        total = d["openings"] or 0
        d["closure_percentage"] = round((d["placed"] / total) * 100, 1) if total else 0.0
        d["remaining_openings"] = max(0, total - d["placed"])
        out.append(d)
    return out


def recruiter_performance(company_id=None, recruiter_id=None, since=None):
    """Per-recruiter metrics for the leaderboard and individual dashboards.

    ``since`` is an ISO date; monthly/quarterly views just pass a different one.
    Each metric deduplicates by record id, so an application with two interviews
    still counts once for "interviews completed".
    """
    where = "WHERE 1=1"
    args = []
    if company_id:
        where += " AND j.company_id = ?"
        args.append(company_id)
    with connect() as conn:
        recruiters = _rows(conn.execute(
            "SELECT u.id, u.email, u.nickname FROM users u WHERE u.role IN "
            "('recruiter','recruitment_manager','super_admin') ORDER BY u.email").fetchall())
        out = []
        for r in recruiters:
            rid = r["id"]
            if recruiter_id and rid != recruiter_id:
                continue
            date_filter = " AND a.applied_at >= ?" if since else ""
            date_args = [since] if since else []
            apps = _scalar(conn, f"""SELECT COUNT(DISTINCT a.id) FROM applications a
                                     JOIN jobs j ON j.id = a.job_id
                                     WHERE a.recruiter_id = ? {date_filter} {where}""",
                           [rid] + date_args + args)
            candidates = _scalar(conn, "SELECT COUNT(*) FROM candidate_profiles WHERE recruiter_id = ?",
                                 (rid,))
            interviews = _scalar(conn, f"""SELECT COUNT(DISTINCT i.id) FROM interviews i
                                           JOIN jobs j ON j.id = i.job_id
                                           WHERE i.recruiter_id = ? {('AND i.scheduled_at >= ?' if since else '')} {where}""",
                                [rid] + date_args + args)
            completed = _scalar(conn, f"""SELECT COUNT(DISTINCT i.id) FROM interviews i
                                          JOIN jobs j ON j.id = i.job_id
                                          WHERE i.recruiter_id = ? AND i.status = 'completed'
                                          {('AND i.scheduled_at >= ?' if since else '')} {where}""",
                                [rid] + date_args + args)
            offers = _scalar(conn, f"""SELECT COUNT(*) FROM offers o
                                       JOIN jobs j ON j.id = o.job_id
                                       WHERE o.offered_by = ? {('AND o.created_at >= ?' if since else '')} {where}""",
                             [rid] + date_args + args)
            accepted = _scalar(conn, f"""SELECT COUNT(*) FROM offers o
                                         JOIN jobs j ON j.id = o.job_id
                                         WHERE o.offered_by = ? AND o.status = 'accepted'
                                         {('AND o.created_at >= ?' if since else '')} {where}""",
                               [rid] + date_args + args)
            placed = _scalar(conn, f"""SELECT COUNT(*) FROM placements p
                                       JOIN jobs j ON j.id = p.job_id
                                       WHERE p.recruiter_id = ? AND p.status IN ('joined','successfully_placed')
                                       {('AND p.created_at >= ?' if since else '')} {where}""",
                              [rid] + date_args + args)
            jobs_closed = _scalar(conn, f"""SELECT COUNT(*) FROM jobs j
                                            WHERE j.recruiter_id = ? AND j.status = 'filled' {where}""",
                                  [rid] + args)
            out.append({
                "id": rid, "name": r["nickname"] or r["email"].split("@")[0], "email": r["email"],
                "candidates_added": candidates, "applications": apps,
                "interviews_scheduled": interviews, "interviews_completed": completed,
                "offers_released": offers, "offers_accepted": accepted,
                "placements": placed, "jobs_closed": jobs_closed,
                "interview_to_offer": round((offers / completed) * 100, 1) if completed else 0.0,
                "offer_to_joining": round((placed / accepted) * 100, 1) if accepted else 0.0,
            })
    return out


def analytics(company_id=None, recruiter_id=None, period="month"):
    since = None
    now = time.localtime()
    if period == "month":
        since = time.strftime("%Y-%m-01T00:00:00", now)
    elif period == "quarter":
        q_start = ((now.tm_mon - 1) // 3) * 3 + 1
        since = time.strftime(f"%Y-{q_start:02d}-01T00:00:00", now)
    dash = dashboard(company_id=company_id, recruiter_id=recruiter_id)
    perf = recruiter_performance(company_id=company_id, since=since)
    # Average time to fill, measured from first application to confirmed joining.
    with connect() as conn:
        rows = conn.execute(
            """SELECT j.id, MIN(a.applied_at) AS first_app,
                      MIN(COALESCE(p.actual_joining_date, p.placement_date)) AS joined
               FROM jobs j JOIN applications a ON a.job_id = j.id
               JOIN placements p ON p.job_id = j.id
               WHERE p.status IN ('joined','successfully_placed')
               GROUP BY j.id""").fetchall()
        days = []
        from datetime import datetime
        for r in rows:
            try:
                days.append((datetime.fromisoformat(r["joined"]) -
                             datetime.fromisoformat(r["first_app"])).days)
            except (TypeError, ValueError):
                continue
    dash["recruiter_performance"] = perf
    dash["leaderboard"] = sorted(perf, key=lambda x: (x["placements"], x["offers_accepted"]),
                                 reverse=True)
    dash["avg_time_to_fill_days"] = round(sum(days) / len(days), 1) if days else 0.0
    dash["period"] = period
    return dash


# --------------------------------------------------------------------------
# global search & CSV
# --------------------------------------------------------------------------

def global_search(q, company_id=None, limit=8):
    like = _like(q)
    out = {"candidates": [], "jobs": [], "companies": [], "applications": [],
           "interviews": [], "placements": []}
    if not q.strip():
        return out
    with connect() as conn:
        out["candidates"] = _rows(conn.execute(
            """SELECT id, full_name AS name, email, professional_title AS subtitle,
                      location FROM candidate_profiles
               WHERE full_name LIKE ? OR email LIKE ? OR phone LIKE ?
               ORDER BY full_name LIMIT ?""", (like, like, like, limit)).fetchall())
        out["companies"] = _rows(conn.execute(
            "SELECT id, name, industry AS subtitle, location FROM companies "
            "WHERE name LIKE ? OR industry LIKE ? LIMIT ?", (like, like, limit)).fetchall())
        job_where = "WHERE (j.title LIKE ? OR j.job_code LIKE ? OR j.location LIKE ?)"
        args = [like, like, like]
        if company_id:
            job_where += " AND j.company_id = ?"
            args.append(company_id)
        out["jobs"] = _rows(conn.execute(
            f"SELECT j.id, j.title AS name, j.job_code AS subtitle, c.name AS company_name "
            f"FROM jobs j LEFT JOIN companies c ON c.id = j.company_id {job_where} "
            f"ORDER BY j.created_at DESC LIMIT ?", args + [limit]).fetchall())
        app_where = "WHERE (cp.full_name LIKE ? OR j.title LIKE ?)"
        app_args = [like, like]
        if company_id:
            app_where += " AND j.company_id = ?"
            app_args.append(company_id)
        out["applications"] = _rows(conn.execute(
            f"""SELECT a.id, cp.full_name AS name, j.title AS subtitle,
                       COALESCE(a.stage,'sourced') AS stage
                FROM applications a JOIN jobs j ON j.id = a.job_id
                LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id
                {app_where} ORDER BY a.updated_at DESC LIMIT ?""",
            app_args + [limit]).fetchall())
        int_where = "WHERE (cp.full_name LIKE ? OR i.interviewer_name LIKE ?)"
        int_args = [like, like]
        if company_id:
            int_where += " AND j.company_id = ?"
            int_args.append(company_id)
        out["interviews"] = _rows(conn.execute(
            f"""SELECT i.id, cp.full_name AS name, i.scheduled_at AS subtitle, i.status
                FROM interviews i JOIN jobs j ON j.id = i.job_id
                LEFT JOIN candidate_profiles cp ON cp.user_id = i.candidate_user_id
                {int_where} ORDER BY i.scheduled_at DESC LIMIT ?""",
            int_args + [limit]).fetchall())
        pl_where = "WHERE (cp.full_name LIKE ? OR j.title LIKE ?)"
        pl_args = [like, like]
        if company_id:
            pl_where += " AND j.company_id = ?"
            pl_args.append(company_id)
        out["placements"] = _rows(conn.execute(
            f"""SELECT p.id, cp.full_name AS name, j.title AS subtitle, p.status
                FROM placements p JOIN jobs j ON j.id = p.job_id
                LEFT JOIN candidate_profiles cp ON cp.user_id = p.candidate_user_id
                {pl_where} ORDER BY p.created_at DESC LIMIT ?""",
            pl_args + [limit]).fetchall())
    return out


def export_csv(entity, company_id=None):
    """Return (filename, csv_text) for the requested entity."""
    entity = (entity or "").lower()
    approved = {"candidates", "jobs", "applications", "interviews", "placements", "tasks",
                "companies", "offers"}
    if entity not in approved:
        return None, None
    if entity == "candidates":
        rows, _ = list_candidates(company_id=company_id, per_page=100000)
        cols = ["id", "full_name", "email", "phone", "location", "current_company",
                "current_designation", "total_experience", "current_ctc", "expected_ctc",
                "notice_period", "status", "source"]
    elif entity == "jobs":
        rows, _ = list_jobs(company_id=company_id, per_page=100000)
        cols = ["id", "job_code", "title", "company_name", "department", "location",
                "employment_type", "work_mode", "openings", "placed", "remaining_openings",
                "closure_percentage", "status", "priority"]
    elif entity == "applications":
        rows, _ = list_applications(company_id=company_id, per_page=100000)
        cols = ["id", "candidate_name", "candidate_email", "job_title", "company_name",
                "stage", "recruiter_name", "applied_at", "days_in_stage"]
    elif entity == "interviews":
        rows, _ = list_interviews(company_id=company_id, per_page=100000)
        cols = ["id", "candidate_name", "job_title", "interview_type", "round",
                "scheduled_at", "status", "interviewer_name", "meeting_link"]
    elif entity == "placements":
        rows, _ = list_placements(company_id=company_id, per_page=100000)
        cols = ["id", "candidate_name", "job_title", "company_name", "offered_ctc",
                "joining_date", "actual_joining_date", "status", "billing_status",
                "placement_fee"]
    elif entity == "tasks":
        rows, _ = list_tasks(company_id=company_id, per_page=100000)
        cols = ["id", "title", "status", "priority", "due_date", "reminder_date"]
    elif entity == "companies":
        rows, _ = list_companies(per_page=100000)
        cols = ["id", "name", "industry", "location", "contact_email", "contact_phone",
                "status", "job_count", "placement_count"]
    else:
        rows, _ = list_offers(company_id=company_id, per_page=100000)
        cols = ["id", "candidate_name", "job_title", "company_name", "designation",
                "salary", "status", "start_date", "expiry_date"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return f"{entity}-{time.strftime('%Y%m%d')}.csv", buf.getvalue()
