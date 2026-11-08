"""Recruiter portal API: company, jobs, applications, pipeline, candidates, tasks,
interviews and the calendar (events, candidate-proposed slots).

Self-contained on purpose: it depends only on `auth`, not on dashboard.py, so it
cannot drift out of step with that file. dashboard.py mounts it with two lines,
placed straight after `app = FastAPI(...)` so these routes win over the older
same-path recruiter routes further down (first registered route matches):

    import recruiter_api
    app.include_router(recruiter_api.router)

Times are stored as UTC ISO strings ("2026-12-01T10:00:00Z"); the browser sends and
renders them in the viewer's own time zone.
"""
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import auth

router = APIRouter()

JOB_STATUSES = ("draft", "published", "paused", "closed")
APP_STATUSES = ("applied", "ai_screening", "recruiter_review", "shortlisted", "interview_scheduled",
                "interview_completed", "technical_round", "hr_round", "offer_released", "offer_accepted",
                "hired", "rejected", "withdrawn", "on_hold", "offer_declined")
TASK_STATUSES = ("pending", "in_progress", "completed", "cancelled")
INTERVIEW_STATUSES = ("scheduled", "rescheduled", "completed", "cancelled", "no_show")
READY_STATUS = "ready_for_interview"  # candidate-form status the dashboard counts
MAX_SLOTS = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recruiter_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL, start_at TEXT NOT NULL, end_at TEXT, notes TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS interview_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    candidate_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    start_at TEXT NOT NULL, duration_minutes INTEGER NOT NULL DEFAULT 60,
    status TEXT NOT NULL DEFAULT 'proposed', -- proposed, booked, declined
    created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_slots_app ON interview_slots(application_id);
"""
_ready: set = set()


def _db():
    """Connection with this module's tables guaranteed (the DB path changes under tests)."""
    conn = auth.connect()
    if auth.DB_FILE not in _ready:
        conn.executescript(_SCHEMA)
        _ready.add(auth.DB_FILE)
    return conn


def _iso(value: str) -> str:
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date/time")
    if dt.tzinfo is None:  # shortcut: naive input is read as UTC, the UI always sends Z
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _candidate(request: Request) -> dict:
    user = _user(request)
    if user.get("role") not in ("candidate", "super_admin"):
        raise HTTPException(status_code=403, detail="Candidates only")
    return user


def _cid(user: dict) -> int | None:
    p = auth.get_recruiter_profile(user["id"])
    return p.get("company_id") if p else None


def _need_company(user: dict) -> int:
    cid = _cid(user)
    if not cid:
        raise HTTPException(status_code=400, detail="Set up your company first")
    return cid


def _set(conn, table: str, row_id: int, allowed: tuple, data: dict, stamp: bool = True) -> None:
    """Update only the allowed keys the caller sent, so a partial edit never blanks the rest."""
    sets = {k: data[k] for k in allowed if k in data}
    if stamp:
        sets["updated_at"] = auth._now()
    if sets:
        conn.execute(f"UPDATE {table} SET " + ", ".join(f"{k} = ?" for k in sets) + " WHERE id = ?",
                     (*sets.values(), row_id))


# ---- company & profile ----------------------------------------------------
class CompanyIn(BaseModel):
    name: str = ""
    website: str = ""
    industry: str = ""
    description: str = ""
    location: str = ""
    logo_url: str = ""


@router.get("/api/recruiter/company")
async def get_company(user: dict = Depends(_recruiter)):
    profile = auth.get_recruiter_profile(user["id"])
    company = auth.get_company(profile["company_id"]) if profile and profile.get("company_id") else None
    return {"ok": True, "company": company, "profile": profile}


@router.post("/api/recruiter/company")
async def save_company(req: CompanyIn, user: dict = Depends(_recruiter)):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Company name required")
    fields = req.model_dump(exclude={"name"})
    cid = _cid(user)
    with _db() as conn:
        if cid:  # a second save edits the company instead of creating another
            _set(conn, "companies", cid, ("name", "website", "industry", "description", "location", "logo_url"),
                 {"name": req.name.strip(), **fields}, stamp=False)
        else:
            cid = auth.create_company(req.name.strip(), created_by=user["id"], **fields)["id"]
            conn.execute(
                "INSERT INTO recruiter_profiles (user_id, company_id, created_at) VALUES (?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET company_id = excluded.company_id",
                (user["id"], cid, auth._now()))
    return {"ok": True, "company": auth.get_company(cid)}


class ProfileIn(BaseModel):
    full_name: str = ""
    title: str = ""
    phone: str = ""


def _identity(user: dict) -> dict:
    return {"nickname": user.get("nickname") or "", "display_name": auth.display_name(user),
            "avatar": user.get("avatar") or auth.DEFAULT_AVATAR, "email": user["email"]}


@router.get("/api/recruiter/profile")
async def get_profile(user: dict = Depends(_recruiter)):
    p = auth.get_recruiter_profile(user["id"]) or {}
    return {"ok": True, "profile": {**p, "full_name": user.get("nickname") or ""}, "identity": _identity(user)}


@router.post("/api/recruiter/profile")
async def save_profile(req: ProfileIn, user: dict = Depends(_recruiter)):
    with _db() as conn:
        conn.execute(
            "INSERT INTO recruiter_profiles (user_id, title, phone, created_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET title = excluded.title, phone = excluded.phone",
            (user["id"], req.title.strip(), req.phone.strip(), auth._now()))
    # The full name is the display name shown in the header and the account menu.
    user = auth.set_identity(user["id"], nickname=req.full_name) if req.full_name.strip() else user
    return {"ok": True, "profile": auth.get_recruiter_profile(user["id"]), "identity": _identity(user)}


# ---- jobs -----------------------------------------------------------------
class JobIn(BaseModel):
    title: str | None = None
    department: str | None = None
    description: str | None = None
    required_skills: list | None = None
    preferred_skills: list | None = None
    min_experience: int | None = None
    max_experience: int | None = None
    education_req: str | None = None
    employment_type: str | None = None
    work_mode: str | None = None
    location: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    currency: str | None = None
    openings: int | None = None
    application_deadline: str | None = None
    status: str | None = None


class StatusIn(BaseModel):
    status: str = ""
    note: str = ""


_JOB_FIELDS = ("title", "department", "description", "min_experience", "max_experience", "education_req",
               "employment_type", "work_mode", "location", "salary_min", "salary_max", "currency", "openings",
               "application_deadline", "status", "required_skills", "preferred_skills")


def _own_job(user: dict, job_id: int) -> dict:
    cid = _cid(user)
    with _db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not cid or row["company_id"] != cid:
        raise HTTPException(status_code=404, detail="Job not found")
    return dict(row)


def _get_job(job_id: int) -> dict:
    with _db() as conn:
        return dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


@router.get("/api/recruiter/jobs")
async def list_jobs(status: str | None = None, user: dict = Depends(_recruiter)):
    cid = _cid(user)
    return {"ok": True, "jobs": auth.list_jobs(company_id=cid, status=status) if cid else []}


@router.post("/api/recruiter/jobs")
async def create_job(req: JobIn, user: dict = Depends(_recruiter)):
    cid = _need_company(user)
    data = req.model_dump(exclude_none=True)
    if not (data.get("title") or "").strip() or not (data.get("description") or "").strip():
        raise HTTPException(status_code=400, detail="Title and description required")
    if data.get("status", "draft") not in JOB_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    return {"ok": True, "job": auth.create_job(cid, user["id"], data)}


@router.post("/api/recruiter/jobs/{job_id}/update")
async def update_job(job_id: int, req: JobIn, user: dict = Depends(_recruiter)):
    _own_job(user, job_id)
    data = req.model_dump(exclude_unset=True)
    for k in ("title", "description"):
        if k in data and not (data[k] or "").strip():
            raise HTTPException(status_code=400, detail="Title and description required")
    if "status" in data and data["status"] not in JOB_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    for k in ("required_skills", "preferred_skills"):
        if k in data:
            data[k] = json.dumps(data[k] or [])
    with _db() as conn:
        _set(conn, "jobs", job_id, _JOB_FIELDS, data)
    return {"ok": True, "job": _get_job(job_id)}


@router.post("/api/recruiter/jobs/{job_id}/status")
async def set_job_status(job_id: int, req: StatusIn, user: dict = Depends(_recruiter)):
    _own_job(user, job_id)
    if req.status not in JOB_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    with _db() as conn:
        _set(conn, "jobs", job_id, ("status",), {"status": req.status})
    return {"ok": True, "job": _get_job(job_id)}


# ---- applications & dashboard ---------------------------------------------
def _applications(company_id: int, job_id: int | None = None) -> list[dict]:
    sql = ("SELECT a.*, j.title AS job_title, u.email AS candidate_email, "
           "COALESCE(cp.full_name, u.nickname, u.email) AS candidate_name, "
           "cp.professional_title AS candidate_title, cp.location AS candidate_location "
           "FROM applications a JOIN jobs j ON j.id = a.job_id JOIN users u ON u.id = a.candidate_user_id "
           "LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id WHERE j.company_id = ?")
    args = [company_id]
    if job_id:
        sql += " AND a.job_id = ?"
        args.append(job_id)
    with _db() as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY a.applied_at DESC", args)]


@router.get("/api/recruiter/applications")
async def list_applications(job_id: int | None = None, user: dict = Depends(_recruiter)):
    cid = _cid(user)
    return {"ok": True, "applications": _applications(cid, job_id) if cid else []}


@router.post("/api/recruiter/applications/{app_id}/status")
async def set_application_status(app_id: int, req: StatusIn, user: dict = Depends(_recruiter)):
    if req.status not in APP_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    cid = _need_company(user)
    with _db() as conn:
        n = conn.execute(
            "UPDATE applications SET status = ?, updated_at = ? WHERE id = ? AND job_id IN "
            "(SELECT id FROM jobs WHERE company_id = ?)", (req.status, auth._now(), app_id, cid)).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Application not found")
    return {"ok": True}


@router.get("/api/recruiter/dashboard/stats")
async def dashboard_stats(user: dict = Depends(_recruiter)):
    cid = _cid(user)
    jobs = auth.list_jobs(company_id=cid) if cid else []
    apps = _applications(cid) if cid else []
    n = lambda st: sum(1 for a in apps if a["status"] == st)
    with _db() as conn:
        cands = [r["status"] for r in conn.execute("SELECT status FROM candidates WHERE user_id = ?", (user["id"],))]
        pending = conn.execute(
            "SELECT COUNT(*) FROM interview_slots s JOIN applications a ON a.id = s.application_id "
            "JOIN jobs j ON j.id = a.job_id WHERE s.status = 'proposed' AND j.company_id = ?", (cid,)).fetchone()[0] if cid else 0
    stats = {
        "total_jobs": len(jobs), "active_jobs": sum(1 for j in jobs if j["status"] == "published"),
        "total_applications": len(apps), "new_applications": n("applied"), "shortlisted": n("shortlisted"),
        "interview_scheduled": n("interview_scheduled"), "offers_released": n("offer_released"), "hired": n("hired"),
        "candidates": len(cands), "candidates_ready": cands.count(READY_STATUS), "pending_slots": pending,
    }
    return {"ok": True, "stats": stats, "recent_applications": apps[:10], "jobs": jobs}


# ---- candidates (the recruiter's own list; replaces the broken duplicate routes) ----
class CandidateIn(BaseModel):
    id: str | None = None
    name: str = ""
    current_location: str = ""
    expected_ctc: str = ""
    current_ctc: str = ""
    email: str = ""
    phone: str = ""
    availability: str = ""
    status: str = "working"
    notes: str = ""


@router.get("/api/candidates")
async def list_candidates(status: str | None = None, user: dict = Depends(_recruiter)):
    sql, args = "SELECT * FROM candidates WHERE user_id = ?", [user["id"]]
    if status:
        sql += " AND status = ?"
        args.append(status)
    with _db() as conn:
        return {"ok": True, "candidates": [dict(r) for r in conn.execute(sql + " ORDER BY created_at DESC", args)]}


@router.post("/api/candidates")
async def save_candidate(req: CandidateIn, user: dict = Depends(_recruiter)):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    now, cid = auth._now(), req.id or auth.secrets.token_hex(6)
    vals = (req.name.strip(), req.current_location.strip() or None, req.expected_ctc.strip() or None,
            req.current_ctc.strip() or None, req.email.strip() or None, req.phone.strip() or None,
            req.availability.strip() or None, req.status or "working", req.notes.strip() or None, now)
    with _db() as conn:
        if conn.execute("SELECT 1 FROM candidates WHERE user_id = ? AND id = ?", (user["id"], cid)).fetchone():
            conn.execute("UPDATE candidates SET name=?, current_location=?, expected_ctc=?, current_ctc=?, email=?, "
                         "phone=?, availability=?, status=?, notes=?, updated_at=? WHERE user_id=? AND id=?",
                         (*vals, user["id"], cid))
        else:
            conn.execute("INSERT INTO candidates (name, current_location, expected_ctc, current_ctc, email, phone, "
                         "availability, status, notes, updated_at, user_id, id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (*vals, user["id"], cid, now))
    return {"ok": True, "id": cid}


@router.delete("/api/candidates/{cand_id}")
async def delete_candidate(cand_id: str, user: dict = Depends(_recruiter)):
    with _db() as conn:
        n = conn.execute("DELETE FROM candidates WHERE user_id = ? AND id = ?", (user["id"], cand_id)).rowcount
    return {"ok": bool(n)}


# ---- tasks ----------------------------------------------------------------
class TaskIn(BaseModel):
    title: str | None = None
    description: str | None = None
    due_date: str | None = None
    status: str | None = None
    priority: str | None = None


@router.get("/api/recruiter/tasks")
async def list_tasks(user: dict = Depends(_recruiter)):
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE recruiter_id = ? ORDER BY (status IN ('completed','cancelled')), "
            "COALESCE(due_date, '9999'), id DESC", (user["id"],)).fetchall()
    return {"ok": True, "tasks": [dict(r) for r in rows]}


@router.post("/api/recruiter/tasks")
async def create_task(req: TaskIn, user: dict = Depends(_recruiter)):
    if not (req.title or "").strip():
        raise HTTPException(status_code=400, detail="Title required")
    now = auth._now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (recruiter_id, company_id, title, description, due_date, status, priority, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (user["id"], _cid(user), req.title.strip(), req.description, req.due_date or None,
             "pending", req.priority or "medium", now, now))
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (cur.lastrowid,)).fetchone()
    return {"ok": True, "task": dict(row)}


@router.post("/api/recruiter/tasks/{task_id}/update")
async def update_task(task_id: int, req: TaskIn, user: dict = Depends(_recruiter)):
    data = req.model_dump(exclude_unset=True)
    if "status" in data and data["status"] not in TASK_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    with _db() as conn:
        if not conn.execute("SELECT 1 FROM tasks WHERE id = ? AND recruiter_id = ?", (task_id, user["id"])).fetchone():
            raise HTTPException(status_code=404, detail="Task not found")
        _set(conn, "tasks", task_id, ("title", "description", "due_date", "status", "priority"), data)
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return {"ok": True, "task": dict(row)}


@router.delete("/api/recruiter/tasks/{task_id}")
async def delete_task(task_id: int, user: dict = Depends(_recruiter)):
    with _db() as conn:
        n = conn.execute("DELETE FROM tasks WHERE id = ? AND recruiter_id = ?", (task_id, user["id"])).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True}


# ---- interviews -----------------------------------------------------------
class InterviewIn(BaseModel):
    application_id: int | None = None
    interview_type: str | None = None
    scheduled_at: str | None = None
    duration_minutes: int | None = None
    meeting_link: str | None = None
    location: str | None = None
    instructions: str | None = None
    status: str | None = None


def _interviews(company_id: int) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT i.*, j.title AS job_title, u.email AS candidate_email, "
            "COALESCE(cp.full_name, u.nickname, u.email) AS candidate_name "
            "FROM interviews i JOIN jobs j ON j.id = i.job_id JOIN users u ON u.id = i.candidate_user_id "
            "LEFT JOIN candidate_profiles cp ON cp.user_id = i.candidate_user_id "
            "WHERE j.company_id = ? ORDER BY i.scheduled_at", (company_id,)).fetchall()
    return [dict(r) for r in rows]


def _book(conn, company_id: int, recruiter_id: int, application_id: int, scheduled_at: str, d: dict) -> dict | None:
    """Create the interview and move the application to interview_scheduled; None if it is not this company's."""
    app = conn.execute(
        "SELECT a.id, a.job_id, a.candidate_user_id FROM applications a JOIN jobs j ON j.id = a.job_id "
        "WHERE a.id = ? AND j.company_id = ?", (application_id, company_id)).fetchone()
    if not app:
        return None
    now = auth._now()
    cur = conn.execute(
        "INSERT INTO interviews (application_id, job_id, candidate_user_id, recruiter_id, interview_type, scheduled_at, "
        "duration_minutes, meeting_link, location, instructions, status, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,'scheduled',?,?)",
        (app["id"], app["job_id"], app["candidate_user_id"], recruiter_id, d.get("interview_type") or "phone_screening",
         scheduled_at, d.get("duration_minutes") or 60, d.get("meeting_link") or None, d.get("location") or None,
         d.get("instructions") or None, now, now))
    conn.execute("UPDATE applications SET status = 'interview_scheduled', updated_at = ? WHERE id = ?", (now, app["id"]))
    return dict(conn.execute("SELECT * FROM interviews WHERE id = ?", (cur.lastrowid,)).fetchone())


@router.get("/api/recruiter/interviews")
async def list_interviews(user: dict = Depends(_recruiter)):
    cid = _cid(user)
    return {"ok": True, "interviews": _interviews(cid) if cid else []}


@router.post("/api/recruiter/interviews")
async def create_interview(req: InterviewIn, user: dict = Depends(_recruiter)):
    if not req.application_id or not req.scheduled_at:
        raise HTTPException(status_code=400, detail="Application and time required")
    cid = _need_company(user)
    with _db() as conn:
        iv = _book(conn, cid, user["id"], req.application_id, _iso(req.scheduled_at), req.model_dump())
    if not iv:
        raise HTTPException(status_code=404, detail="Application not found")
    return {"ok": True, "interview": iv}


@router.post("/api/recruiter/interviews/{interview_id}/update")
async def update_interview(interview_id: int, req: InterviewIn, user: dict = Depends(_recruiter)):
    data = req.model_dump(exclude_unset=True, exclude={"application_id"})
    if "status" in data and data["status"] not in INTERVIEW_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status")
    if "scheduled_at" in data:
        data["scheduled_at"] = _iso(data["scheduled_at"])
    cid = _need_company(user)
    with _db() as conn:
        if not conn.execute("SELECT 1 FROM interviews i JOIN jobs j ON j.id = i.job_id WHERE i.id = ? AND j.company_id = ?",
                            (interview_id, cid)).fetchone():
            raise HTTPException(status_code=404, detail="Interview not found")
        _set(conn, "interviews", interview_id, ("interview_type", "scheduled_at", "duration_minutes", "meeting_link",
                                                 "location", "instructions", "status"), data)
        row = conn.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,)).fetchone()
    return {"ok": True, "interview": dict(row)}


# ---- calendar: interviews + candidate-proposed slots + own events ----------
class EventIn(BaseModel):
    title: str = ""
    start_at: str = ""
    end_at: str | None = None
    notes: str | None = None


@router.get("/api/recruiter/calendar")
async def calendar(user: dict = Depends(_recruiter)):
    cid = _cid(user)
    with _db() as conn:
        events = [dict(r) for r in conn.execute(
            "SELECT * FROM calendar_events WHERE recruiter_id = ? ORDER BY start_at", (user["id"],))]
        slots = [dict(r) for r in conn.execute(
            "SELECT s.*, j.title AS job_title, u.email AS candidate_email, "
            "COALESCE(cp.full_name, u.nickname, u.email) AS candidate_name "
            "FROM interview_slots s JOIN applications a ON a.id = s.application_id JOIN jobs j ON j.id = a.job_id "
            "JOIN users u ON u.id = s.candidate_user_id LEFT JOIN candidate_profiles cp ON cp.user_id = s.candidate_user_id "
            "WHERE s.status = 'proposed' AND j.company_id = ? ORDER BY s.start_at", (cid,))] if cid else []
    return {"ok": True, "events": events, "slots": slots, "interviews": _interviews(cid) if cid else []}


@router.post("/api/recruiter/events")
async def create_event(req: EventIn, user: dict = Depends(_recruiter)):
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title required")
    with _db() as conn:
        cur = conn.execute("INSERT INTO calendar_events (recruiter_id, title, start_at, end_at, notes, created_at) VALUES (?,?,?,?,?,?)",
                           (user["id"], req.title.strip(), _iso(req.start_at), _iso(req.end_at) if req.end_at else None,
                            req.notes, auth._now()))
        row = conn.execute("SELECT * FROM calendar_events WHERE id = ?", (cur.lastrowid,)).fetchone()
    return {"ok": True, "event": dict(row)}


@router.delete("/api/recruiter/events/{event_id}")
async def delete_event(event_id: int, user: dict = Depends(_recruiter)):
    with _db() as conn:
        n = conn.execute("DELETE FROM calendar_events WHERE id = ? AND recruiter_id = ?", (event_id, user["id"])).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"ok": True}


def _own_slot(conn, company_id: int, slot_id: int):
    return conn.execute(
        "SELECT s.* FROM interview_slots s JOIN applications a ON a.id = s.application_id JOIN jobs j ON j.id = a.job_id "
        "WHERE s.id = ? AND j.company_id = ? AND s.status = 'proposed'", (slot_id, company_id)).fetchone()


@router.post("/api/recruiter/slots/{slot_id}/confirm")
async def confirm_slot(slot_id: int, req: InterviewIn, user: dict = Depends(_recruiter)):
    """Accept a candidate's slot: books the interview and releases their other proposed slots."""
    cid = _need_company(user)
    with _db() as conn:
        slot = _own_slot(conn, cid, slot_id)
        if not slot:
            raise HTTPException(status_code=404, detail="Slot not found or already handled")
        d = req.model_dump(exclude_unset=True)
        d["duration_minutes"] = d.get("duration_minutes") or slot["duration_minutes"]
        iv = _book(conn, cid, user["id"], slot["application_id"], slot["start_at"], d)
        conn.execute("UPDATE interview_slots SET status = CASE WHEN id = ? THEN 'booked' ELSE 'declined' END "
                     "WHERE application_id = ? AND status = 'proposed'", (slot_id, slot["application_id"]))
    return {"ok": True, "interview": iv}


@router.post("/api/recruiter/slots/{slot_id}/decline")
async def decline_slot(slot_id: int, user: dict = Depends(_recruiter)):
    cid = _need_company(user)
    with _db() as conn:
        if not _own_slot(conn, cid, slot_id):
            raise HTTPException(status_code=404, detail="Slot not found or already handled")
        conn.execute("UPDATE interview_slots SET status = 'declined' WHERE id = ?", (slot_id,))
    return {"ok": True}


# ---- candidate side: see my applications, offer interview slots ------------
class SlotsIn(BaseModel):
    slots: list[dict] = []  # [{start_at, duration_minutes}]


@router.get("/api/candidate/applications")
async def my_applications(user: dict = Depends(_candidate)):
    with _db() as conn:
        apps = [dict(r) for r in conn.execute(
            "SELECT a.id, a.status, a.applied_at, j.title AS job_title, c.name AS company_name "
            "FROM applications a JOIN jobs j ON j.id = a.job_id LEFT JOIN companies c ON c.id = j.company_id "
            "WHERE a.candidate_user_id = ? ORDER BY a.applied_at DESC", (user["id"],))]
        for a in apps:
            a["slots"] = [dict(r) for r in conn.execute(
                "SELECT id, start_at, duration_minutes, status FROM interview_slots WHERE application_id = ? ORDER BY start_at", (a["id"],))]
            iv = conn.execute("SELECT scheduled_at, duration_minutes, meeting_link, location, status FROM interviews "
                              "WHERE application_id = ? AND status IN ('scheduled','rescheduled') ORDER BY scheduled_at LIMIT 1",
                              (a["id"],)).fetchone()
            a["interview"] = dict(iv) if iv else None
    return {"ok": True, "applications": apps}


@router.post("/api/candidate/applications/{app_id}/slots")
async def propose_slots(app_id: int, req: SlotsIn, user: dict = Depends(_candidate)):
    if not req.slots or len(req.slots) > MAX_SLOTS:
        raise HTTPException(status_code=400, detail=f"Send 1 to {MAX_SLOTS} slots")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    clean = []
    for s in req.slots:
        start = _iso(s.get("start_at", ""))
        if start <= now_iso:
            raise HTTPException(status_code=400, detail="Slots must be in the future")
        dur = int(s.get("duration_minutes") or 60)
        clean.append((start, max(15, min(dur, 480))))
    with _db() as conn:
        if not conn.execute("SELECT 1 FROM applications WHERE id = ? AND candidate_user_id = ?", (app_id, user["id"])).fetchone():
            raise HTTPException(status_code=404, detail="Application not found")
        for start, dur in clean:
            conn.execute("INSERT INTO interview_slots (application_id, candidate_user_id, start_at, duration_minutes, created_at) "
                         "SELECT ?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM interview_slots WHERE application_id = ? "
                         "AND start_at = ? AND status = 'proposed')",
                         (app_id, user["id"], start, dur, auth._now(), app_id, start))
    return {"ok": True}


@router.delete("/api/candidate/slots/{slot_id}")
async def withdraw_slot(slot_id: int, user: dict = Depends(_candidate)):
    with _db() as conn:
        n = conn.execute("DELETE FROM interview_slots WHERE id = ? AND candidate_user_id = ? AND status = 'proposed'",
                         (slot_id, user["id"])).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Slot not found")
    return {"ok": True}
