"""
auth.py — accounts, sessions and outbound email for JobScope.

Everything a person owns lives here: their account row, the CV "profiles" (the
sidebar sessions that used to share one profiles.json), the login sessions, and
the single-use links used to verify an address or reset a password.

Storage is SQLite from the standard library — no ORM, no extra dependency. The
project already reads Gmail with imaplib rather than adding a mail library, so
this stays in the same spirit: smtplib for sending, hashlib for hashing.

Passwords
    hashlib.scrypt with a per-password random salt. scrypt is memory-hard, so
    a stolen database is expensive to crack on GPUs, and it is in the stdlib
    (bcrypt/argon2 would mean a new dependency). The stored format is
    ``scrypt$n$r$p$<salt b64>$<hash b64>`` so the parameters travel with the
    hash and can be raised later without invalidating old passwords.

Sessions
    A random 32-byte token is handed to the browser in an HttpOnly cookie; only
    its SHA-256 digest is stored, so a leaked database cannot be replayed as a
    login. The session row also records which CV profile that browser has open,
    which is why "active profile" is per-session rather than a process-wide
    variable.
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(HERE, "users.db")

SESSION_COOKIE = "js_session"
SESSION_DAYS = 30

# Idle timeout. The absolute SESSION_DAYS cap answers "how long may this cookie
# ever live"; this answers "how long may it live with nobody using it", which is
# the case that matters on a shared or walked-away-from machine. A session idle
# past this is treated as gone: session_user returns None and the row is
# deleted, so the next sign-in starts from nothing rather than inheriting the
# previous session's open CV.
SESSION_IDLE_MINUTES = 25
SESSION_EXTEND_MINUTES = 5
SESSION_IDLE_SECONDS = SESSION_IDLE_MINUTES * 60
SESSION_EXTEND_SECONDS = SESSION_EXTEND_MINUTES * 60
VERIFY_TOKEN_HOURS = 24
RESET_TOKEN_HOURS = 2

# scrypt cost. n=2**14 with r=8 needs ~16 MB per hash: heavy enough to make
# offline cracking expensive, light enough that a login stays snappy.
SCRYPT_N = 1 << 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    verified_at   TEXT,
    is_owner      INTEGER NOT NULL DEFAULT 0,
    -- Display identity. The dashboard greets a nickname and shows a chosen
    -- avatar instead of the login address, which is the one field of this app
    -- that gets read aloud on a screen share. Both are nullable and fall back to
    -- the address: an account created before these columns existed keeps
    -- working untouched (see _migrate).
    nickname      TEXT,
    avatar        TEXT,
    role          TEXT    NOT NULL DEFAULT 'candidate'  -- 'candidate' or 'recruiter'
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash    TEXT    PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at    TEXT    NOT NULL,
    expires_at    TEXT    NOT NULL,
    active_profile TEXT,
    -- When the cookie was last presented, as epoch seconds. The idle timeout is
    -- measured from this, not from expires_at, so an active user is never
    -- logged out mid-task and a stale one is dropped even though their absolute
    -- expiry is weeks off. Epoch rather than the ISO timestamps the rest of this
    -- schema uses, because the UI needs the remaining seconds and subtracting
    -- strings is how "19:58 left" happens.
    last_seen     REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS google_identities (
    subject TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS google_account_states (
    state_hash TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_hash TEXT,
    nonce TEXT NOT NULL,
    verifier TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS google_gmail_connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    client_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    email TEXT NOT NULL,
    refresh_token TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS google_gmail_alerts (
    connection_id INTEGER NOT NULL REFERENCES google_gmail_connections(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    sender TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    listings TEXT NOT NULL,
    PRIMARY KEY (connection_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_google_gmail_sent ON google_gmail_alerts(connection_id, sent_at);

-- One-time links: 'verify' to confirm an address, 'reset' to set a new
-- password. Storing only the digest means the database alone is useless.
CREATE TABLE IF NOT EXISTS email_tokens (
    token_hash TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    purpose    TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    expires_at TEXT    NOT NULL,
    used_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON email_tokens(user_id);

-- The sidebar CV sessions, one row per profile, scoped to its owner. `data`
-- holds the same JSON shape profiles.json used, so a migration is a copy and
-- the rest of the app keeps reading the fields it always did.
CREATE TABLE IF NOT EXISTS profiles (
    id         TEXT    NOT NULL,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name       TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    is_default INTEGER NOT NULL DEFAULT 0,
    data       TEXT    NOT NULL,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id);

-- The candidate's own document library: everything they upload for tailoring
-- (CVs, project write-ups, certificates). Retrieval reads ONLY from this table,
-- so a generated resume can never assert a claim that is not in a document the
-- user actually uploaded - the no-invention rule holds by construction rather
-- than by trusting a generator not to embellish.
--
-- Chunks are derived at read time instead of being stored: a stored index can
-- drift out of sync with `text` after an edit, and at this corpus size (a
-- handful of documents, hundreds of chunks) re-chunking costs milliseconds.
CREATE TABLE IF NOT EXISTS rag_documents (
    id         TEXT    NOT NULL,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name       TEXT    NOT NULL,
    kind       TEXT    NOT NULL DEFAULT 'cv',
    text       TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX IF NOT EXISTS idx_rag_docs_user ON rag_documents(user_id);

-- Recruitment / candidate tracking (legacy)
CREATE TABLE IF NOT EXISTS candidates (
    id                TEXT    NOT NULL,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name              TEXT    NOT NULL,
    current_location  TEXT,
    expected_ctc      TEXT,
    current_ctc       TEXT,
    email             TEXT,
    phone             TEXT,
    availability      TEXT,
    status            TEXT    NOT NULL DEFAULT 'working', -- 'working' or 'recruiter_entry'
    notes             TEXT,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX IF NOT EXISTS idx_candidates_user ON candidates(user_id);

-- Recruiter CRM Schema
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    website     TEXT,
    logo_url    TEXT,
    industry    TEXT,
    description TEXT,
    location    TEXT,
    created_at  TEXT    NOT NULL,
    created_by  INTEGER REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS recruiter_profiles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    company_id   INTEGER REFERENCES companies(id),
    title        TEXT,
    phone        TEXT,
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_profiles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    full_name         TEXT,
    professional_title TEXT,
    location          TEXT,
    phone             TEXT,
    profile_photo     TEXT,
    professional_summary TEXT,
    skills            TEXT,  -- JSON array
    experience        TEXT,  -- JSON array
    education         TEXT,  -- JSON array
    certifications    TEXT,  -- JSON array
    projects          TEXT,  -- JSON array
    portfolio_links   TEXT,  -- JSON array
    availability      TEXT,
    profile_completeness INTEGER DEFAULT 0,
    updated_at        TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id        INTEGER REFERENCES companies(id),
    recruiter_id      INTEGER REFERENCES users(id),
    title             TEXT    NOT NULL,
    department        TEXT,
    description       TEXT    NOT NULL,
    required_skills   TEXT,  -- JSON array
    preferred_skills  TEXT,  -- JSON array
    min_experience    INTEGER,
    max_experience    INTEGER,
    education_req     TEXT,
    employment_type   TEXT,  -- full-time, part-time, contract, internship
    work_mode         TEXT,  -- remote, hybrid, onsite
    location          TEXT,
    salary_min        INTEGER,
    salary_max        INTEGER,
    currency          TEXT DEFAULT 'INR',
    openings          INTEGER DEFAULT 1,
    application_deadline TEXT,
    status            TEXT    NOT NULL DEFAULT 'draft', -- draft, published, paused, closed
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS applications (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    candidate_profile_id INTEGER REFERENCES candidate_profiles(id),
    resume_id         TEXT,
    status            TEXT    NOT NULL DEFAULT 'applied', -- applied, ai_screening, recruiter_review, shortlisted, interview_scheduled, interview_completed, technical_round, hr_round, offer_released, offer_accepted, hired, rejected, withdrawn, on_hold, offer_declined
    rejection_reason  TEXT,
    cover_letter      TEXT,
    applied_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    UNIQUE(job_id, candidate_user_id)
);
CREATE INDEX IF NOT EXISTS idx_applications_job ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_applications_candidate ON applications(candidate_user_id);

CREATE TABLE IF NOT EXISTS application_status_emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    attempted_at REAL
);
CREATE INDEX IF NOT EXISTS idx_application_status_emails ON application_status_emails(application_id, id);

CREATE TABLE IF NOT EXISTS talent_pools (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER REFERENCES companies(id),
    recruiter_id INTEGER REFERENCES users(id),
    name        TEXT    NOT NULL,
    description TEXT,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS talent_pool_members (
    pool_id     INTEGER NOT NULL REFERENCES talent_pools(id) ON DELETE CASCADE,
    candidate_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    added_at    TEXT    NOT NULL,
    added_by    INTEGER REFERENCES users(id),
    PRIMARY KEY (pool_id, candidate_user_id)
);

CREATE TABLE IF NOT EXISTS interviews (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id    INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    job_id            INTEGER REFERENCES jobs(id),
    candidate_user_id INTEGER REFERENCES users(id),
    recruiter_id      INTEGER REFERENCES users(id),
    interview_type    TEXT    NOT NULL, -- phone_screening, technical_interview, hr_interview, managerial_interview, final_interview
    scheduled_at      TEXT    NOT NULL,
    duration_minutes  INTEGER DEFAULT 60,
    meeting_link      TEXT,
    location          TEXT,
    instructions      TEXT,
    status            TEXT    NOT NULL DEFAULT 'scheduled', -- scheduled, rescheduled, completed, cancelled, no_show
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS interview_feedback (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id      INTEGER NOT NULL REFERENCES interviews(id) ON DELETE CASCADE,
    interviewer_id    INTEGER REFERENCES users(id),
    rating            INTEGER,
    technical_score   INTEGER,
    communication_score INTEGER,
    overall_score     INTEGER,
    strengths         TEXT,
    improvements     TEXT,
    comments          TEXT,
    recommendation    TEXT, -- hire, maybe, no
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS offers (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id    INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    job_id            INTEGER REFERENCES jobs(id),
    candidate_user_id INTEGER REFERENCES users(id),
    offered_by        INTEGER REFERENCES users(id),
    salary            INTEGER,
    currency          TEXT DEFAULT 'INR',
    offer_letter      TEXT,
    benefits          TEXT,
    start_date        TEXT,
    expiry_date       TEXT,
    status            TEXT    NOT NULL DEFAULT 'draft', -- draft, released, accepted, declined, withdrawn
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS recruiter_notes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recruiter_id      INTEGER NOT NULL REFERENCES users(id),
    candidate_user_id INTEGER NOT NULL REFERENCES users(id),
    application_id    INTEGER REFERENCES applications(id),
    note              TEXT    NOT NULL,
    is_private        INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS communications (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recruiter_id      INTEGER REFERENCES users(id),
    candidate_user_id INTEGER REFERENCES users(id),
    application_id    INTEGER REFERENCES applications(id),
    type              TEXT    NOT NULL, -- email, message, call
    subject           TEXT,
    message           TEXT    NOT NULL,
    direction         TEXT    NOT NULL, -- outbound, inbound
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS activities (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER REFERENCES users(id),
    company_id        INTEGER REFERENCES companies(id),
    type              TEXT    NOT NULL,
    description       TEXT    NOT NULL,
    entity_type       TEXT,
    entity_id         INTEGER,
    metadata          TEXT, -- JSON
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recruiter_id      INTEGER NOT NULL REFERENCES users(id),
    company_id        INTEGER REFERENCES companies(id),
    title             TEXT    NOT NULL,
    description       TEXT,
    due_date          TEXT,
    status            TEXT    NOT NULL DEFAULT 'pending', -- pending, in_progress, completed, cancelled
    priority          TEXT    DEFAULT 'medium', -- low, medium, high
    related_to_type   TEXT,
    related_to_id     INTEGER,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(id),
    type              TEXT    NOT NULL,
    title             TEXT    NOT NULL,
    message           TEXT    NOT NULL,
    link              TEXT,
    is_read           INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL
);

-- --------------------------------------------------------------------------
-- Recruiter ATS: normalised records behind the candidate portfolio.
--
-- The master candidate lives in candidate_profiles; everything that only makes
-- sense for one job hangs off applications. That split is what lets one person
-- sit in Screening for one job and Interview for another.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS company_contacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    designation TEXT,
    email       TEXT,
    phone       TEXT,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    notes       TEXT,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contacts_company ON company_contacts(company_id);

CREATE TABLE IF NOT EXISTS candidate_documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidate_profiles(id) ON DELETE CASCADE,
    name         TEXT    NOT NULL,
    kind         TEXT    NOT NULL DEFAULT 'resume', -- resume, certificate, identity, other
    mime         TEXT,
    size         INTEGER NOT NULL DEFAULT 0,
    storage_path TEXT,       -- relative name under UPLOAD_DIR; NULL when text only
    version      INTEGER NOT NULL DEFAULT 1,
    uploaded_by  INTEGER REFERENCES users(id),
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_candidate ON candidate_documents(candidate_id);

CREATE TABLE IF NOT EXISTS candidate_experiences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  INTEGER NOT NULL REFERENCES candidate_profiles(id) ON DELETE CASCADE,
    company_name  TEXT    NOT NULL,
    designation   TEXT,
    start_date    TEXT,
    end_date      TEXT,
    responsibilities TEXT,
    technologies  TEXT,
    reason_for_leaving TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiences_candidate ON candidate_experiences(candidate_id);

CREATE TABLE IF NOT EXISTS candidate_education (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id   INTEGER NOT NULL REFERENCES candidate_profiles(id) ON DELETE CASCADE,
    qualification  TEXT    NOT NULL,
    institution    TEXT,
    field          TEXT,
    graduation_year TEXT,
    grade          TEXT,
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_education_candidate ON candidate_education(candidate_id);

CREATE TABLE IF NOT EXISTS candidate_skills (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidate_profiles(id) ON DELETE CASCADE,
    skill        TEXT    NOT NULL,
    level        TEXT,          -- beginner, intermediate, advanced, expert
    years        REAL,
    assessment   TEXT,
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skills_candidate ON candidate_skills(candidate_id);

CREATE TABLE IF NOT EXISTS job_assignments (
    job_id       INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    recruiter_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at   TEXT    NOT NULL,
    PRIMARY KEY (job_id, recruiter_id)
);

CREATE TABLE IF NOT EXISTS application_stage_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    from_stage     TEXT,
    to_stage       TEXT    NOT NULL,
    note           TEXT,
    reason         TEXT,
    changed_by     INTEGER REFERENCES users(id),
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stage_history_app ON application_stage_history(application_id);

CREATE TABLE IF NOT EXISTS offer_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id    INTEGER NOT NULL REFERENCES offers(id) ON DELETE CASCADE,
    status      TEXT    NOT NULL,
    note        TEXT,
    changed_by  INTEGER REFERENCES users(id),
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_offer_history_offer ON offer_history(offer_id);

CREATE TABLE IF NOT EXISTS placements (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id       INTEGER NOT NULL UNIQUE REFERENCES applications(id) ON DELETE CASCADE,
    job_id               INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    company_id           INTEGER REFERENCES companies(id),
    recruiter_id         INTEGER REFERENCES users(id),
    offered_ctc          TEXT,
    currency             TEXT DEFAULT 'INR',
    joining_date         TEXT,
    actual_joining_date  TEXT,
    placement_date       TEXT,
    status               TEXT NOT NULL DEFAULT 'selected',
    replacement_period   TEXT,
    guarantee_period     TEXT,
    client_confirmation  TEXT,
    billing_status       TEXT DEFAULT 'pending',
    placement_fee        REAL,
    invoice_number       TEXT,
    invoice_details      TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_placements_job ON placements(job_id);
CREATE INDEX IF NOT EXISTS idx_placements_recruiter ON placements(recruiter_id);

-- Security trail, separate from `activities`: the feed is for humans, this is
-- the append-only record of who changed what.
CREATE TABLE IF NOT EXISTS audit_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    action      TEXT    NOT NULL,
    entity_type TEXT,
    entity_id   INTEGER,
    details     TEXT,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);

-- Full posting text, keyed by URL, fetched in the background so the first render
-- does not wait on a browser.
--
-- Deliberately NOT scoped to a user_id. A job description is public text about a
-- posting, identical for everyone who opens that URL, and the expensive part is
-- the fetch: sharing one row means the second person to generate a resume for a
-- posting gets it instantly instead of paying for another Chromium run. Nothing
-- private is stored here - no cookie, no account, no candidate data.
--
-- `text` may legitimately be empty: that row is the record of "we already tried
-- this posting and there was no description to get", which is what stops every
-- click from launching another browser against a wall.
CREATE TABLE IF NOT EXISTS job_desc_cache (
    url        TEXT    PRIMARY KEY,
    text       TEXT    NOT NULL DEFAULT '',
    source     TEXT    NOT NULL DEFAULT 'none',
    fetched_at REAL    NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    """A connection with row access by name and foreign keys enforced.

    check_same_thread is off because FastAPI runs sync work on a thread pool;
    the statements here are short, and writes are serialised by SQLite itself.
    """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after an account was first created.

    CREATE TABLE IF NOT EXISTS is a no-op on a table that already exists, so
    every column added to SCHEMA after v1 silently does not exist in a real
    users.db and the first query that names it fails with "no such column". Each
    addition has to be applied explicitly, by comparing what the live table has
    against what SCHEMA now declares. Additive only: no column is ever dropped or
    retyped, because that would lose the data already in it.
    """
    for table, column, decl in (("users", "nickname", "TEXT"),
                                ("users", "avatar", "TEXT"),
                                ("users", "role", "TEXT NOT NULL DEFAULT 'candidate'"),
                                ("sessions", "last_seen", "REAL NOT NULL DEFAULT 0"),
                                # companies: client CRM fields
                                ("companies", "address", "TEXT"),
                                ("companies", "company_size", "TEXT"),
                                ("companies", "contact_email", "TEXT"),
                                ("companies", "contact_phone", "TEXT"),
                                ("companies", "status", "TEXT DEFAULT 'active'"),
                                ("companies", "account_manager", "INTEGER"),
                                # candidate master profile
                                ("candidate_profiles", "email", "TEXT"),
                                ("candidate_profiles", "alt_phone", "TEXT"),
                                ("candidate_profiles", "current_company", "TEXT"),
                                ("candidate_profiles", "current_designation", "TEXT"),
                                ("candidate_profiles", "total_experience", "REAL"),
                                ("candidate_profiles", "relevant_experience", "REAL"),
                                ("candidate_profiles", "current_ctc", "TEXT"),
                                ("candidate_profiles", "expected_ctc", "TEXT"),
                                ("candidate_profiles", "notice_period", "TEXT"),
                                ("candidate_profiles", "preferred_location", "TEXT"),
                                ("candidate_profiles", "employment_type", "TEXT"),
                                ("candidate_profiles", "primary_skills", "TEXT"),
                                ("candidate_profiles", "secondary_skills", "TEXT"),
                                ("candidate_profiles", "linkedin", "TEXT"),
                                ("candidate_profiles", "portfolio", "TEXT"),
                                ("candidate_profiles", "resume_url", "TEXT"),
                                ("candidate_profiles", "recruiter_id", "INTEGER"),
                                ("candidate_profiles", "source", "TEXT"),
                                ("candidate_profiles", "status", "TEXT DEFAULT 'active'"),
                                ("candidate_profiles", "created_at", "TEXT"),
                                # jobs: requirement fields
                                ("jobs", "job_code", "TEXT"),
                                ("jobs", "priority", "TEXT DEFAULT 'medium'"),
                                ("jobs", "mandatory_skills", "TEXT"),
                                ("jobs", "good_to_have_skills", "TEXT"),
                                ("jobs", "hiring_manager", "TEXT"),
                                ("jobs", "client_contact_id", "INTEGER"),
                                ("jobs", "target_closure_date", "TEXT"),
                                # applications: job-specific pipeline state
                                ("applications", "stage", "TEXT DEFAULT 'sourced'"),
                                ("applications", "recruiter_id", "INTEGER"),
                                ("applications", "source", "TEXT"),
                                ("applications", "stage_entered_at", "TEXT"),
                                ("applications", "rejection_reason", "TEXT"),
                                ("application_status_emails", "attempted_at", "REAL"),
                                # interviews: scheduling detail
                                ("interviews", "round", "TEXT"),
                                ("interviews", "mode", "TEXT"),
                                ("interviews", "timezone", "TEXT"),
                                ("interviews", "interviewer_name", "TEXT"),
                                ("interviews", "interviewer_email", "TEXT"),
                                ("interviews", "interviewer_contact", "TEXT"),
                                # offers: compensation detail
                                ("offers", "designation", "TEXT"),
                                ("offers", "fixed_ctc", "REAL"),
                                ("offers", "variable_ctc", "REAL"),
                                ("offers", "joining_bonus", "REAL"),
                                ("offers", "notice_period", "TEXT"),
                                ("offers", "expected_joining_date", "TEXT"),
                                # tasks: reminders
                                ("tasks", "reminder_date", "TEXT")):
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    # Ensure candidates table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id                TEXT    NOT NULL,
            user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name              TEXT    NOT NULL,
            current_location  TEXT,
            expected_ctc      TEXT,
            current_ctc       TEXT,
            email             TEXT,
            phone             TEXT,
            availability      TEXT,
            status            TEXT    NOT NULL DEFAULT 'working',
            notes             TEXT,
            created_at        TEXT    NOT NULL,
            updated_at        TEXT    NOT NULL,
            PRIMARY KEY (user_id, id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_user ON candidates(user_id)")


def init_db() -> None:
    """Create the schema. Safe to call on every boot."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _in(hours: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + hours * 3600))


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------

MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    """Hash a password for storage. Raises ValueError on a too-short one."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("password must be at least %d characters"
                         % MIN_PASSWORD_LENGTH)
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N,
                        r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN)
    return "scrypt${}${}${}${}${}".format(
        SCRYPT_N, SCRYPT_R, SCRYPT_P,
        base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a stored hash.

    Returns False (never raises) for a malformed record, so a corrupt row can
    never crash the login endpoint.
    """
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"),
                            salt=base64.b64decode(salt_b64), n=int(n), r=int(r),
                            p=int(p), dklen=len(base64.b64decode(hash_b64)))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except (ValueError, TypeError, AttributeError):
        return False


def authenticate(email: str, password: str) -> dict | None:
    """The account for these credentials, or None.

    An unknown address and a wrong password both return None, so the login form
    cannot be used to discover which addresses are registered. The dummy hash
    below keeps the timing comparable in the miss case.
    """
    user = get_user_by_email(email)
    if not user:
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


# A real scrypt hash of an unguessable value, used only to spend the same time
# on a miss as on a hit. Generated once at import; it protects no account.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(32))


# --------------------------------------------------------------------------
# users
# --------------------------------------------------------------------------

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _looks_like_email(email: str) -> bool:
    """Cheap shape check. Deliberately not RFC 5322 - the only authority that
    matters is whether a confirmation mail arrives, and guessing at exotic valid
    forms only rejects addresses that would have worked."""
    if email.count("@") != 1:
        return False
    local, _, domain = email.partition("@")
    if not local or " " in local:
        return False
    if "." not in domain or domain.startswith(".") or domain.endswith("."):
        return False
    return " " not in domain


def create_user(email: str, password: str, owner: bool = False, role: str = "candidate",
                *, verified: bool = False, google_subject: str | None = None) -> dict:
    """Create an account and give it one empty default profile.

    An explicit owner request succeeds only if there is no existing owner.
    """
    role = role if role in ("candidate", "recruiter", "super_admin") else "candidate"
    email = normalize_email(email)
    if not _looks_like_email(email):
        raise ValueError("that does not look like an email address")
    pwd_hash = hash_password(password)
    now = _now()
    claimed = False
    with connect() as conn:
        if owner:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM users WHERE is_owner = 1 LIMIT 1").fetchone():
                raise ValueError("an owner account already exists")
        try:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, created_at, is_owner, role) "
                "VALUES (?, ?, ?, ?, ?)",
                (email, pwd_hash, now, 1 if owner else 0, role))
            uid = cur.lastrowid
        except sqlite3.IntegrityError:
            if google_subject is not None:
                raise ValueError("an account with that email already exists") from None
            # google_forms_api creates unverified applicant accounts with an
            # unusable password. Signing up with that address claims the row
            # instead of dead-ending, so a form respondent can still join.
            row = conn.execute(
                "SELECT id FROM users WHERE email = ? COLLATE NOCASE AND verified_at IS NULL "
                "AND password_hash LIKE '!%'", (email,)).fetchone()
            if not row:
                raise ValueError("an account with that email already exists")
            uid, claimed = row["id"], True
            conn.execute("UPDATE users SET password_hash = ?, role = ? WHERE id = ?",
                         (pwd_hash, role, uid))
        if owner and claimed:
            conn.execute("UPDATE users SET is_owner = 1 WHERE id = ?", (uid,))
        if verified:
            conn.execute("UPDATE users SET verified_at = ? WHERE id = ?", (now, uid))
        if google_subject is not None:
            conn.execute("INSERT INTO google_identities (subject, user_id) VALUES (?, ?)",
                         (google_subject, uid))
        conn.execute(
            ("INSERT OR IGNORE" if claimed else "INSERT") + " INTO profiles "
            "(id, user_id, name, created_at, is_default, data) VALUES (?, ?, ?, ?, 1, ?)",
            ("default", uid, "Default", now,
             json.dumps({"id": "default", "name": "Default", "created_at": now,
                         "resume_text": "", "github_url": "", "skills": [],
                         "cities": [], "roles": [], "role_suggestions": [],
                         "work_modes": [], "regions": [], "remote_only": False})))

    return get_user(uid)


def get_user(uid: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                           (normalize_email(email),)).fetchone()
    return dict(row) if row else None


def delete_user(uid: int) -> bool:
    """Delete an account and the rows that point at it.

    Most tables cascade from users(), but the recruitment tables reference a
    user without ON DELETE CASCADE. With foreign_keys=ON, deleting the users row
    alone fails on the first such row, so the owned rows are removed first and
    leaves come before their parents (notes/communications before the
    applications they mention, applications before jobs).
    """
    own_apps = ("SELECT id FROM applications WHERE candidate_user_id = ? "
                "OR job_id IN (SELECT id FROM jobs WHERE recruiter_id = ?)")
    with connect() as conn:
        conn.execute("DELETE FROM interview_feedback WHERE interviewer_id = ?", (uid,))
        conn.execute("DELETE FROM interviews WHERE candidate_user_id = ? OR recruiter_id = ?",
                     (uid, uid))
        conn.execute("DELETE FROM offers WHERE candidate_user_id = ? OR offered_by = ?",
                     (uid, uid))
        conn.execute(f"DELETE FROM recruiter_notes WHERE recruiter_id = ? "
                     f"OR candidate_user_id = ? OR application_id IN ({own_apps})",
                     (uid, uid, uid, uid))
        conn.execute(f"DELETE FROM communications WHERE recruiter_id = ? "
                     f"OR candidate_user_id = ? OR application_id IN ({own_apps})",
                     (uid, uid, uid, uid))
        conn.execute(f"DELETE FROM applications WHERE id IN ({own_apps})", (uid, uid))
        conn.execute("DELETE FROM talent_pool_members WHERE candidate_user_id = ? "
                     "OR added_by = ?", (uid, uid))
        conn.execute("DELETE FROM talent_pools WHERE recruiter_id = ?", (uid,))
        conn.execute("DELETE FROM tasks WHERE recruiter_id = ?", (uid,))
        conn.execute("DELETE FROM notifications WHERE user_id = ?", (uid,))
        conn.execute("DELETE FROM activities WHERE user_id = ?", (uid,))
        conn.execute("DELETE FROM jobs WHERE recruiter_id = ?", (uid,))
        # companies.created_by is nullable and a company outlives its creator
        conn.execute("UPDATE companies SET created_by = NULL WHERE created_by = ?", (uid,))
        cur = conn.execute("DELETE FROM users WHERE id = ?", (uid,))
        return cur.rowcount > 0


# --------------------------------------------------------------------------
# candidates (recruitment)
# --------------------------------------------------------------------------


# Recruiter CRM helpers
def create_company(name: str, created_by: int = None, **kwargs) -> dict:
    now = _now()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO companies (name, website, logo_url, industry, description, location, created_at, created_by) VALUES (?,?,?,?,?,?,?,?)",
            (name, kwargs.get('website'), kwargs.get('logo_url'), kwargs.get('industry'), kwargs.get('description'), kwargs.get('location'), now, created_by))
        cid = cur.lastrowid
        row = conn.execute("SELECT * FROM companies WHERE id = ?", (cid,)).fetchone()
    return dict(row) if row else {"id": cid}


def get_company(cid: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM companies WHERE id = ?", (cid,)).fetchone()
    return dict(row) if row else None


def create_recruiter_profile(user_id: int, company_id: int = None, **kwargs) -> dict:
    now = _now()
    with connect() as conn:
        conn.execute(
            """INSERT INTO recruiter_profiles (user_id, company_id, title, phone, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET
                 company_id=COALESCE(excluded.company_id, company_id),
                 title=COALESCE(excluded.title, title),
                 phone=COALESCE(excluded.phone, phone)""",
            (user_id, company_id, kwargs.get('title'), kwargs.get('phone'), now))
        row = conn.execute("SELECT * FROM recruiter_profiles WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else {}


def get_recruiter_profile(user_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM recruiter_profiles WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_job(company_id: int, recruiter_id: int, data: dict) -> dict:
    now = _now()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO jobs (company_id, recruiter_id, title, department, description, required_skills, preferred_skills,
               min_experience, max_experience, education_req, employment_type, work_mode, location, salary_min, salary_max,
               currency, openings, application_deadline, status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?, ?)""",
            (company_id, recruiter_id, data.get('title'), data.get('department'), data.get('description'),
             json.dumps(data.get('required_skills') or []), json.dumps(data.get('preferred_skills') or []),
             data.get('min_experience'), data.get('max_experience'), data.get('education_req'),
             data.get('employment_type'), data.get('work_mode'), data.get('location'),
             data.get('salary_min'), data.get('salary_max'), data.get('currency','INR'),
             data.get('openings',1), data.get('application_deadline'), data.get('status','draft'), now, now))
        jid = cur.lastrowid
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
    return dict(row) if row else {"id": jid}


def list_jobs(company_id: int = None, status: str = None) -> list[dict]:
    with connect() as conn:
        if company_id and status:
            rows = conn.execute("SELECT * FROM jobs WHERE company_id = ? AND status = ? ORDER BY created_at DESC", (company_id, status)).fetchall()
        elif company_id:
            rows = conn.execute("SELECT * FROM jobs WHERE company_id = ? ORDER BY created_at DESC", (company_id,)).fetchall()
        elif status:
            rows = conn.execute("SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def apply_to_job(job_id: int, candidate_user_id: int, resume_id: str = None,
                 cover_letter: str = None, *, published_only: bool = False) -> dict:
    now = _now()
    with connect() as conn:
        if published_only:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute("SELECT 1 FROM jobs WHERE id = ? AND status = 'published'",
                                (job_id,)).fetchone():
                return {}
        try:
            cur = conn.execute(
                "INSERT INTO applications (job_id, candidate_user_id, resume_id, cover_letter, status, applied_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (job_id, candidate_user_id, resume_id, cover_letter, 'applied', now, now))
            aid = cur.lastrowid
        except sqlite3.IntegrityError:
            row = conn.execute("SELECT * FROM applications WHERE job_id = ? AND candidate_user_id = ?", (job_id, candidate_user_id)).fetchone()
            return dict(row) if row else {}
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (aid,)).fetchone()
    return dict(row) if row else {"id": aid}


def list_applications(job_id: int = None, candidate_user_id: int = None) -> list[dict]:
    with connect() as conn:
        if job_id and candidate_user_id:
            rows = conn.execute("SELECT * FROM applications WHERE job_id = ? AND candidate_user_id = ?", (job_id, candidate_user_id)).fetchall()
        elif job_id:
            rows = conn.execute("SELECT * FROM applications WHERE job_id = ? ORDER BY applied_at DESC", (job_id,)).fetchall()
        elif candidate_user_id:
            rows = conn.execute("SELECT * FROM applications WHERE candidate_user_id = ? ORDER BY applied_at DESC", (candidate_user_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM applications ORDER BY applied_at DESC").fetchall()
    return [dict(r) for r in rows]


def update_application_status(application_id: int, status: str) -> bool:
    now = _now()
    with connect() as conn:
        cur = conn.execute("UPDATE applications SET status = ?, updated_at = ? WHERE id = ?", (status, now, application_id))
        return cur.rowcount > 0


# --- Recruiter hiring loop -------------------------------------------------
_COMPANY_FIELDS = ("name", "website", "logo_url", "industry", "description", "location")
_JOB_FIELDS = ("title", "department", "description", "min_experience", "max_experience",
               "education_req", "employment_type", "work_mode", "location", "salary_min",
               "salary_max", "currency", "openings", "application_deadline", "status")
JOB_STATUSES = ("draft", "published", "paused", "closed")
APP_STATUSES = ("applied", "ai_screening", "recruiter_review", "shortlisted", "interview_scheduled",
                "interview_completed", "technical_round", "hr_round", "offer_released", "offer_accepted",
                "hired", "rejected", "withdrawn", "on_hold", "offer_declined")
TASK_STATUSES = ("pending", "in_progress", "completed", "cancelled")
INTERVIEW_STATUSES = ("scheduled", "rescheduled", "completed", "cancelled", "no_show")


def _update(conn, table: str, row_id: int, allowed: tuple, data: dict, stamp: bool = True) -> None:
    """Set only the allowed keys present in data, so a partial edit never blanks the rest."""
    sets = {k: data[k] for k in allowed if k in data}
    if stamp:
        sets["updated_at"] = _now()
    if not sets:
        return
    conn.execute(f"UPDATE {table} SET " + ", ".join(f"{k} = ?" for k in sets) + " WHERE id = ?",
                 (*sets.values(), row_id))


def update_company(cid: int, data: dict) -> dict | None:
    with connect() as conn:
        _update(conn, "companies", cid, _COMPANY_FIELDS, data, stamp=False)
    return get_company(cid)


def get_job(jid: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
    return dict(row) if row else None


def update_job(jid: int, data: dict) -> dict | None:
    data = dict(data)
    for k in ("required_skills", "preferred_skills"):
        if k in data:
            data[k] = json.dumps(data[k] or [])
    with connect() as conn:
        _update(conn, "jobs", jid, _JOB_FIELDS + ("required_skills", "preferred_skills"), data)
    return get_job(jid)


def list_company_applications(company_id: int, job_id: int = None) -> list[dict]:
    """Applications for one company's jobs, with the candidate and job names joined in."""
    sql = ("SELECT a.*, j.title AS job_title, u.email AS candidate_email, "
           "COALESCE(cp.full_name, u.nickname, u.email) AS candidate_name, "
           "cp.professional_title AS candidate_title, cp.location AS candidate_location "
           "FROM applications a JOIN jobs j ON j.id = a.job_id "
           "JOIN users u ON u.id = a.candidate_user_id "
           "LEFT JOIN candidate_profiles cp ON cp.user_id = a.candidate_user_id "
           "WHERE j.company_id = ?")
    args = [company_id]
    if job_id:
        sql += " AND a.job_id = ?"
        args.append(job_id)
    with connect() as conn:
        rows = conn.execute(sql + " ORDER BY a.applied_at DESC", args).fetchall()
    return [dict(r) for r in rows]


def set_company_application_status(company_id: int, application_id: int, status: str) -> bool:
    """Like update_application_status, but only for an application on this company's jobs."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE applications SET status = ?, updated_at = ? WHERE id = ? AND job_id IN "
            "(SELECT id FROM jobs WHERE company_id = ?)", (status, _now(), application_id, company_id))
        return cur.rowcount > 0


def list_tasks(recruiter_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE recruiter_id = ? ORDER BY (status IN ('completed','cancelled')), "
            "COALESCE(due_date, '9999'), id DESC", (recruiter_id,)).fetchall()
    return [dict(r) for r in rows]


def create_task(recruiter_id: int, company_id: int | None, data: dict) -> dict:
    now = _now()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (recruiter_id, company_id, title, description, due_date, status, priority, "
            "related_to_type, related_to_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (recruiter_id, company_id, data["title"], data.get("description"), data.get("due_date") or None,
             data.get("status") or "pending", data.get("priority") or "medium",
             data.get("related_to_type") or None, data.get("related_to_id"), now, now))
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_task(recruiter_id: int, task_id: int, data: dict) -> dict | None:
    with connect() as conn:
        if not conn.execute("SELECT 1 FROM tasks WHERE id = ? AND recruiter_id = ?", (task_id, recruiter_id)).fetchone():
            return None
        _update(conn, "tasks", task_id, ("title", "description", "due_date", "status", "priority"), data)
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row)


def delete_task(recruiter_id: int, task_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM tasks WHERE id = ? AND recruiter_id = ?", (task_id, recruiter_id)).rowcount > 0


def list_interviews(company_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT i.*, j.title AS job_title, u.email AS candidate_email, "
            "COALESCE(cp.full_name, u.nickname, u.email) AS candidate_name "
            "FROM interviews i JOIN jobs j ON j.id = i.job_id JOIN users u ON u.id = i.candidate_user_id "
            "LEFT JOIN candidate_profiles cp ON cp.user_id = i.candidate_user_id "
            "WHERE j.company_id = ? ORDER BY i.scheduled_at", (company_id,)).fetchall()
    return [dict(r) for r in rows]


def create_interview(company_id: int, recruiter_id: int, data: dict) -> dict | None:
    """Schedule an interview and move the application to interview_scheduled. None if the
    application is not on one of this company's jobs."""
    now = _now()
    with connect() as conn:
        app = conn.execute(
            "SELECT a.id, a.job_id, a.candidate_user_id FROM applications a JOIN jobs j ON j.id = a.job_id "
            "WHERE a.id = ? AND j.company_id = ?", (data["application_id"], company_id)).fetchone()
        if not app:
            return None
        cur = conn.execute(
            "INSERT INTO interviews (application_id, job_id, candidate_user_id, recruiter_id, interview_type, "
            "scheduled_at, duration_minutes, meeting_link, location, instructions, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,'scheduled',?,?)",
            (app["id"], app["job_id"], app["candidate_user_id"], recruiter_id,
             data.get("interview_type") or "phone_screening", data["scheduled_at"],
             data.get("duration_minutes") or 60, data.get("meeting_link") or None,
             data.get("location") or None, data.get("instructions") or None, now, now))
        conn.execute("UPDATE applications SET status = 'interview_scheduled', updated_at = ? WHERE id = ?",
                     (now, app["id"]))
        row = conn.execute("SELECT * FROM interviews WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_interview(company_id: int, interview_id: int, data: dict) -> dict | None:
    with connect() as conn:
        if not conn.execute(
                "SELECT 1 FROM interviews i JOIN jobs j ON j.id = i.job_id WHERE i.id = ? AND j.company_id = ?",
                (interview_id, company_id)).fetchone():
            return None
        _update(conn, "interviews", interview_id,
                ("interview_type", "scheduled_at", "duration_minutes", "meeting_link", "location",
                 "instructions", "status"), data)
        row = conn.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,)).fetchone()
    return dict(row)


def owner_user() -> dict | None:
    """The account that owns the migrated CV sessions, if there is one."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE is_owner = 1 ORDER BY id LIMIT 1").fetchone()
    return dict(row) if row else None


def count_users() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]


def set_password(uid: int, password: str) -> None:
    """Replace a password and sign the account out everywhere.

    Killing the sessions here rather than in the caller means a stolen cookie
    stops working the moment the real owner resets, whichever route did it.
    """
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (hash_password(password), uid))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))


def mark_verified(uid: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE users SET verified_at = COALESCE(verified_at, ?) "
                     "WHERE id = ?", (_now(), uid))


# --------------------------------------------------------------------------
# display identity: nickname + avatar
# --------------------------------------------------------------------------

# The avatar choices offered in the UI. Kept server-side so the ids stored in
# the database are validated against one list: the UI cannot invent an id, and
# an old row naming an avatar that has since been removed degrades to the
# default instead of rendering a blank circle.
AVATARS = ("aurora", "ember", "forest", "harbor", "iris", "sand", "slate",
           "violet", "coral", "mint", "dusk", "cobalt")

DEFAULT_AVATAR = "aurora"

MAX_NICKNAME = 32


def display_name(user: dict) -> str:
    """What the dashboard calls this person.

    Falls back to the local part of the address so a user who never picked a
    nickname still gets something that is not a full email in the header.
    """
    nick = (user.get("nickname") or "").strip()
    if nick:
        return nick
    email = (user.get("email") or "").strip()
    return email.split("@")[0] if email else "there"


def set_identity(uid: int, nickname: str | None = None,
                 avatar: str | None = None) -> dict:
    """Update the display identity. Each field is optional and independent.

    Nicknames are trimmed and length-capped, and control characters are dropped
    rather than escaped: this string is rendered into the header, into a
    document title and into LaTeX output, and a nickname is a thing a person
    types, not markup. An empty nickname is stored as NULL, which reads as
    "use the fallback" instead of persisting a blank that overrides the address.
    """
    sets, vals = [], []
    if nickname is not None:
        clean = "".join(ch for ch in nickname.strip() if ch.isprintable())
        clean = re.sub(r"\s+", " ", clean)[:MAX_NICKNAME].strip()
        sets.append("nickname = ?")
        vals.append(clean or None)
    if avatar is not None:
        sets.append("avatar = ?")
        vals.append(avatar if avatar in AVATARS else DEFAULT_AVATAR)
    if not sets:
        return get_user(uid) or {}
    with connect() as conn:
        conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?",
                     (*vals, uid))
    return get_user(uid) or {}


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(uid: int) -> tuple[str, str]:
    """Start a login session. Returns (token, expires_at)."""
    token = secrets.token_urlsafe(32)
    expires = _in(SESSION_DAYS)
    with connect() as conn:
        conn.execute("INSERT INTO sessions (token_hash, user_id, created_at, "
                     "expires_at, last_seen) VALUES (?, ?, ?, ?, ?)",
                     (_token_hash(token), uid, _now(), expires, time.time()))
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
    return token, expires


def session_row(token: str) -> dict | None:
    """The session row JOINED to its account, ignoring the idle timeout.

    Exposed separately so /api/auth/session can report "19 minutes left" and the
    signed-in display name to a user who is very much still signed in;
    session_user is the one that decides whether they are. This one deliberately
    does NOT touch last_seen, because the countdown polls it and a poll is not
    activity - a session refreshed by its own countdown would never expire.

    The join is what makes it usable for identity: the nickname and avatar live
    on users, not sessions, so a sessions-only row would report "there" for every
    signed-in user.
    """
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT s.*, u.email, u.nickname, u.avatar, u.verified_at, u.is_owner, u.role "
            "FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ?",
            (_token_hash(token),)).fetchone()
    return dict(row) if row else None


def seconds_left(token: str) -> int:
    """Seconds of idle allowance this session has left, 0 when there is none.

    The number the countdown renders, so the UI and the server agree on when the
    logout happens instead of the browser guessing from its own clock. Reads
    without touching, for the same reason as session_row.
    """
    row = session_row(token)
    if not row:
        return 0
    # A row written before the migration has last_seen = 0, which reads as 1970
    # and would instantly expire every pre-existing session. Treat it as "used
    # just now" so a deploy cannot log out everyone who was mid-task.
    last = row.get("last_seen") or time.time()
    return max(0, int(SESSION_IDLE_SECONDS - (time.time() - last)))


def extend_session(token: str, minutes: int = SESSION_EXTEND_MINUTES) -> int:
    """Push the idle deadline out and return the new remaining seconds, or 0 if
    the session is already gone.

    The "+5 minutes" button. Two rules it must not break:

    It cannot revive an expired session. Refreshing last_seen on a row that is
    already past the idle deadline would let anyone holding a dead cookie keep it
    alive by clicking the button - the timeout would be advisory rather than
    enforced. An expired row is deleted and 0 is returned instead.

    It cannot make a session live forever. It advances last_seen to now, which
    restores the full allowance rather than adding to it, so repeated clicks
    cannot build an unbounded session.
    """
    if not token:
        return 0
    now = time.time()
    with connect() as conn:
        row = conn.execute(
            "SELECT last_seen, expires_at FROM sessions WHERE token_hash = ?",
            (_token_hash(token),)).fetchone()
        if not row or row["expires_at"] <= _now():
            return 0
        last = row["last_seen"] or now
        if now - last > SESSION_IDLE_SECONDS:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?",
                         (_token_hash(token),))
            return 0
        conn.execute("UPDATE sessions SET last_seen = ? WHERE token_hash = ?",
                     (now, _token_hash(token)))
    return SESSION_IDLE_SECONDS


def session_user(token: str) -> dict | None:
    """The account behind a session cookie, or None if absent/expired/idle.

    Every authenticated request lands here, so this is where the idle timeout is
    actually enforced: a cookie nobody has presented for SESSION_IDLE_MINUTES is
    deleted rather than merely refused, which is what makes "no data is carried
    into the next session" true instead of aspirational. Touching last_seen on
    the way through is what makes the timeout an *idle* one.
    """
    if not token:
        return None
    now = time.time()
    with connect() as conn:
        row = conn.execute(
            "SELECT u.*, s.last_seen, s.token_hash AS _th FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (_token_hash(token), _now())).fetchone()
        if not row:
            return None
        last = row["last_seen"] or now
        if now - last > SESSION_IDLE_SECONDS:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?",
                         (_token_hash(token),))
            return None
        conn.execute("UPDATE sessions SET last_seen = ? WHERE token_hash = ?",
                     (now, _token_hash(token)))
    return dict(row)


def set_session_profile(token: str, profile_id: str) -> None:
    """Remember which CV profile this browser has open."""
    with connect() as conn:
        conn.execute("UPDATE sessions SET active_profile = ? WHERE token_hash = ?",
                     (profile_id, _token_hash(token)))


def session_profile(token: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT active_profile FROM sessions WHERE token_hash = ?",
                           (_token_hash(token),)).fetchone()
    return (row["active_profile"] if row else None) or None


def destroy_session(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?",
                     (_token_hash(token),))


def destroy_user_sessions(uid: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))


# --------------------------------------------------------------------------
# one-time email links
# --------------------------------------------------------------------------

def issue_token(uid: int, purpose: str, hours: int, *, email: str | None = None) -> str | None:
    """Mint a single-use link token. Only its digest is kept."""
    if purpose not in ("verify", "reset"):
        raise ValueError("unknown token purpose")
    token = secrets.token_urlsafe(32)
    with connect() as conn:
        if email is not None:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute("SELECT 1 FROM users WHERE id = ? AND email = ?",
                                (uid, email)).fetchone():
                return None
        # Issuing a new link retires the old one for that purpose.
        conn.execute("UPDATE email_tokens SET used_at = ? "
                     "WHERE user_id = ? AND purpose = ? AND used_at IS NULL",
                     (_now(), uid, purpose))
        conn.execute("INSERT INTO email_tokens (token_hash, user_id, purpose, "
                     "created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                     (_token_hash(token), uid, purpose, _now(), _in(hours)))
    return token


def _consume_token(conn: sqlite3.Connection, token: str, purpose: str) -> dict | None:
    if not token:
        return None
    now = _now()
    row = conn.execute(
        "UPDATE email_tokens SET used_at = ? "
        "WHERE token_hash = ? AND purpose = ? AND used_at IS NULL AND expires_at > ? "
        "RETURNING user_id",
        (now, _token_hash(token), purpose, now)).fetchone()
    if not row:
        return None
    user = conn.execute("SELECT id, email FROM users WHERE id = ?", (row["user_id"],)).fetchone()
    return dict(user) if user else None


def consume_token(token: str, purpose: str) -> dict | None:
    """Redeem a link, returning the account. Single use, expiry enforced."""
    with connect() as conn:
        return _consume_token(conn, token, purpose)


def verify_email(token: str) -> dict | None:
    with connect() as conn:
        user = _consume_token(conn, token, "verify")
        if user:
            conn.execute("UPDATE users SET verified_at = COALESCE(verified_at, ?) WHERE id = ?",
                         (_now(), user["id"]))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))
        return user


def reset_password(token: str, password: str) -> dict | None:
    password_hash = hash_password(password)
    with connect() as conn:
        user = _consume_token(conn, token, "reset")
        if user:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                         (password_hash, user["id"]))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))
        return user


# --------------------------------------------------------------------------
# profiles (the sidebar CV sessions)
# --------------------------------------------------------------------------

def list_profiles(uid: int) -> list[dict]:
    """Every profile for one user: the default first, then creation order.

    The tiebreaker is rowid, not id. `created_at` only has second resolution,
    so two sessions saved in the same second compared equal and fell through
    to the id — a random 8-hex string. That made the sidebar order of
    same-second sessions arbitrary, and it shuffled between calls, which is
    what made the *Detect* endpoint's "which session answered" answer change
    for no reason. rowid is SQLite's monotonic insertion counter, so it is a
    true creation sequence and needs no schema change.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, data, is_default FROM profiles WHERE user_id = ? "
            "ORDER BY is_default DESC, created_at ASC, rowid ASC",
            (uid,)).fetchall()
    out = []
    for r in rows:
        rec = json.loads(r["data"])
        rec["id"] = r["id"]
        out.append(rec)
    return out


def get_profile(uid: int, profile_id: str) -> dict | None:
    """A profile, or None. Scoped to the owner, so one user can never read
    another's CV by guessing an id."""
    with connect() as conn:
        row = conn.execute(
            "SELECT data FROM profiles WHERE user_id = ? AND id = ?",
            (uid, profile_id)).fetchone()
    if not row:
        return None
    rec = json.loads(row["data"])
    rec["id"] = profile_id
    return rec


def save_profile(uid: int, profile_id: str, data: dict) -> None:
    """Insert or update a profile's JSON payload for one user."""
    payload = dict(data)
    payload["id"] = profile_id
    with connect() as conn:
        conn.execute(
            "INSERT INTO profiles (id, user_id, name, created_at, is_default, data) "
            "VALUES (?, ?, ?, ?, 0, ?) ON CONFLICT(user_id, id) DO UPDATE SET "
            "name = excluded.name, data = excluded.data",
            (profile_id, uid, payload.get("name") or profile_id,
             payload.get("created_at") or _now(), json.dumps(payload)))


def blank_profile() -> dict:
    """An empty profile payload: no CV, no skills, no roles, no filters.

    One definition of "empty", because the shape was being written out in
    three places (create_profile, the default-profile rehome, and the
    whole-account reset) and they drifted: a field added to one and not the
    others came back from the dead on the next load."""
    return {"name": "Default", "resume_text": "", "github_url": "",
            "skills": [], "cities": [], "roles": [], "role_suggestions": [],
            "work_modes": [], "regions": [], "countries": [],
            "remote_only": False}


def create_profile(uid: int, name: str, data: dict | None = None) -> str:
    """Add a profile and return its id."""
    pid = secrets.token_hex(5)
    payload = {"id": pid, "name": name or "New session", "created_at": _now()}
    payload.update(blank_profile())
    payload["name"] = name or "New session"
    if data:
        payload.update(data)
        payload["id"] = pid
    save_profile(uid, pid, payload)
    return pid


def delete_profile(uid: int, profile_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM profiles WHERE user_id = ? AND id = ? "
                           "AND is_default = 0", (uid, profile_id))
    return cur.rowcount > 0


# --------------------------------------------------------------------------
# candidates (recruitment tracking)
# --------------------------------------------------------------------------


def list_candidates(uid: int, status: str | None = None) -> list[dict]:
    with connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE user_id = ? AND status = ? "
                "ORDER BY created_at DESC",
                (uid, status)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE user_id = ? "
                "ORDER BY created_at DESC",
                (uid,)).fetchall()
    return [dict(r) for r in rows]


def get_candidate(uid: int, candidate_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM candidates WHERE user_id = ? AND id = ?",
            (uid, candidate_id)).fetchone()
    return dict(row) if row else None


def upsert_candidate(uid: int, candidate_id: str | None = None,
                     name: str = "", current_location: str = "",
                     expected_ctc: str = "", current_ctc: str = "",
                     email: str = "", phone: str = "",
                     availability: str = "", status: str = "working",
                     notes: str = "") -> str:
    cid = candidate_id or secrets.token_hex(6)
    now = _now()
    if candidate_id:
        existing = get_candidate(uid, candidate_id)
        created_at = existing["created_at"] if existing else now
    else:
        created_at = now
    with connect() as conn:
        conn.execute(
            "INSERT INTO candidates (id, user_id, name, current_location, "
            "expected_ctc, current_ctc, email, phone, availability, status, "
            "notes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, id) DO UPDATE SET "
            "name=excluded.name, current_location=excluded.current_location, "
            "expected_ctc=excluded.expected_ctc, current_ctc=excluded.current_ctc, "
            "email=excluded.email, phone=excluded.phone, "
            "availability=excluded.availability, status=excluded.status, "
            "notes=excluded.notes, updated_at=excluded.updated_at",
            (cid, uid, name.strip(), current_location.strip() or None,
             expected_ctc.strip() or None, current_ctc.strip() or None,
             email.strip() or None, phone.strip() or None,
             availability.strip() or None, status or "working",
             notes.strip() or None, created_at, now))
    return cid


def delete_candidate(uid: int, candidate_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM candidates WHERE user_id = ? AND id = ?",
                           (uid, candidate_id))
    return cur.rowcount > 0
