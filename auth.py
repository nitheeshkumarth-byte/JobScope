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
    avatar        TEXT
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
                                ("sessions", "last_seen", "REAL NOT NULL DEFAULT 0")):
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


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


def create_user(email: str, password: str, owner: bool = False) -> dict:
    """Create an account and give it one empty default profile.

    Raises ValueError on a duplicate address or a bad password. The first
    account created becomes the owner, which is how the pre-existing CV
    sessions get claimed.
    """
    email = normalize_email(email)
    if not _looks_like_email(email):
        raise ValueError("that does not look like an email address")
    pwd_hash = hash_password(password)
    now = _now()
    with connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, created_at, is_owner) "
                "VALUES (?, ?, ?, ?)",
                (email, pwd_hash, now, 1 if owner else 0))
        except sqlite3.IntegrityError:
            raise ValueError("an account with that email already exists")
        uid = cur.lastrowid
        if owner:
            # Belt and braces: exactly one owner even if two registrations race.
            conn.execute("UPDATE users SET is_owner = 0 WHERE id != ?", (uid,))
            conn.execute("UPDATE users SET is_owner = 1 WHERE id = ?", (uid,))
        conn.execute(
            "INSERT INTO profiles (id, user_id, name, created_at, is_default, data) "
            "VALUES (?, ?, ?, ?, 1, ?)",
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
            "SELECT s.*, u.email, u.nickname, u.avatar, u.verified_at, u.is_owner "
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

def issue_token(uid: int, purpose: str, hours: int) -> str:
    """Mint a single-use link token. Only its digest is kept."""
    if purpose not in ("verify", "reset"):
        raise ValueError("unknown token purpose")
    token = secrets.token_urlsafe(32)
    with connect() as conn:
        # Issuing a new link retires the old one for that purpose.
        conn.execute("UPDATE email_tokens SET used_at = ? "
                     "WHERE user_id = ? AND purpose = ? AND used_at IS NULL",
                     (_now(), uid, purpose))
        conn.execute("INSERT INTO email_tokens (token_hash, user_id, purpose, "
                     "created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                     (_token_hash(token), uid, purpose, _now(), _in(hours)))
    return token


def consume_token(token: str, purpose: str) -> dict | None:
    """Redeem a link, returning the account. Single use, expiry enforced."""
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT t.user_id, u.email FROM email_tokens t "
            "JOIN users u ON u.id = t.user_id "
            "WHERE t.token_hash = ? AND t.purpose = ? AND t.used_at IS NULL "
            "AND t.expires_at > ?",
            (_token_hash(token), purpose, _now())).fetchone()
        if not row:
            return None
        conn.execute("UPDATE email_tokens SET used_at = ? WHERE token_hash = ?",
                     (_now(), _token_hash(token)))
    return {"id": row["user_id"], "email": row["email"]}


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
