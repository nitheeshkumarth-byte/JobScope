"""
gmail_index.py — a local cache of job-alert mail, queried instead of IMAP.

Why this exists
---------------
Searching Gmail per query meant every single search opened a fresh IMAP
connection, ran a `FROM "..." SINCE "..."` search, re-fetched full RFC822
messages and re-parsed them. On this account that is 5,095 messages in mailbox
1 and 10,308 in mailbox 2, and the alert mail is a small fraction of it. The
consequences were all bad and all measured:

  * It did not work. `jobs-noreply@linkedin.com` matches 1 message; the address
    that actually carries LinkedIn job alerts is `messages-noreply@linkedin.com`
    (378 + 103 messages). A per-query IMAP search therefore returned "no
    emails" while hundreds of relevant alerts sat in the inbox unread by the
    tool.
  * It was slow. Fetching full RFC822 per hit means downloading every MIME
    part, including the HTML alternative of every digest, only to discard most
    of it.
  * It was fragile. IMAP is rate-limited by Google and a search per query is
    the fastest way to get a connection refused mid-run.

So: fetch the alert mail ONCE into SQLite, then answer every query locally.

What is stored
--------------
One row per *message* (`mail_messages`) and one row per *listing* parsed out of
it (`mail_listings`). They are separate because an Indeed alert carries many
jobs and a Google Careers digest may carry one: a single "search this sender"
question is answerable from messages, while a search for a role is answerable
from listings, and keeping both means neither has to re-parse.

Message-ID is the primary key. IMAP has no stable per-message id across
mailboxes, and the Message-ID header is the only identifier that is stable
across a re-sync *and* unique across two accounts that both received a forward
of the same alert. Listing rows are keyed (message_id, ordinal) so re-parsing a
message replaces its listings instead of duplicating them.

Nothing is ever deleted
-----------------------
`sync` only inserts or replaces. Mail is deleted by the user, and a cached
listing outliving its email is preferable to silently dropping a job the user
may still want to apply to; the `synced_at` column is what a caller checks for
staleness, not row absence.

Time is stored as an ISO-8601 string, not an IMAP date. IMAP's INTERNALDATE is
only available per-message via FETCH and the Date header is what the user sees
on the alert, so parsing it once at sync time keeps queries a plain string
comparison.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "gmail_cache.db")

# Resolved against THIS file, never against the process CWD.
#
# The MCP server runs as a subprocess launched by the agent, whose CWD is the
# directory it was started from and not necessarily the project root. Resolving
# the default next to this module means the subprocess opens the same database
# the dashboard and any direct script use, instead of silently creating a
# second, empty one and reporting "no results" from it.
_DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

SCHEMA = """
CREATE TABLE IF NOT EXISTS mail_messages (
    msg_id      TEXT PRIMARY KEY,
    account     INTEGER NOT NULL DEFAULT 1,
    sender      TEXT NOT NULL DEFAULT '',
    subject     TEXT NOT NULL DEFAULT '',
    sent_at     TEXT NOT NULL DEFAULT '',
    body        TEXT NOT NULL DEFAULT '',
    listing_ct  INTEGER NOT NULL DEFAULT 0,
    synced_at   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mail_sender ON mail_messages(sender);
CREATE INDEX IF NOT EXISTS idx_mail_sent   ON mail_messages(sent_at);
CREATE INDEX IF NOT EXISTS idx_mail_synced ON mail_messages(synced_at);

CREATE TABLE IF NOT EXISTS mail_listings (
    msg_id    TEXT NOT NULL,
    ordinal   INTEGER NOT NULL,
    title     TEXT NOT NULL DEFAULT '',
    company   TEXT NOT NULL DEFAULT '',
    location  TEXT NOT NULL DEFAULT '',
    link      TEXT NOT NULL DEFAULT '',
    sender    TEXT NOT NULL DEFAULT '',
    sent_at   TEXT NOT NULL DEFAULT '',
    account   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (msg_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_listing_sent ON mail_listings(sent_at);

CREATE TABLE IF NOT EXISTS mail_sync_state (
    account    INTEGER PRIMARY KEY,
    last_sync  REAL NOT NULL DEFAULT 0,
    msg_count  INTEGER NOT NULL DEFAULT 0,
    last_error TEXT    NOT NULL DEFAULT ''
);
"""

# One writer at a time. sqlite3 would serialise anyway, but two syncs racing on
# REPLACE can still produce a transient "database is locked" that surfaces as a
# failed search, and the lock costs nothing here.
_write_lock = threading.Lock()


def db_path() -> str:
    return os.environ.get("GMAIL_CACHE_DB") or DEFAULT_DB


@contextmanager
def _conn(path: str | None = None):
    conn = sqlite3.connect(path or db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init(path: str | None = None) -> None:
    """Create the schema. Safe to call on every process start."""
    with _conn(path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def upsert_messages(rows: list[dict], path: str | None = None) -> int:
    """Store parsed messages and their listings. Returns messages written.

    `rows` is what `mcp_server_gmail.parse_message` produces: msg_id, account,
    sender, subject, sent_at, body, listings[{title, company, location, link}].
    """
    if not rows:
        return 0
    init(path)
    now = time.time()
    written = 0
    with _write_lock, _conn(path) as conn:
        for r in rows:
            conn.execute(
                "INSERT INTO mail_messages"
                " (msg_id, account, sender, subject, sent_at, body, listing_ct,"
                "  synced_at)"
                " VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(msg_id) DO UPDATE SET"
                "   account=excluded.account, sender=excluded.sender,"
                "   subject=excluded.subject, sent_at=excluded.sent_at,"
                "   body=excluded.body, listing_ct=excluded.listing_ct,"
                "   synced_at=excluded.synced_at",
                (r["msg_id"], r.get("account", 1), r.get("sender", ""),
                 r.get("subject", ""), r.get("sent_at", ""), r.get("body", ""),
                 len(r.get("listings") or []), now))
            # Replace, not append: re-parsing a message must not double its
            # listings, or the same job would appear as two cards.
            conn.execute("DELETE FROM mail_listings WHERE msg_id = ?",
                         (r["msg_id"],))
            for i, j in enumerate(r.get("listings") or []):
                conn.execute(
                    "INSERT INTO mail_listings"
                    " (msg_id, ordinal, title, company, location, link, sender,"
                    "  sent_at, account) VALUES (?,?,?,?,?,?,?,?,?)",
                    (r["msg_id"], i, j.get("title", ""), j.get("company", ""),
                     j.get("location", ""), j.get("link", ""),
                     r.get("sender", ""), r.get("sent_at", ""),
                     r.get("account", 1)))
            written += 1
    return written


def record_sync(account: int, msg_count: int, error: str = "",
                path: str | None = None) -> None:
    init(path)
    with _write_lock, _conn(path) as conn:
        conn.execute(
            "INSERT INTO mail_sync_state (account, last_sync, msg_count,"
            " last_error) VALUES (?,?,?,?)"
            " ON CONFLICT(account) DO UPDATE SET"
            "   last_sync=excluded.last_sync, msg_count=excluded.msg_count,"
            "   last_error=excluded.last_error",
            (account, time.time(), msg_count, error[:300]))


def sync_state(account: int, path: str | None = None) -> dict:
    init(path)
    with _conn(path) as conn:
        row = conn.execute(
            "SELECT last_sync, msg_count, last_error FROM mail_sync_state"
            " WHERE account = ?", (account,)).fetchone()
    if row is None:
        return {"last_sync": 0.0, "msg_count": 0, "last_error": ""}
    return {"last_sync": row["last_sync"], "msg_count": row["msg_count"],
            "last_error": row["last_error"]}


def is_stale(account: int, max_age_s: float = 3600.0,
             path: str | None = None) -> bool:
    """True when this mailbox has never been synced, or the copy is old.

    An empty `last_error` is not required for freshness: a sync that failed is
    still the newest thing we know, and treating it as stale would retry the
    broken login on every single query instead of once per interval.
    """
    st = sync_state(account, path)
    if not st["last_sync"]:
        return True
    return (time.time() - st["last_sync"]) > max_age_s


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def search_listings(sender: str = "", account: int | None = None,
                    query: str = "", days_back: int | None = None,
                    limit: int = 50, path: str | None = None) -> list[dict]:
    """Listings matching the filters, newest first.

    `query` matches title/company/location as substrings of any word, so
    "devops" finds "DevOps Engineer" and "cloud aws" finds a card carrying
    either term. Matching on substrings of words rather than the whole phrase is
    deliberate: a user types a role, not a job title.
    """
    init(path)
    sql = ["SELECT title, company, location, link, sender, sent_at, account"
           " FROM mail_listings WHERE 1=1"]
    args: list = []
    if sender:
        sql.append("AND sender = ?")
        args.append(sender.strip().lower())
    if account is not None:
        sql.append("AND account = ?")
        args.append(account)
    if days_back:
        sql.append("AND sent_at >= ?")
        args.append(cutoff_iso(days_back))
    if query:
        # Every whitespace-separated term must appear somewhere in the card.
        for term in query.split():
            sql.append("AND (lower(title) LIKE ? OR lower(company) LIKE ?"
                       " OR lower(location) LIKE ?)")
            like = f"%{term.lower()}%"
            args += [like, like, like]
    sql.append("ORDER BY sent_at DESC, msg_id DESC, ordinal ASC LIMIT ?")
    args.append(max(1, int(limit)))
    with _conn(path) as conn:
        return [dict(r) for r in conn.execute(" ".join(sql), args)]


def count_listings(account: int | None = None,
                   path: str | None = None) -> int:
    init(path)
    sql = "SELECT COUNT(*) AS n FROM mail_listings"
    args: list = []
    if account is not None:
        sql += " WHERE account = ?"
        args.append(account)
    with _conn(path) as conn:
        return int(conn.execute(sql, args).fetchone()["n"])


def search_messages(sender: str = "", account: int | None = None,
                    days_back: int | None = None, limit: int = 20,
                    path: str | None = None) -> list[dict]:
    """Matching messages with their subject/date, newest first.

    Returned for the "which alerts do I actually have?" question, which is not
    the same as the role search: an alert can carry four jobs and none of them
    match, and the user still deserves to know the digest arrived.
    """
    init(path)
    sql = ["SELECT msg_id, account, sender, subject, sent_at, listing_ct,"
           " synced_at FROM mail_messages WHERE 1=1"]
    args: list = []
    if sender:
        sql.append("AND sender = ?")
        args.append(sender.strip().lower())
    if account is not None:
        sql.append("AND account = ?")
        args.append(account)
    if days_back:
        sql.append("AND sent_at >= ?")
        args.append(cutoff_iso(days_back))
    sql.append("ORDER BY sent_at DESC, msg_id DESC LIMIT ?")
    args.append(max(1, int(limit)))
    with _conn(path) as conn:
        return [dict(r) for r in conn.execute(" ".join(sql), args)]


def sender_breakdown(path: str | None = None) -> list[dict]:
    """Per-sender message/listing counts. Used to show what was indexed."""
    init(path)
    with _conn(path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT sender, account, COUNT(*) AS msgs,"
            " SUM(listing_ct) AS listings FROM mail_messages"
            " GROUP BY sender, account ORDER BY msgs DESC")]


def cutoff_iso(days_back: int) -> str:
    """The oldest `sent_at` still in range, as an ISO string.

    Same format as stored `sent_at`, so this is a plain string comparison in
    SQLite rather than a date function — which also means a message with an
    unparseable Date header sorts oldest and ages out, instead of being
    permanently "recent".
    """
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(days=max(0, days_back))) \
        .strftime("%Y-%m-%dT%H:%M:%S")


def clear(path: str | None = None) -> None:
    """Drop every cached row. Used by the reindex path, never by a search."""
    init(path)
    with _write_lock, _conn(path) as conn:
        conn.execute("DELETE FROM mail_listings")
        conn.execute("DELETE FROM mail_messages")
        conn.execute("DELETE FROM mail_sync_state")


if __name__ == "__main__":
    init()
    print(json.dumps({"db": db_path(),
                      "listings": count_listings(),
                      "senders": sender_breakdown()}, indent=2))