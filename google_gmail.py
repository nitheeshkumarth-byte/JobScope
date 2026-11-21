"""Sync only the job-alert listings belonging to a connected candidate."""
import json
from collections import Counter

from fastapi import HTTPException
from google.auth.exceptions import GoogleAuthError, RefreshError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import auth
import filters
import gmail_index
import google_accounts
import mcp_server_gmail


def _new_service(connection):
    credentials = google_accounts.gmail_credentials(connection)
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def sync(connection, days_back: int) -> dict:
    senders = list(dict.fromkeys(filters.gmail_senders_for_account(1) + filters.gmail_senders_for_account(2)))
    try:
        rows, _ = mcp_server_gmail.fetch_job_alerts(_new_service(connection), senders, days_back)
    except RefreshError as exc:
        raise HTTPException(502, "Gmail access expired or was revoked. Reconnect Gmail to fetch new alerts.") from exc
    except (GoogleAuthError, HttpError, OSError) as exc:
        raise HTTPException(502, "Gmail could not sync the job alerts. Try again later.") from exc
    connection_id = connection["id"]
    with auth.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM google_gmail_connections WHERE id = ? AND user_id = ?",
                            (connection_id, connection["user_id"])).fetchone():
            raise HTTPException(409, "This Gmail connection changed. Try again.")
        conn.executemany(
            "INSERT INTO google_gmail_alerts (connection_id, message_id, sender, sent_at, listings) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(connection_id, message_id) DO UPDATE SET "
            "sender = excluded.sender, sent_at = excluded.sent_at, listings = excluded.listings",
            [(connection_id, row["msg_id"], row["sender"], row["sent_at"], json.dumps(row["listings"]))
             for row in rows],
        )
        cached = conn.execute("SELECT sender, listings FROM google_gmail_alerts WHERE connection_id = ?",
                              (connection_id,)).fetchall()
    total = sum(len(json.loads(row["listings"])) for row in cached)
    return {"ok": True, "days_back": days_back,
            "summary": f"Synced {len(rows)} job-alert email(s).", "cached_total": total}


def listings(connection, days_back: int, query: str, limit: int) -> dict:
    with auth.connect() as conn:
        rows = conn.execute(
            "SELECT a.sender, a.sent_at, a.listings FROM google_gmail_alerts a "
            "JOIN google_gmail_connections c ON a.connection_id = c.id "
            "WHERE c.id = ? AND c.user_id = ? ORDER BY a.sent_at DESC, a.message_id",
            (connection["id"], connection["user_id"]),
        ).fetchall()
    cutoff = gmail_index.cutoff_iso(days_back)
    terms = query.casefold().split()
    jobs, cached_total = [], 0
    for row in rows:
        alerts = json.loads(row["listings"])
        cached_total += len(alerts)
        if row["sent_at"] < cutoff:
            continue
        for job in alerts:
            text = " ".join(job.get(key, "") for key in ("title", "company", "location")).casefold()
            if all(term in text for term in terms):
                jobs.append({**job, "sender": row["sender"], "source": row["sender"] or "Gmail"})
    jobs = gmail_index.diverse_listings(jobs, limit)
    sources = Counter(job["source"] for job in jobs)
    return {"sources": dict(sources), "total": len(jobs), "jobs": jobs,
            "days_back": days_back, "cached_total": cached_total}
