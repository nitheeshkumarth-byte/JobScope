"""Tests for the idle session timeout and the display identity.

The timeout is a security control, so the tests here are about enforcement
rather than appearance: a session nobody has touched must stop working even
though its absolute expiry is weeks away, an idle one must NOT stop working, and
an expired row must be deleted rather than left sitting in the database.

The identity tests are mostly about what a nickname is NOT allowed to be. It is
typed by a person and rendered into the header, a document title and LaTeX
output, so control characters and markup have to be stripped rather than escaped.
"""

import time

import pytest

import auth


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "users.db")
    monkeypatch.setattr(auth, "DB_FILE", path)
    auth.init_db()
    return path


def _user(db, email="a@b.com", password="a-good-password", owner=False):
    return auth.create_user(email, password, owner=owner)


def _signed_in(db, email="a@b.com"):
    u = _user(db, email)
    token, _ = auth.create_session(u["id"])
    return u, token


def _age_session(token, seconds):
    """Backdate a session's last_seen, as if it had been idle that long."""
    with auth.connect() as conn:
        conn.execute(
            "UPDATE sessions SET last_seen = ? WHERE token_hash = ?",
            (time.time() - seconds, auth._token_hash(token)))


# --------------------------------------------------------------------------
# idle timeout
# --------------------------------------------------------------------------

def test_a_fresh_session_has_the_full_allowance(db):
    _u, token = _signed_in(db)
    left = auth.seconds_left(token)
    assert auth.SESSION_IDLE_SECONDS - 5 <= left <= auth.SESSION_IDLE_SECONDS


def test_the_default_allowance_is_25_minutes():
    assert auth.SESSION_IDLE_MINUTES == 25
    assert auth.SESSION_IDLE_SECONDS == 1500
    assert auth.SESSION_EXTEND_MINUTES == 5


def test_using_a_session_resets_its_idle_clock(db):
    """The timeout is an IDLE timeout: activity must push it back.

    Without this the app logs out someone mid-task purely because they spent
    twenty-five minutes in one long-running board scrape.
    """
    _u, token = _signed_in(db)
    _age_session(token, auth.SESSION_IDLE_SECONDS - 20)
    assert auth.session_user(token) is not None, "session died while still in use"
    assert auth.seconds_left(token) > auth.SESSION_IDLE_SECONDS - 5


def test_an_idle_session_stops_authenticating(db):
    _u, token = _signed_in(db)
    _age_session(token, auth.SESSION_IDLE_SECONDS + 5)
    assert auth.session_user(token) is None


def test_the_idle_timeout_fires_long_before_the_absolute_expiry(db):
    """A stale cookie must not survive just because SESSION_DAYS is generous."""
    _u, token = _signed_in(db)
    _age_session(token, auth.SESSION_IDLE_SECONDS + 5)
    row = auth.session_row(token)
    assert row is not None, "absolute expiry already passed, test is meaningless"
    assert auth.session_user(token) is None


def test_an_expired_session_row_is_deleted_not_just_refused(db):
    """Deleting it is what makes 'no data carries into the next session' true."""
    _u, token = _signed_in(db)
    _age_session(token, auth.SESSION_IDLE_SECONDS + 5)
    assert auth.session_user(token) is None
    assert auth.session_row(token) is None
    with auth.connect() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
    assert n == 0


def test_idling_out_ends_the_session_without_touching_the_account(db):
    u = _user(db)
    token, _ = auth.create_session(u["id"])
    _age_session(token, auth.SESSION_IDLE_SECONDS + 5)
    assert auth.session_user(token) is None
    # The account survives; only the session went.
    assert auth.get_user(u["id"]) is not None
    assert auth.count_users() == 1


def test_extend_restores_the_full_allowance(db):
    _u, token = _signed_in(db)
    _age_session(token, auth.SESSION_IDLE_SECONDS - 30)
    left = auth.extend_session(token)
    assert auth.SESSION_IDLE_SECONDS - 5 <= left <= auth.SESSION_IDLE_SECONDS
    assert auth.seconds_left(token) > auth.SESSION_IDLE_SECONDS - 5


def test_extend_cannot_revive_an_expired_session(db):
    """Otherwise the +5 button would be a way to keep a stolen cookie alive."""
    _u, token = _signed_in(db)
    _age_session(token, auth.SESSION_IDLE_SECONDS + 5)
    assert auth.session_user(token) is None
    assert auth.extend_session(token) == 0
    assert auth.session_user(token) is None


def test_extend_restores_rather_than_accumulates(db):
    """Two clicks leave the normal allowance, not a doubled one.

    Otherwise clicking +5 repeatedly would build an unbounded session, which is
    precisely what the idle timeout exists to prevent.
    """
    _u, token = _signed_in(db)
    _age_session(token, 10)
    first = auth.extend_session(token)
    second = auth.extend_session(token)
    assert first == second


def test_seconds_left_is_zero_for_an_unknown_token(db):
    assert auth.seconds_left("nope") == 0
    assert auth.seconds_left("") == 0
    assert auth.session_row("nope") is None


def test_a_pre_migration_session_is_not_instantly_expired(db):
    """last_seen = 0 reads as 1970; a deploy must not log everyone out."""
    _u, token = _signed_in(db)
    with auth.connect() as conn:
        conn.execute("UPDATE sessions SET last_seen = 0 WHERE token_hash = ?",
                     (auth._token_hash(token),))
    assert auth.session_user(token) is not None
    assert auth.seconds_left(token) > 0


def test_one_users_session_going_idle_leaves_the_others_alone(db):
    u = _user(db)
    laptop, _ = auth.create_session(u["id"])
    phone, _ = auth.create_session(u["id"])
    _age_session(laptop, auth.SESSION_IDLE_SECONDS + 5)
    assert auth.session_user(laptop) is None
    assert auth.session_user(phone) is not None


# --------------------------------------------------------------------------
# schema migration
# --------------------------------------------------------------------------

def test_migration_adds_the_new_columns_to_an_existing_database(tmp_path, monkeypatch):
    """The whole point of _migrate: CREATE TABLE IF NOT EXISTS would skip these.

    A users.db created by an earlier build has no nickname/avatar column and no
    sessions.last_seen, and the first query naming one raises "no such column".
    """
    import sqlite3
    path = str(tmp_path / "users.db")
    legacy = sqlite3.connect(path)
    legacy.executescript(
        "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT "
        "NOT NULL UNIQUE, password_hash TEXT NOT NULL, created_at TEXT NOT NULL, "
        "verified_at TEXT, is_owner INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT "
        "NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, "
        "active_profile TEXT);")
    legacy.commit()
    legacy.close()

    monkeypatch.setattr(auth, "DB_FILE", path)
    auth.init_db()

    with auth.connect() as conn:
        users = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        sessions = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert {"nickname", "avatar"} <= users
    assert "last_seen" in sessions


def test_migration_is_idempotent(db):
    for _ in range(3):
        auth.init_db()
    with auth.connect() as conn:
        users = [r["name"] for r in conn.execute("PRAGMA table_info(users)")]
    assert users.count("nickname") == 1


def test_migration_keeps_existing_rows(db):
    u = _user(db)
    auth.init_db()  # a boot after the upgrade
    assert auth.get_user(u["id"])["email"] == u["email"]


# --------------------------------------------------------------------------
# nickname + avatar
# --------------------------------------------------------------------------

def test_a_nickname_is_saved_and_returned(db):
    u = _user(db)
    got = auth.set_identity(u["id"], nickname="Ada Lovelace")
    assert got["nickname"] == "Ada Lovelace"
    assert auth.get_user(u["id"])["nickname"] == "Ada Lovelace"


def test_a_nickname_is_trimmed_and_collapsed(db):
    u = _user(db)
    got = auth.set_identity(u["id"], nickname="   Ada    Lovelace  \n")
    assert got["nickname"] == "Ada Lovelace"


def test_control_characters_are_dropped_from_a_nickname(db):
    """A nickname is typed text, not markup, and it reaches LaTeX output."""
    u = _user(db)
    got = auth.set_identity(u["id"], nickname="Ada\x00\x07Lovelace")
    assert "\x00" not in (got["nickname"] or "")
    assert "\x07" not in (got["nickname"] or "")


def test_an_over_long_nickname_is_capped(db):
    u = _user(db)
    got = auth.set_identity(u["id"], nickname="x" * 500)
    assert len(got["nickname"]) == auth.MAX_NICKNAME


def test_an_empty_nickname_falls_back_to_the_address(db):
    u = _user(db)
    got = auth.set_identity(u["id"], nickname="   ")
    assert got["nickname"] is None
    assert auth.display_name(got) == "a"


def test_display_name_prefers_the_nickname(db):
    u = _user(db)
    got = auth.set_identity(u["id"], nickname="Ada")
    assert auth.display_name(got) == "Ada"


def test_display_name_falls_back_to_the_local_part(db):
    """Never the whole address: that is the one field read aloud on a share."""
    _user(db, "someone@example.com")
    got = auth.get_user_by_email("someone@example.com")
    assert auth.display_name(got) == "someone"
    assert "@" not in auth.display_name(got)


def test_display_name_survives_an_account_with_nothing_set(db):
    assert auth.display_name({}) == "there"
    assert auth.display_name({"email": ""}) == "there"


def test_a_known_avatar_is_accepted(db):
    u = _user(db)
    got = auth.set_identity(u["id"], avatar="forest")
    assert got["avatar"] == "forest"


def test_an_unknown_avatar_falls_back_to_the_default(db):
    """The UI cannot smuggle in an id the renderer does not know."""
    u = _user(db)
    for bad in ("../../etc/passwd", "<script>", "", "not-a-real-avatar"):
        got = auth.set_identity(u["id"], avatar=bad)
        assert got["avatar"] == auth.DEFAULT_AVATAR


def test_every_advertised_avatar_is_accepted(db):
    """auth.AVATARS is the contract; a typo in it would silently 500 the picker."""
    u = _user(db)
    for name in auth.AVATARS:
        assert auth.set_identity(u["id"], avatar=name)["avatar"] == name


def test_nickname_and_avatar_update_independently(db):
    u = _user(db)
    auth.set_identity(u["id"], nickname="Ada", avatar="ember")
    got = auth.set_identity(u["id"], avatar="cobalt")
    assert got["nickname"] == "Ada", "avatar change wiped the nickname"
    assert got["avatar"] == "cobalt"


def test_a_null_field_is_left_alone(db):
    u = _user(db)
    auth.set_identity(u["id"], nickname="Ada", avatar="ember")
    got = auth.set_identity(u["id"], nickname=None, avatar=None)
    assert got["nickname"] == "Ada"
    assert got["avatar"] == "ember"


def test_identity_is_per_account(db):
    a = _user(db, "a@b.com")
    b = _user(db, "c@d.com")
    auth.set_identity(a["id"], nickname="Ada", avatar="ember")
    assert auth.get_user(b["id"])["nickname"] is None
    assert auth.get_user(b["id"])["avatar"] is None


def test_set_identity_returns_the_user_unchanged_when_nothing_is_passed(db):
    u = _user(db)
    got = auth.set_identity(u["id"])
    assert got["id"] == u["id"]
