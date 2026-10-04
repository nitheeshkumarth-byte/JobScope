"""Tests for the accounts layer.

Everything here is a data-isolation test in disguise. The previous design kept
one profiles.json and one agent bundle for the whole process, so a second
person using the same server would silently get the first person's CV, search
history and filters. The store is now keyed by owner on every read, and these
tests exist to keep it that way.
"""

import os
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


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------

def test_a_password_is_never_stored_in_the_clear(db):
    _user(db, password="correct horse battery")
    row = auth.get_user_by_email("a@b.com")
    assert row["password_hash"] != "correct horse battery"
    assert "correct horse" not in row["password_hash"]


def test_two_accounts_with_the_same_password_get_different_hashes(db):
    _user(db, email="a@b.com", password="same-password")
    _user(db, email="c@d.com", password="same-password")
    one = auth.get_user_by_email("a@b.com")["password_hash"]
    two = auth.get_user_by_email("c@d.com")["password_hash"]
    # a per-password salt, so identical passwords are not recognisable
    assert one != two


def test_the_stored_hash_records_its_own_parameters(db):
    """So the scrypt cost can be raised later without invalidating anyone."""
    _user(db)
    stored = auth.get_user_by_email("a@b.com")["password_hash"]
    scheme, n, r, p, _salt, _hash = stored.split("$")
    assert scheme == "scrypt"
    assert (int(n), int(r), int(p)) == (auth.SCRYPT_N, auth.SCRYPT_R,
                                         auth.SCRYPT_P)
    assert auth.verify_password("a-good-password", stored) is True
    assert auth.verify_password("a-good-passwore", stored) is False


def test_verify_rejects_a_hash_it_cannot_parse(db):
    """A row written by some future version must not raise on login."""
    for junk in ("garbage", "", "bcrypt$1$2$3$a$b", "scrypt$x$y$z$a$b"):
        assert auth.verify_password("anything", junk) is False


def test_short_passwords_are_refused(db):
    with pytest.raises(ValueError):
        auth.hash_password("short")
    with pytest.raises(ValueError):
        auth.hash_password("")
    _user(db, password="a-good-password")
    with pytest.raises(ValueError):
        auth.create_user("b@b.com", "1234567")
    assert auth.get_user_by_email("b@b.com") is None


def test_a_wrong_password_and_an_unknown_address_are_indistinguishable(db):
    """Otherwise the login form discloses which addresses have accounts."""
    _user(db)
    assert auth.authenticate("a@b.com", "wrong") is None
    assert auth.authenticate("ghost@b.com", "wrong") is None


def test_authenticate_returns_the_account_for_good_credentials(db):
    _user(db)
    assert auth.authenticate("a@b.com", "a-good-password")["email"] == "a@b.com"


# --------------------------------------------------------------------------
# accounts
# --------------------------------------------------------------------------

def test_email_is_matched_case_insensitively(db):
    _user(db, email="Person@Example.COM")
    assert auth.get_user_by_email("person@example.com") is not None
    assert auth.get_user_by_email("PERSON@example.com") is not None


def test_addresses_without_a_domain_are_refused(db):
    for bad in ("", "not-an-email", "a@b", "@b.com"):
        with pytest.raises(ValueError):
            auth.create_user(bad, "a-good-password")


def test_duplicate_emails_are_refused(db):
    _user(db, email="a@b.com")
    with pytest.raises(ValueError):
        auth.create_user("a@b.com", "a-good-password")
    with pytest.raises(ValueError):
        auth.create_user("A@B.com", "a-good-password")


def test_the_first_account_is_the_owner_and_later_ones_are_not(db):
    first = auth.create_user("first@b.com", "a-good-password", owner=True)
    second = auth.create_user("second@b.com", "a-good-password")
    assert first["is_owner"] == 1 and second["is_owner"] == 0
    assert auth.owner_user()["email"] == "first@b.com"


def test_marking_verified_stores_a_timestamp_once(db):
    user = _user(db)
    assert user["verified_at"] is None
    auth.mark_verified(user["id"])
    first = auth.get_user(user["id"])["verified_at"]
    assert first is not None
    auth.mark_verified(user["id"])
    assert auth.get_user(user["id"])["verified_at"] == first


def test_count_users_drives_the_bootstrap_prompt(db):
    assert auth.count_users() == 0
    _user(db, email="a@b.com")
    assert auth.count_users() == 1


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

def test_a_session_resolves_to_its_user(db):
    user = _user(db)
    token, expires = auth.create_session(user["id"])
    assert auth.session_user(token)["email"] == "a@b.com"
    assert expires > time.strftime("%Y-%m-%dT%H:%M:%S")


def test_the_session_token_is_stored_hashed_not_raw(db):
    """A leaked database must not be replayable as a login."""
    user = _user(db)
    token, _ = auth.create_session(user["id"])
    with auth.connect() as conn:
        stored = conn.execute("SELECT token_hash FROM sessions "
                              "WHERE user_id = ?", (user["id"],)).fetchone()[0]
    assert stored != token
    assert stored == auth._token_hash(token)
    assert auth.session_user(token) is not None


def test_a_random_or_tampered_cookie_is_rejected(db):
    user = _user(db)
    token, _ = auth.create_session(user["id"])
    assert auth.session_user("not-a-real-token") is None
    assert auth.session_user(token[:-4] + "AAAA") is None
    assert auth.session_user("") is None
    assert auth.session_user(None) is None


def test_logout_kills_the_session_immediately(db):
    user = _user(db)
    token, _ = auth.create_session(user["id"])
    auth.destroy_session(token)
    assert auth.session_user(token) is None


def test_logging_out_of_one_device_leaves_the_other_signed_in(db):
    user = _user(db)
    laptop, _ = auth.create_session(user["id"])
    phone, _ = auth.create_session(user["id"])
    auth.destroy_session(laptop)
    assert auth.session_user(laptop) is None
    assert auth.session_user(phone) is not None


def test_expired_sessions_are_swept_and_not_accepted(db):
    user = _user(db)
    token, _ = auth.create_session(user["id"])
    with auth.connect() as conn:
        conn.execute("UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
                     ("2000-01-01T00:00:00", auth._token_hash(token)))
    assert auth.session_user(token) is None
    # the next login clears it out rather than accumulating dead rows
    auth.create_session(user["id"])
    with auth.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions "
                            "WHERE token_hash = ?",
                            (auth._token_hash(token),)).fetchone()[0] == 0


def test_sessions_do_not_cross_accounts(db):
    a = _user(db, email="a@b.com")
    b = _user(db, email="b@b.com")
    token_a, _ = auth.create_session(a["id"])
    token_b, _ = auth.create_session(b["id"])
    assert auth.session_user(token_a)["id"] == a["id"]
    assert auth.session_user(token_b)["id"] == b["id"]


def test_destroying_all_sessions_signs_out_every_device(db):
    user = _user(db)
    one, _ = auth.create_session(user["id"])
    two, _ = auth.create_session(user["id"])
    auth.destroy_user_sessions(user["id"])
    assert auth.session_user(one) is None
    assert auth.session_user(two) is None


def test_the_active_profile_travels_with_the_session(db):
    user = _user(db)
    pid = auth.create_profile(user["id"], "Second CV")
    token, _ = auth.create_session(user["id"])
    assert auth.session_profile(token) is None
    auth.set_session_profile(token, pid)
    assert auth.session_profile(token) == pid


# --------------------------------------------------------------------------
# one-time email links
# --------------------------------------------------------------------------

def test_a_verification_token_works_once_and_only_once(db):
    user = _user(db)
    token = auth.issue_token(user["id"], "verify", auth.VERIFY_TOKEN_HOURS)
    assert auth.consume_token(token, "verify")["id"] == user["id"]
    assert auth.consume_token(token, "verify") is None


def test_a_token_of_the_wrong_kind_is_refused(db):
    """A reset link must not double as a verification link, or vice versa."""
    user = _user(db)
    reset = auth.issue_token(user["id"], "reset", auth.RESET_TOKEN_HOURS)
    assert auth.consume_token(reset, "verify") is None
    assert auth.consume_token(reset, "reset")["id"] == user["id"]


def test_an_expired_token_is_refused(db):
    user = _user(db)
    token = auth.issue_token(user["id"], "verify", hours=-1)
    assert auth.consume_token(token, "verify") is None


def test_issued_tokens_are_not_stored_in_the_clear(db):
    user = _user(db)
    token = auth.issue_token(user["id"], "verify", auth.VERIFY_TOKEN_HOURS)
    with auth.connect() as conn:
        stored = conn.execute("SELECT token_hash FROM email_tokens").fetchone()[0]
    assert stored != token
    assert stored == auth._token_hash(token)


def test_issuing_a_new_link_retires_the_previous_one(db):
    """Otherwise an old reset link mailed days ago would still work."""
    user = _user(db)
    first = auth.issue_token(user["id"], "reset", auth.RESET_TOKEN_HOURS)
    second = auth.issue_token(user["id"], "reset", auth.RESET_TOKEN_HOURS)
    assert auth.consume_token(first, "reset") is None
    assert auth.consume_token(second, "reset") is not None


def test_an_unknown_purpose_is_refused(db):
    user = _user(db)
    with pytest.raises(ValueError):
        auth.issue_token(user["id"], "take_over", 1)


def test_empty_or_garbage_tokens_are_refused(db):
    assert auth.consume_token("", "verify") is None
    assert auth.consume_token("nonsense", "verify") is None


# --------------------------------------------------------------------------
# password reset
# --------------------------------------------------------------------------

def test_a_reset_replaces_the_password_and_kills_existing_sessions(db):
    """Otherwise a stolen cookie keeps working after the owner resets."""
    user = _user(db, password="old-password")
    token, _ = auth.create_session(user["id"])
    auth.set_password(user["id"], "a-brand-new-password")
    assert auth.authenticate("a@b.com", "a-brand-new-password") is not None
    assert auth.authenticate("a@b.com", "old-password") is None
    assert auth.session_user(token) is None


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------

def test_a_new_account_starts_with_one_empty_default_session(db):
    user = _user(db)
    profiles = auth.list_profiles(user["id"])
    assert [p["id"] for p in profiles] == ["default"]
    assert profiles[0]["skills"] == [] and profiles[0]["work_modes"] == []


def test_profiles_are_scoped_to_their_owner(db):
    a = _user(db, email="a@b.com")
    b = _user(db, email="b@b.com")
    pid = auth.create_profile(a["id"], "A's CV")
    assert auth.get_profile(a["id"], pid)["name"] == "A's CV"
    # b can neither read nor delete a's row
    assert auth.get_profile(b["id"], pid) is None
    assert auth.delete_profile(b["id"], pid) is False
    assert auth.get_profile(a["id"], pid) is not None


def test_writing_a_profile_under_another_accounts_id_does_not_leak(db):
    """Profile ids are random, and a save is an upsert, so one account can
    write a row under an id another account happens to hold. The guarantee is
    about reads: the id must still resolve to the owner's own data."""
    a = _user(db, email="a@b.com")
    b = _user(db, email="b@b.com")
    b_pid = auth.create_profile(b["id"], "B's CV")
    auth.save_profile(a["id"], b_pid, {"name": "a writes here"})

    assert auth.get_profile(b["id"], b_pid)["name"] == "B's CV"
    assert auth.get_profile(a["id"], b_pid)["name"] == "a writes here"
    # and the two rows are genuinely separate
    assert [p["name"] for p in auth.list_profiles(b["id"])] == [
        "Default", "B's CV"]


def test_listing_profiles_leaks_nothing_from_another_account(db):
    a = _user(db, email="a@b.com")
    b = _user(db, email="b@b.com")
    auth.create_profile(a["id"], "A's CV")
    assert [p["name"] for p in auth.list_profiles(b["id"])] == ["Default"]


def test_the_default_session_is_listed_first(db):
    user = _user(db)
    auth.create_profile(user["id"], "Newer CV")
    assert [p["id"] for p in auth.list_profiles(user["id"])][0] == "default"


def test_the_default_session_cannot_be_deleted(db):
    user = _user(db)
    assert auth.delete_profile(user["id"], "default") is False
    assert auth.get_profile(user["id"], "default") is not None


def test_deleting_a_session_leaves_the_default_intact(db):
    user = _user(db)
    pid = auth.create_profile(user["id"], "Scratch")
    assert auth.delete_profile(user["id"], pid) is True
    assert [p["id"] for p in auth.list_profiles(user["id"])] == ["default"]


def test_a_saved_profile_round_trips_every_field(db):
    user = _user(db)
    auth.save_profile(user["id"], "default", {
        "name": "Samsudheen", "roles": ["AI Engineer"], "cities": ["Hyderabad"],
        "skills": ["SQL", "Power BI"], "work_modes": ["remote"],
        "regions": ["europe"], "remote_only": True,
        "resume_text": "CV text", "github_url": "https://github.com/x"})
    rec = auth.get_profile(user["id"], "default")
    assert rec["name"] == "Samsudheen"
    assert rec["roles"] == ["AI Engineer"] and rec["skills"] == ["SQL", "Power BI"]
    assert rec["work_modes"] == ["remote"] and rec["regions"] == ["europe"]
    assert rec["resume_text"] == "CV text"
    # the id is authoritative from the column, not the payload
    assert rec["id"] == "default"


def test_a_saved_profile_cannot_forge_its_own_id(db):
    user = _user(db)
    auth.save_profile(user["id"], "default", {"id": "someone-elses",
                                              "name": "S"})
    assert auth.get_profile(user["id"], "someone-elses") is None
    assert auth.get_profile(user["id"], "default")["name"] == "S"


# --------------------------------------------------------------------------
# storage details
# --------------------------------------------------------------------------

def test_init_db_is_safe_to_call_twice(db):
    user = _user(db)
    auth.create_profile(user["id"], "CV")
    auth.init_db()            # e.g. a second worker booting on the same file
    assert len(auth.list_profiles(user["id"])) == 2


def test_deleting_an_account_cascades_to_sessions_and_profiles(db):
    user = _user(db)
    auth.create_profile(user["id"], "CV")
    token, _ = auth.create_session(user["id"])
    auth.issue_token(user["id"], "verify", 1)
    with auth.connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
        for table in ("sessions", "email_tokens", "profiles"):
            n = conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
            assert n == 0, table
    assert auth.session_user(token) is None


def test_the_default_db_path_is_a_gitignored_file_beside_the_source():
    """Password hashes and session cookies must never reach version control.

    The conftest guard redirects DB_FILE for every test, so this reads the
    shipped default out of the module source rather than the live attribute.
    """
    import inspect
    import re

    source = inspect.getsource(auth)
    assert re.search(r'^DB_FILE = os\.path\.join\(HERE, "users\.db"\)$',
                     source, re.M)

    root = os.path.dirname(os.path.abspath(auth.__file__))
    ignored = [line.strip() for line in
               open(os.path.join(root, ".gitignore"), encoding="utf-8")]
    assert "users.db" in ignored
