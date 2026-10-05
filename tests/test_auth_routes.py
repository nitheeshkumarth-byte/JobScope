"""End-to-end tests for the HTTP surface: the gate, the session cookie, and
the per-account isolation of every endpoint behind it.

These use the real router through TestClient with a temporary database and a
stubbed agent, so they cover the parts unit tests cannot: that a 401 actually
reaches the browser, that the cookie flags are right, and that two clients with
two cookies really do see two different sets of CV sessions.
"""

import pytest
from fastapi.testclient import TestClient

import auth
import dashboard as d
import mailer


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient on a throwaway database, with the agent stubbed out."""
    monkeypatch.setattr(auth, "DB_FILE", str(tmp_path / "users.db"))
    auth.init_db()
    # Never let a test migrate the real profiles.json into its throwaway
    # database; tests that care about the migration point this at a fixture
    # file of their own.
    monkeypatch.setattr(d, "PROFILES_FILE", str(tmp_path / "no-profiles.json"))

    class _Bundle:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tools = []

    async def _fake_create_agent(cfg=None, runtime=None):
        return _Bundle(cfg)

    monkeypatch.setattr(d, "create_agent", _fake_create_agent)
    monkeypatch.setattr(d, "create_runtime", _noop_runtime)
    monkeypatch.setattr(d.app.state, "bundles", {}, raising=False)
    d.app.state.bundles = {}
    monkeypatch.setattr(d.app.state, "runtime", object(), raising=False)

    with TestClient(d.app) as c:
        c.headers["host"] = "jobs.test"
        yield c


async def _noop_runtime():
    return object()


@pytest.fixture(autouse=True)
def no_smtp(monkeypatch):
    """Default every test to 'no mail server', so the dev-link path is used."""
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                 "SMTP_FROM", "AUTH_DEV_LINKS"):
        monkeypatch.delenv(name, raising=False)


def register(client, email="owner@jobs.test", password="a-good-password",
             verify=True):
    """Create an account and, unless asked otherwise, confirm it.

    With no SMTP the server self-confirms, so this is a plain registration."""
    r = client.post("/api/auth/register",
                    json={"email": email, "password": password})
    assert r.status_code in (200, 202), r.text
    body = r.json()
    if verify and body.get("needs_verification"):
        token = body["dev_link"].split("verify=")[-1]
        assert client.get("/api/auth/verify?token=" + token).status_code == 200
    return body


def sign_in(client, email="owner@jobs.test", password="a-good-password"):
    r = client.post("/api/auth/login",
                    json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r


def _capture(box):
    """A stand-in sender that records the link it was handed."""
    def _send(to, link):
        box["to"], box["link"] = to, link
        return mailer.MailResult(True, "sent")
    return _send


# --------------------------------------------------------------------------
# bootstrap and state
# --------------------------------------------------------------------------

def test_a_fresh_server_asks_for_the_first_account(client):
    st = client.get("/api/auth/state").json()
    # Subset check, not dict equality: this response now also carries the display
    # identity and the idle-timeout numbers, and adding a field must not break a
    # test about bootstrapping. Every key it did assert is still asserted.
    for key, want in (("needs_bootstrap", True), ("authenticated", False),
                      ("email", ""), ("verified", False),
                      ("smtp_configured", False)):
        assert st[key] == want, f"{key} was {st[key]!r}"
    # Nobody is signed in, so there is no identity and no time on the clock.
    assert st["nickname"] == ""
    assert st["needs_nickname"] is False
    assert st["seconds_left"] == 0
    assert st["idle_minutes"] == 25
    assert st["extend_minutes"] == 5


def test_the_state_response_reports_the_identity_and_countdown_once_signed_in(client):
    """One call has to carry everything the header renders, so a cold page load
    does not need a second round trip just to learn who is there."""
    reg = client.post("/api/auth/register",
                      json={"email": "ident@b.com", "password": "a-good-password"})
    assert reg.status_code == 200
    st = client.get("/api/auth/state").json()
    assert st["authenticated"] is True
    assert st["email"] == "ident@b.com"
    assert st["avatar"] == "aurora"          # the default, no pick made yet
    assert st["display_name"] == "ident"     # local part until a nickname is set
    assert st["needs_nickname"] is True
    assert st["seconds_left"] > 1400


def test_the_ui_is_served_before_anyone_signs_in(client):
    """The gate is in the page, so the browser can render it immediately."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="gate"' in r.text
    assert "bootAuth" in r.text


def test_after_signing_in_the_state_reports_the_account(client):
    register(client)
    sign_in(client)
    st = client.get("/api/auth/state").json()
    assert st["needs_bootstrap"] is False
    assert st["authenticated"] is True
    assert st["verified"] is True
    assert st["email"] == "owner@jobs.test"


def test_every_data_endpoint_is_closed_to_a_stranger(client):
    register(client)                      # the account exists...
    client.cookies.clear()                # ...but this visitor is not it
    for path in ("/api/config", "/api/profiles", "/api/logs"):
        r = client.get(path)
        assert r.status_code == 401, path
    assert client.post("/api/run", json={"message": "hi"}).status_code == 401
    assert client.post("/api/profiles").status_code == 401
    assert client.post("/api/resume/detect").status_code == 401
    assert client.post("/api/config", json={"model": "x"}).status_code == 401
    assert client.post("/api/auth/resend").status_code == 401


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def test_the_first_account_becomes_the_owner(client):
    body = register(client, verify=False)
    assert body["ok"] is True
    assert auth.owner_user()["email"] == "owner@jobs.test"


def test_registering_without_smtp_leaves_the_account_usable(client):
    """No mail server, so the server confirms the address itself rather than
    handing out an account nobody could ever get into."""
    body = register(client, verify=False)
    assert body["needs_verification"] is False
    assert body["verified"] is True
    # a session was issued, so the app is reachable straight away
    assert client.get("/api/auth/state").json()["authenticated"] is True
    assert client.get("/api/config").status_code == 200


def test_with_smtp_configured_a_link_is_mailed_and_nothing_is_returned(
        client, monkeypatch):
    captured = {}
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_verification", _capture(captured))

    body = client.post("/api/auth/register", json={
        "email": "owner@jobs.test", "password": "a-good-password"}).json()
    assert body["needs_verification"] is True
    assert body["dev_link"] == ""
    # the link points at the app so following it lands in the UI
    assert captured["link"].startswith("http://jobs.test/?verify=")


def test_an_unconfirmed_account_cannot_use_the_app(client, monkeypatch):
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_verification",
                        lambda to, link: mailer.MailResult(True, "sent"))
    client.post("/api/auth/register", json={
        "email": "owner@jobs.test", "password": "a-good-password"})

    # the session from registration works, so the person can be told what is
    # missing and can ask for another link...
    st = client.get("/api/auth/state").json()
    assert st["authenticated"] is True and st["verified"] is False
    # ...but the app itself is closed until the address is confirmed
    assert client.get("/api/config").status_code == 403
    assert client.get("/api/profiles").status_code == 403
    assert client.post("/api/run", json={"message": "hi"}).status_code == 403


def test_a_fresh_link_can_be_asked_for_but_only_while_signed_in(client,
                                                                monkeypatch):
    captured = {}
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_verification", _capture(captured))
    client.post("/api/auth/register", json={
        "email": "owner@jobs.test", "password": "a-good-password"})

    r = client.post("/api/auth/resend")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert captured["link"].startswith("http://jobs.test/?verify=")

    client.cookies.clear()
    assert client.post("/api/auth/resend").status_code == 401


def test_resending_retires_the_previous_link(client, monkeypatch):
    """Otherwise an old link keeps working after a newer one is requested."""
    links = []
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_verification",
                        lambda to, link: links.append(link)
                        or mailer.MailResult(True, "sent"))
    client.post("/api/auth/register", json={
        "email": "owner@jobs.test", "password": "a-good-password"})
    client.post("/api/auth/resend")

    stale = links[0].split("verify=")[-1]
    assert client.get("/api/auth/verify?token=" + stale).status_code == 400
    assert client.get("/api/auth/verify?token=" +
                      links[1].split("verify=")[-1]).status_code == 200


def test_following_the_link_opens_the_account_and_drops_old_cookies(
        client, monkeypatch):
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    captured = {}
    monkeypatch.setattr(d.mailer, "send_verification", _capture(captured))
    client.post("/api/auth/register", json={
        "email": "owner@jobs.test", "password": "a-good-password"})
    sign_in(client)                      # a session made before verification
    token = captured["link"].split("verify=")[-1]

    assert client.get("/api/auth/verify?token=" + token).json()["ok"] is True
    # the pre-verification cookie was destroyed, so the UI asks them to sign in
    assert client.get("/api/config").status_code == 401
    sign_in(client)
    assert client.get("/api/config").status_code == 200


def test_a_verification_link_only_works_once(client):
    body = register(client, verify=False)
    token = body["dev_link"].split("verify=")[-1]
    assert client.get("/api/auth/verify?token=" + token).status_code == 200
    assert client.get("/api/auth/verify?token=" + token).status_code == 400


def test_a_duplicate_registration_is_refused(client):
    register(client, verify=False)
    r = client.post("/api/auth/register",
                    json={"email": "owner@jobs.test",
                          "password": "another-password"})
    assert r.status_code == 400
    assert "already exists" in r.json()["error"]


def test_a_weak_password_is_refused_with_a_reason(client):
    r = client.post("/api/auth/register",
                    json={"email": "a@b.com", "password": "short"})
    assert r.status_code == 400
    assert "8 characters" in r.json()["error"]


# --------------------------------------------------------------------------
# the session cookie
# --------------------------------------------------------------------------

def test_signing_in_sets_an_httponly_cookie(client):
    register(client)
    r = sign_in(client)
    cookie = r.headers["set-cookie"]
    assert auth.SESSION_COOKIE in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie.replace("Lax", "lax")
    assert "Path=/" in cookie


def test_the_cookie_value_is_the_opaque_token_not_the_account(client):
    register(client)
    raw = sign_in(client).headers["set-cookie"]
    token = raw.split(auth.SESSION_COOKIE + "=")[1].split(";")[0]
    with auth.connect() as conn:
        stored = [r[0] for r in conn.execute("SELECT token_hash FROM sessions")]
    # the raw token is nowhere in the file, only its digest is
    assert token not in stored
    assert auth._token_hash(token) in stored
    # 32 random bytes, URL-safe
    assert len(token) >= 40
    assert token.replace("-", "").replace("_", "").isalnum()


def test_two_sessions_for_one_account_have_different_tokens(client):
    """A per-session token is what makes 'log out this device' possible."""
    register(client)
    first = sign_in(client).headers["set-cookie"]
    second = sign_in(client).headers["set-cookie"]
    pick = lambda raw: raw.split(auth.SESSION_COOKIE + "=")[1].split(";")[0]
    assert pick(first) != pick(second)


def test_signing_out_clears_the_cookie(client):
    register(client)
    sign_in(client)
    r = client.post("/api/auth/logout")
    assert r.status_code == 200
    assert 'js_session=""' in r.headers["set-cookie"].replace(" ", "")
    assert client.get("/api/config").status_code == 401


def test_a_wrong_password_is_refused_without_saying_which_part_was_wrong(client):
    register(client)
    r = client.post("/api/auth/login",
                    json={"email": "owner@jobs.test", "password": "wrong-one"})
    assert r.status_code == 401
    assert r.json()["error"] == "wrong email or password"
    # an unknown address gets the identical answer
    other = client.post("/api/auth/login",
                        json={"email": "ghost@jobs.test", "password": "wrong-one"})
    assert other.status_code == 401
    assert other.json()["error"] == r.json()["error"]


def test_the_login_page_is_still_reachable_after_signing_out(client):
    register(client)
    sign_in(client)
    client.post("/api/auth/logout")
    assert client.get("/").status_code == 200
    assert client.get("/api/auth/state").json()["authenticated"] is False


# --------------------------------------------------------------------------
# reset everything
# --------------------------------------------------------------------------

def test_reset_leaves_one_empty_session_and_nothing_else(client):
    """The whole-page reset: every session gone, the default rebuilt blank."""
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    saved = client.post("/api/profiles", json={"name": "Data Analyst"})
    d._patch_profile(1, saved.json()["id"], resume_text="Jane Doe\nPython",
                     skills=["Python"], cities=["Austin"],
                     roles=["Backend Engineer"], github_url="https://github.com/j")

    body = client.post("/api/reset").json()
    assert body["ok"] is True
    assert body["removed"] == 1

    cfg = client.get("/api/config").json()
    assert cfg["roles"] == []
    assert cfg["skills"] == []
    assert cfg["cities"] == []
    assert cfg["github_url"] == ""
    assert [p["id"] for p in client.get("/api/profiles").json()["profiles"]] == ["default"]
    assert client.get("/api/profiles").json()["profiles"][0]["resume_chars"] == 0


def test_reset_clears_the_cached_bundle_for_deleted_sessions(client):
    """The bundle cache was only pruned for the default id, so a deleted
    session's compiled graph - which captured the old config - survived."""
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    pid = client.post("/api/profiles", json={"name": "Data Analyst"}).json()["id"]
    client.get("/api/config")  # build the bundle while the session exists
    assert d._bundle_cache().get((1, pid)) is not None

    client.post("/api/reset")
    assert d._bundle_cache().get((1, pid)) is None
    assert d._bundle_cache().get((1, "default")) is None


def test_reset_clears_the_run_history_the_sidebar_lists(client):
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    d._remember_run({"query": "python", "total": 3}, uid=1)
    assert client.get("/api/logs").json()["runs"]

    client.post("/api/reset")
    assert client.get("/api/logs").json()["runs"] == []


def test_sessions_saved_in_the_same_second_keep_creation_order(client):
    """created_at only has second resolution, so sessions saved in the same
    second tied and the sort fell through to the profile id - a random hex
    string. The sidebar order of those sessions was therefore arbitrary AND
    unstable between calls, which made the Detect endpoint report a different
    "scanned" session for the same request. rowid is the creation sequence."""
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    ids = [client.post("/api/profiles", json={"name": f"S{i}"}).json()["id"]
           for i in range(6)]

    listed = [p["id"] for p in client.get("/api/profiles").json()["profiles"]]
    assert listed[0] == "default"
    # Same order, every single time - repeat because the old bug was random.
    for _ in range(5):
        assert [p["id"] for p in client.get("/api/profiles").json()["profiles"]] \
            == listed
    assert listed[1:] == ids


def test_reset_does_not_touch_the_login_or_another_account(client):
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    pid = client.post("/api/profiles", json={"name": "Theirs"}).json()["id"]

    client.cookies.clear()
    register(client, email="two@jobs.test")
    sign_in(client, "two@jobs.test")
    client.post("/api/profiles", json={"name": "Keep me"})
    d._patch_profile(1, "default", roles=["Backend Engineer"])

    client.post("/api/reset")
    assert client.get("/api/auth/state").json()["authenticated"] is True
    assert [p["id"] for p in client.get("/api/profiles").json()["profiles"]] == ["default"]

    client.cookies.clear()
    sign_in(client, "one@jobs.test")
    assert pid in [p["id"] for p in client.get("/api/profiles").json()["profiles"]]


# --------------------------------------------------------------------------
# two accounts, two servers' worth of state
# --------------------------------------------------------------------------

def test_two_accounts_cannot_see_each_others_sessions(client):
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    pid = client.post("/api/profiles").json()["id"]
    client.post("/api/resume", files={"file": ("cv.txt", b"one's CV", "text/plain")})

    client.cookies.clear()
    register(client, email="two@jobs.test")
    sign_in(client, "two@jobs.test")

    mine = client.get("/api/profiles").json()["profiles"]
    assert [p["id"] for p in mine] == ["default"]
    assert all("one" not in str(p) for p in mine)
    # and the other account's id is not addressable
    assert client.post("/api/profiles/activate", json={"id": pid}).status_code == 404
    assert client.post("/api/profiles/delete", json={"id": pid}).status_code == 404


def test_two_accounts_have_separate_configs(client, tmp_path, monkeypatch):
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    client.post("/api/config", json={"model": "llama3.1", "max_results": 11,
                                     "days_back": 7, "num_ctx": 2048,
                                     "ollama_url": "http://127.0.0.1:11434",
                                     "indeed_url": ""})
    assert client.get("/api/config").json()["max_results"] == 11

    client.cookies.clear()
    register(client, email="two@jobs.test")
    sign_in(client, "two@jobs.test")
    fresh = client.get("/api/config").json()
    assert fresh["max_results"] != 11
    assert fresh["days_back"] != 7
    assert fresh["email"] == "two@jobs.test"

    # and the first account still has its own
    client.cookies.clear()
    sign_in(client, "one@jobs.test")
    assert client.get("/api/config").json()["max_results"] == 11


def test_one_accounts_resume_is_not_leaked_into_another(client):
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    r = client.post("/api/resume", files={"file": ("cv.txt", b"SECRET CV TEXT",
                                                   "text/plain")})
    assert r.status_code == 200

    client.cookies.clear()
    register(client, email="two@jobs.test")
    sign_in(client, "two@jobs.test")
    assert "SECRET CV TEXT" not in client.get("/api/config").text
    assert "SECRET" not in str(client.post("/api/resume/detect").json())
    assert [p["resume_chars"] for p in
            client.get("/api/profiles").json()["profiles"]] == [0]


def test_detect_does_not_borrow_a_link_from_another_saved_session(client):
    """Detect reads the ACTIVE session only.

    It used to fall back to the account's other CVs, on the reasoning that the
    link belongs to the person rather than the session. That is what put a
    GitHub URL on a CV that has none: the answer was saved into the active
    profile's github_url, which the generated resume read ahead of the CV, so
    every document produced afterwards carried an unrelated account. A blank
    answer is the correct one.
    """
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    saved = client.post("/api/profiles", json={"name": "With CV"})
    cv_pid = saved.json()["id"]
    d._patch_profile(1, cv_pid,
                     resume_text="Jane Doe\nhttps://github.com/janedoe\nPython")

    # A fresh session whose CV has no GitHub of its own.
    blank = client.post("/api/profiles", json={"name": "Blank"}).json()["id"]
    d._patch_profile(1, blank, resume_text="Jane Doe\njane@example.com\nPython")

    body = client.post("/api/resume/detect").json()
    assert body["ok"] is True
    assert body["github_url"] == ""
    assert body["scanned"] == blank
    assert cv_pid != body["scanned"]


def test_detect_still_finds_the_link_in_the_active_session(client):
    """The narrowed scope must not break the button's actual job."""
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    saved = client.post("/api/profiles", json={"name": "With CV"})
    cv_pid = saved.json()["id"]
    d._patch_profile(1, cv_pid,
                     resume_text="Jane Doe\nhttps://github.com/janedoe\nPython")

    body = client.post("/api/resume/detect").json()
    assert body["github_url"] == "https://github.com/janedoe"
    assert body["scanned"] == cv_pid


def test_detect_uses_the_newly_uploaded_cv_not_a_stale_cached_bundle(client):
    """It reads the stored record, not ctx.cfg: the agent bundle is cached per
    (account, profile), so a CV written after the bundle was built is not in it.
    """
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    client.get("/api/config")  # build the cache before the upload
    uploaded = client.post("/api/resume", files={"file": (
        "cv.txt", b"Jane Doe\nhttps://github.com/fresh\n", "text/plain")}).json()
    assert uploaded["ok"] is True

    found = client.post("/api/resume/detect").json()
    assert found["github_url"] == "https://github.com/fresh"

    # With no CV anywhere, it says so rather than inventing a link.
    client.cookies.clear()
    register(client, email="two@jobs.test")
    sign_in(client, "two@jobs.test")
    empty = client.post("/api/resume/detect").json()
    assert empty["ok"] is True
    assert empty["github_url"] == ""
    assert empty["scanned"] == ""


def test_run_history_is_per_account(client):
    register(client, email="one@jobs.test")
    sign_in(client, "one@jobs.test")
    d._remember_run({"message": "one's hunt", "status": "ok", "tools": [],
                     "jobs": 1, "duration_s": 1.0, "error": None}, 1)

    client.cookies.clear()
    register(client, email="two@jobs.test")
    sign_in(client, "two@jobs.test")
    assert client.get("/api/logs").json()["runs"] == []


# --------------------------------------------------------------------------
# profiles over HTTP
# --------------------------------------------------------------------------

def test_creating_activating_and_deleting_a_session(client):
    register(client)
    sign_in(client)
    pid = client.post("/api/profiles").json()["id"]
    listing = client.get("/api/profiles").json()
    assert [p["id"] for p in listing["profiles"]] == ["default", pid]
    assert listing["active"] == pid          # creating it opens it

    assert client.post("/api/profiles/activate", json={"id": "default"}).status_code == 200
    assert client.get("/api/profiles").json()["active"] == "default"
    assert client.post("/api/profiles/activate", json={"id": "nope"}).status_code == 404
    assert client.post("/api/profiles/delete", json={"id": pid}).status_code == 200
    assert client.post("/api/profiles/delete", json={"id": pid}).status_code == 404


def test_deleting_the_open_session_falls_back_to_the_default(client):
    register(client)
    sign_in(client)
    pid = client.post("/api/profiles").json()["id"]
    client.post("/api/profiles/delete", json={"id": pid})
    assert client.get("/api/profiles").json()["active"] == "default"
    assert client.get("/api/config").status_code == 200


def test_the_default_session_cannot_be_deleted(client):
    register(client)
    sign_in(client)
    r = client.post("/api/profiles/delete", json={"id": "default"})
    assert r.status_code == 400
    assert "default" in r.json()["error"].lower()
    assert [p["id"] for p in
            client.get("/api/profiles").json()["profiles"]] == ["default"]


# --------------------------------------------------------------------------
# password reset
# --------------------------------------------------------------------------

def test_a_reset_link_sets_a_new_password_and_signs_everyone_out(client,
                                                                  monkeypatch):
    captured = {}
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_password_reset", _capture(captured))
    register(client)
    sign_in(client)
    stolen = client.cookies[auth.SESSION_COOKIE]

    client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    token = captured["link"].split("token=")[1].split("&")[0]

    r = client.post("/api/auth/reset", json={"token": token,
                                             "password": "a-newer-password"})
    assert r.status_code == 200
    # the cookie captured before the reset is dead
    assert client.get("/api/config").status_code == 401
    client.cookies.clear()
    sign_in(client, password="a-newer-password")


def test_the_reset_link_carries_the_address_so_the_form_can_prefill(client,
                                                                    monkeypatch):
    captured = {}
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_password_reset", _capture(captured))
    register(client)
    client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    assert "email=owner%40jobs.test" in captured["link"] or \
        "email=owner@jobs.test" in captured["link"]


def test_a_reset_token_only_works_once(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_password_reset", _capture(captured))
    register(client)
    client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    token = captured["link"].split("token=")[1].split("&")[0]
    assert client.post("/api/auth/reset", json={"token": token,
                                                 "password": "first-new-pass"}).status_code == 200
    assert client.post("/api/auth/reset", json={"token": token,
                                                 "password": "second-new-pass"}).status_code == 400


def test_a_garbage_reset_token_is_refused(client):
    register(client)
    r = client.post("/api/auth/reset", json={"token": "made-up",
                                             "password": "a-good-password"})
    assert r.status_code == 400
    assert "invalid or has expired" in r.json()["error"]


def test_forgot_never_reveals_whether_an_address_is_registered(client):
    """Same status code, same body, either way - otherwise this endpoint
    enumerates every account on the server."""
    register(client)
    known = client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    unknown = client.post("/api/auth/forgot", json={"email": "ghost@jobs.test"})
    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json() == {"ok": True, "sent": False}


def test_forgot_hands_back_the_link_only_when_explicitly_allowed(client,
                                                                 monkeypatch):
    """The escape hatch for a checkout with no mail server, off by default
    because returning a link to whoever asked is an enumeration oracle."""
    register(client)
    r = client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    assert "dev_link" not in r.json()

    monkeypatch.setenv("AUTH_DEV_LINKS", "1")
    r = client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    assert r.status_code == 202
    assert "token=" in r.json()["dev_link"]


def test_forgot_issues_a_reset_token_even_when_it_stays_silent(client):
    register(client)
    client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    with auth.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM email_tokens WHERE purpose = 'reset'"
        ).fetchone()[0] == 1


def test_a_reset_weak_password_is_refused_without_burning_the_token(client,
                                                                    monkeypatch):
    """Otherwise a typo would consume the link and strand the user."""
    captured = {}
    monkeypatch.setattr(d.mailer, "configured", lambda: True)
    monkeypatch.setattr(d.mailer, "send_password_reset", _capture(captured))
    register(client)
    client.post("/api/auth/forgot", json={"email": "owner@jobs.test"})
    token = captured["link"].split("token=")[1].split("&")[0]

    assert client.post("/api/auth/reset", json={"token": token,
                                                 "password": "short"}).status_code == 400
    assert client.post("/api/auth/reset", json={"token": token,
                                                 "password": "a-good-new-one"}).status_code == 200





# --------------------------------------------------------------------------
# migrating the old single-user install
# --------------------------------------------------------------------------

def test_the_legacy_sessions_land_in_the_first_account(client, tmp_path,
                                                       monkeypatch):
    legacy = tmp_path / "profiles.json"
    legacy.write_text(
        '{"default": {"id": "default", "name": "Samsudheen", '
        '"roles": ["AI Engineer"], "cities": ["Hyderabad"], '
        '"skills": ["SQL"], "work_modes": ["remote"], "regions": ["europe"], '
        '"remote_only": true, "resume_text": "MY CV"},'
        ' "abc123": {"id": "abc123", "name": "Second CV", "roles": [], '
        '"skills": []}}', encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))

    body = register(client, verify=False)
    # "abc123" is an empty record, so it is dropped rather than imported as a
    # blank sidebar entry; the populated one is kept.
    assert body["profiles_adopted"] == 1

    sign_in(client)
    profiles = client.get("/api/profiles").json()["profiles"]
    # The account opens on the blank template; the old CV is beside it.
    assert client.get("/api/profiles").json()["active"] == "default"
    assert [p["id"] for p in profiles][0] == "default"
    assert profiles[0]["resume_chars"] == 0
    assert profiles[0]["skills"] == 0

    imported = [p for p in profiles if p["name"] == "Samsudheen"]
    assert len(imported) == 1
    assert imported[0]["resume_chars"] == len("MY CV")
    assert imported[0]["skills"] == 1

    # The active (blank) session starts with no filters applied.
    cfg = client.get("/api/config").json()
    assert not cfg["work_modes"]
    assert not cfg["regions"]
    assert not cfg["cities"]
    assert not cfg["roles"]


def test_the_imported_legacy_session_keeps_its_filters(client, tmp_path,
                                                       monkeypatch):
    """Opening an imported CV must restore the way that install hunted."""
    legacy = tmp_path / "profiles.json"
    legacy.write_text(
        '{"default": {"id": "default", "name": "Samsudheen", '
        '"roles": ["AI Engineer"], "cities": ["Hyderabad"], '
        '"skills": ["SQL"], "work_modes": ["remote"], "regions": ["europe"], '
        '"remote_only": true, "resume_text": "MY CV"}}', encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))

    register(client, verify=False)
    sign_in(client)
    profiles = client.get("/api/profiles").json()["profiles"]
    imported = [p for p in profiles if p["name"] == "Samsudheen"][0]

    opened = client.post("/api/profiles/activate",
                         json={"id": imported["id"]}).json()
    assert opened["ok"] is True
    assert opened["roles"] == ["AI Engineer"]
    assert opened["cities"] == ["Hyderabad"]
    assert opened["work_modes"] == ["remote"]
    assert opened["regions"] == ["europe"]
    assert opened["skills"] == ["SQL"]


def test_the_legacy_file_is_untouched_by_the_migration(client, tmp_path,
                                                       monkeypatch):
    legacy = tmp_path / "profiles.json"
    original = '{"default": {"id": "default", "name": "S", "roles": [], "skills": []}}'
    legacy.write_text(original, encoding="utf-8")
    monkeypatch.setattr(d, "PROFILES_FILE", str(legacy))
    register(client, verify=False)
    assert legacy.read_text(encoding="utf-8") == original

# --------------------------------------------------------------------------
# session countdown and display identity
# --------------------------------------------------------------------------

def test_the_session_route_reports_the_countdown_to_a_signed_in_user(client):
    register(client)
    st = client.get("/api/auth/session").json()
    assert st["ok"] is True and st["signed_in"] is True
    assert st["seconds_left"] > 0
    assert st["idle_minutes"] == 25
    assert st["extend_minutes"] == 5


def test_the_session_route_reports_signed_out_without_a_cookie(client):
    st = client.get("/api/auth/session").json()
    assert st["signed_in"] is False
    assert st["seconds_left"] == 0


def test_the_session_route_does_not_extend_the_clock_by_being_read(client):
    """Polling it must not count as activity.

    The countdown polls this endpoint; if the poll refreshed last_seen, the
    session would be kept alive by the very widget meant to report it going
    stale, and would never log out.
    """
    register(client)
    token = client.cookies.get(auth.SESSION_COOKIE)
    _age_session(token, auth.SESSION_IDLE_SECONDS - 30)
    before = client.get("/api/auth/session").json()["seconds_left"]
    for _ in range(5):
        client.get("/api/auth/session")
    after = client.get("/api/auth/session").json()["seconds_left"]
    assert after <= before, "reading the clock pushed the deadline out"


def test_extending_pushes_the_deadline_back_out(client):
    register(client)
    _age_session(client.cookies.get(auth.SESSION_COOKIE),
                 auth.SESSION_IDLE_SECONDS - 20)
    r = client.post("/api/auth/extend").json()
    assert r["ok"] is True
    assert r["seconds_left"] > auth.SESSION_IDLE_SECONDS - 5


def test_extending_an_expired_session_reports_failure_rather_than_resurrecting_it(client):
    """A 200 here would leave the user watching a dashboard that 401s silently."""
    register(client)
    _age_session(client.cookies.get(auth.SESSION_COOKIE),
                 auth.SESSION_IDLE_SECONDS + 5)
    r = client.post("/api/auth/extend").json()
    assert r["ok"] is False
    assert r["signed_in"] is False
    assert r["seconds_left"] == 0


def test_an_idled_out_cookie_is_refused_by_the_protected_endpoints(client):
    register(client)
    _age_session(client.cookies.get(auth.SESSION_COOKIE),
                 auth.SESSION_IDLE_SECONDS + 5)
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/auth/session").json()["signed_in"] is False


def test_the_identity_can_be_set_and_is_returned_everywhere(client):
    register(client, email="ident@jobs.test")
    r = client.post("/api/auth/identity",
                    json={"nickname": "Ada", "avatar": "forest"}).json()
    assert r["ok"] is True
    assert r["nickname"] == "Ada" and r["avatar"] == "forest"
    # Both the lightweight session poll and the full state agree.
    assert client.get("/api/auth/session").json()["display_name"] == "Ada"
    assert client.get("/api/auth/state").json()["display_name"] == "Ada"


def test_setting_a_nickname_clears_the_needs_nickname_flag(client):
    register(client)
    assert client.get("/api/auth/state").json()["needs_nickname"] is True
    client.post("/api/auth/identity", json={"nickname": "Ada"})
    assert client.get("/api/auth/state").json()["needs_nickname"] is False


def test_an_unknown_avatar_is_refused_by_the_route(client):
    register(client)
    r = client.post("/api/auth/identity",
                    json={"avatar": "<script>"}).json()
    assert r["avatar"] == auth.DEFAULT_AVATAR


def test_the_identity_route_requires_a_session(client):
    assert client.post("/api/auth/identity", json={"nickname": "X"}).status_code == 401


def test_one_account_cannot_read_or_change_another_identity(client):
    """Two clients, two cookies, two accounts - the isolation that matters."""
    register(client, email="one@jobs.test")
    client.post("/api/auth/identity", json={"nickname": "One", "avatar": "ember"})
    cookie = dict(client.cookies)

    other = TestClient(d.app)
    other.headers["host"] = "jobs.test"
    other.post("/api/auth/register",
               json={"email": "two@jobs.test", "password": "a-good-password"})
    assert other.get("/api/auth/session").json()["display_name"] == "two"

    other.post("/api/auth/identity", json={"nickname": "Two"})
    client.cookies.clear()
    client.cookies.update(cookie)
    assert client.get("/api/auth/session").json()["display_name"] == "One"


def test_the_avatar_list_is_served_and_never_exceeds_the_valid_set(client):
    r = client.get("/api/auth/avatars").json()
    assert r["ok"] is True
    ids = [a["id"] for a in r["avatars"]]
    assert ids and set(ids) == set(auth.AVATARS)
    assert r["default"] == auth.DEFAULT_AVATAR
    for a in r["avatars"]:
        assert a["from"].startswith("#") and a["to"].startswith("#"), \
            f"{a['id']} has no gradient stops, so the swatch would render blank"


def test_every_avatar_has_a_colour_pair_in_the_server(client):
    """The picker renders what this returns, so a missing pair is a blank circle."""
    for name in auth.AVATARS:
        assert name in d._AVATARS, f"{name} has no colour defined"


def test_signing_out_drops_the_session_and_the_countdown(client):
    register(client)
    assert client.post("/api/auth/logout").json()["ok"] is True
    assert client.get("/api/auth/session").json()["signed_in"] is False


def _age_session(token, seconds):
    import time
    with auth.connect() as conn:
        conn.execute("UPDATE sessions SET last_seen = ? WHERE token_hash = ?",
                     (time.time() - seconds, auth._token_hash(token)))
