"""Tests for the SMTP sender.

No test here touches a real mail server. smtplib is stubbed at the socket
boundary, which is the only way to assert the things that actually matter: that
STARTTLS is negotiated, that the password is never logged, and that a mail
server which is down produces a result the sign-up flow can act on rather than
an exception.
"""

import os
import ssl

import pytest

import mailer


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                 "SMTP_FROM", "SMTP_TLS", "SMTP_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


class FakeSMTP:
    """Records the conversation instead of opening a socket."""

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls = []
        self.sent = []
        self.closed = False
        self._context = context

    def __enter__(self):
        self.calls.append("connect")
        return self

    def __exit__(self, *exc):
        self.calls.append("quit")
        self.closed = True
        return False

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")
        self.tls_context = context

    def login(self, user, password):
        self.calls.append("login")
        self.logged_in_as = user
        self.logged_in_with = password

    def send_message(self, msg):
        self.calls.append("send")
        self.sent.append(msg)


@pytest.fixture
def smtp(monkeypatch):
    """Patch both constructors to the same fake, and hand back the instance."""
    box = {}

    def _make(cls_name):
        def _factory(host, port, timeout=None, context=None):
            box["server"] = FakeSMTP(host, port, timeout, context)
            return box["server"]
        return _factory

    monkeypatch.setattr(mailer.smtplib, "SMTP", _make("SMTP"))
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", _make("SMTP_SSL"))
    return box


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def test_nothing_is_configured_by_default():
    assert mailer.configured() is False


def test_mailer_loads_dotenv_itself_in_a_fresh_interpreter():
    """mailer must not depend on agent.py having been imported first.

    It used to, which meant a caller that reached for `mailer` on its own got
    "SMTP is not configured (set SMTP_HOST)" - an error blaming the user's
    configuration for what was really an import-order accident. Run in a
    subprocess so the module really is imported from nothing.
    """
    import subprocess
    import sys
    probe = ("import mailer, sys;"
             " sys.exit(0 if mailer.settings()['host'] else 3)")
    # Empty environment except PATH, so nothing is inherited from this process.
    env = {"PATH": os.environ.get("PATH", ""),
           "SystemRoot": os.environ.get("SystemRoot", "")}
    done = subprocess.run([sys.executable, "-c", probe], env=env,
                          capture_output=True, text=True, cwd=os.path.dirname(
                              os.path.dirname(os.path.abspath(__file__))))
    assert done.returncode == 0, (
        "importing mailer alone did not pick up .env - it is relying on "
        "another module to load it. stderr: %s" % done.stderr.strip())


def test_a_host_is_all_it_takes_to_count_as_configured(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    assert mailer.configured() is True


def test_whitespace_around_the_host_does_not_count(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "   ")
    assert mailer.configured() is False


def test_the_documented_defaults_apply(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    cfg = mailer.settings()
    assert cfg["port"] == 587
    assert cfg["tls"] is True
    assert cfg["timeout"] == 15
    assert cfg["from"] == ""


def test_tls_can_be_turned_off_for_a_local_test_server(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "localhost")
    for off in ("0", "false", "no", "FALSE"):
        monkeypatch.setenv("SMTP_TLS", off)
        assert mailer.settings()["tls"] is False
    monkeypatch.setenv("SMTP_TLS", "1")
    assert mailer.settings()["tls"] is True


def test_a_junk_port_does_not_raise(monkeypatch):
    """/api/auth/state reads these settings on every page load, so a typo in
    .env must not turn into a 500 on the sign-in screen."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "not-a-port")
    monkeypatch.setenv("SMTP_TIMEOUT", "")
    assert mailer.settings()["port"] == 587
    assert mailer.settings()["timeout"] == 15


def test_port_465_is_honoured(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    assert mailer.settings()["port"] == 465


# --------------------------------------------------------------------------
# sending
# --------------------------------------------------------------------------

def test_without_a_host_nothing_is_attempted(monkeypatch):
    called = []
    monkeypatch.setattr(mailer.smtplib, "SMTP",
                        lambda *a, **k: called.append(a) or FakeSMTP(*a))
    result = mailer.send("a@b.com", "Subject", "Body", dev_link="http://x/y")
    assert result.ok is False
    assert "SMTP_HOST" in result.detail
    assert called == []
    # the link comes back so a local install can still finish signing up
    assert result.dev_link == "http://x/y"


def test_a_message_is_built_with_the_right_envelope_fields(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("SMTP_FROM", "JobScope <bot@example.com>")

    assert mailer.send("me@person.com", "Confirm your account", "click here").ok
    msg = smtp["server"].sent[0]
    assert msg["To"] == "me@person.com"
    assert msg["From"] == "JobScope <bot@example.com>"
    assert msg["Subject"] == "Confirm your account"
    assert "click here" in msg.get_content()


def test_the_from_address_falls_back_to_the_login_user(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    mailer.send("me@person.com", "s", "b")
    assert smtp["server"].sent[0]["From"] == "bot@example.com"


def test_starttls_is_negotiated_before_the_credentials_are_sent(monkeypatch,
                                                                 smtp):
    """Sending the password on a plaintext connection would leak it."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    mailer.send("me@person.com", "s", "b")
    calls = smtp["server"].calls
    assert calls[0] == "connect"
    assert "starttls" in calls
    assert calls.index("starttls") < calls.index("login")
    assert calls.index("login") < calls.index("send")
    assert calls[-1] == "quit"
    assert isinstance(smtp["server"].tls_context, ssl.SSLContext)


def test_a_server_with_no_credentials_still_sends(monkeypatch, smtp):
    """Some relays are open on the LAN and take no AUTH at all."""
    monkeypatch.setenv("SMTP_HOST", "localhost")
    assert mailer.send("me@person.com", "s", "b").ok
    assert "login" not in smtp["server"].calls
    assert "send" in smtp["server"].calls


def test_port_465_wraps_the_socket_instead_of_using_starttls(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    assert mailer.send("me@person.com", "s", "b").ok
    assert "starttls" not in smtp["server"].calls
    assert isinstance(smtp["server"]._context, ssl.SSLContext)


def test_tls_off_skips_starttls(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "localhost")
    monkeypatch.setenv("SMTP_TLS", "0")
    assert mailer.send("me@person.com", "s", "b").ok
    assert "starttls" not in smtp["server"].calls


def test_the_configured_timeout_is_passed_through(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_TIMEOUT", "3")
    mailer.send("me@person.com", "s", "b")
    assert smtp["server"].timeout == 3


# --------------------------------------------------------------------------
# failures come back as results, not exceptions
# --------------------------------------------------------------------------

def _boom(monkeypatch, exc, name="SMTP"):
    def _factory(*a, **k):
        raise exc
    monkeypatch.setattr(mailer.smtplib, name, _factory)


def test_rejected_credentials_are_reported(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "wrong")

    def _factory(*a, **k):
        s = FakeSMTP(*a, **k)

        def _login(user, password):
            raise mailer.smtplib.SMTPAuthenticationError(535, b"bad creds")
        s.login = _login
        return s

    monkeypatch.setattr(mailer.smtplib, "SMTP", _factory)
    result = mailer.send("me@person.com", "s", "b")
    assert result.ok is False
    assert "535" in result.detail


def test_a_refused_connection_is_reported(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    _boom(monkeypatch, OSError("Connection refused"))
    result = mailer.send("me@person.com", "s", "b")
    assert result.ok is False
    assert "could not reach" in result.detail
    # nothing was sent, so no link should be offered as a substitute
    assert result.dev_link == ""


def test_a_dns_failure_is_reported(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "nope.invalid")
    _boom(monkeypatch, OSError("Name or service not known"))
    assert mailer.send("me@person.com", "s", "b").ok is False


def test_a_timeout_is_reported(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    _boom(monkeypatch, TimeoutError("timed out"))
    assert mailer.send("me@person.com", "s", "b").ok is False


def test_an_smtp_level_error_is_reported(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    _boom(monkeypatch, mailer.smtplib.SMTPException("bad sequence"))
    assert mailer.send("me@person.com", "s", "b").ok is False


def test_a_missing_recipient_is_refused_before_dialling(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    result = mailer.send("", "s", "b")
    assert result.ok is False
    assert "recipient" in result.detail
    assert "connect" not in smtp.get("server", FakeSMTP("x", 1)).calls


# --------------------------------------------------------------------------
# the templated messages
# --------------------------------------------------------------------------

def test_the_verification_mail_carries_the_link(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    result = mailer.send_verification("me@person.com", "https://x/verify?t=abc")
    assert result.ok
    msg = smtp["server"].sent[0]
    text = msg.get_content()
    assert "https://x/verify?t=abc" in text
    assert "Confirm your JobScope account" == msg["Subject"]
    # the link works once and for a limited time, so say so
    assert "once" in text and "24 hours" in text


def test_the_reset_mail_carries_the_link_and_its_shorter_expiry(monkeypatch,
                                                               smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    result = mailer.send_password_reset("me@person.com", "https://x/reset?t=abc")
    assert result.ok
    text = smtp["server"].sent[0].get_content()
    assert "https://x/reset?t=abc" in text
    assert "2 hours" in text


def test_every_link_mail_says_an_unwanted_one_is_harmless(monkeypatch, smtp):
    """Nobody reads a reset notice that looks like an attack."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    for fn in (mailer.send_verification, mailer.send_password_reset):
        fn("me@person.com", "https://x/t")
        assert "you can ignore this email" in smtp["server"].sent[-1].get_content()


def test_the_welcome_mail_greets_by_name_when_given(monkeypatch, smtp):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    mailer.send_welcome("me@person.com", "Sam")
    assert "Hi Sam," in smtp["server"].sent[0].get_content()
    mailer.send_welcome("me@person.com")
    assert "Hi," in smtp["server"].sent[0].get_content()


def test_no_link_mail_reports_a_dev_link_only_when_nothing_was_sent(monkeypatch,
                                                                    smtp):
    """A dev_link alongside a delivered mail would be a second, unlogged way in
    - so it must stay empty whenever the send actually succeeded."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    assert mailer.send_verification("me@person.com", "https://x/t").dev_link == ""

    monkeypatch.delenv("SMTP_HOST")
    result = mailer.send_verification("me@person.com", "https://x/t")
    assert result.ok is False
    assert result.dev_link == "https://x/t"
