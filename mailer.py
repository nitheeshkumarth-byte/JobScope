"""
mailer.py — outbound email over SMTP, using only the standard library.

JobScope already reads job-alert mail with imaplib rather than pulling in an
email package, so sending goes through smtplib for the same reason: no new
dependency, and the project keeps working offline.

Configuration comes from .env, which this module loads itself on import —
the same thing every entry point in the project (step2..step6) already does.
It used to rely on agent.py having been imported first, which meant any
caller that reached for `mailer` on its own reported "SMTP is not
configured (set SMTP_HOST)" and blamed the configuration for what was
really an import-order accident:

    SMTP_HOST      e.g. smtp.gmail.com          (required to send)
    SMTP_PORT      587                          (default; 465 = implicit TLS)
    SMTP_USER      full address for login      (optional; blank = no AUTH)
    SMTP_PASSWORD  app password / API key
    SMTP_FROM      envelope + From address     (default: SMTP_USER)
    SMTP_TLS       1 = STARTTLS on 587, 0 = plain (default 1)
    SMTP_TIMEOUT   seconds                      (default 15)

The load is deliberately done once at import rather than inside settings():
python-dotenv does not overwrite variables that already exist, so loading per
call would quietly repopulate SMTP_HOST behind a test that had just deleted
it, and "no SMTP configured" could never be reproduced.

Nothing here raises at import time. A missing or wrong configuration is
reported as a result the caller can log, because a broken mail server must
never take the dashboard down.
"""

import os
import smtplib
import ssl

from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

PORT = 587
TIMEOUT = 15


class MailResult:
    """Outcome of a send. `ok` False carries a reason safe to show a user."""

    def __init__(self, ok: bool, detail: str = "", dev_link: str = ""):
        self.ok = ok
        self.detail = detail
        # When SMTP is not configured we return the link instead of mailing it,
        # so a fresh checkout can still complete signup. Only ever populated
        # when nothing was actually sent.
        self.dev_link = dev_link

    def __repr__(self):
        return "MailResult(ok=%r, detail=%r)" % (self.ok, self.detail)


def _int_env(name: str, default: int) -> int:
    """Read an integer setting, ignoring anything unparseable.

    A typo in .env must not take the dashboard down: settings() is called by
    /api/auth/state, which the sign-in screen needs on every load.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def settings() -> dict:
    """Read SMTP settings from the environment, applying defaults."""
    return {
        "host": os.environ.get("SMTP_HOST", "").strip(),
        "port": _int_env("SMTP_PORT", PORT),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "from": os.environ.get("SMTP_FROM", "").strip(),
        "tls": os.environ.get("SMTP_TLS", "1").strip().lower() not in ("0", "false", "no"),
        "timeout": _int_env("SMTP_TIMEOUT", TIMEOUT),
    }


def configured() -> bool:
    return bool(settings()["host"])


def dev_links_allowed() -> bool:
    """Whether an unsent link may be handed back to the browser.

    Off unless explicitly enabled. Handing a link to whoever asked is only safe
    where the requester has already proved they own the address; where they have
    not, the difference between "here is your link" and "no such account" is an
    account-enumeration oracle. Registration is the one place it is safe, since
    signup necessarily reveals that an address is taken.
    """
    return os.environ.get("AUTH_DEV_LINKS", "").strip().lower() in (
        "1", "true", "yes", "on")


def send(to: str, subject: str, body: str, dev_link: str = "") -> MailResult:
    """Send one plain-text email.

    Returns a MailResult rather than raising: the caller decides whether a
    failure is fatal. A delivery problem must not leave the user staring at a
    500 with no idea what happened.
    """
    cfg = settings()
    if not cfg["host"]:
        return MailResult(False, "SMTP is not configured (set SMTP_HOST)", dev_link)
    if not to:
        return MailResult(False, "no recipient address")

    sender = cfg["from"] or cfg["user"] or "no-reply@localhost"
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if cfg["port"] == 465:
            # Implicit TLS: the socket is wrapped before the SMTP greeting.
            server = smtplib.SMTP_SSL(cfg["host"], cfg["port"],
                                      timeout=cfg["timeout"], context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=cfg["timeout"])
        with server:
            server.ehlo()
            if cfg["port"] != 465 and cfg["tls"]:
                server.starttls(context=ssl.create_default_context())
                server.ehlo()          # the capabilities changed after STARTTLS
            if cfg["user"] and cfg["password"]:
                server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        return MailResult(False, "SMTP rejected the credentials (%s)" % exc.smtp_code)
    except smtplib.SMTPException as exc:
        return MailResult(False, "SMTP error: %s" % exc)
    except OSError as exc:
        # DNS failure, refused connection, timeout.
        return MailResult(False, "could not reach the SMTP server: %s" % exc)
    return MailResult(True)


def _link_body(greeting: str, action: str, link: str, note: str = "") -> str:
    lines = [greeting, "", action, "", link, ""]
    if note:
        lines += [note, ""]
    lines += ["If you did not request this, you can ignore this email and "
              "nothing will change.", ""]
    return "\n".join(lines)


def send_verification(to: str, link: str) -> MailResult:
    """Confirm an address before the account can be used."""
    body = _link_body(
        "Welcome to JobScope,",
        "Confirm your email address to finish setting up your account:",
        link,
        "The link works once and expires in %d hours." % (24))
    return send(to, "Confirm your JobScope account", body, dev_link=link)


def send_password_reset(to: str, link: str) -> MailResult:
    body = _link_body(
        "Hi,",
        "Use this link to choose a new JobScope password:",
        link,
        "The link works once and expires in %d hours." % 2)
    return send(to, "Reset your JobScope password", body, dev_link=link)


def send_welcome(to: str, name: str = "") -> MailResult:
    who = (" %s" % name) if name else ""
    return send(to, "Your JobScope account is ready",
                "Hi%s,\n\nYour email is confirmed and your JobScope account is "
                "ready to use.\n\nUpload a CV to get your first set of "
                "suggested roles and skills.\n" % who)
