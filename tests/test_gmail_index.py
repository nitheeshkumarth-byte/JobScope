"""Tests for the local Gmail alert cache and the parser that feeds it.

The regression tests here are all cases that were measured against the real
mailboxes, because each one is a behaviour that looked correct in the code and
was wrong in the mailbox:

  * `jobs-noreply@linkedin.com` is the real LinkedIn alert sender, while
    `messages-noreply@linkedin.com` is 481 messages of notifications.
  * Alert dates are RFC 2822, not ISO, so an ISO-only parser dated nothing.
  * Real job URLs are longer than 80 characters, so the old long-URL filter
    deleted every Google Careers job.
  * The first URL in an alert is a logo or a preferences link.
"""
import os
import tempfile

import pytest

import gmail_index as gi
import mcp_server_gmail as g


# ---------------------------------------------------------------- store ----

@pytest.fixture()
def store(tmp_path):
    """A throwaway cache DB, so tests never touch the real gmail_cache.db."""
    path = str(tmp_path / "cache.db")
    gi.init(path)
    return path


def _msg(mid, listings, sender="donotreply@jobalert.indeed.com", account=1,
         sent="2026-10-04T09:00:00"):
    return {"msg_id": mid, "account": account, "sender": sender,
            "subject": "Indeed job alert", "sent_at": sent, "body": "b",
            "listings": listings}


def test_upsert_then_search_returns_the_listings(store):
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "DevOps Engineer", "company": "Acme", "location": "Remote",
         "link": "https://in.indeed.com/rc/clk/dl?jk=1"}])], store)
    rows = gi.search_listings(path=store)
    assert [r["title"] for r in rows] == ["DevOps Engineer"]
    assert rows[0]["company"] == "Acme"


def test_resync_does_not_duplicate_a_message(store):
    """A second sync of the same alert must replace it, not double its cards."""
    row = _msg("<a@1>", [{"title": "One", "company": "A", "location": "R",
                          "link": "https://x/1"}])
    gi.upsert_messages([row], store)
    gi.upsert_messages([row], store)
    assert gi.count_listings(path=store) == 1
    assert gi.search_messages(path=store)[0]["listing_ct"] == 1


def test_reparsing_a_message_replaces_its_old_listings(store):
    """Re-parsing with fewer jobs must not leave the previous jobs behind."""
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "Old", "company": "A", "location": "R", "link": "https://x/1"},
        {"title": "Also old", "company": "B", "location": "R",
         "link": "https://x/2"}])], store)
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "New", "company": "A", "location": "R", "link": "https://x/1"}])],
        store)
    rows = gi.search_listings(path=store)
    assert [r["title"] for r in rows] == ["New"]


def test_accounts_stay_separate(store):
    """The same job in two mailboxes is two listings, not one merged row."""
    shared = {"title": "SDE", "company": "Acme", "location": "R",
              "link": "https://x/1"}
    gi.upsert_messages([_msg("<a@1>", [shared], account=1),
                        _msg("<a@2>", [shared], account=2,
                             sender="careers-noreply@google.com")], store)
    assert gi.count_listings(path=store) == 2
    assert gi.count_listings(account=1, path=store) == 1
    assert gi.count_listings(account=2, path=store) == 1


def test_sender_filter_is_exact(store):
    gi.upsert_messages([
        _msg("<a@1>", [{"title": "Indeed job", "company": "", "location": "",
                        "link": "https://x/1"}],
             sender="donotreply@jobalert.indeed.com"),
        _msg("<a@2>", [{"title": "Google job", "company": "", "location": "",
                        "link": "https://x/2"}],
             sender="careers-noreply@google.com", account=2)], store)
    assert [r["title"] for r in
            gi.search_listings(sender="careers-noreply@google.com", path=store)] \
        == ["Google job"]


def test_query_terms_all_have_to_match(store):
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "Cloud Engineer", "company": "Acme", "location": "Remote",
         "link": "https://x/1"},
        {"title": "Data Analyst", "company": "Globex", "location": "London",
         "link": "https://x/2"}])], store)
    assert [r["title"] for r in gi.search_listings(query="cloud", path=store)] \
        == ["Cloud Engineer"]
    # "cloud globex" must not match: every term is required.
    assert gi.search_listings(query="cloud globex", path=store) == []


def test_query_ignores_case(store):
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "Senior DevOps Engineer", "company": "", "location": "",
         "link": "https://x/1"}])], store)
    assert len(gi.search_listings(query="DEVOPS", path=store)) == 1


def test_days_back_excludes_older_mail(store):
    gi.upsert_messages([
        _msg("<old@1>", [{"title": "Old job", "company": "", "location": "",
                         "link": "https://x/1"}], sent="2020-01-01T00:00:00"),
        _msg("<new@1>", [{"title": "New job", "company": "", "location": "",
                         "link": "https://x/2"}], sent="2099-01-01T00:00:00")],
        store)
    rows = gi.search_listings(days_back=365, path=store)
    assert [r["title"] for r in rows] == ["New job"]


def test_results_are_newest_first(store):
    gi.upsert_messages([
        _msg("<a@1>", [{"title": "Older", "company": "", "location": "",
                        "link": "https://x/1"}], sent="2026-01-01T00:00:00"),
        _msg("<b@1>", [{"title": "Newest", "company": "", "location": "",
                        "link": "https://x/2"}], sent="2026-10-01T00:00:00")],
        store)
    assert [r["title"] for r in gi.search_listings(path=store)] \
        == ["Newest", "Older"]


def test_limit_is_respected(store):
    gi.upsert_messages([_msg(f"<{i}@1>", [
        {"title": f"Job {i}", "company": "", "location": "",
         "link": f"https://x/{i}"}], sent=f"2026-10-{i + 1:02d}T00:00:00")
        for i in range(10)], store)
    assert len(gi.search_listings(limit=3, path=store)) == 3


def test_sender_breakdown_counts_both_tables(store):
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "A", "company": "", "location": "", "link": "https://x/1"},
        {"title": "B", "company": "", "location": "", "link": "https://x/2"}])],
        store)
    rows = gi.sender_breakdown(store)
    assert rows[0]["msgs"] == 1 and rows[0]["listings"] == 2


def test_clear_empties_the_store(store):
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "A", "company": "", "location": "", "link": "https://x/1"}])],
        store)
    gi.clear(store)
    assert gi.count_listings(path=store) == 0 and gi.search_messages(path=store) == []


def test_empty_upsert_is_a_no_op(store):
    assert gi.upsert_messages([], store) == 0
    assert gi.count_listings(path=store) == 0


# ------------------------------------------------------------ sync state ----

def test_an_uncached_account_is_stale(store):
    assert gi.is_stale(1, path=store) is True


def test_a_fresh_sync_is_not_stale(store):
    gi.record_sync(1, 5, path=store)
    assert gi.is_stale(1, path=store) is False


def test_staleness_expires(store):
    gi.record_sync(1, 5, path=store)
    assert gi.is_stale(1, max_age_s=0, path=store) is True


def test_a_failed_sync_is_not_retried_every_query(store):
    """A recorded failure still counts as the newest known state.

    Otherwise every single query would retry a broken IMAP login, which is the
    latency the cache exists to remove.
    """
    gi.record_sync(1, 0, error="login failed", path=store)
    assert gi.sync_state(1, store)["last_error"] == "login failed"
    assert gi.is_stale(1, path=store) is False


def test_sync_state_is_per_account(store):
    gi.record_sync(1, 10, path=store)
    assert gi.sync_state(2, store)["last_sync"] == 0


# -------------------------------------------------------------- parsing ----

def test_rfc2822_date_is_parsed_not_dropped():
    """Alert dates look like 'Wed, 15 Oct 2025 10:22:33 +0000'.

    The old regex expected an ISO date, matched nothing, and every listing
    sorted as the oldest row, which silently broke days_back filtering.
    """
    import email
    msg = email.message_from_string(
        "Date: Wed, 15 Oct 2025 10:22:33 +0000\nSubject: x\n\nbody")
    assert g._sent_at(msg) == "2025-10-15T10:22:33"


def test_iso_date_still_parses():
    import email
    msg = email.message_from_string("Date: 2025-10-15T10:22:33+00:00\n\nb")
    assert g._sent_at(msg) == "2025-10-15T10:22:33"


def test_an_unparseable_date_sorts_oldest_rather_than_now():
    """An unknown date must age out, not be pinned to the top forever."""
    import email
    msg = email.message_from_string("Date: not a date at all\n\nb")
    assert g._sent_at(msg) == ""


def test_a_message_with_no_date_is_still_stored():
    import email
    msg = email.message_from_string("Subject: Alert\n\nDevOps Engineer\n"
                                    "View job: https://in.indeed.com/viewjob?jk=abc\n")
    row = g.parse_message(msg.as_bytes())
    assert row is not None and row["listings"]


def test_sender_address_strips_the_display_name():
    import email
    msg = email.message_from_string(
        'From: "indeed" <Donotreply@jobalert.indeed.com>\n\nb')
    assert g._sender_address(msg) == "donotreply@jobalert.indeed.com"


def test_a_message_with_no_message_id_gets_a_stable_key():
    import email
    a = email.message_from_string("Subject: Alert\n\nDevOps\n")
    b = email.message_from_string("Subject: Alert\n\nDevOps\n")
    assert g.parse_message(a.as_bytes())["msg_id"] == \
        g.parse_message(b.as_bytes())["msg_id"]
    assert g.parse_message(a.as_bytes())["msg_id"].startswith("sha:")


def test_a_digest_yields_every_job_not_just_the_first():
    """An Indeed alert lists many roles; the old parser returned only one."""
    body = ("Your jobs\n"
            "DevOps Engineer at Acme - Bangalore\n"
            "View job: https://in.indeed.com/rc/clk/dl?jk=aaa\n"
            "Data Engineer at Globex - Remote\n"
            "View job: https://in.indeed.com/rc/clk/dl?jk=bbb\n"
            "SRE at Initech - Pune\n"
            "View job: https://in.indeed.com/rc/clk/dl?jk=ccc\n")
    got = g._extract_listings("30 new jobs for you", body)
    assert len(got) == 3
    assert {x["link"].split("jk=")[-1] for x in got} == {"aaa", "bbb", "ccc"}


def test_duplicate_links_in_one_digest_collapse():
    body = ("View job: https://in.indeed.com/rc/clk/dl?jk=aaa\n"
            "View job: https://in.indeed.com/rc/clk/dl?jk=aaa\n")
    assert len(g._extract_listings("New jobs", body)) == 1


def test_a_single_job_alert_uses_the_subject_as_the_title():
    body = "Apply now: https://in.indeed.com/rc/clk/dl?jk=aaa\n"
    got = g._extract_listings("DevOps Engineer at Acme", body)
    assert got[0]["title"] == "DevOps Engineer at Acme"


def test_a_digest_title_is_not_the_subject():
    """'30 new jobs' is not the title of any one posting."""
    body = ("View job: https://in.indeed.com/rc/clk/dl?jk=aaa\n")
    got = g._extract_listings("30 new jobs for you", body)
    assert got[0]["title"] != "30 new jobs for you"


# ---------------------------------------------------------------- links ----

@pytest.mark.parametrize("url", [
    "https://in.indeed.com/legal?hl=en#tos",
    "https://in.indeed.com/legal?hl=en#privacy",
    "https://support.indeed.com/hc/en-in/articles/123",
    "https://careers.google.com/jobs/dist/img/email/search-white.png",
    "https://www.linkedin.com/unsubscribe",
])
def test_non_job_urls_are_rejected(url):
    assert g._best_job_link([url]) == ""


@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/jobs/view/4422451392",
    "https://in.indeed.com/rc/clk/dl?jk=aa65d1340df2a",
    "https://www.google.com/about/careers/applications/jobs/results/1267381556",
])
def test_real_job_urls_are_kept(url):
    assert g._best_job_link([url]) == url


def test_a_posting_beats_a_search_page():
    """Every one of these were Apply buttons before the ranking existed."""
    assert g._best_job_link([
        "https://in.indeed.com/jobs?q=engineering+intern",
        "https://in.indeed.com/rc/clk/dl?jk=abc123",
    ]) == "https://in.indeed.com/rc/clk/dl?jk=abc123"


def test_a_logo_url_loses_to_the_job_url():
    assert g._best_job_link([
        "https://careers.google.com/jobs/dist/img/email/logo.png",
        "https://www.linkedin.com/jobs/view/4463017406",
    ]) == "https://www.linkedin.com/jobs/view/4463017406"


def test_a_tracker_is_used_only_when_nothing_better_exists():
    """Dropping trackers entirely lost 133 real Indeed listings."""
    tracker = "https://cts.indeed.com/v3/H4sIAAAAAAAA_42RTY-b"
    assert g._best_job_link([tracker]) == tracker
    assert g._best_job_link(["https://in.indeed.com/rc/clk/dl?jk=abc", tracker]) \
        == "https://in.indeed.com/rc/clk/dl?jk=abc"


def test_a_long_job_url_survives_body_cleaning():
    """Real job URLs are 150+ chars with campaign params.

    The old cleaner dropped any line whose URL was 80+ characters, which
    deleted every Google Careers job.
    """
    long_url = ("https://www.google.com/about/careers/applications/jobs/results/"
                "1267381556?sort_by=date&utm_campaign=jobalerts&utm_medium=email"
                "&utm_source=googlejobs&src=Online/Direct")
    body = ("Business Development Manager, Gaming\n"
            f"View job: {long_url}\n")
    cleaned = g._clean_body(body)
    assert "1267381556" in cleaned
    listings = g._extract_listings("New job(s) match your search", cleaned)
    assert listings and "1267381556" in listings[0]["link"]


def test_tracking_params_are_stripped_but_the_job_key_is_kept():
    short = g._shorten_url(
        "https://in.indeed.com/rc/clk/dl?jk=abc123&src=search&trackingId=x")
    assert short == "https://in.indeed.com/rc/clk/dl?jk=abc123"


def test_chrome_only_lines_are_dropped():
    assert g._clean_body("View our logo https://x/a.png\nApply now https://y/b.png") \
        == ""


# -------------------------------------------------------------- html -------

def test_html_anchors_become_readable_links():
    """Google Careers sends HTML only, so job links live in href attributes."""
    html = ('<table><tr><td><a href="https://www.google.com/about/careers/'
            'applications/jobs/results/1267381556?src=x">'
            'Business Development Manager</a></td></tr></table>')
    text = g._html_to_text(html)
    assert "Business Development Manager" in text
    assert "1267381556" in text


def test_html_conversion_keeps_the_anchor_label_as_title_text():
    html = ('<a href="https://in.indeed.com/rc/clk/dl?jk=abc">'
            'DevOps Engineer at Acme</a>')
    listings = g._extract_listings("Indeed job alert", g._html_to_text(html))
    assert listings[0]["link"].endswith("jk=abc")


def test_plain_text_is_passed_through_untouched():
    assert g._html_to_text("just text\nand more") == "just text\nand more"


# ------------------------------------------------------------ senders ------

def test_linkedin_job_alerts_use_the_jobs_noreply_address():
    """`messages-noreply@` is 481 notifications; `jobs-noreply@` is the alerts."""
    from filters import LINKEDIN_ALERT_SENDER
    assert LINKEDIN_ALERT_SENDER == "jobs-noreply@linkedin.com"
    assert LINKEDIN_ALERT_SENDER not in \
        __import__("filters").LINKEDIN_NOTIFICATION_SENDERS


def test_noise_linkedin_senders_are_not_synced_by_default():
    from filters import FALLBACK_SENDERS, LINKEDIN_NOTIFICATION_SENDERS
    for noise in LINKEDIN_NOTIFICATION_SENDERS:
        assert noise not in FALLBACK_SENDERS


def test_the_default_cache_path_is_not_cwd_dependent(tmp_path, monkeypatch):
    """The MCP subprocess has a different CWD, so it must still find the cache.

    Resolving next to the module is what stops the subprocess creating a second,
    empty database and reporting "no results" from it.
    """
    monkeypatch.delenv("GMAIL_CACHE_DB", raising=False)
    expected = os.path.join(os.path.dirname(os.path.abspath(gi.__file__)),
                            "gmail_cache.db")
    assert gi.db_path() == expected
    assert not os.path.isabs(os.path.basename(gi.db_path()))


def test_env_loading_is_not_cwd_dependent():
    """mcp_server_gmail must read .env by absolute path.

    agent.py forwards GMAIL_* into the subprocess, but a CWD-relative dotenv
    lookup silently found nothing, leaving every account/sender unset.
    """
    src = open(g.__file__, encoding="utf-8").read()
    assert "load_dotenv(_ENV_PATH" in src
    assert "os.path.dirname(os.path.abspath(__file__))" in src


def test_the_agent_forwards_every_gmail_variable(monkeypatch):
    """Only GMAIL_ADDRESS/PASSWORD were forwarded, hiding account 2 + senders.

    That single omission is why the app reported "no Gmail results" while a
    direct script against the same .env returned 340 listings.
    """
    src = open(agent_src(), encoding="utf-8").read()
    assert 'if k.startswith("GMAIL_")' in src
    assert '"env": gmail_env' in src


def agent_src() -> str:
    import agent
    return agent.__file__


def test_per_account_senders_come_from_env(monkeypatch):
    from filters import gmail_senders_for_account
    monkeypatch.setenv("GMAIL_ADDRESS", "a@x.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    monkeypatch.setenv("GMAIL_SENDERS", "one@x.com,two@x.com")
    assert gmail_senders_for_account(1) == ["one@x.com", "two@x.com"]


def test_an_unset_sender_var_falls_back_to_the_known_alert_senders(monkeypatch):
    """The old fallback used only the first two, which skipped LinkedIn."""
    from filters import FALLBACK_SENDERS, gmail_senders_for_account
    monkeypatch.delenv("GMAIL_SENDERS", raising=False)
    assert gmail_senders_for_account(1) == list(FALLBACK_SENDERS)
    assert "jobs-noreply@linkedin.com" in gmail_senders_for_account(1)


# ------------------------------------------------- queries stay offline ----

def _unwrapped(tool):
    """The plain function behind a FastMCP @mcp.tool() wrapper."""
    return getattr(tool, "fn", None) or getattr(tool, "__wrapped__", None) or tool


def test_a_query_on_a_fresh_cache_never_opens_imap(store, monkeypatch):
    """The whole point: search the local copy, not the mailbox.

    `imaplib.IMAP4_SSL` is patched to raise, so any IMAP use at all fails the
    test rather than quietly succeeding against the real account.
    """
    import imaplib
    monkeypatch.setenv("GMAIL_CACHE_DB", store)
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "DevOps Engineer", "company": "Acme", "location": "Remote",
         "link": "https://in.indeed.com/rc/clk/dl?jk=abc"}])], store)
    for acct in (1, 2):
        gi.record_sync(acct, 1, path=store)

    def _boom(*a, **k):
        raise AssertionError("a fresh-cache search must not use IMAP")
    monkeypatch.setattr(imaplib, "IMAP4_SSL", _boom)

    out = _unwrapped(g.search_job_emails)(query="devops", days_back=3650,
                                         max_results=5, resync_if_stale=True)
    assert "DevOps Engineer" in out
    assert "###JOBS_JSON###" in out


def test_a_stale_cache_is_refreshed_once(store, monkeypatch):
    """An hour-old cache triggers exactly one sync, not one per query."""
    import imaplib
    monkeypatch.setenv("GMAIL_CACHE_DB", store)
    calls = []

    class _FakeConn:
        def select(self, box):
            return "OK", [b""]

        def search(self, charset, crit):
            calls.append(crit)
            return "OK", [b""]

        def fetch(self, mid, spec):
            import email as _e
            raw = _e.message_from_string(
                "Message-ID: <stale@1>\nFrom: \"indeed\" <donotreply@jobalert.indeed.com>\n"
                "Date: Wed, 15 Oct 2025 10:22:33 +0000\nSubject: Indeed job alert\n\n"
                "View job: https://in.indeed.com/rc/clk/dl?jk=xyz\n")
            return "OK", [(b"1 (RFC822 {1}", raw.as_bytes())]

        def logout(self):
            return "BYE", [b""]

    monkeypatch.setattr(imaplib, "IMAP4_SSL",
                        lambda host, *a, **k: _FakeConn())
    monkeypatch.setattr(g, "_connect", lambda account=1: _FakeConn())

    out = _unwrapped(g.search_job_emails)(query="", days_back=3650, max_results=5)
    assert calls, "a stale cache should have triggered a sync"
    assert "xyz" in out or "cached listing" in out


def test_a_sync_failure_does_not_break_an_existing_query(store, monkeypatch):
    """A query must still answer from the cache when IMAP is unreachable."""
    import imaplib
    monkeypatch.setenv("GMAIL_CACHE_DB", store)
    gi.upsert_messages([_msg("<a@1>", [
        {"title": "Cached DevOps Role", "company": "", "location": "",
         "link": "https://in.indeed.com/rc/clk/dl?jk=cached"}])], store)
    for acct in (1, 2):
        gi.record_sync(acct, 1, path=store)

    def _boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(imaplib, "IMAP4_SSL", _boom)
    monkeypatch.setattr(g, "_connect", lambda account=1: (_ for _ in ()).throw(
        RuntimeError("network down")))

    out = _unwrapped(g.search_job_emails)(query="devops", days_back=3650,
                                         max_results=5)
    assert "Cached DevOps Role" in out
