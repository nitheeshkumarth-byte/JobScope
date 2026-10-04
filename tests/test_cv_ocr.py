"""Tests for the image-only CV recovery path (cv_ocr) and its wiring.

OCR is the one reader in this project that can be wrong in a way the user
cannot see: it reads pixels, so a misread digit in a phone number or a swapped
character in an email looks like a real value and is then used for every
application this profile sends. Two things are therefore load-bearing here.

First, the decision to run OCR at all. A real text-based CV must never take the
slow path, and a container OCR cannot read must never be sent to it.

Second, the contact read. The recognised address has to come back exactly, and
the year-range / number ambiguity is the specific failure the phone pattern has
to avoid.

Almost everything below runs against a stub engine so the suite stays fast. One
test at the end uses the real RapidOCR to prove the contact fields survive a
genuine render-and-recognise round trip.
"""

import io
import os
import pathlib
import re
import sys
from types import SimpleNamespace
import types

import pytest

import cv_ocr
import dashboard as d


# --------------------------------------------------------------------------
# Deciding whether OCR should run at all
# --------------------------------------------------------------------------

def test_a_real_cv_is_never_handed_to_ocr():
    """The fast path has to stay the fast path. A CV with a proper text layer
    already parsed, and paying seconds of OCR on top of it would make every
    ordinary upload feel broken."""
    cv = ("Priya Sharma\nSenior Backend Engineer\n"
          "Technical Skills: Python, Go, PostgreSQL, Kafka\n"
          "Experience\nBuilt payment services for 8 years.\n") * 3
    assert cv_ocr.needs_ocr(cv) is False


@pytest.mark.parametrize("thin", [
    "",
    "   ",
    "\x0c",                       # a page break with no text on it
    "1",
    "Page 2",                     # only a page number survived
])
def test_an_empty_extraction_is_handed_to_ocr(thin):
    assert cv_ocr.needs_ocr(thin) is True


def test_needs_ocr_counts_letters_not_characters():
    """A PDF can hand back plenty of characters and still carry no prose - form
    feeds and layout glyphs are characters too, so the threshold has to be
    measuring readable letters."""
    junk = "\x0c" * 4000
    assert len(junk) > cv_ocr.MIN_CHARS_FOR_TRUST
    assert cv_ocr.needs_ocr(junk) is True


@pytest.mark.parametrize("name,expected", [
    ("cv.pdf", True),
    ("scan.PNG", True),
    ("photo.jpeg", True),
    ("page.webp", True),
    ("notes.txt", False),
    ("cv.docx", False),
    ("resume.md", False),
    ("", False),
])
def test_only_containers_ocr_can_read_are_supported(name, expected):
    assert cv_ocr.supported(name) is expected


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"])
def test_every_declared_image_extension_is_really_supported(ext):
    assert cv_ocr.supported("cv" + ext) is True


def test_off_switch_disables_the_whole_path(monkeypatch):
    monkeypatch.setenv("JOBSCOPE_OCR", "off")
    assert cv_ocr.mode() == "off"
    assert cv_ocr.available() is False
    assert cv_ocr.needs_ocr("") is False
    res = cv_ocr.ocr_document(b"%PDF-1.4 whatever", "cv.pdf")
    assert res.ok is False
    assert "off" in res.detail.lower()


def test_auto_and_on_are_distinct_but_both_run():
    for value in ("auto", "on"):
        os.environ["JOBSCOPE_OCR"] = value
        try:
            assert cv_ocr.mode() == value
            assert cv_ocr.available() is True
        finally:
            del os.environ["JOBSCOPE_OCR"]


def test_a_nonsense_mode_falls_back_to_auto(monkeypatch):
    """A typo in .env must not silently disable scanning, and must not raise
    either - auto is the only safe reading of an unrecognised value."""
    monkeypatch.setenv("JOBSCOPE_OCR", "maybe")
    assert cv_ocr.mode() == "auto"


# --------------------------------------------------------------------------
# Configuration bounds
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("9", 9), ("1", 1), ("999", 25), ("0", 1), ("abc", 4), ("", 4),
])
def test_page_cap_is_clamped_to_something_a_request_can_survive(raw, expected, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_OCR_MAX_PAGES", raw)
    assert cv_ocr.max_pages() == expected


@pytest.mark.parametrize("raw,expected", [
    ("600", 400), ("1", 72), ("nonsense", 150), ("", 150),
])
def test_dpi_is_clamped_to_a_range_the_ocr_actually_handles(raw, expected, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_OCR_DPI", raw)
    assert cv_ocr.dpi() == expected


def test_confidence_floor_is_clamped(monkeypatch):
    monkeypatch.setenv("JOBSCOPE_OCR_MIN_CONF", "5")
    assert cv_ocr.min_conf() == 1.0
    monkeypatch.setenv("JOBSCOPE_OCR_MIN_CONF", "-2")
    assert cv_ocr.min_conf() == 0.0


# --------------------------------------------------------------------------
# Contact extraction - the field the whole app acts on
# --------------------------------------------------------------------------

def test_email_is_pulled_out_exactly():
    text = "Priya Sharma\npriya.sharma@example.com | +91 98765 43210\n"
    email, _ = cv_ocr._collect_contact(text)
    assert email == "priya.sharma@example.com"


def test_an_obfuscated_email_is_left_alone_rather_than_guessed():
    """OCR reads what is on the page. If a CV says "name [at] domain [dot] com"
    there is no address to recover, and inventing one would put a fake address
    on every application."""
    text = "priya.sharma [at] example [dot] com\n"
    assert cv_ocr._collect_contact(text)[0] == ""


@pytest.mark.parametrize("text,expected", [
    ("Call +91 98765 43210 today", "+91 98765 43210"),
    ("Call 9876543210 today", "9876543210"),
    ("Call +1 (555) 123-4567 today", "+1 (555) 123-4567"),
    ("Tel: 020-7946-0958", "020-7946-0958"),
])
def test_phone_shapes_are_read_in_full(text, expected):
    """The digit floor used to accept the tail of a number instead of all of
    it, which reported "765 43210" for "+91 98765 43210" - a wrong phone
    number is worse than no phone number."""
    assert cv_ocr._find_phone(text) == expected


@pytest.mark.parametrize("text", [
    "Senior Engineer, Razorpay 2018 - 2021",
    "Reduced latency by 38 percent in 2021 to 2022",
    "B.Tech 2014 - 2018, IIT Madras",
])
def test_a_year_range_alone_yields_no_phone(text):
    """A dash between two four-digit years is eight digits in a row to a naive
    pattern, and reporting that as a phone number is worse than reporting none."""
    assert cv_ocr._find_phone(text) == ""


def test_too_few_digits_is_not_a_phone_number():
    assert cv_ocr._find_phone("Serial 12345") == ""
    assert cv_ocr._find_phone("PIN 9876") == ""


def test_international_form_beats_a_longer_national_number():
    """A candidate written with a leading + is unambiguous, so it wins even if
    some other run of digits elsewhere is longer."""
    text = "1234567890123 ref, or call +91 98765 43210"
    assert cv_ocr._find_phone(text) == "+91 98765 43210"


def test_missing_contact_is_reported_rather_than_defaulted():
    assert cv_ocr._collect_contact("Jane Doe\nData Analyst\n") == ("", "")


# --------------------------------------------------------------------------
# Reading a document, against a stub engine
# --------------------------------------------------------------------------

class _FakeOutput:
    def __init__(self, txts, scores):
        self.txts = tuple(txts)
        self.scores = tuple(scores)


class _FakeEngine:
    """Stands in for RapidOCR. Records calls so page counts can be asserted."""

    def __init__(self, lines_per_page=None, score=0.99, delay=0.0):
        self.lines_per_page = lines_per_page or ["Jane Doe", "Python, SQL"]
        self.score = score
        self.delay = delay
        self.calls = 0

    def __call__(self, payload):
        import time
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return _FakeOutput(self.lines_per_page, [self.score] * len(self.lines_per_page))


@pytest.fixture
def stub_engine(monkeypatch):
    """Install a fake engine and reset the module-level singleton."""
    def _install(engine):
        monkeypatch.setattr(cv_ocr, "_ENGINE", engine)
        return engine
    yield _install
    monkeypatch.setattr(cv_ocr, "_ENGINE", None)


def _image_only_pdf(pages=1):
    """A PDF whose only content is pixels, which is what pypdf cannot read."""
    pymupdf = pytest.importorskip("pymupdf")
    out = pymupdf.open()
    for _ in range(pages):
        page = out.new_page(width=300, height=400)
        page.insert_text((20, 40), "Jane Doe", fontsize=14)
    rendered = pymupdf.open()
    for page in out:
        pix = page.get_pixmap(dpi=72)
        new = rendered.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    data = rendered.tobytes()
    rendered.close()
    out.close()
    return data


def test_a_page_cap_stops_the_read_and_says_so(stub_engine):
    engine = stub_engine(_FakeEngine())
    res = cv_ocr.ocr_document(_image_only_pdf(6), "cv.pdf")
    assert res.ok is True
    assert res.pages == cv_ocr.max_pages()
    assert res.truncated is True
    assert engine.calls == cv_ocr.max_pages()
    assert any("first" in w for w in res.warnings)


def test_a_short_document_is_not_reported_as_truncated(stub_engine):
    stub_engine(_FakeEngine(lines_per_page=["jane@example.com", "Python, SQL"]))
    res = cv_ocr.ocr_document(_image_only_pdf(1), "cv.pdf")
    assert res.ok is True
    assert res.truncated is False
    assert not any("first" in w or "limit" in w for w in res.warnings)


def test_the_time_budget_stops_the_read_and_says_so(stub_engine, monkeypatch):
    # The page cap has to be lifted out of the way, or it would trip first and
    # the budget would never be the thing under test.
    monkeypatch.setattr(cv_ocr, "max_pages", lambda: 25)
    monkeypatch.setattr(cv_ocr, "seconds_budget", lambda: 5)
    engine = stub_engine(_FakeEngine(delay=1.4))
    res = cv_ocr.ocr_document(_image_only_pdf(8), "cv.pdf")
    assert res.truncated is True
    assert engine.calls < 8
    assert any("limit" in w for w in res.warnings)


def test_low_confidence_lines_are_dropped_and_counted(stub_engine, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_OCR_MIN_CONF", "0.8")
    engine = _FakeEngine(lines_per_page=["sure", "guess"], score=0.5)
    stub_engine(engine)
    res = cv_ocr.ocr_document(_image_only_pdf(1), "cv.pdf")
    assert res.dropped_lines == 2
    assert res.text == ""
    assert res.ok is False


def test_a_mixed_confidence_page_keeps_only_the_readable_lines(stub_engine, monkeypatch):
    monkeypatch.setenv("JOBSCOPE_OCR_MIN_CONF", "0.8")

    class _Mixed(_FakeEngine):
        def __call__(self, payload):
            self.calls += 1
            return _FakeOutput(["Jane Doe", "j4ne d0e"], [0.99, 0.20])

    stub_engine(_Mixed())
    res = cv_ocr.ocr_document(_image_only_pdf(1), "cv.pdf")
    assert res.text == "Jane Doe"
    assert res.dropped_lines == 1
    assert any("low-confidence" in w for w in res.warnings)


def test_a_missing_email_is_flagged_for_review(stub_engine):
    stub_engine(_FakeEngine(lines_per_page=["Jane Doe", "Data Analyst"]))
    res = cv_ocr.ocr_document(_image_only_pdf(1), "cv.pdf")
    assert res.ok is True
    assert res.email == ""
    assert any("email" in w.lower() for w in res.warnings)


def test_reported_confidence_and_timing_are_populated(stub_engine):
    stub_engine(_FakeEngine(lines_per_page=["a", "b", "c"], score=0.9))
    res = cv_ocr.ocr_document(_image_only_pdf(1), "cv.pdf")
    assert res.mean_conf == pytest.approx(0.9)
    assert res.seconds >= 0.0
    assert res.pages == 1


@pytest.mark.parametrize("data,filename,fragment", [
    (b"", "cv.pdf", "empty"),
    (b"hello", "notes.txt", "pdfs and images"),
])
def test_unreadable_input_fails_with_a_reason_not_an_exception(data, filename, fragment):
    res = cv_ocr.ocr_document(data, filename)
    assert res.ok is False
    assert fragment in res.detail.lower()


def test_a_corrupt_pdf_does_not_escape_as_an_exception(stub_engine, monkeypatch):
    """A missing optional dependency must never turn an upload into a 500."""
    stub_engine(_FakeEngine())
    res = cv_ocr.ocr_document(b"%PDF-1.4 not really a pdf", "cv.pdf")
    assert res.ok is False
    assert res.detail


def test_the_engine_is_only_built_once(monkeypatch):
    """Loading the ONNX models costs about a second, so a per-upload instance
    would make every OCR upload pay for it again."""
    import threading

    built = []

    class _Builder:
        def __init__(self):
            built.append(1)

        def __call__(self, payload):
            return _FakeOutput(["Jane Doe"], [0.99])

    monkeypatch.setattr(cv_ocr, "_ENGINE", None)
    monkeypatch.setattr(cv_ocr, "_ENGINE_LOCK", threading.Lock())
    monkeypatch.setitem(sys.modules, "rapidocr",
                        types.SimpleNamespace(RapidOCR=_Builder))
    first = cv_ocr._engine()
    second = cv_ocr._engine()
    assert first is second
    assert len(built) == 1


# --------------------------------------------------------------------------
# Dashboard wiring
# --------------------------------------------------------------------------

def test_a_parsed_cv_skips_ocr_entirely():
    """The ordinary path must not touch OCR - both the attempt and the run."""
    called = []

    def _spy(*args, **kwargs):
        called.append(1)
        return cv_ocr.OcrResult(ok=True, text="x")

    original = cv_ocr.ocr_document
    cv_ocr.ocr_document = _spy
    try:
        cv = ("Jane Doe\nSenior Backend Engineer\n"
              "Technical Skills: Python, SQL, Docker, Kubernetes\n"
              "Eight years building payment services and data pipelines.\n") * 3
        text, info = d._recover_text_by_ocr(b"%PDF-1.4", "cv.pdf", cv)
    finally:
        cv_ocr.ocr_document = original
    assert called == []
    assert info == {}
    assert "Jane Doe" in text


def test_an_unreadable_pdf_is_handed_to_ocr(monkeypatch):
    seen = {}

    def _fake(data, filename):
        seen["args"] = (data, filename)
        return cv_ocr.OcrResult(ok=True, text="Jane Doe\nPython, SQL",
                                pages=1, seconds=4.2, mean_conf=0.99,
                                email="jane@example.com", phone="+91 98765 43210")

    monkeypatch.setattr(cv_ocr, "ocr_document", _fake)
    text, info = d._recover_text_by_ocr(b"%PDF-1.4", "cv.pdf", "")
    assert seen["args"] == (b"%PDF-1.4", "cv.pdf")
    assert text == "Jane Doe\nPython, SQL"
    assert info["used"] is True
    assert info["email"] == "jane@example.com"
    assert info["phone"] == "+91 98765 43210"
    assert info["pages"] == 1


def test_a_text_container_is_not_handed_to_ocr(monkeypatch):
    called = []
    monkeypatch.setattr(cv_ocr, "ocr_document",
                        lambda *a: called.append(1) or cv_ocr.OcrResult())
    text, info = d._recover_text_by_ocr(b"hi", "notes.txt", "hi")
    assert called == []
    assert info == {}
    assert text == "hi"


def test_a_failed_ocr_keeps_the_original_text_and_reports_why(monkeypatch):
    monkeypatch.setattr(cv_ocr, "ocr_document",
                        lambda *a: cv_ocr.OcrResult(ok=False, detail="OCR is unavailable."))
    text, info = d._recover_text_by_ocr(b"%PDF-1.4", "cv.pdf", "")
    assert text == ""
    assert info["used"] is False
    assert info["detail"] == "OCR is unavailable."


def test_links_survive_the_ocr_recovery(monkeypatch):
    """A scan can still declare a hyperlink, and pypdf is the only reader that
    sees it - so the recovered text must not cost the user their GitHub link."""
    pytest.importorskip("pypdf")
    pymupdf = pytest.importorskip("pymupdf")

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((20, 40), "Jane Doe", fontsize=14)
    page.insert_link({
        "kind": pymupdf.LINK_URI,
        "from": pymupdf.Rect(20, 60, 200, 80),
        "uri": "https://github.com/janedoe",
    })
    data = doc.tobytes()
    doc.close()

    monkeypatch.setattr(cv_ocr, "ocr_document", lambda *a: cv_ocr.OcrResult(
        ok=True, text="Jane Doe\nPython", pages=1))
    text, _ = d._recover_text_by_ocr(data, "cv.pdf", "")
    assert "https://github.com/janedoe" in text


def test_the_ocr_payload_has_the_shape_the_client_expects():
    res = cv_ocr.OcrResult(ok=True, text="x", pages=2, seconds=8.5,
                           mean_conf=0.97, dropped_lines=1, truncated=True,
                           email="a@b.com", phone="+91 98765 43210",
                           warnings=["careful"], detail="")
    payload = d._ocr_payload(res)
    for key in ("used", "attempted", "pages", "seconds", "confidence",
                "dropped_lines", "truncated", "warnings", "email", "phone",
                "detail"):
        assert key in payload, key
    assert payload["used"] is True
    assert payload["confidence"] == 0.97
    assert payload["warnings"] == ["careful"]


def test_the_api_route_declares_the_ocr_field():
    """The upload response is the contract the dashboard renders from, so the
    OCR block has to be part of it on both the success and failure paths."""
    source = pathlib.Path(d.__file__).read_text(encoding="utf-8")
    assert '"ocr": ocr_info' in source
    assert "run_in_threadpool" in source, \
        "OCR must not run inline or it blocks the event loop"


# --------------------------------------------------------------------------
# The real thing, once
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_real_ocr_reads_the_contact_line_exactly(stub_engine):
    """End-to-end against actual RapidOCR on a genuinely image-only PDF.

    Everything above is stubbed, so this is the one test that proves the
    premise holds: a scan comes back with a usable email and phone number, not
    a plausible-looking invention.
    """
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("rapidocr")

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 70), "Priya Sharma", fontsize=20, fontname="hebo")
    page.insert_text((60, 100), "priya.sharma@example.com | +91 98765 43210",
                     fontsize=11, fontname="cour")
    page.insert_text((60, 130), "Technical Skills: Python, Go, Kafka", fontsize=11)
    src = doc.tobytes()
    doc.close()

    src_doc = pymupdf.open(stream=src, filetype="pdf")
    out = pymupdf.open()
    for sp in src_doc:
        pix = sp.get_pixmap(dpi=150)
        new = out.new_page(width=sp.rect.width, height=sp.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    scanned = out.tobytes()
    out.close()
    src_doc.close()

    # pypdf genuinely cannot read it, which is the whole reason this path exists.
    from pypdf import PdfReader
    assert not (PdfReader(io.BytesIO(scanned)).pages[0].extract_text() or "").strip()

    cv_ocr._ENGINE = None
    res = cv_ocr.ocr_document(scanned, "scan.pdf")
    assert res.ok is True, res.detail
    assert res.email == "priya.sharma@example.com"
    assert re.sub(r"\D", "", res.phone) == "919876543210"
    assert "Priya Sharma" in res.text


# --------------------------------------------------------------------------
# The whole route, in process
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_upload_route_accepts_a_scanned_cv(tmp_path, monkeypatch):
    """End-to-end through the real handler: a session, a real multipart upload
    of an image-only PDF, and a profile built from what OCR read.

    The skill extractor is stubbed because it is a model call, but nothing
    between the HTTP layer and the pixels is. A regression that left the route
    returning the old "Could not read any text" would fail here.
    """
    from starlette.testclient import TestClient

    pytest.importorskip("pymupdf")
    pytest.importorskip("rapidocr")

    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "users.db"
    monkeypatch.setattr(d.auth, "DB_FILE", str(path))
    d.auth.init_db()
    account = d.auth.create_user("scan@example.com", "hunter2hunter2")

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 70), "Priya Sharma", fontsize=20, fontname="hebo")
    page.insert_text((60, 100), "priya.sharma@example.com | +91 98765 43210",
                     fontsize=11, fontname="cour")
    page.insert_text((60, 130), "Technical Skills: Python, Go, Kafka",
                     fontsize=11, fontname="helv")
    src = doc.tobytes()
    doc.close()

    s = pymupdf.open(stream=src, filetype="pdf")
    o = pymupdf.open()
    for sp in s:
        pix = sp.get_pixmap(dpi=150)
        new = o.new_page(width=sp.rect.width, height=sp.rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("jpeg", jpg_quality=70))
    scanned = o.tobytes()
    o.close()
    s.close()

    async def _fake_skills(text, model, num_ctx):
        # Whatever OCR produced has to reach the extractor, and the recognised
        # contact line has to arrive intact or the profile is built on a lie.
        assert "priya.sharma@example.com" in text
        assert "Priya Sharma" in text
        return ["Python", "Go", "Kafka"]

    monkeypatch.setattr(d, "extract_skills", _fake_skills)
    async def _fake_agent(cfg, runtime=None):
        async def _never():
            yield {}
        return SimpleNamespace(cfg=cfg, astream=_never)
    monkeypatch.setattr(d, "create_agent", _fake_agent)
    d.app.state.runtime = object()
    d._bundle_cache().clear()
    monkeypatch.setattr(d, "detect_links", lambda text: {})

    client = TestClient(d.app)
    d.auth.mark_verified(account["id"])
    token, _expires = d.auth.create_session(account["id"])
    client.cookies.set(d.auth.SESSION_COOKIE, token)
    cv_ocr._ENGINE = None

    response = client.post("/api/resume",
                           files={"file": ("scan.pdf", scanned, "application/pdf")})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["ok"] is True, body
    assert body["ocr"]["used"] is True
    assert body["ocr"]["pages"] == 1
    assert body["ocr"]["email"] == "priya.sharma@example.com"
    assert sorted(body["skills"]) == ["Go", "Kafka", "Python"]

    stored = d.auth.list_profiles(account["id"])
    assert len(stored) == 2, "the upload should open its own new session"
    fresh = [p for p in stored if p["id"] != "default"][0]
    assert "priya.sharma@example.com" in fresh["resume_text"]


def test_a_text_upload_reports_no_ocr_block(tmp_path, monkeypatch):
    """A normal .txt CV must come back with no OCR block at all, so the client
    knows there is nothing to review."""
    from starlette.testclient import TestClient

    path = tmp_path / "users.db"
    monkeypatch.setattr(d.auth, "DB_FILE", str(path))
    d.auth.init_db()
    account = d.auth.create_user("txt@example.com", "hunter2hunter2")

    async def _fake_skills(text, model, num_ctx):
        return ["Python"]

    monkeypatch.setattr(d, "extract_skills", _fake_skills)
    async def _fake_agent(cfg, runtime=None):
        async def _never():
            yield {}
        return SimpleNamespace(cfg=cfg, astream=_never)
    monkeypatch.setattr(d, "create_agent", _fake_agent)
    d.app.state.runtime = object()
    d._bundle_cache().clear()
    monkeypatch.setattr(d, "detect_links", lambda text: {})

    body_text = ("Jane Doe\nData Analyst\nTechnical Skills: Python, SQL\n"
                 "Experience\nFive years of analytics work across teams.\n") * 3
    client = TestClient(d.app)
    d.auth.mark_verified(account["id"])
    token, _expires = d.auth.create_session(account["id"])
    client.cookies.set(d.auth.SESSION_COOKIE, token)
    response = client.post("/api/resume",
                           files={"file": ("cv.txt", body_text.encode(), "text/plain")})
    assert response.status_code == 200, response.text
    assert response.json()["ocr"] == {}