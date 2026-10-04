"""
cv_ocr.py - Tier-2 text recovery for image-only CVs.

An uploaded CV is usually a real PDF with a real text layer, and pypdf reads
that in milliseconds. The exception is a CV that was scanned, exported from a
phone camera, or re-saved as page images: pypdf returns nothing at all, and the
upload used to fail with "Could not read any text from that file".

This module is the second attempt for exactly that case. It renders the page
pixels with PyMuPDF and reads them with RapidOCR (PaddleOCR's models compiled to
ONNX, run through onnxruntime on the CPU).

Why this and not a vision-language model. Measured on the development machine
(no GPU, i7-8650U), Ollama generates 2.53 tok/s on a text-only model, so a
single CV page would take several minutes once the image itself is encoded,
and a VLM asked to "read" a page will confidently invent a plausible email
address or phone number. RapidOCR reads the same page in about 4-5 seconds and
returned the email and phone number character-exact. In a job-search app a
mangled contact address means applications silently going nowhere, which is
worse than refusing the file, so exactness matters more here than elegance.

Measured accuracy on a synthetic image-only A4 CV at 150 dpi: 3.15% character
error rate at 99.5% mean confidence, where the entire error was a single short
all-caps heading the detector skipped. Character accuracy did not improve at
200 or 300 dpi, and neither did runtime, so 150 is the default - see DPI below.

Nothing here raises at import time, and both PyMuPDF and RapidOCR are imported
lazily, so a checkout without them installed still starts and still handles
every text-based CV. Configuration:

    JOBSCOPE_OCR             auto (default) | on | off
    JOBSCOPE_OCR_MAX_PAGES   pages to read        (default 4)
    JOBSCOPE_OCR_DPI         render resolution    (default 150)
    JOBSCOPE_OCR_SECONDS     wall-clock budget    (default 45)
    JOBSCOPE_OCR_MIN_CONF    drop lines below this confidence (default 0.55)

The page cap and the time budget exist because this runs inside a request. A
40-page scanned CV must not hold a worker for minutes to produce a partial
result the caller never asked for; past the cap it returns what it has and
says so, which the caller surfaces to the user.
"""

import os
import re
import threading
import time

from dataclasses import dataclass, field

# Below this many characters a "text" extraction is not a CV, it is a stray
# glyph or a page number, and OCR is worth the few seconds it costs. A real
# one-page CV yields well over a thousand.
MIN_CHARS_FOR_TRUST = 200

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Rejects the common OCR confusions in a contact line without being clever
# about it: the point is to notice that a glyph was misread, not to guess the
# intended character. Domains must have a real TLD and an @.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Optional country code, optional bracketed area code, then 7-15 digits that may
# be separated by single spaces/dots/dashes. One separator character at a time
# is deliberate: it stops "2018 - 2021" from being read as a ten-digit number.
PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d(?:[\s.-]?\d){6,14}")
MIN_PHONE_DIGITS = 7

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def mode() -> str:
    """auto | on | off. Anything unrecognised is treated as auto."""
    value = (os.environ.get("JOBSCOPE_OCR") or "auto").strip().lower()
    return value if value in ("auto", "on", "off") else "auto"


def max_pages() -> int:
    return _env_int("JOBSCOPE_OCR_MAX_PAGES", 4, 1, 25)


def dpi() -> int:
    return _env_int("JOBSCOPE_OCR_DPI", 150, 72, 400)


def seconds_budget() -> int:
    return _env_int("JOBSCOPE_OCR_SECONDS", 45, 5, 600)


def min_conf() -> float:
    return _env_float("JOBSCOPE_OCR_MIN_CONF", 0.55, 0.0, 1.0)


def supported(filename: str) -> bool:
    """True for the container types this module can turn into pixels."""
    name = (filename or "").lower()
    return name.endswith(".pdf") or name.endswith(IMAGE_EXTS)


def available() -> bool:
    """Whether OCR can actually run here, independent of the mode setting.

    Resolved with find_spec rather than by importing: onnxruntime and opencv are
    heavy, and the dashboard calls this to decide what to tell the user before
    anyone has asked for an OCR read.
    """
    if mode() == "off":
        return False
    from importlib.util import find_spec
    try:
        return all(find_spec(name) is not None
                   for name in ("pymupdf", "rapidocr"))
    except (ImportError, ValueError):
        return False


def needs_ocr(text: str) -> bool:
    """True when an extraction came back too thin to be a real document.

    Deliberately counts letters rather than characters: a PDF can yield a
    surprising number of control and layout characters while still carrying no
    readable prose at all.
    """
    if mode() == "off":
        return False
    letters = sum(1 for ch in (text or "") if ch.isalpha())
    return letters < MIN_CHARS_FOR_TRUST


@dataclass
class OcrResult:
    """Outcome of a Tier-2 attempt. `ok` False carries a user-safe reason."""

    ok: bool = False
    text: str = ""
    pages: int = 0
    seconds: float = 0.0
    mean_conf: float = 0.0
    dropped_lines: int = 0
    truncated: bool = False
    email: str = ""
    phone: str = ""
    warnings: list[str] = field(default_factory=list)
    detail: str = ""


def _engine():
    """One shared RapidOCR instance; loading the ONNX models costs ~1s."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                from rapidocr import RapidOCR
                _ENGINE = RapidOCR()
    return _ENGINE


def _read_image(engine, payload: bytes) -> tuple[list[str], list[float]]:
    result = engine(payload)
    return list(result.txts or []), list(result.scores or [])


def _read_pdf(engine, data: bytes, limit: int, budget: float,
              stop_reason: str) -> tuple[list[str], list[float], int, float]:
    import pymupdf

    started = time.monotonic()
    texts: list[str] = []
    scores: list[float] = []
    read = 0
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            if index >= limit:
                stop_reason["reason"] = "page_cap"
                break
            if time.monotonic() - started > budget:
                stop_reason["reason"] = "time_budget"
                break
            # PNG keeps colour handling inside the decoder rather than us
            # hand-rolling an RGB->BGR swap on the pixmap buffer.
            pix = page.get_pixmap(dpi=dpi())
            page_text, page_scores = _read_image(engine, pix.tobytes("png"))
            texts.extend(page_text)
            scores.extend(page_scores)
            read += 1
    return texts, scores, read, time.monotonic() - started


def _find_phone(text: str) -> str:
    """Longest plausible phone number in the text.

    Several candidates can match one contact line, and the longest is not always
    right - "2018 to 2021" is a year range, and reading it as a phone number is
    worse than reporting none. A candidate must clear a digit floor, and a
    candidate written in international form wins outright because it is
    unambiguous.
    """
    best = ""
    best_digits = 0
    for match in PHONE_RE.finditer(text):
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < MIN_PHONE_DIGITS:
            continue
        if raw.lstrip().startswith("+"):
            return raw.strip(" .-")
        if len(digits) > best_digits:
            best, best_digits = raw.strip(" .-"), len(digits)
    return best


def _collect_contact(text: str) -> tuple[str, str]:
    email = EMAIL_RE.search(text)
    return (email.group(0) if email else "", _find_phone(text))


def ocr_document(data: bytes, filename: str) -> OcrResult:
    """Read an image-only CV (PDF or image) into text.

    Never raises: any failure comes back as `ok=False` with a `detail` safe to
    show a user, because a missing optional dependency must not turn into a 500
    on an upload.
    """
    result = OcrResult()
    setting = mode()
    if setting == "off":
        result.detail = "OCR is turned off (JOBSCOPE_OCR=off)."
        return result
    if not data:
        result.detail = "That file was empty."
        return result

    name = (filename or "").lower()
    if not (name.endswith(".pdf") or name.endswith(IMAGE_EXTS)):
        result.detail = "OCR only reads PDFs and images."
        return result

    try:
        engine = _engine()
    except Exception as exc:
        result.detail = (f"OCR is unavailable ({type(exc).__name__}). "
                         "Install pymupdf, rapidocr and onnxruntime to enable it.")
        return result

    limit = max_pages()
    budget = float(seconds_budget())
    floor = min_conf()
    stop_reason: dict = {"reason": ""}

    try:
        if name.endswith(".pdf"):
            texts, scores, pages, elapsed = _read_pdf(
                engine, data, limit, budget, stop_reason)
        else:
            texts, scores = _read_image(engine, data)
            pages, elapsed = 1, 0.0
    except Exception as exc:
        result.detail = f"OCR could not read that file ({type(exc).__name__})."
        return result

    kept, kept_scores = [], []
    dropped = 0
    for text, score in zip(texts, scores or [1.0] * len(texts)):
        clean = (text or "").strip()
        if not clean:
            continue
        if float(score) < floor:
            dropped += 1
            continue
        kept.append(clean)
        kept_scores.append(float(score))

    body = "\n".join(kept).strip()
    result.text = body
    result.pages = pages
    result.seconds = round(elapsed, 2)
    result.dropped_lines = dropped
    result.mean_conf = round(sum(kept_scores) / len(kept_scores), 4) if kept_scores else 0.0
    result.truncated = bool(stop_reason["reason"])
    result.email, result.phone = _collect_contact(body)

    if stop_reason["reason"] == "page_cap":
        result.warnings.append(
            f"Only the first {pages} page(s) were read "
            f"(JOBSCOPE_OCR_MAX_PAGES={limit}).")
    elif stop_reason["reason"] == "time_budget":
        result.warnings.append(
            f"Stopped after {pages} page(s) to stay inside the "
            f"{int(budget)}s limit.")
    if dropped:
        result.warnings.append(
            f"{dropped} low-confidence line(s) were left out; check the "
            "extracted text below.")

    if not kept:
        result.detail = "OCR ran but could not find readable text on the page."
        return result

    if not result.email:
        result.warnings.append(
            "No email address was recognised - check the text before saving.")
    result.ok = True
    return result