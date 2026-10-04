"""Tests for RAG-driven tailoring in the generated resume.

Two properties matter more than the ranking itself:

1. Nothing is invented. Retrieval may move a bullet, never create one, and
   never alter its wording. resume_generator refuses to invent a skills section
   for exactly this reason, and switching retrieval on must not weaken that.
2. Without retrieval the document is unchanged. Every pre-RAG caller and test
   depends on build_resume's output being identical when no hits are passed.
"""

import re

import pytest

from agent import AgentConfig
import rag_store as R
import resume_generator as G

CV = """John Doe
john@example.com | +91 98765 43210 | Bengaluru, India
linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Data analyst who builds ETL pipelines and ships dashboards.

TECHNICAL SKILLS
Languages: Python, SQL, Java

EXPERIENCE
Acme Analytics - Data Analyst (2023 - present)
- Owned the PostgreSQL schema for 12M-row fact tables
- Built a Spark ETL pipeline that cut nightly batch time from 6h to 40m
- Mentored two interns on dbt and Airflow

Globex - Reporting Analyst (2022 - 2023)
- Rebuilt the monthly finance pack in Power BI

PROJECTS
Retail Forecast
- Modelled weekly demand with XGBoost, reaching MAPE 8.1%
- Built an interactive React dashboard for store managers
"""

JOB = {"title": "Data Engineer", "company": "Acme", "location": "Remote",
       "link": "https://example.com/job/1", "source": "Indeed"}


@pytest.fixture
def cfg():
    c = AgentConfig()
    c.resume_text = CV
    return c


def _hits(query: str, text: str = CV, doc: str = "John Doe CV.pdf"):
    chunks = [R.Chunk("cv", doc, "cv", i, c)
              for i, c in enumerate(R.chunk_text(text))]
    return R.rank(query, chunks)


def _bullets(tex: str) -> list[str]:
    # Experience/project bullets only; the skills block renders as \item
    # \textbf{...} and is not part of the reordering.
    return re.findall(r"\\item (?!\\textbf)(.+)", tex)


# --------------------------------------------------------------------------
# nothing changes when retrieval is off
# --------------------------------------------------------------------------

def test_no_hits_means_no_tailoring_line(cfg):
    tex, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline")
    assert "Tailored from" not in tex


def test_an_empty_hit_list_behaves_exactly_like_no_hits(cfg):
    with_none, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline")
    with_empty, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline", rag_hits=[])
    assert with_none == with_empty


def test_the_default_signature_is_unchanged_for_existing_callers(cfg):
    # Positional call, exactly as the old code and every test used it.
    tex, name = G.build_resume(cfg, JOB, "Spark ETL", "https://github.com/jd")
    assert tex.startswith("\\documentclass") or "\\begin{document}" in tex
    assert name


# --------------------------------------------------------------------------
# reordering
# --------------------------------------------------------------------------

def test_the_relevant_achievement_leads_its_entry(cfg):
    hits = _hits("Spark ETL pipeline experience")
    plain, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline experience")
    tailored, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline experience",
                                 rag_hits=hits)
    assert _bullets(plain)[1].startswith("Built a Spark ETL")
    assert _bullets(tailored)[0].startswith("Built a Spark ETL")


def test_a_different_posting_reorders_a_different_bullet(cfg):
    hits = _hits("React dashboard for store managers")
    tailored, _ = G.build_resume(cfg, JOB, "React dashboard",
                                 rag_hits=hits)
    projects = [b for b in _bullets(tailored) if "React" in b or "XGBoost" in b]
    assert projects[0].startswith("Built an interactive React dashboard")


def test_entry_order_is_never_changed(cfg):
    # Chronology is a fact about the candidate, not a relevance signal.
    hits = _hits("Power BI finance reporting")
    tex, _ = G.build_resume(cfg, JOB, "Power BI finance reporting", rag_hits=hits)
    assert tex.index("Acme Analytics") < tex.index("Globex")
    assert tex.index("Retail Forecast") > tex.index("Globex")


def test_bullets_keep_their_item_formatting(cfg):
    # Regression: re-emitting cleaned bullet bodies stripped the "- " marker,
    # so every reordered bullet silently became a bold paragraph line.
    hits = _hits("Spark ETL pipeline")
    tex, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline", rag_hits=hits)
    assert "Mentored two interns on dbt and Airflow" in tex
    # 3 under Acme + 1 under Globex + 2 under Retail Forecast.
    assert len(_bullets(tex)) == 6


def test_an_empty_posting_leaves_the_order_alone(cfg):
    hits = _hits("Spark ETL pipeline")
    plain, _ = G.build_resume(cfg, JOB, "")
    tailored, _ = G.build_resume(cfg, JOB, "", rag_hits=hits)
    assert _bullets(plain) == _bullets(tailored)


def test_reordering_is_stable_across_repeated_builds(cfg):
    hits = _hits("data analyst reporting")
    first, _ = G.build_resume(cfg, JOB, "data analyst reporting", rag_hits=hits)
    second, _ = G.build_resume(cfg, JOB, "data analyst reporting", rag_hits=hits)
    assert first == second


# --------------------------------------------------------------------------
# the no-invention guarantee
# --------------------------------------------------------------------------

def test_no_bullet_appears_that_the_cv_did_not_contain(cfg):
    for jd in ("Spark ETL pipeline", "React dashboard", "Power BI finance",
               "kubernetes rust", ""):
        hits = _hits(jd)
        tex, _ = G.build_resume(cfg, JOB, jd, rag_hits=hits)
        for bullet in _bullets(tex):
            # Strip the LaTeX escaping the renderer applies, then require the
            # wording to be the CV's own.
            plain = (bullet.replace("\\%", "%").replace("{", "")
                     .replace("}", "").replace("~", " ").strip())
            assert plain.rstrip(".") in CV.replace("\n", " ") or \
                any(plain.rstrip(".") in line
                    for line in CV.splitlines())


def test_a_document_the_cv_never_mentioned_is_not_invented(cfg):
    # Retrieval hits drawn from a different uploaded file may reorder, but
    # must never contribute their wording to the document. Scoped to the
    # bullets: the posting's own text legitimately appears in the Objective's
    # "Role focus from the posting" line, which is the JD echoing itself.
    hits = _hits("kubernetes autoscaling rust")
    tex, _ = G.build_resume(cfg, JOB, "kubernetes autoscaling rust",
                            rag_hits=hits)
    body = "\n".join(_bullets(tex))
    for word in ("Kubernetes", "Rust", "autoscaling"):
        assert word not in body


def test_an_unrelated_posting_leaves_the_resume_intact(cfg):
    hits = _hits("kitchen pastry chef")
    plain, _ = G.build_resume(cfg, JOB, "kitchen pastry chef")
    tailored, _ = G.build_resume(cfg, JOB, "kitchen pastry chef", rag_hits=hits)
    assert _bullets(plain) == _bullets(tailored)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def test_the_tailored_from_line_names_the_matched_document(cfg):
    hits = _hits("Spark ETL pipeline", doc="John Doe CV.pdf")
    tex, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline", rag_hits=hits)
    assert "Tailored from" in tex
    assert "John Doe CV.pdf" in tex


def test_the_tailored_from_line_lists_each_document_once(cfg):
    chunks = [R.Chunk("cv", "CV.pdf", "cv", i, c)
              for i, c in enumerate(R.chunk_text(CV))]
    chunks += [R.Chunk("cv2", "CV.pdf", "cv", 99, "- also matched here")]
    hits = R.rank("Spark ETL data analyst", chunks)
    tex, _ = G.build_resume(cfg, JOB, "Spark ETL data analyst", rag_hits=hits)
    assert tex.count("Tailored from") == 1
    assert tex.count("CV.pdf") == 1


def test_no_provenance_line_when_nothing_matched(cfg):
    tex, _ = G.build_resume(cfg, JOB, "kitchen pastry chef",
                            rag_hits=_hits("kitchen pastry chef"))
    assert "Tailored from" not in tex


def test_the_provenance_line_ends_with_a_real_newline(cfg):
    """It once ended in a literal backslash-n.

    The f-string was written "\\n" instead of "\n", so the two characters
    backslash and n were typeset into the finished document.
    """
    hits = _hits("Spark ETL pipeline", doc="John Doe CV.pdf")
    tex, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline", rag_hits=hits)
    line = next(l for l in tex.splitlines() if "Tailored from" in l)
    assert not line.endswith("\\n")
    assert "CV.pdf\\n" not in tex
    # A real newline means the line after it exists in the output at all.
    idx = tex.index("Tailored from")
    assert tex[idx:].splitlines()[0].endswith("John Doe CV.pdf")


def test_no_unsubstituted_template_token_reaches_the_document(cfg):
    """@ROLE@ and friends must never survive into a finished .tex.

    The CV-assembled path rebuilds the preamble from ATS_TEMPLATE, whose banner
    carries those tokens; only the canonical path fills them in.
    """
    hits = _hits("Spark ETL pipeline", doc="John Doe CV.pdf")
    tex, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline", rag_hits=hits)
    for token in ("@ROLE@", "@COMPANY@", "@SOURCE@", "@LINK@", "@NAME@",
                  "@CONTACT@", "@OBJECTIVE@", "@SKILLS@", "@EXPERIENCE@",
                  "@PROJECTS@", "@EDUCATION@", "@CERTIFICATIONS@"):
        assert token not in tex, token


def test_the_document_built_away_from_the_template_still_compiles_shaped(cfg):
    """Dropping the banner must not cost the preamble it introduced."""
    hits = _hits("Spark ETL pipeline", doc="John Doe CV.pdf")
    tex, _ = G.build_resume(cfg, JOB, "Spark ETL pipeline", rag_hits=hits)
    assert tex.startswith("\\documentclass")
    for needed in ("\\usepackage{enumitem}", "\\usepackage{hyperref}",
                   "\\definecolor{accent}", "\\begin{document}",
                   "\\end{document}"):
        assert needed in tex, needed
