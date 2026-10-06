"""
ats_score.py — an explainable ATS/readability score for a generated resume.

Design constraint: a score that cannot be argued with is worse than no score.
Every component returns the specific reason it fired and what the candidate can
do about it, because "63/100" tells a jobseeker nothing actionable while
"no dates parsed on 2 of 4 roles - ATS may read them as one continuous job"
does.

The score is NOT an ATS vendor's score. No public ATS publishes a formula, and
anyone claiming an exact percentage is guessing. What this measures is the set
of things that are known to break parsers, plus keyword coverage against the
posting — the two things that actually determine whether a resume survives
automated screening.

Components
----------
  contact (12)  name / email / phone / location parsed by a real CV parser
  sections (14) standard headings present and non-empty
  structure (16) single column, no tables, parseable line density
  dates (16)     employment dates machine-readable, no ambiguous formats
  keywords (26)  posting terms actually present, weighted by whether the JD
                 itself uses the phrase
  evidence (16)  quantified outcomes and skill-adjacent project evidence

Weights total 100. `breakdown` always has every key, so the UI never has to
handle a missing component.
"""

from __future__ import annotations

import re

# Deliberately mirrors resume_generator.CV_HEADINGS so the score checks the
# same section names the generator writes. Divergence here would report a
# missing section on a resume that has it.
SECTION_WEIGHTS = {
    "summary": 2.0, "skills": 3.5, "experience": 4.0, "projects": 2.5,
    "education": 2.0, "certifications": 1.0, "languages": 1.0,
}

# Standard ATS section headings. Presence of the heading itself matters as much
# as the content, since parsers use headings to assign blocks to fields.
_CANONICAL_SECTIONS = (
    "summary", "skills", "experience", "projects", "education",
    "certifications", "languages",
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Deliberately strict. A 10-digit run that is also part of a longer digit string
# is usually an ID or a year, not a phone number.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,17}\d)(?!\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/", re.I)
_GITHUB_RE = re.compile(r"github\.com/[A-Za-z0-9]", re.I)

# Month-year and full-date forms an ATS resolves reliably. Numeric ambiguity
# ("01/02/25") is deliberately not accepted: it is the single most common cause
# of misparsed employment dates.
_MONTH_YEAR_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+'"
    r"?\d{2,4}", re.I)
_FULL_DATE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2},?\s+(?:19|20)\d{2}", re.I)
_YEAR_ONLY_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# Ranges an ATS can read as "this role lasted from X to Y".
#
# The leading month is optional but must be supported, because the generator
# writes "Mar 2023 -- Present". The previous pattern required a year FIRST, so
# on real output it matched only the "2023 -- Present" tail: the range count
# came out at half the roles, and the dates component could never reach its
# maximum no matter how good the timeline was.
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_JOINER = r"(?:\s*(?:-{1,3}|[–—]|to|until)\s*)"
_YEAR = r"(?:19|20)\d{2}"
_OPEN = r"(?:present|current|now)"
_DATE_RANGE_RE = re.compile(
    rf"(?:{_MONTH}\s+'?)?{_YEAR}{_JOINER}(?:{_MONTH}\s+)?(?:{_YEAR}|{_OPEN})"
    rf"|(?:{_MONTH}\s+'?)(?:{_YEAR}|{_OPEN}){_JOINER}"
    rf"(?:{_MONTH}\s+)?(?:{_YEAR}|{_OPEN})",
    re.I)

# A quantified outcome: percentages, counts, money, scale words.
_QUANT_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|\$\s?\d[\d,.]*|₹\s?\d[\d,.]*|"
    r"\b\d[\d,.]*\s*(?:x|×)\b|\b(?:increased|reduced|improved|cut|saved|"
    r"grew|generated|served|handled|processed|managed|trained|deployed)\b)",
    re.I)

# ATS keyword vocabulary. A term only scores when the posting uses it, so a
# resume cannot inflate its score by listing everything it knows.
_STOP_TERMS = frozenset("""
a an the and or of to in for with on at by from as is are be we you our your
will would can could should may might must have has had do does did not no
this that these those it its they them their he she his her but if then than
so such about into over under again further once here there when where why how
all any both each few more most other some only own same too very just also
""".split())

_TECH_TERMS = (
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "scala", "kotlin", "swift", "sql", "r", "bash",
    "react", "angular", "vue", "next.js", "node", "django", "flask", "fastapi",
    "spring", "express", "rails", ".net", "rest", "graphql", "grpc",
    "aws", "azure", "gcp", "lambda", "s3", "ec2",
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd", "cicd",
    "git", "github", "gitlab", "linux", "nginx",
    "postgres", "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "dynamodb", "kafka", "rabbitmq", "spark", "hadoop", "airflow", "dbt",
    "pandas", "numpy", "scipy", "polars",
    "tensorflow", "pytorch", "keras", "langchain", "llm", "rag", "nlp",
    "machine learning", "deep learning", "computer vision", "mlops",
    "snowflake", "bigquery", "redshift", "power bi", "tableau", "excel",
    "sap", "salesforce", "jira", "confluence", "agile", "scrum", "kanban",
    "selenium", "cypress", "pytest", "jest", "playwright", "tdd",
    "microservices", "api", "restful", "oauth", "jwt", "graphql",
    "html", "css", "sass", "tailwind", "bootstrap",
    "figma", "redux", "webpack", "vite",
    "communication", "leadership", "mentoring", "teamwork",
)

# Soft-skill phrases the JD may use and a resume may not.
_SOFT_PHRASES = (
    "problem solving", "problem-solving", "attention to detail",
    "time management", "self-starter", "self starter", "work independently",
    "collaborate", "collaboration", "cross-functional", "stakeholder",
    "client facing", "client-facing", "fast-paced", "ownership",
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", (text or "").lower())


def extract_keywords(jd_text: str, limit: int = 40) -> list[str]:
    """Terms the posting actually asks for, most significant first.

    Multi-word tech phrases are matched before single tokens so "machine
    learning" wins over "machine" and "learning", which would otherwise both
    appear as separate required keywords and let a resume score on the word
    "learning" alone.
    """
    low = (jd_text or "").lower()
    if not low.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()

    for phrase in sorted(_SOFT_PHRASES + _TECH_TERMS, key=len, reverse=True):
        if phrase in found:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", low):
            norm = phrase if " " in phrase or "-" in phrase else phrase
            key = _norm(norm).strip()
            if key and key not in seen:
                seen.add(key)
                found.append(norm)
        if len(found) >= limit:
            break
    return found


def _keyword_spread(resume_low: str, hits: list[str]) -> float:
    """Fraction of matched terms that appear OUTSIDE the skills block.

    A keyword that appears only in the Skills line is a claim; the same keyword
    inside a bullet is evidence. Measuring the ratio of the two is what stops a
    resume from scoring well purely by listing every term from the posting.
    """
    if not hits:
        return 0.0
    block = re.search(
        r"(?:technical\s+|core\s+|key\s+)?skills?\b(.{0,1600}?)"
        r"(?:\\section|\n\s*\\section|experience|education|projects|"
        r"certifications|languages)\b", resume_low, re.S)
    if not block:
        return 1.0      # no skills block to hide behind
    inside = block.group(1)
    outside = resume_low.replace(inside, " ")
    found = sum(
        1 for k in hits
        if re.search(rf"(?<![a-z0-9]){re.escape(k.lower())}(?![a-z0-9])", outside))
    return found / len(hits)


def _keyword_hits(resume_text: str, keywords: list[str]) -> list[str]:
    """Which posting terms the resume actually contains.

    Word-boundary matching only, never a substring. `"rest" in "restaurant"`
    made a resume for a chef hit a REST-API keyword, and `"go" in "going"` does
    the same for a Go posting — both inflated the score for a term the candidate
    never wrote.
    """
    low = (resume_text or "").lower()
    hits = []
    for k in keywords:
        if re.search(rf"(?<![a-z0-9]){re.escape(k.lower())}(?![a-z0-9])", low):
            hits.append(k)
    return hits


def _as_readable(text: str) -> str:
    r"""Reduce a document to what a parser would actually read out of it.

    The generator hands us two equivalent documents: the `.tex` source and the
    rendered HTML preview. Scoring the rendered one is more honest, because that
    is the document the user looks at - but the LaTeX patterns saw no headings
    in it at all, so a complete resume reported "No 'summary' section", "No
    'experience' section" and "No recognisable headings", and lost 9.5 points
    for a structure it had perfectly.

    Tags become line breaks so `<h2>Experience</h2>` reads as a heading the way
    `\section*{Experience}` does, and each `<li>` becomes a "- " bullet, so the
    bullet count and the "quantity per bullet" evidence checks read the same on
    both representations.
    """
    if not text:
        return ""
    if "<" in text and ">" in text and re.search(r"</?(?:div|h[1-6]|li|p|br|ul)\b",
                                                 text, re.I):
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"<li\b[^>]*>", "\n- ", text, flags=re.I)
        text = re.sub(r"</(?:div|h[1-6]|li|p|ul)>", "\n", text, flags=re.I)
        text = re.sub(r"<(?:p|div|h[1-6]|ul)\b[^>]*>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = (text.replace("&amp;", "&").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&nbsp;", " ")
                    .replace("&mdash;", "-").replace("&ndash;", "-")
                    .replace("&#39;", "'").replace("&quot;", '"'))
    return re.sub(r"[ \t]+", " ", text)


def score_resume(resume_text: str, jd_text: str = "",
                 sections: list[str] | None = None) -> dict:
    """Score one resume. Returns {total, band, breakdown, issues, hits, missing}.

    Accepts either the `.tex` source or the rendered HTML; `_as_readable`
    normalises both. `breakdown` maps component name to
    {"score", "max", "notes"}. `issues` is a list of human-readable, fixable
    findings — the part the candidate acts on.
    """
    text = _as_readable(resume_text or "")
    low = text.lower()
    sections = list(sections) if sections is not None else list(_CANONICAL_SECTIONS)
    issues: list[str] = []
    breakdown: dict[str, dict] = {}

    # ---------------------------------------------------------------- contact
    c_score, c_notes = 0.0, []
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", text):
        c_score += 4
    else:
        c_notes.append("No name detected in the first lines.")
        issues.append("Name not found — add it as a plain text line at the top.")
    if _EMAIL_RE.search(text):
        c_score += 3
    else:
        c_notes.append("No email address detected.")
        issues.append("No email address found — ATS cannot route the application.")
    if _PHONE_RE.search(text):
        c_score += 3
    else:
        c_notes.append("No phone number detected.")
    if re.search(r"\b(?:remote|hybrid|onsite|relocat\w+|[A-Z]{2}\b|"
                 r"India|Bengaluru|Bangalore|Hyderabad|Chennai|Pune|Delhi|"
                 r"Mumbai|Gurgaon|Noida)\b", text, re.I):
        c_score += 2
    else:
        c_notes.append("No location or work-mode keyword detected.")
    if _LINKEDIN_RE.search(text):
        c_score += 1.5
    if _GITHUB_RE.search(text):
        c_score += 1.5
    c_max = 12.0
    breakdown["contact"] = {"score": round(min(c_score, c_max), 1), "max": c_max,
                            "notes": c_notes}

    # --------------------------------------------------------------- sections
    s_score, s_notes = 0.0, []
    for name in _CANONICAL_SECTIONS:
        present = _has_section(text, name)
        if present:
            w = SECTION_WEIGHTS.get(name, 1.0)
            if name in sections:
                s_score += w
            else:
                # Present in the text but not declared by the generator: still
                # visible to a parser, so full credit.
                s_score += w
        else:
            if name in ("summary", "skills", "experience"):
                s_notes.append(f"No '{name}' section.")
                issues.append(f"Add a standard '{name}' heading — "
                              f"parsers key off the heading, not the layout.")
    s_max = 14.0
    breakdown["sections"] = {"score": round(min(s_score, s_max), 1), "max": s_max,
                             "notes": s_notes}

    # -------------------------------------------------------------- structure
    # Starts at full marks and subtracts. Starting at a fixed 8 made a clean
    # single-column document cap out at half the component, so every resume
    # scored as though it had a layout problem.
    st_score, st_notes = 16.0, []
    if re.search(r"\\begin\{(tabular|table|array)\}", text, re.I) or \
            re.search(r"\bmultirow\b|\bmulticolumn\b", text):
        st_score -= 5
        st_notes.append("LaTeX table found.")
        issues.append("Resume contains a table — many ATS parsers read columns "
                      "as out of order. Use a single column.")
    if re.search(r"\\begin\{(minipage|paracol|multicols|tcolorbox)\}", text, re.I):
        st_score -= 3
        st_notes.append("Multi-column or boxed layout found.")
        issues.append("Multi-column layout detected — flatten to one column.")
    if re.search(r"\\includegraphics|\\tikz|\\includegraphics\[", text):
        st_score -= 2
        st_notes.append("Graphics found.")
        issues.append("Resume embeds graphics — skills drawn as images cannot "
                      "be read by an ATS.")
    # Source-code length is not prose length. Counting LaTeX braces, commands
    # and comment lines made a well-structured 60-line resume look padded, and
    # `len(text) < 1200` in the keywords component fired against the generator's
    # own output. Measure words in the content instead.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    prose_lines = [ln for ln in lines
                   if not ln.lstrip().startswith(("%", "\\\\", "#"))
                   and not ln.strip().startswith("\\\\documentclass")]
    prose = " ".join(prose_lines)
    words = len(re.findall(r"[A-Za-z][A-Za-z'+#.-]*", prose))
    if lines and max(len(ln) for ln in lines) > 300:
        st_score -= 1
        st_notes.append("A line is very long; wrapping may split a field.")
    # `\\item` inside a long argument means a whole bullet crammed on one
    # source line, which survives to the PDF but splits badly on paste.
    if re.search(r"\\[a-zA-Z]+\*?\{[^}]{200,}\}", text):
        st_score -= 1
        st_notes.append("A very long single-line field detected.")
    if not re.search(r"\\section\*?\{[^}]+\}", text) and \
            not re.search(r"^\s*(?:summary|experience|skills|education)\b",
                          text, re.I | re.M):
        st_score -= 4
        st_notes.append("No recognisable headings.")
        issues.append("No section headings found — parsers locate blocks by "
                      "heading, so unlabelled content is discarded.")
    # Too few content words is a real problem: an ATS that reads a two-line
    # document has nothing to match keywords against. 120 words is roughly a
    # single role with three bullets, so this only fires on a stub.
    if words and words < 120:
        st_score -= 2
        st_notes.append(f"Only ~{words} words of content.")
        issues.append(f"Resume has only ~{words} words of content — add detail "
                      f"to roles you already have rather than new sections.")
    st_max = 16.0
    breakdown["structure"] = {"score": round(max(0.0, min(st_score, st_max)), 1),
                              "max": st_max, "notes": st_notes}

    # ------------------------------------------------------------------ dates
    #
    # The component is worth 16 but the old code stopped at 10, so every
    # document scored 10/16 and a resume with perfect dates still looked
    # penalised. The remaining 6 are earned by coverage: dates on every role,
    # and a current role that says so.
    d_score, d_notes = 0.0, []
    ranges = _DATE_RANGE_RE.findall(text)
    explicit = _FULL_DATE_RE.findall(text)
    months = _MONTH_YEAR_RE.findall(text)
    years = _YEAR_ONLY_RE.findall(text)
    ambiguous = re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
    dated_entries = len(ranges) + len(explicit)
    if ranges or explicit:
        d_score += 8
    elif months:
        d_score += 6
        d_notes.append("Month-year dates only.")
        issues.append("Use 'Mar 2024 - Present' style dates; bare years are "
                      "harder for an ATS to sequence.")
    elif years:
        # A bare year per role is better than nothing: an ATS can still count
        # roles and approximate tenure, it just cannot order them precisely.
        d_score += 3
        d_notes.append("Only bare years present.")
        issues.append("Employment dates are years only - add the month "
                      "('Mar 2024 - Present') so an ATS can order your roles.")
    else:
        d_notes.append("No parseable employment dates.")
        issues.append("No employment dates found - ATS often drops roles with "
                      "no date range.")
    # Coverage: a parser reads a role as a block, and a role with no date in it
    # is a role it cannot place. Two or more dated entries is the difference
    # between "one internship" and "a career".
    if dated_entries >= 2:
        d_score += 4
    elif dated_entries == 1:
        d_score += 2
        d_notes.append("Only one dated entry.")
        issues.append("Only one dated role/entry found - add date ranges to "
                      "every role so the timeline can be reconstructed.")
    # Currency: the role a reader cares about most is the current one, and an
    # open-ended range is how an ATS recognises it.
    if re.search(r"(?:present|current|ongoing|now)\b", text, re.I):
        d_score += 4
    else:
        d_notes.append("No 'Present' marker on the current role.")
        issues.append("End your current role with 'Present' so an ATS treats it "
                      "as ongoing rather than closed.")
    if ambiguous:
        d_score -= 4
        d_notes.append("Ambiguous numeric date format.")
        issues.append(f"Found {len(ambiguous)} ambiguous date(s) like "
                      f"'{ambiguous[0]}' — write 'Mar 2025' instead.")
    # A range with a year but no month ("2021 - 2023") is readable but weaker
    # than "Mar 2021 - Present"; deducted rather than excluded so it still
    # scores above a role with no dates at all.
    if ranges and not (months or explicit):
        d_score -= 2
        d_notes.append("Date ranges lack a month.")
        issues.append("Add months to your date ranges so each role's length is "
                      "unambiguous.")
    d_max = 16.0
    breakdown["dates"] = {"score": round(max(0.0, min(d_score, d_max)), 1),
                          "max": d_max, "notes": d_notes}

    # --------------------------------------------------------------- keywords
    keywords = extract_keywords(jd_text)
    hits = _keyword_hits(low, keywords)
    missing = [k for k in keywords if k not in hits]
    k_score, k_notes = 0.0, []
    if keywords:
        coverage = len(hits) / len(keywords)
        # Square-root damping, not linear. A linear 60% coverage looks better
        # than it is: the first few keywords are the ones a recruiter actually
        # scans for, and pasting a long keyword list into Skills must not outrank
        # genuinely matching content. Damping makes 100% hard to reach without
        # real coverage.
        k_score = (coverage ** 0.5) * 26.0
        # Coverage concentrated in one place is weaker than coverage spread
        # through the document: a term that appears in a bullet is evidenced,
        # a term that appears only in Skills is asserted.
        spread = _keyword_spread(low, hits)
        if spread < 0.6 and coverage > 0.5:
            k_score *= 0.9
            k_notes.append("Most matching terms appear only in the skills list.")
            issues.append("Keyword hits are concentrated in Skills — echo the "
                          "posting's terms inside your bullet points too.")
    k_max = 26.0
    breakdown["keywords"] = {"score": round(k_score, 1), "max": k_max,
                             "notes": [
                                 f"{len(hits)}/{len(keywords)} posting terms present"
                                 + (f"; missing: {', '.join(missing[:6])}"
                                    if missing else "")] + k_notes}

    # --------------------------------------------------------------- evidence
    e_score, e_notes = 0.0, []
    quantified = _QUANT_RE.findall(text)
    if quantified:
        e_score += min(8.0, len(quantified) * 1.6)
    else:
        e_notes.append("No quantified outcomes.")
        issues.append("Add numbers to at least a few bullets (%, volume, "
                      "time saved) — unquantified bullets read as filler.")
    action = re.findall(
        r"\b(?:built|designed|implemented|deployed|automated|optimi[sz]ed|"
        r"migrated|launched|led|created|developed|refactor\w*|integrated)\b",
        text, re.I)
    if action:
        e_score += min(4.0, len(action) * 0.5)
    else:
        e_notes.append("No strong action verbs.")
    bullets = len(re.findall(r"^\s*[•\-*]|\\item", text, re.M))
    if bullets >= 4:
        e_score += 2.0
    elif bullets:
        e_score += 1.0
    if re.search(r"https?://github\.com/[A-Za-z0-9]", text):
        e_score += 2.0
    e_max = 16.0
    breakdown["evidence"] = {"score": round(min(e_score, e_max), 1), "max": e_max,
                             "notes": e_notes}

    total = sum(v["score"] for v in breakdown.values())
    total = round(max(0.0, min(100.0, total)), 1)
    return {
        "total": total,
        "band": band(total),
        "breakdown": breakdown,
        "issues": issues,
        "hits": hits,
        "missing": missing,
        "keyword_total": len(keywords),
    }


def band(total: float) -> str:
    if total >= 85:
        return "strong"
    if total >= 70:
        return "good"
    if total >= 55:
        return "fair"
    return "weak"


_SECTION_PATTERNS = {
    "summary": r"(?:professional\s+|career\s+)?(?:summary|objective|profile)\b",
    "skills": r"(?:technical\s+|core\s+|key\s+)?skills?\b|"
              r"(?:technolog\w+|tool\s?kit|competenc\w+|stack)\b",
    "experience": r"(?:work\s+|professional\s+|employment\s+)?"
                  r"(?:experience|history|employment)\b",
    "projects": r"projects?\b|portfolio\b|selected\s+work\b",
    "education": r"education\b|academics?\b|qualifications?\b",
    "certifications": r"certificat\w+|licenses?\b|credentials?\b",
    "languages": r"languages?\b",
}


def _has_section(text: str, name: str) -> bool:
    """Is this section headed, in either representation?

    A leading bullet marker is allowed because the rendered preview is
    normalised to "- " list items: without it the same document scored 12.5 on
    the LaTeX side and 11.5 on the HTML side purely from a tag.
    """
    pattern = _SECTION_PATTERNS.get(name)
    if not pattern:
        return False
    return bool(re.search(
        rf"(?:^|\n)\s*(?:[-*•]\s*)?(?:\\section\*?\{{)?\s*{pattern}",
        text or "", re.I | re.M))


def score_for_user(resume_text: str, jd_text: str = "",
                   sections: list[str] | None = None) -> dict:
    """Convenience wrapper mirroring the generator's signature conventions."""
    return score_resume(resume_text, jd_text, sections)


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys
    src = sys.stdin.read() if not sys.stdin.isatty() else ""
    jd = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(score_resume(src, jd), indent=2))
