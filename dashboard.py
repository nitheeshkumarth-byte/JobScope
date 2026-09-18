"""
dashboard.py — FastAPI backend for the JobScope dashboard.

Serves static/index.html and streams agent runs to the browser over SSE.

Endpoints
  GET  /             dashboard UI
  GET  /api/config   current agent configuration + tools
  POST /api/config   rebuild the agent with new settings (role, model, skills…)
  POST /api/resume   upload a resume/CV (txt, md, pdf…) -> skill keywords,
                     role suggestions and location inference (India / remote)
  POST /api/resume/gen   build a job-tailored LaTeX resume (+ HTML preview)
                     for a single listing (see resume_generator.py)
  POST /api/run      run the agent; SSE stream of token/tool/jobs/error events
  GET  /api/logs     recent run history + errors (for the History drawer)

Logging: every run and every error is appended as JSONL under logs/
(logs/runs-YYYYMMDD.jsonl and logs/errors.jsonl) so you can audit failures
even after the server restarts.
"""

import asyncio
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
import uuid

from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from agent import (AgentConfig, JOBS_MARKER, _text_content, create_agent,
                   load_config, system_message)
import resume_data
import resume_generator

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

MAX_CTX_LEN = 10000          # resume text cap sent to the skill extractor
MAX_RUNS_KEPT = 60           # in-memory run history

# Fallback skill lexicon (used when the local LLM extraction fails).
FALLBACK_SKILLS = [
    "python", "sql", "pandas", "numpy", "excel", "power bi", "tableau",
    "statistics", "machine learning", "data analysis", "data visualization",
    "git", "github", "aws", "powerpoint", "communication", "teamwork",
    "problem solving", "mysql", "postgresql", "django", "flask", "javascript",
    "html", "css", "react", "jupyter", "matplotlib", "seaborn", "scikit-learn",
]

SKILL_SYS_PROMPT = (
    "Extract the concrete skill keywords from this resume. Include both "
    "technical skills (languages, tools, libraries, databases) and soft/ "
    "interpersonal skills. Reply with ONLY a plain comma-separated list, "
    "8-20 items, single words or short phrases, no numbering, no bullets, "
    "no extra text."
)

# --------------------------------------------------------------------------
# Skill -> role suggestions (matches the user's actual skills to job titles,
# instead of guessing a role), and city/location inference from the CV.
# --------------------------------------------------------------------------

ROLE_RUBRIC = [
    ("Data Analyst", ["sql", "python", "pandas", "excel", "power bi", "tableau",
                      "statistics", "looker", "google sheets", "numpy", "matplotlib"]),
    ("Business Intelligence Analyst", ["power bi", "tableau", "sql", "excel", "dax",
                                       "data visualization", "looker"]),
    ("Data Engineer", ["python", "sql", "etl", "spark", "airflow", "aws",
                       "databricks", "kafka", "snowflake", "docker"]),
    ("Business Analyst", ["excel", "sql", "requirements", "stakeholder",
                          "agile", "jira", "powerpoint"]),
    ("SQL / Database Developer", ["sql", "mysql", "postgresql", "tsql", "mongodb", "database"]),
    ("Data Scientist", ["python", "statistics", "machine learning", "scikit-learn",
                        "pytorch", "tensorflow", "nlp", "pandas"]),
    ("ML Engineer", ["python", "scikit-learn", "pytorch", "tensorflow", "ml",
                     "docker", "aws"]),
    ("Python Developer", ["python", "django", "flask", "fastapi", "sql", "rest", "docker"]),
]

INDIAN_CITIES = ["hyderabad", "bangalore", "bengaluru", "mumbai", "pune", "delhi",
                 "noida", "gurugram", "gurgaon", "chennai", "kolkata", "ahmedabad",
                 "coimbatore", "indore", "jaipur", "lucknow"]
REMOTE_HINTS = ["remote", "work from home", "wfh", "fully remote", "home office"]


def suggest_roles(skills: list[str]) -> list[dict]:
    low = {s.lower() for s in skills}
    best = []
    for role, needs in ROLE_RUBRIC:
        matched = [n for n in needs if n in low]
        if not matched:
            continue
        # coverage of the role's needs + weight of matched skill count
        score = round(len(matched) / max(len(needs), 1), 2)
        best.append({"role": role, "score": score, "matched": matched})
    best.sort(key=lambda r: (r["score"], len(r["matched"])), reverse=True)
    return best[:4]


def infer_location(resume_text: str) -> dict:
    low = resume_text.lower()
    city = next((c for c in INDIAN_CITIES if c in low), "")
    remote = any(h in low for h in REMOTE_HINTS)
    country = "India" if city or re.search(r"\bindia\b", low) else ""
    loc = (city.title() + ", " if city else "") + (country if country else "global")
    return {"location": loc.rstrip(", ") or "not found",
            "city": city.title() or "", "country": country or "",
            "remote_only": remote}


class RunRequest(BaseModel):
    message: str


class ResumeGenRequest(BaseModel):
    """Flashcard fields for one listing — used to build its tailored resume."""
    title: str = ""
    company: str = ""
    location: str = ""
    link: str = ""
    source: str = ""


class ConfigRequest(BaseModel):
    target_role: str = ""
    model: str
    num_ctx: int = 8192
    days_back: int = 60
    max_results: int = 10
    skills: list[str] = []
    remote_only: bool = True
    cities: list[str] = []
    roles: list[str] = []


# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

RUNS: list[dict] = []


def _log(record: dict, kind: str = "runs") -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        with open(os.path.join(LOG_DIR, f"{kind}-{time.strftime('%Y%m%d')}.jsonl"),
                  "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # logging must never take the app down


def _remember_run(summary: dict) -> None:
    RUNS.insert(0, summary)
    del RUNS[MAX_RUNS_KEPT:]


# --------------------------------------------------------------------------
# Skill extraction (resume/CV -> search keywords)
# --------------------------------------------------------------------------

def _extract_text(data: bytes, filename: str) -> str:
    """Best-effort text extraction from txt/md/csv/html/pdf uploads."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if text.strip():
                return text
        except Exception:
            pass
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="ignore")


def _parse_skill_list(text: str) -> list[str]:
    parts = re.split(r"[,;\n]+", text or "")
    out = []
    for p in parts:
        p = p.strip().strip("*-#•").strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) < 2 or len(p) > 30:
            continue
        if re.fullmatch(r"[\w\s.\-#+/&()]+", p) and not re.fullmatch(r"\d+.*", p):
            out.append(p)
    seen, result = set(), []
    for p in out:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result[:20]


# Heading markers for a resume's explicit skills block (matched as a prefix —
# the list often shares the same line via a colon, or a pipe-separated table
# row). STRICT ones name a skills/technology area; GENERIC ("skills") is used
# only as a last resort.
_TECH_HEADING_STRICT = re.compile(
    r"^\s*(?:technical\s+skills?|skills?\s*(?:&\s*|\s+and\s+)?(?:tools?|technolog\w+)"
    r"|tools?\s*(?:&\s*|\s+and\s+)?technolog\w*|technolog\w*|technical\s+expertise"
    r"|core\s+competenc\w*|key\s+skills?|skills?\s*summary|tech\s+stack)\b",
    re.IGNORECASE,
)
_TECH_HEADING_GENERIC = re.compile(r"^\s*skills?\b", re.IGNORECASE)

# Common next-section headings that end a skills list even when they aren't
# rendered as all-caps or with a trailing colon.
_SECTION_WORDS = {
    "projects", "experience", "work experience", "professional experience",
    "education", "certifications", "courses", "accomplishments",
    "achievements", "internships", "training", "languages", "interests",
    "activities", "personal projects", "contact", "references",
}

# Filler leading/trailing tokens inside one skill item ("working knowledge of
# Python" -> "Python") and words that are never a skill themselves.
_SKILL_FILLER_RE = re.compile(
    r"(?i)^(?:proficient\s+in|working\s+knowledge\s+of|knowledge\s+of|"
    r"experience\s+in|familiar\s+with|basic\s+|strong\s+|using\s+|in\s+|"
    r"with\s+|and\s+|etc\.?)\s*"
)
_NOT_A_SKILL = {"and", "etc", "etc.", "tools", "skills", "work experience",
                "education", "certifications", "soft skills", "languages",
                "projects", "experience", "summary", "profile"}


def _is_heading_break(line: str) -> bool:
    """True when a line ends the skills list (looks like the NEXT section:
    a known section word, all-caps, or a short title-cased line + colon)."""
    s = line.strip()
    if not s or s.startswith(("*", "-", "•")) or "|" in s or "," in s or ";" in s:
        return False
    if s.lower() in _SECTION_WORDS:
        return True
    return (s.isupper() or s.endswith(":")) and len(s) < 45


def _technical_skill_rows(text: str) -> list[str]:
    """Take EVERY entry from the resume's explicit skills block.

    The user's CV lists its toolkit in a "Technical Skills" column; whatever
    the PDF text layer spits out (colons, pipe-separated table cells, bullet
    lines), the whole block belongs in the keyword list — not just the slice
    that happens to match a known lexicon."""
    lines = (text or "").splitlines()
    anchor = None
    for i, ln in enumerate(lines):
        if _TECH_HEADING_STRICT.search(ln):
            anchor = i
            break
    if anchor is None:
        for i, ln in enumerate(lines):
            if _TECH_HEADING_GENERIC.search(ln):
                anchor = i
                break
    if anchor is None:
        return []

    head = lines[anchor]
    m = _TECH_HEADING_STRICT.search(head) or _TECH_HEADING_GENERIC.search(head)
    heading_end = m.end() if m else len(head)
    collected = []
    if "|" in head:
        # a table row like "Technical Skills|Python|SQL|Power BI"
        cells = [c.strip() for c in head.split("|") if c.strip()]
        if len(cells) > 1:
            collected.append(", ".join(cells[1:]))
    else:
        rest = head[heading_end:].lstrip(":|\t ).").strip()
        if rest:
            collected.append(rest)
    for ln in lines[anchor + 1:]:
        if _is_heading_break(ln):
            break
        collected.append(ln)

    out = []
    for chunk in collected:
        for p in re.split(r"[,;|\n•]+", chunk):
            p = p.strip().strip("*-#").strip()
            p = _SKILL_FILLER_RE.sub("", p).strip()
            p = p.rstrip(".,;:").strip()
            p = re.sub(r"\s+", " ", p)
            if (2 <= len(p) <= 40 and not re.fullmatch(r"\d+.*", p)
                    and p.lower() not in _NOT_A_SKILL
                    and re.fullmatch(r"[\w\s.\-#+/&()]+", p)):
                out.append(p)
    seen, result = set(), []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            result.append(p)
    return result[:60]


def _merge_skills(*groups: list[str]) -> list[str]:
    """Explicit technical-skills first (user's exact words), then the rest.
    Dedupes case-insensitively, drops a later item that is redundant with an
    earlier one as a standalone word (lexicon "excel" behind "Advanced
    Excel"), and caps so the prompt/search stay lean."""
    seen, out = set(), []
    for gi, group in enumerate(groups):
        for s in group:
            k = (s or "").strip().lower()
            if not k or k in seen:
                continue
            if gi > 0 and any(re.search(rf"\b{re.escape(k)}\b", prev) for prev in seen):
                continue
            seen.add(k)
            out.append(s.strip())
    return out[:60]


# Lexicon = explicit fallback list + the canonical resume keyword groups, so
# the fast path still catches backend/Python skills a Data-Analyst-flavoured
# rubric would miss. Sorted for deterministic output.
ENRICHED_LEXICON = tuple(sorted(set(FALLBACK_SKILLS) | set(resume_data.flat_skill_keywords())))

# Cache extracted skills keyed by (model, resume text hash): re-uploading the
# same CV must be instant instead of another slow local-LLM call.
SKILL_CACHE: dict[str, list[str]] = {}
SKILL_CACHE_MAX = 32


def _lexicon_skills(resume_text: str) -> list[str]:
    low = resume_text.lower()
    return [s for s in ENRICHED_LEXICON if s in low]


def _cache_skills(key: str, skills: list[str]) -> list[str]:
    if key not in SKILL_CACHE and len(SKILL_CACHE) >= SKILL_CACHE_MAX:
        SKILL_CACHE.pop(next(iter(SKILL_CACHE)))  # evict one, keep the cache bounded
    SKILL_CACHE[key] = skills
    return skills


async def extract_skills(resume_text: str, model: str, num_ctx: int = 8192) -> list[str]:
    """Deterministic skill extraction — instant, no LLM by default.

    `strict` (the default) uses ONLY keywords that literally appear in the
    resume: the whole "Technical Skills" block if the CV has one, otherwise
    known-skill mentions found verbatim in the text. `auto` additionally
    enriches with lexicon terms, `fast` is lexicon-only, and `llm` always
    runs the local model. Results are cached."""
    key = hashlib.sha256(f"{model}\n{resume_text}".encode("utf-8", "ignore")).hexdigest()
    if key in SKILL_CACHE:
        return SKILL_CACHE[key]

    lexicon = _lexicon_skills(resume_text)
    explicit = _technical_skill_rows(resume_text)
    mode = os.environ.get("RESUME_EXTRACT_MODE", "strict").lower()
    if mode == "strict":
        base = explicit or lexicon
        if base:
            return _cache_skills(key, _merge_skills(base))
    if mode == "fast":
        return _cache_skills(key, _merge_skills(lexicon))
    if mode == "auto" and (explicit or lexicon):
        return _cache_skills(key, _merge_skills(explicit, lexicon))

    try:
        # Use the SAME num_ctx as the agent so Ollama reuses the resident
        # model slot instead of reloading a second context size.
        llm = ChatOllama(model=model, num_ctx=num_ctx)
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=SKILL_SYS_PROMPT),
                HumanMessage(content=resume_text[:MAX_CTX_LEN]),
            ]),
            timeout=180,
        )
        parsed = _parse_skill_list(_text_content(resp.content))
        if parsed:
            return _cache_skills(key, _merge_skills(explicit, parsed))
    except Exception as exc:
        _log({"event": "skill_extractor_error", "detail": str(exc)}, kind="errors")
    return _cache_skills(key, _merge_skills(explicit, lexicon))


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook: build the shared agent (boots the MCP server
    subprocesses) once at startup and release it on shutdown."""
    app.state.bundle = await create_agent()
    app.state.resume_text = ""
    yield
    app.state.bundle = None


app = FastAPI(title="JobScope", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def _no_cache(request, call_next):
    """Never serve the previous session's page/CSS/JS. The skills/cities/roles
    chips change the page between deploys, and a stale cached index.html shows
    a UI that doesn't match the API (broken Save, missing inputs)."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def _frame(obj: dict) -> str:
    """Wrap a dict as one Server-Sent Events frame (data: json\n\n)."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    """Serve the single-page JobScope UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/config")
async def get_config():
    """Current agent configuration, exposed so the UI can render its state."""
    bundle = app.state.bundle
    return {
        "target_role": bundle.cfg.target_role,
        "model": bundle.cfg.model,
        "num_ctx": bundle.cfg.num_ctx,
        "days_back": bundle.cfg.days_back,
        "max_results": bundle.cfg.max_results,
        "skills": bundle.cfg.skills,
        "role_suggestions": bundle.cfg.role_suggestions,
        "remote_only": bundle.cfg.remote_only,
        "location": bundle.cfg.indeed_location,
        "cities": bundle.cfg.preferred_cities,
        "roles": bundle.cfg.target_roles or [bundle.cfg.target_role],
        "tools": [t.name for t in bundle.tools],
    }


@app.post("/api/config")
async def update_config(req: ConfigRequest):
    """Replace the running agent with a new configuration.

    Roles are multi-select: the first is the primary (drives the board query),
    the rest are kept on the list for matching/context. Never throws a bare
    500 — restart failures come back as a normal JSON payload so the UI can
    show what happened instead of a vague network error."""
    cities = [c.strip() for c in req.cities if c.strip()]
    roles = [r.strip() for r in req.roles if r.strip()]
    previous = app.state.bundle.cfg
    if not roles:
        # user cleared the chips: fall back to what the hunt is currently using
        roles = [r for r in (previous.target_roles or [previous.target_role]) if r]
    if not roles:
        roles = [req.target_role.strip()] if req.target_role.strip() else []
    if not roles:
        roles = ["Junior Data Analyst / entry-level, 0-2 years experience"]
    primary = roles[0]
    try:
        cfg = AgentConfig(
            target_role=primary,
            target_roles=roles,
            model=req.model,
            num_ctx=req.num_ctx,
            days_back=req.days_back,
            max_results=req.max_results,
            indeed_query=primary,
            skills=[s for s in req.skills if s],
            resume_text=app.state.resume_text or previous.resume_text,
            remote_only=req.remote_only,
            # with remote_only on the boards search "Remote" and remote India
            # postings are already filtered; with it off the first chosen city
            # becomes the boards' location instead of the stale asset default.
            indeed_location=cities[0] if (cities and not req.remote_only) else "Remote",
            preferred_cities=cities,
            role_suggestions=previous.role_suggestions,
        )
        app.state.bundle = await create_agent(cfg)
    except Exception as exc:
        _log({"event": "config_restart_error", "detail": str(exc)}, kind="errors")
        return {"ok": False, "error": f"Agent restart failed: {exc}"}
    _log({"event": "config_updated", "model": cfg.model, "num_ctx": cfg.num_ctx,
          "skills": cfg.skills, "remote_only": cfg.remote_only,
          "cities": cities, "roles": roles, "role": primary})
    return {"ok": True, "target_role": primary, "roles": cfg.target_roles,
            "model": cfg.model, "num_ctx": cfg.num_ctx, "skills": cfg.skills,
            "remote_only": cfg.remote_only, "cities": cfg.preferred_cities,
            "role_suggestions": cfg.role_suggestions}


@app.post("/api/resume")
async def upload_resume(file: UploadFile = File(...)):
    """Crunch a resume/CV into skill keywords, suggest roles for it, infer
    the user's location/remote preference, and feed all of it to the agent.

    A new upload ADDS to the existing keyword set instead of replacing it —
    multiple CVs/resumes accumulate (deduped), so the search keeps every
    tool/language the user has ever listed."""
    data = await file.read()
    text = _extract_text(data, file.filename or "")
    if not text.strip():
        return {"ok": False, "error": "Could not read any text from that file."}

    new_skills = await extract_skills(text, app.state.bundle.cfg.model,
                                      app.state.bundle.cfg.num_ctx)
    if not new_skills:
        return {"ok": False, "error": "No skills could be extracted.",
                "skills": list(app.state.bundle.cfg.skills),
                "preview": text[:300]}

    cfg = app.state.bundle.cfg
    existing = list(cfg.skills or [])
    merged = _merge_skills(new_skills, existing)
    # previous uploads (or manual chips) keep driving the role choice
    first_upload = not existing
    suggestions = [{**s, "role": f"{s['role']} / entry-level, remote"}
                   for s in suggest_roles(merged)]
    inferred = infer_location(text)
    remote_only = bool(suggestions) or cfg.remote_only
    if inferred.get("remote_only"):
        remote_only = True

    top_role = (suggestions[0]["role"] if first_upload and suggestions
                else cfg.target_role)
    # Seed the city chips from the CV only on the FIRST upload — afterwards the
    # user's manually chosen cities (or an earlier CV's city) stay put, and the
    # "default Hyderabad" trap (stale resume_data asset) can't override them.
    seen_c, cities = set(), []
    for c_ in (cfg.preferred_cities + ([inferred["city"]] if
               first_upload and inferred.get("city") else [])):
        if c_ and c_.lower() not in seen_c:
            seen_c.add(c_.lower())
            cities.append(c_)
    app.state.bundle.cfg = replace(
        cfg,
        skills=merged,
        resume_text=(cfg.resume_text + "\n\n" + text).strip()
        if cfg.resume_text else text,
        role_suggestions=[s["role"] for s in suggestions] or cfg.role_suggestions,
        target_role=top_role,
        target_roles=cfg.target_roles or [top_role],
        indeed_query=top_role,
        indeed_location="Remote",
        indeed_domain="www.indeed.com",
        remote_only=remote_only,
        preferred_cities=cities,
    )
    app.state.resume_text = app.state.bundle.cfg.resume_text
    _log({"event": "resume_upload", "file": file.filename, "chars": len(text),
          "skills_added": len(new_skills), "skills_total": len(merged),
          "first_upload": first_upload, "suggested_role": top_role,
          "inferred": inferred, "cities": cities})
    return {"ok": True, "skills": merged, "suggestions": suggestions,
            "inferred": inferred, "target_role": top_role,
            "cities": cities, "preview": text[:400], "chars": len(text),
            "skills_added": len(new_skills)}


@app.post("/api/resume/gen")
async def generate_resume(req: ResumeGenRequest):
    """Build a job-tailored LaTeX resume for a single listing.

    Fetches a description snippet of the posting (best-effort), ranks the
    canonical skill groups against it, and returns both the .tex source and
    a rendered HTML preview so the UI can show a resume without a TeX
    install. The description fetch is run off the event loop (blocking I/O).
    """
    job = {"title": req.title, "company": req.company, "location": req.location,
           "link": req.link, "source": req.source}
    try:
        desc = await asyncio.to_thread(resume_generator.desc_snippet, req.link)
        cfg = app.state.bundle.cfg
        tex, fname = await asyncio.to_thread(
            resume_generator.build_resume, cfg, job, desc)
        preview = await asyncio.to_thread(resume_generator.tex_to_html, tex)
    except Exception as exc:
        _log({"event": "resume_gen_error", "detail": str(exc)}, kind="errors")
        return {"ok": False, "error": str(exc)}
    _log({"event": "resume_generated", "job": job, "desc_chars": len(desc)})
    return {"ok": True, "filename": fname, "tex": tex, "preview": preview,
            "desc_fetched": bool(desc)}


@app.get("/api/logs")
async def get_logs():
    return {"runs": RUNS[:50], "errors": _recent_errors(kind="errors", n=20)}


def _recent_errors(kind: str, n: int) -> list[dict]:
    """Tail the day's jsonl logs as a plain list (read-only, safe)."""
    path = os.path.join(LOG_DIR, f"{kind}-{time.strftime('%Y%m%d')}.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        return lines[-n:]
    except (OSError, json.JSONDecodeError):
        return []


# --------------------------------------------------------------------------
# Run (SSE)
# --------------------------------------------------------------------------

async def _run_stream(message: str, run_id: str, t0: float):
    bundle = app.state.bundle
    skills = bundle.cfg.skills
    yield _frame({"type": "status", "stage": "starting",
                  "message": "Agent starting…"})
    _log({"run": run_id, "event": "start", "message": message,
          "skills": skills, "model": bundle.cfg.model,
          "remote_only": bundle.cfg.remote_only})

    payload = {"messages": [system_message(bundle.cfg),
                            HumanMessage(content=message)]}
    summary = {"run": run_id, "message": message, "tools": [],
               "token_chars": 0, "duration_s": None,
               "status": "ok", "error": None, "jobs": 0}
    try:
        async for mode, data in bundle.app.astream(
            payload, stream_mode=["messages", "updates"],
            config={"recursion_limit": 60},
        ):
            if mode == "messages":
                chunk, meta = data
                if isinstance(chunk, AIMessageChunk):
                    text = _text_content(chunk.content)
                    if text:
                        summary["token_chars"] += len(text)
                        yield _frame({"type": "token", "text": text})

            elif mode == "updates":
                for node, update in data.items():
                    msgs = update.get("messages") or []
                    if node == "agent":
                        for msg in msgs:
                            for tc in getattr(msg, "tool_calls", None) or []:
                                name = tc.get("name", "")
                                args = tc.get("args", {})
                                summary["tools"].append(name)
                                yield _frame({"type": "tool_start", "name": name,
                                              "args": json.dumps(args, default=str)})
                    elif node == "tools":
                        for msg in msgs:
                            if isinstance(msg, ToolMessage):
                                tname = (getattr(msg, "name", "") or "").lower()
                                full = _text_content(msg.content)
                                yield _frame({"type": "tool_end",
                                              "name": getattr(msg, "name", "") or "tool",
                                              "output": full[:900]})
                                # ONLY the boards scraper feeds the deck here.
                                # Gmail results are speculative — boards and
                                # Gmail now run in parallel, and "prep" decides
                                # which source actually wins.
                                if ("job_boards" in tname or "scrape" in tname) and JOBS_MARKER in full:
                                    try:
                                        deck = json.loads(full.split(JOBS_MARKER, 1)[1].strip())
                                        summary["jobs"] += deck.get("total", 0)
                                        yield _frame({"type": "jobs", "payload": deck})
                                    except (json.JSONDecodeError, ValueError):
                                        pass
                    elif node in ("fallback", "prep"):
                        # "fallback" = legacy sequential path; "prep" = the
                        # parallel-collect path. Both append ###JOBS_JSON### to
                        # the message they hand the summary LLM.
                        for msg in msgs:
                            # prep/fallback return the human payload plus any
                            # RemoveMessage cleanup entries — skip non-content.
                            if not isinstance(msg, HumanMessage):
                                continue
                            full = _text_content(getattr(msg, "content", msg))
                            if full.startswith("Job boards responded with listings"):
                                continue  # board decks are already live
                            yield _frame({"type": "fallback", "message": (
                                "Job boards were blocked — using Gmail job alerts.")})
                            if JOBS_MARKER in full:
                                try:
                                    deck = json.loads(full.split(JOBS_MARKER, 1)[1].strip())
                                    summary["jobs"] += deck.get("total", 0)
                                    yield _frame({"type": "jobs", "payload": deck})
                                except (json.JSONDecodeError, ValueError):
                                    pass

        summary["duration_s"] = round(time.time() - t0, 1)
        yield _frame({"type": "status", "stage": "done"})
        _log({"run": run_id, "event": "done", **summary})
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = str(exc)[:500]
        summary["duration_s"] = round(time.time() - t0, 1)
        detail = traceback.format_exc()
        _log({"run": run_id, "event": "error", **summary}, kind="errors")
        _log({"run": run_id, "event": "error", "exception": detail})
        yield _frame({"type": "error", "message": str(exc), "run": run_id})
    finally:
        _remember_run(summary)


@app.post("/api/run")
async def run(req: RunRequest):
    run_id = uuid.uuid4().hex[:8]
    return StreamingResponse(
        _run_stream(req.message, run_id, time.time()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    uvicorn.run("dashboard:app", host="127.0.0.1", port=port, reload=False)