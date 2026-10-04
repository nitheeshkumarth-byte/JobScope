"""
dashboard.py — FastAPI backend for the JobScope dashboard.

Serves static/index.html and streams agent runs to the browser over SSE.

Endpoints
  GET  /                    dashboard UI
  GET  /api/config          current agent configuration + tools
  POST /api/config          rebuild the agent with new settings (role, model,
                            skills, github link…)
  GET  /api/profiles        sidebar sessions — one profile per CV
  POST /api/profiles        create a new (empty) profile/session and open it
  POST /api/profiles/activate   switch to another saved profile
  POST /api/profiles/delete delete a saved profile
  POST /api/resume          upload a resume/CV (txt, md, pdf…) -> skill keywords,
                            role suggestions, location + GitHub link inference
  POST /api/resume/gen      build a job-tailored LaTeX resume (+ HTML preview)
                            for a single listing (see resume_generator.py)
  POST /api/run             run the agent; SSE stream of token/tool/jobs/error events
  GET  /api/logs            recent run history + errors (for the History drawer)

Sessions: each uploaded CV becomes a profile in the sidebar; switching a
profile swaps the active search/resume context back into it. Profiles are
persisted to profiles.json so they survive server restarts.

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

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from agent import (AgentConfig, JOBS_MARKER, _text_content, create_agent,
                   create_runtime, load_config, system_message)
import auth
import cv_ocr
import filters
import mailer
import resume_data
import resume_generator
import rag_store

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

MAX_CTX_LEN = 10000          # resume text cap sent to the skill extractor
MAX_RUNS_KEPT = 60           # in-memory run history, per account
MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # resume/CV upload ceiling (8 MB)
PROFILES_FILE = os.path.join(HERE, "profiles.json")   # legacy store, read once
LOG_SCAN_DAYS = 7            # how many days of JSONL the History drawer reads

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

# The full list behind the role dropdown. It is a SUGGESTION catalog: adding a
# row here makes the role selectable, it does not put the role in anybody's
# search. A role becomes a live filter only when the user picks it in the
# drawer and saves. Grouped by family purely so the <optgroup> headers read
# sensibly; ordering inside a group is roughly entry-level first.
JOB_ROLE_OPTIONS = [
    "Software Developer", "Frontend Developer", "Backend Developer",
    "Full Stack Developer", "Web Developer", "Mobile App Developer",
    "Android Developer", "iOS Developer", "Software Engineer",
    "Game Developer", "Embedded Software Engineer", "Desktop Application Developer",
    "DevOps Engineer", "Cloud Engineer", "Site Reliability Engineer",
    "Platform Engineer", "Kubernetes Administrator", "Infrastructure Engineer",
    "Data Analyst", "Business Intelligence Analyst", "Data Engineer",
    "Data Scientist", "Machine Learning Engineer", "AI Engineer",
    "MLOps Engineer", "Data Science Intern", "Analytics Engineer",
    "QA Engineer", "Test Automation Engineer", "SDET", "Performance Tester",
    "Security Engineer", "Cyber Security Analyst", "Application Security Engineer",
    "Network Engineer", "Systems Administrator", "Database Administrator",
    "Cloud Support Engineer", "Technical Support Engineer",
    "Product Manager", "Project Manager", "Program Manager", "Business Analyst",
    "Scrum Master", "Product Owner", "Solution Architect", "Technical Architect",
    "Solutions Engineer", "Pre-Sales Engineer",
    "Salesforce Developer", "Salesforce Administrator", "SAP Consultant",
    "SAP ABAP Developer", "ERP Consultant", "CRM Consultant", "ServiceNow Developer",
    "Software Engineering Intern", "Software Developer Intern",
    "Frontend Developer Intern", "Backend Developer Intern",
    "Full Stack Developer Intern", "QA Engineer Intern",
    "Business Analyst Intern", "Summer Internship", "Graduate Trainee",
    "Graduate Software Engineer", "Junior Software Developer",
    "Associate Software Engineer", "Trainee Software Engineer",
    "Firmware Engineer", "Embedded Systems Engineer", "Hardware Engineer",
    "Design Engineer", "UI/UX Designer", "Product Designer",
    "Technical Writer", "Content Writer", "Customer Success Manager",
]


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
    # An Indian city is the strongest signal there is, so it decides the country
    # as before. Otherwise name the country only when the CV names exactly ONE
    # — a CV that mentions "France" in some project line and "United Kingdom"
    # in another tells us nothing about where the author lives, and guessing
    # there would preselect a country filter that hides every result.
    if city or re.search(r"\bindia\b", low):
        country = "India"
    else:
        named = filters.identify_countries(resume_text)
        country = named[0] if len(named) == 1 else ""
    loc = (city.title() + ", " if city else "") + (country if country else "global")
    return {"location": loc.rstrip(", ") or "not found",
            "city": city.title() or "", "country": country or "",
            "remote_only": remote}


class RunRequest(BaseModel):
    message: str


class ResumeGenRequest(BaseModel):
    """Flashcard fields for one listing - used to build its tailored resume."""
    title: str = ""
    company: str = ""
    location: str = ""
    link: str = ""
    source: str = ""
    github_url: str = ""   # optional override; falls back to the profile's
    # The posting's own text, pasted by the user. This is the reliable JD
    # source: desc_snippet scrapes <=900 chars and boards answer 403 often
    # enough that an un-pasted JD silently degrades tailoring to matching on
    # the title alone. Pasted text wins; the link fetch is the fallback.
    jd_text: str = ""


class ConfigRequest(BaseModel):
    target_role: str = ""
    model: str
    num_ctx: int = 8192
    days_back: int = 60
    max_results: int = 10
    skills: list[str] = []
    # Retired. Still accepted so an older cached frontend does not 422, but the
    # value is ignored — the India-tied drop must only happen when the user asks
    # for it in chat, never because of a default nobody can see.
    remote_only: bool = False
    cities: list[str] = []
    roles: list[str] = []
    github_url: str = ""
    # Work arrangement + geography filters. Empty lists mean "don't filter",
    # which reproduces the pre-feature behaviour exactly.
    work_modes: list[str] | None = None
    # Retired with the region chips; accepted and ignored for the same reason
    # as remote_only.
    regions: list[str] | None = None
    countries: list[str] | None = None


class ProfileRequest(BaseModel):
    name: str = ""


class ProfileIdRequest(BaseModel):
    id: str = "default"


def _load_legacy_profiles() -> dict:
    """Read the pre-accounts profiles.json (best-effort; missing/corrupt -> {}).

    Only used once, to hand the existing CV sessions to the first account.
    """
    try:
        with open(PROFILES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _is_pristine(rec: dict) -> bool:
    """True for a session nobody has actually filled in.

    Used to tell a real CV from a freshly seeded one, in both directions: a
    legacy record with nothing in it is not worth importing, and a stored
    profile with content in it must never be overwritten.
    """
    return not (rec.get("resume_text") or rec.get("roles") or rec.get("skills")
                or rec.get("cities"))


def _find_imported_from(uid: int, source_id: str) -> dict | None:
    """The already-imported copy of a legacy record, if the import ran before.

    The imported CVs are what stops the migration being destructive, but a
    re-run (a retry after a failed sign-up, say) would otherwise duplicate all
    of them. Only the legacy "default" is re-id'd, so this marker is what makes
    that re-run idempotent.
    """
    for rec in auth.list_profiles(uid):
        if rec.get("migrated_from") == source_id:
            return rec
    return None


def _migrate_legacy_profiles(uid: int) -> int:
    """Import every profiles.json record to `uid`. Returns how many moved.

    Run when the first account is created. The old CVs are preserved, but they
    are NOT made the active session: the account opens on the blank template
    created by create_user, and the imported CVs sit alongside it in the
    sidebar to be opened when wanted. Landing on somebody's old, half-filled
    profile is what made the first screen look pre-populated.
    """
    legacy = _load_legacy_profiles()
    if not legacy:
        return 0
    moved = 0
    for pid, rec in legacy.items():
        if not isinstance(rec, dict) or _is_pristine(rec):
            continue
        payload = dict(rec)
        payload.setdefault("name", pid)
        payload.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

        if pid == "default":
            # "default" is reserved for the blank session every account starts
            # on, so the legacy default is imported under a new id instead of
            # overwriting it. is_default stays 0: the imported CV is a
            # reference in the sidebar, not the session to hunt in.
            if _find_imported_from(uid, pid):
                continue
            payload["migrated_from"] = pid
            payload.pop("id", None)
            auth.create_profile(uid, payload.get("name") or "Imported CV", payload)
            moved += 1
            continue

        existing = auth.get_profile(uid, pid)
        if existing and not _is_pristine(existing):
            continue                       # already real data; don't clobber
        payload["id"] = pid
        auth.save_profile(uid, pid, payload)
        moved += 1
    return moved


def _rehome_polluted_default(uid: int) -> str:
    """Make sure `default` really is the blank starting session. Returns the id
    of the session the old contents were moved to, or "" if nothing to do.

    Accounts created before the blank-default fix adopted somebody's CV AS
    `default`, so logging in opened a session that already had 40+ skills, two
    roles and a city in it — the "why is everything pre-filled?" complaint. The
    data is never deleted: it is copied to a new sidebar session first, and
    `default` is then reset to the blank template.

    One-shot by construction: after this runs `default` is pristine, so it never
    fires again, and anything the user later puts in their default session is
    theirs and stays.
    """
    rec = auth.get_profile(uid, "default")
    if not rec or _is_pristine(rec):
        return ""
    if _find_imported_from(uid, "default"):
        return ""                       # already re-homed on an earlier login
    payload = dict(rec)
    name = payload.get("name") or "Imported CV"
    payload["migrated_from"] = "default"
    payload.pop("id", None)
    pid = auth.create_profile(uid, name, payload)
    blank = auth.blank_profile()
    blank.update({"id": "default", "name": "Default",
                  "created_at": rec.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S")})
    auth.save_profile(uid, "default", blank)
    _log({"event": "default_profile_rehomed", "user_id": uid,
          "moved_to": pid, "skills": len(payload.get("skills") or [])})
    return pid


def _profile_for(uid: int, profile_id: str) -> dict:
    """A user's profile, falling back to their default if the id is unknown.

    Scoped by uid, so a request can never reach another account's CV even by
    guessing an id. The old code silently rewrote an unknown id to the shared
    "default", which leaked one person's session into another's.
    """
    rec = auth.get_profile(uid, profile_id) if profile_id else None
    if rec:
        return rec
    rec = auth.get_profile(uid, "default")
    if rec:
        return rec
    # Should not happen (create_user seeds a default) but never leave a request
    # without a profile to work with.
    pid = auth.create_profile(uid, "Default")
    return auth.get_profile(uid, pid) or {}


def _patch_profile(uid: int, profile_id: str, **fields) -> dict:
    """Merge fields into a stored profile and return the updated record.

    Only the keys the caller passes are touched; resume_text/work_modes and
    friends are stored verbatim so a restart restores exactly what was saved.
    """
    rec = _profile_for(uid, profile_id)
    rec.update(fields)
    rec["id"] = rec.get("id") or profile_id
    auth.save_profile(uid, rec["id"], rec)
    return rec


def _cfg_from_profile(base: AgentConfig, snap: dict) -> AgentConfig:
    """Build an AgentConfig for one profile, layered over the shared base.

    `base` only supplies the process-wide defaults (model, context window,
    lookback, result cap) that are not personal to anyone. Every search-related
    field comes from the profile, so one account's roles/cities/filters can
    never bleed into another's.
    """
    roles = [r for r in (snap.get("roles") or []) if r]
    # A brand-new session has no roles yet, and it must stay that way: falling
    # back to the process-wide default here is what made a fresh account look
    # like it had already been set up ("Junior Data Analyst / entry-level").
    # An empty role is not an error - the agent asks what the user wants, and
    # search_query_from falls back to a generic query if a hunt starts anyway.
    primary = roles[0] if roles else ""
    cities = [c for c in (snap.get("cities") or []) if c]
    skills = [s for s in (snap.get("skills") or []) if s]
    work_modes = filters.normalize_work_modes(snap.get("work_modes"))
    regions = filters.normalize_regions(snap.get("regions"))
    countries = filters.normalize_countries(snap.get("countries"))
    # The outside-India-only rule was retired, so the flag is forced off on
    # load rather than trusted. Reading it is what kept it alive: a profile
    # saved before the removal still carried remote_only=true, and loading
    # one would silently start dropping every India-tied job again. It stays
    # in the stored dict (and in the API response) purely so old files round
    # trip without a migration; it can no longer affect a search.
    remote_only = False
    return replace(
        base,
        target_role=primary,
        target_roles=roles, indeed_query=primary,
        skills=skills, resume_text=snap.get("resume_text") or "",
        remote_only=remote_only,
        # Same rule the config endpoint uses, so switching profiles and saving
        # by hand can't disagree about where to search.
        indeed_location=filters.boards_location(work_modes, cities, remote_only,
                                                countries),
        preferred_cities=cities,
        role_suggestions=[s for s in (snap.get("role_suggestions") or []) if s],
        work_modes=work_modes,
        regions=regions,
        countries=countries,
        github_url=snap.get("github_url") or "",
        model=snap.get("model") or base.model,
        num_ctx=int(snap.get("num_ctx") or base.num_ctx),
        days_back=int(snap.get("days_back") or base.days_back),
        max_results=int(snap.get("max_results") or base.max_results),
    )


# --------------------------------------------------------------------------
# Link inference from a resume's plain text (GitHub / LinkedIn / portfolio).
# --------------------------------------------------------------------------

# Links as they actually come out of a CV, not as they appear in a browser
# address bar. A LaTeX CV puts the contact block in a two-column table, so the
# text layer reads "a.com/x|b.com/y|" - no protocol, and framed in pipes.
#
# GitHub and LinkedIn get a scheme-optional pattern each because those two are
# unambiguous by name. The generic portfolio match deliberately REQUIRES an
# explicit http(s):// - without it, "Node.js" in a skills line and the
# "someone.th" in "someone.th@gmail.com" both match, and a bogus URL would end
# up printed on the resume.
_GITHUB_LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)", re.I)
_LINKEDIN_LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9_.-]+)", re.I)
_HTTP_LINK_RE = re.compile(r"https?://[^\s<>\"'|]+", re.I)
_PORTFOLIO_SKIP_RE = re.compile(
    r"(facebook|instagram|twitter|x\.com|t\.me|wa\.me|youtube|shorts|"
    r"api\.|schema\.org|w3\.org|\.png|\.jpg|\.jpeg|\.gif|\.svg|mailto|"
    r"linkedin|github|indeed|glassdoor|naukri|monster)", re.I)


def _clean_link(url: str) -> str:
    """Trim the punctuation a PDF/LaTeX text layer glues onto a URL."""
    return (url or "").strip().strip("|").rstrip(".,;:)]}>\"'").strip("|")


class Ctx:
    """One authenticated request: who is asking, which CV they have open, and
    an agent bundle built for exactly that pair.

    Replaces the process-wide `app.state.bundle` / `app.state.profile_id`. Two
    browsers can now be hunting different roles at the same time, because each
    request resolves its own bundle instead of sharing one global.
    """

    __slots__ = ("user", "uid", "token", "profile_id", "profile", "bundle")

    def __init__(self, user, token, profile_id, profile, bundle):
        self.user = user
        self.uid = user["id"]
        self.token = token
        self.profile_id = profile_id
        self.profile = profile
        self.bundle = bundle

    @property
    def cfg(self) -> AgentConfig:
        return self.bundle.cfg

    def save(self, **fields) -> dict:
        """Patch the open profile and refresh the in-request copy."""
        self.profile = _patch_profile(self.uid, self.profile_id, **fields)
        return self.profile


def _bundle_cache() -> dict:
    """Bundles keyed by (account, profile). Created on first use."""
    if not hasattr(app.state, "bundles"):
        app.state.bundles = {}
    return app.state.bundles


def _drop_bundle(uid: int, profile_id: str) -> None:
    """Forget a cached bundle so the next request rebuilds it from storage."""
    _bundle_cache().pop((uid, profile_id), None)


async def _bundle_for(uid: int, profile_id: str, snap: dict):
    """The agent bundle for one (account, profile), built on demand and cached.

    The MCP runtime is shared - discovering the tools costs ~3.4s and the tool
    set never changes - but the compiled graph captures a specific AgentConfig,
    so it must be per profile. Caching keeps a page refresh from paying the
    graph-compile cost again.
    """
    key = (uid, profile_id)
    cache = _bundle_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit
    cfg = _cfg_from_profile(load_config(), snap)
    # Scope the board scraper's "already shown" memory to this account. The
    # bundle is cached per (account, profile), so this is set once and the
    # memory it keys cannot drift between runs.
    cfg = replace(cfg, seen_scope=f"u{uid}")
    bundle = await create_agent(cfg, runtime=app.state.runtime)
    cache[key] = bundle
    return bundle


async def build_ctx(request: Request) -> Ctx:
    """FastAPI dependency: resolve the caller, or 401.

    Everything below the UI needs an account, so this is the single place the
    session cookie is turned into a usable context. An address that has not been
    confirmed gets 403 rather than 401 - the session is real, the mailbox is not
    proven yet, and the UI turns that into the "check your inbox" screen.
    """
    token = request.cookies.get(auth.SESSION_COOKIE) or ""
    user = auth.session_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="sign in to continue")
    if not user.get("verified_at"):
        raise HTTPException(status_code=403, detail="confirm your email address")
    pid = auth.session_profile(token) or "default"
    snap = _profile_for(user["id"], pid)
    if snap.get("id") and snap["id"] != pid:
        # The session pointed at a profile that no longer exists; realign it.
        pid = snap["id"]
        auth.set_session_profile(token, pid)
    bundle = await _bundle_for(user["id"], pid, snap)
    return Ctx(user, token, pid, snap, bundle)


def detect_links(text: str) -> dict:
    """Pull profile links out of resume text: the first GitHub and LinkedIn
    profile, then the first other site that isn't social/cdn/binary noise.

    The input is PDF text, not markup, so a URL may arrive without a scheme,
    wrapped in table pipes, or split across a line break. All three are handled
    here so the caller can just store what it gets.
    """
    t = _unwrap_broken_urls(text or "")
    out = {"github": "", "linkedin": "", "portfolio": ""}
    gh = _GITHUB_LINK_RE.search(t)
    if gh:
        out["github"] = resume_generator._normalize_url(_clean_link(gh.group(0)))
    li = _LINKEDIN_LINK_RE.search(t)
    if li:
        out["linkedin"] = resume_generator._normalize_url(
            _clean_link(li.group(0)))
    for m in _HTTP_LINK_RE.finditer(t):
        url = _clean_link(m.group(0))
        if not url or _is_not_a_link(t, m.start(), m.end()):
            continue
        if _PORTFOLIO_SKIP_RE.search(url):
            continue
        out["portfolio"] = resume_generator._normalize_url(url)
        break
    return {k: v for k, v in out.items() if v}


def _is_not_a_link(text: str, start: int, end: int) -> bool:
    """Reject a match that is really an email address.

    The pattern requires a scheme, so the only way to get one wrong is text that
    embeds a URL in an address - "mail http://x@corp.example/y" - where the '@'
    lands inside the match rather than beside it.
    """
    match = text[start:end]
    # strip the scheme first: splitting on "/" alone would cut inside "http://"
    host = re.sub(r"^https?://", "", match, flags=re.I).split("/", 1)[0]
    if "@" in host:
        return True
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    return before == "@" or after == "@"


def _unwrap_broken_urls(text: str) -> str:
    """Rejoin a URL that a PDF text layer split across lines or columns.

    A hyperlink in a PDF is often drawn as separate runs, and pypdf keeps the
    line breaks between them, so "https://github.com/" ends up on one line and
    the username on the next. Joining them back before matching is what makes
    detection work on real CVs.
    """
    # A line ending in the scheme/host, with the rest continuing below it.
    text = re.sub(r"(?i)\b((?:https?://)?(?:www\.)?"
                  r"(?:github\.com|linkedin\.com|gitlab\.com|bitbucket\.org)/?)\s*\n\s*",
                  r"\1", text)
    # Table column separators inside a run of link-ish text.
    text = re.sub(r"(?<=[A-Za-z0-9_/.-])\|(?=[A-Za-z0-9])", " ", text)
    return text


# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------

# Run history, kept per account. A single shared list would show one person
# everyone else's searches (their prompt text and results), so the History
# drawer filters by uid.
RUNS: dict[int, list[dict]] = {}


def _user_runs(uid: int) -> list[dict]:
    return RUNS.setdefault(uid, [])


def _user_runs_all(uid: int) -> list[dict]:
    return list(_user_runs(uid))


def _log(record: dict, kind: str = "runs") -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        with open(os.path.join(LOG_DIR, f"{kind}-{time.strftime('%Y%m%d')}.jsonl"),
                  "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # logging must never take the app down


def _remember_run(summary: dict, uid: int | None = None) -> None:
    runs = _user_runs(uid) if uid is not None else RUNS.setdefault(0, [])
    runs.insert(0, summary)
    del runs[MAX_RUNS_KEPT:]


# --------------------------------------------------------------------------
# Skill extraction (resume/CV -> search keywords)
# --------------------------------------------------------------------------

def _pdf_link_targets(reader) -> list[str]:
    """Every hyperlink a PDF declares, whether or not it is visible text.

    A CV built in Word or LaTeX usually shows a labelled link - the text says
    "LinkedIn" while the address lives in the page's link annotation. Plain text
    extraction only returns the label, so those CVs look like they have no
    profile links at all. Reading /Annots recovers the real URLs.
    """
    urls: list[str] = []
    for page in reader.pages:
        try:
            annots = page.get("/Annots") or []
        except Exception:
            continue
        for annot in annots:
            try:
                obj = annot.get_object()
                action = obj.get("/A")
                uri = action.get("/URI") if action else obj.get("/URI")
                if uri:
                    urls.append(str(uri))
            except Exception:
                continue
    return urls


def _extract_text(data: bytes, filename: str) -> str:
    """Best-effort text extraction from txt/md/csv/html/pdf/docx/rtf uploads
    (DOCX via docx2txt and RTF via striprtf, both soft-imported so the app
    still runs if pip install was skipped).

    PDF uploads also get their link annotations appended, because that is the
    only place a clickable profile link exists in most CVs.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if not text.strip():
                return text
            links = _pdf_link_targets(reader)
            if links:
                # appended, so the CV's own body text is not disturbed
                return text + "\n" + "\n".join(links)
            return text
        except Exception:
            pass
    elif name.endswith(".docx"):
        try:
            import docx2txt
            text = docx2txt.process(io.BytesIO(data))
            if text.strip():
                return text
        except Exception:
            pass
    elif name.endswith(".rtf"):
        try:
            from striprtf.striprtf import rtf_to_text
            text = rtf_to_text(data.decode("latin-1"))
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

# A CV's skills block is usually grouped, and the PDF text layer prints the
# group name against the first item with a colon and no space:
#   "•Programming Languages:Python, JavaScript, SQL"
#   "•Backend / Web:Django, Flask, FastAPI-ready REST API design"
#   "•Data Analysis & Visualization:Pandas, NumPy"
# Splitting only on commas made "Programming Languages Python" and "Web
# Technologies HTML" single keyword strings, so searches went out for a phrase
# nobody has ever heard of.
_SKILL_CATEGORY_RE = re.compile(
    r"(?i)\b(languages?|tools?|technolog\w*|frameworks?|databases?|cloud|"
    r"devops|web|programming|concepts?|misc|analytics|testing|design|"
    r"programming\s+languages|soft\s+skills)\b")


def _split_skill_chunk(chunk: str) -> list[str]:
    """One colon-separated group -> its items, with the group label dropped.

    The label is recognised structurally rather than from a fixed word list,
    because these labels are unbounded ("Data Analysis & Visualization",
    "Programming Languages", "Backend / Web"): anything to the left of a colon
    that contains whitespace or a slash is a group heading, while a colon with
    no space in it is part of the skill itself - "Node:JS", "C++". A fixed
    vocabulary caught the first four labels and then invented a fifth rule for
    the next CV.
    """
    if ":" not in chunk:
        return [chunk]
    label, _, rest = chunk.partition(":")
    label = label.strip(" \t*-\u2022")
    rest = rest.strip()
    if not rest or not label:
        return [chunk]
    if re.search(r"\s", label) or "/" in label or _SKILL_CATEGORY_RE.search(label):
        return [rest]
    return [label, rest]


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
        pieces = []
        for part in re.split(r"[,;|\n•]+", chunk):
            pieces.extend(_split_skill_chunk(part))
        for p in pieces:
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
    """FastAPI lifespan hook: open the accounts database and boot the MCP
    subprocesses once.

    The MCP runtime is shared by everyone (discovering the tools costs ~3.4s
    and the set never changes), but there is deliberately NO app.state.bundle
    any more. A bundle captures one person's AgentConfig in its compiled graph,
    so a single global one meant two signed-in users shared a search context.
    Bundles are now built per (account, profile) on first use - see
    _bundle_for()."""
    auth.init_db()
    app.state.runtime = await create_runtime()
    app.state.bundles = {}
    yield
    app.state.bundles = None
    app.state.runtime = None


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


# --------------------------------------------------------------------------
# Accounts: register, verify, sign in, reset
# --------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str = ""
    password: str = ""


class LoginRequest(BaseModel):
    email: str = ""
    password: str = ""


class ForgotRequest(BaseModel):
    email: str = ""


class ResetRequest(BaseModel):
    token: str = ""
    password: str = ""


MIN_PASSWORD = 8


def _base_url(request: Request) -> str:
    """Absolute URL for the links we email out.

    Prefers an explicit APP_URL (set it when the app is behind a proxy or a
    tunnel), otherwise reconstructs from the request.
    """
    return (os.environ.get("APP_URL", "").strip()
            or str(request.base_url)).rstrip("/")


def _set_session_cookie(response, token: str, expires: str,
                        request: Request) -> None:
    """Attach the session cookie.

    Secure is decided from the request's own scheme, because a response object
    has no notion of the URL it is answering. Marking it Secure on a plain-HTTP
    LAN address would make the browser drop the cookie and the sign-in would
    silently never stick.
    """
    response.set_cookie(
        auth.SESSION_COOKIE, token, httponly=True, samesite="lax",
        secure=request.url.scheme == "https",
        expires=expires, path="/")


def _check_password(password: str) -> str | None:
    if len(password or "") < MIN_PASSWORD:
        return "use at least %d characters" % MIN_PASSWORD
    return None


@app.get("/api/auth/state")
async def auth_state(request: Request):
    """What the sign-in screen needs: is anyone set up, is this browser signed
    in, and can the server actually send mail."""
    token = request.cookies.get(auth.SESSION_COOKIE) or ""
    user = auth.session_user(token)
    return {
        "needs_bootstrap": auth.count_users() == 0,
        "authenticated": bool(user),
        "email": (user or {}).get("email", ""),
        "verified": bool(user and user.get("verified_at")),
        "smtp_configured": mailer.configured(),
    }


@app.post("/api/auth/register")
async def register(req: RegisterRequest, request: Request):
    """Create an account.

    The very first account also becomes the owner and adopts the CV sessions
    from the old profiles.json, so an existing install keeps its history. When
    SMTP is configured a confirmation link is emailed and the account stays
    unusable until the link is followed; without SMTP the link is returned in
    the response instead, so a fresh checkout can still finish signing up.
    """
    weak = _check_password(req.password)
    if weak:
        return JSONResponse({"ok": False, "error": weak}, status_code=400)
    first = auth.count_users() == 0
    try:
        user = auth.create_user(req.email, req.password, owner=first)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    adopted = 0
    if first:
        adopted = _migrate_legacy_profiles(user["id"])
        _log({"event": "owner_created", "email": user["email"],
              "profiles_adopted": adopted})
    # An unverified account can sign in far enough to be verified, but nothing
    # else - it has no CV and no settings.
    token = auth.issue_token(user["id"], "verify", auth.VERIFY_TOKEN_HOURS)
    # The link points at the app, not the API: following it should land the
    # person in the UI, not on a page of raw JSON.
    link = f"{_base_url(request)}/?verify={token}"
    if mailer.configured():
        result = mailer.send_verification(user["email"], link)
    else:
        result = mailer.MailResult(False, "SMTP is not configured")

    if not result.ok:
        # No mail sent, so nothing can be verified by email: let them straight
        # in and say why. The alternative is an account nobody can ever use.
        auth.mark_verified(user["id"])
        if not result.dev_link:
            result.dev_link = link

    # A session is issued either way. With SMTP the account stays locked behind
    # 403 until the link is followed, but a session is what lets the person ask
    # for a fresh link; without SMTP it is simply a signed-in account.
    # result.ok is exactly "a confirmation was actually sent", so the inverse
    # tells us whether the address is already usable.
    session, expires = auth.create_session(user["id"])
    auth.set_session_profile(session, "default")
    response = JSONResponse({
        "ok": True,
        "verified": not result.ok,
        "needs_verification": bool(result.ok),
        "email": user["email"],
        "detail": result.detail,
        "dev_link": result.dev_link or "",
        "profiles_adopted": adopted,
    }, status_code=202 if result.ok else 200)
    _set_session_cookie(response, session, expires, request)
    return response


@app.post("/api/auth/resend")
async def resend(request: Request):
    """Send the confirmation link again.

    Requires a session, which is what keeps this from being an enumeration
    oracle: the link can only ever be re-sent to the address of the account the
    caller is already signed in as.
    """
    token = request.cookies.get(auth.SESSION_COOKIE) or ""
    user = auth.session_user(token)
    if not user:
        return JSONResponse({"ok": False, "error": "sign in to continue"},
                            status_code=401)
    if user.get("verified_at"):
        return {"ok": True, "already_verified": True}

    fresh = auth.issue_token(user["id"], "verify", auth.VERIFY_TOKEN_HOURS)
    link = f"{_base_url(request)}/?verify={fresh}"
    if mailer.configured():
        result = mailer.send_verification(user["email"], link)
    else:
        result = mailer.MailResult(False, "SMTP is not configured")
    if not result.ok and mailer.dev_links_allowed():
        result.dev_link = link
    return JSONResponse({
        "ok": result.ok, "sent": result.ok, "detail": result.detail,
        "dev_link": result.dev_link or "",
    }, status_code=200 if result.ok else 503)


@app.get("/api/auth/verify")
async def verify(token: str = ""):
    """Confirm an address from an emailed link. Idempotent enough to be
    clicked twice: the second attempt just reports the outcome."""
    user = auth.consume_token(token, "verify")
    if not user:
        return JSONResponse({"ok": False,
                             "error": "that link is invalid or has expired"},
                            status_code=400)
    auth.mark_verified(user["id"])
    auth.destroy_user_sessions(user["id"])   # old sessions predate verification
    mailer.send_welcome(user["email"])
    return {"ok": True, "email": user["email"], "message": "Email confirmed."}


@app.post("/api/auth/login")
async def login(req: LoginRequest, request: Request):
    """Sign in and set the session cookie.

    The same message covers an unknown address and a wrong password, so this
    cannot be used to discover which addresses have accounts.
    """
    user = auth.authenticate(req.email, req.password)
    if not user:
        _log({"event": "login_failed", "email": auth.normalize_email(req.email)})
        return JSONResponse({"ok": False, "error": "wrong email or password"},
                            status_code=401)
    # Every login opens the blank starting session (see set_session_profile
    # below), so any pre-fill still sitting in `default` from the old build has
    # to be cleared out of the way first. Nothing is deleted.
    _rehome_polluted_default(user["id"])
    token, expires = auth.create_session(user["id"])
    auth.set_session_profile(token, "default")
    _bundle_cache().pop((user["id"], "default"), None)
    response = JSONResponse({"ok": True, "email": user["email"],
                             "verified": bool(user.get("verified_at"))})
    _set_session_cookie(response, token, expires, request)
    return response


@app.post("/api/auth/logout")
async def logout(request: Request):
    token = request.cookies.get(auth.SESSION_COOKIE) or ""
    if token:
        auth.destroy_session(token)
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return response


@app.post("/api/auth/forgot")
async def forgot(req: ForgotRequest, request: Request):
    """Email a password-reset link.

    Always reports success, whether or not the address is known: answering
    differently would turn this into an account-enumeration oracle. That includes
    the status code and the response shape, which is why the unsent link is only
    ever included when AUTH_DEV_LINKS is explicitly turned on.
    """
    user = auth.get_user_by_email(req.email)
    sent = False
    if user:
        token = auth.issue_token(user["id"], "reset", auth.RESET_TOKEN_HOURS)
        # The address is in the link so the reset form can prefill it; the
        # token is the only thing that authorises the change.
        link = f"{_base_url(request)}/?token={token}&email={user['email']}"
        result = mailer.send_password_reset(user["email"], link)
        sent = result.ok
        if result.dev_link and mailer.dev_links_allowed():
            return JSONResponse({"ok": True, "sent": False, "dev_link": link},
                                status_code=202)
        if not result.ok:
            _log({"event": "reset_email_failed", "detail": result.detail},
                 kind="errors")
    return {"ok": True, "sent": sent}


@app.post("/api/auth/reset")
async def reset(req: ResetRequest):
    """Set a new password from an emailed link, then sign every session out."""
    weak = _check_password(req.password)
    if weak:
        return JSONResponse({"ok": False, "error": weak}, status_code=400)
    user = auth.consume_token(req.token, "reset")
    if not user:
        return JSONResponse({"ok": False,
                             "error": "that link is invalid or has expired"},
                            status_code=400)
    # set_password() also drops every session for the account, so a cookie
    # captured before the reset stops working immediately.
    auth.set_password(user["id"], req.password)
    _log({"event": "password_reset", "email": user["email"]})
    return {"ok": True, "message": "Password updated. Sign in with it now."}


@app.get("/")
async def index():
    """Serve the single-page JobScope UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


_BOARD_NAMES_CACHE: dict[str, list[str]] = {}


def _board_names() -> list[str]:
    """Every board the scraper can search, read from the scraper itself.

    Imported lazily: the scraper pulls in Scrapling, which costs a second or
    two, and nothing else in this process needs it. Cached because the
    registry is fixed for the life of the process. Returns [] rather than
    raising if the scraper cannot be imported, so the UI degrades to hiding the
    ticker instead of failing the whole settings request.
    """
    if "all" not in _BOARD_NAMES_CACHE:
        try:
            import mcp_server_indeed_scraper as _s
            _BOARD_NAMES_CACHE["all"] = list(_s.BOARDS)
        except Exception:
            _BOARD_NAMES_CACHE["all"] = []
    return _BOARD_NAMES_CACHE["all"]


def _default_board_names() -> list[str]:
    """The boards a sweep actually probes, which is not all of them."""
    if "default" not in _BOARD_NAMES_CACHE:
        try:
            import mcp_server_indeed_scraper as _s
            _BOARD_NAMES_CACHE["default"] = list(_s.default_boards())
        except Exception:
            _BOARD_NAMES_CACHE["default"] = []
    return _BOARD_NAMES_CACHE["default"]


@app.get("/api/boards")
async def public_boards():
    """Board names for the sign-in screen, with no account.

    Deliberately unauthenticated, because it has to be. The ticker on the
    sign-in gate is the first thing a visitor sees, and it renders before anyone
    has signed in - while /api/config 401s, which left that ticker empty and
    the gate claiming only "Job boards, one search".

    This is not a way around the login: it carries no user state, no queries,
    no results, and nothing that is not already printed on the page in prose.
    A list of which job boards an app can scrape is not private information.
    """
    return {"boards": _board_names(), "default_boards": _default_board_names()}


@app.get("/api/config")
async def get_config(ctx: Ctx = Depends(build_ctx)):
    """Current agent configuration, exposed so the UI can render its state."""
    bundle = ctx.bundle
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
        # No roles on a new session is a real state, so it is reported as an
        # empty list rather than filled in with a placeholder the UI would
        # then render as a chip the user never chose.
        "roles": list(bundle.cfg.target_roles),
        # Suggestion catalog for the role dropdown. Handing it to the UI means
        # the list has one owner; the fallback there is only for an older server.
        "role_options": list(JOB_ROLE_OPTIONS),
        # Suggestion catalog for the Cities picker. Served from filters.py so the
        # list and the geography rules that run against a chosen city cannot
        # disagree. Free text is still accepted — this only makes the common
        # spellings one click instead of typing.
        "city_options": [{"key": c, "region": filters.city_region(c)}
                         for c in filters.known_cities()],
        "work_modes": bundle.cfg.work_modes or [],
        "work_mode_options": [{"key": k, "label": v}
                              for k, v in filters.WORK_MODES.items()],
        "regions": [],
        "region_options": [{"key": k, "label": v}
                           for k, v in filters.REGIONS.items()],
        "countries": bundle.cfg.countries or [],
        "country_options": [{"key": k, "label": k,
                             "region": v[0]}
                            for k, v in filters.COUNTRIES.items()],
        "github_url": bundle.cfg.github_url or "",
        "profile_id": ctx.profile_id,
        "profile_name": (ctx.profile.get("name") or "Default"),
        "email": ctx.user["email"],
        "tools": [t.name for t in bundle.tools],
        # The board registry, served from the scraper itself. The UI used to
        # hardcode its own list of board names, which drifted: it advertised
        # four sites the app cannot scrape while omitting four that work. One
        # owner means the ticker cannot lie about what will be searched.
        "boards": _board_names(),
        "default_boards": _default_board_names(),
    }


@app.post("/api/config")
async def update_config(req: ConfigRequest, ctx: Ctx = Depends(build_ctx)):
    """Replace the running agent with a new configuration.

    Roles are multi-select: the first is the primary (drives the board query),
    the rest are kept on the list for matching/context. Never throws a bare
    500 — restart failures come back as a normal JSON payload so the UI can
    show what happened instead of a vague network error."""
    # Picker or free text, both land here. Unknown cities are kept (a real place
    # with no catalog entry is still a valid place to search); only exact
    # case-insensitive duplicates and blanks are dropped.
    cities = filters.normalize_cities(req.cities)
    roles = [r.strip() for r in req.roles if r.strip()]
    # Unknown chip values are dropped rather than trusted, so a hand-rolled
    # POST can't smuggle in a bogus arrangement or country.
    work_modes = filters.normalize_work_modes(req.work_modes)
    regions: list[str] = []          # retired: never stored, never applied
    countries = filters.normalize_countries(req.countries)
    previous = ctx.cfg
    if not roles:
        # user cleared the chips: fall back to what the hunt is currently using
        roles = [r for r in (previous.target_roles or [previous.target_role]) if r]
    if not roles:
        roles = [req.target_role.strip()] if req.target_role.strip() else []
    # No role is a legitimate state, not a reason to invent one. This used to
    # fall back to the demo "Junior Data Analyst" role, which meant saving an
    # empty session quietly re-filled it.
    primary = roles[0] if roles else ""
    github_url = resume_generator._normalize_url(req.github_url)
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
            resume_text=previous.resume_text,
            # Retired flag: forced off, never taken from the request.
            remote_only=False,
            # An on-site/hybrid hunt needs a real place to search, so the
            # first chosen city becomes the boards' location; otherwise the
            # search stays worldwide. filters.boards_location owns that rule.
            indeed_location=filters.boards_location(work_modes, cities,
                                                    False, countries),
            preferred_cities=cities,
            role_suggestions=previous.role_suggestions,
            work_modes=work_modes,
            regions=regions,
            countries=countries,
            github_url=github_url,
        )
        bundle = await create_agent(cfg, runtime=app.state.runtime)
    except Exception as exc:
        _log({"event": "config_restart_error", "detail": str(exc)}, kind="errors")
        return {"ok": False, "error": f"Agent restart failed: {exc}"}
    # Only swap the cached bundle in once the rebuild actually succeeded, so a
    # failure leaves the caller on a working agent instead of a broken one.
    _bundle_cache()[(ctx.uid, ctx.profile_id)] = bundle
    ctx.bundle = bundle
    # The perf picks live on the profile too, so they survive a restart and
    # follow the user to another machine.
    ctx.save(skills=cfg.skills, roles=cfg.target_roles, cities=cities,
             remote_only=False, github_url=cfg.github_url,
             work_modes=work_modes, regions=[], countries=countries,
             model=cfg.model, num_ctx=cfg.num_ctx, days_back=cfg.days_back,
             max_results=cfg.max_results)
    _log({"event": "config_updated", "user": ctx.user["email"],
          "model": cfg.model, "num_ctx": cfg.num_ctx,
          "skills": cfg.skills, "remote_only": cfg.remote_only,
          "cities": cities, "roles": roles, "role": primary,
          "work_modes": work_modes, "regions": regions,
          "countries": countries,
          "profile": ctx.profile_id,
          "github_url": cfg.github_url})
    return {"ok": True, "target_role": primary, "roles": cfg.target_roles,
            "model": cfg.model, "num_ctx": cfg.num_ctx, "skills": cfg.skills,
            "remote_only": cfg.remote_only, "cities": cfg.preferred_cities,
            "role_suggestions": cfg.role_suggestions,
            "work_modes": cfg.work_modes, "regions": [],
            "work_mode_options": [{"key": k, "label": v}
                                  for k, v in filters.WORK_MODES.items()],
            "region_options": [{"key": k, "label": v}
                               for k, v in filters.REGIONS.items()],
            "role_options": list(JOB_ROLE_OPTIONS),
            "countries": cfg.countries,
            "country_options": [{"key": k, "label": k,
                                 "region": v[0]}
                                for k, v in filters.COUNTRIES.items()],
            "location": cfg.indeed_location,
            "github_url": cfg.github_url}


@app.get("/api/profiles")
async def list_profiles(ctx: Ctx = Depends(build_ctx)):
    """Sidebar sessions: this account's profiles only, with a compact summary
    and which one this browser has open."""
    rows = [{
        "id": p["id"], "name": p.get("name") or p["id"][:8],
        "created_at": p.get("created_at", ""),
        "skills": len(p.get("skills") or []),
        "resume_chars": len(p.get("resume_text") or ""),
        "github": p.get("github_url") or "",
        "active": p["id"] == ctx.profile_id,
    } for p in auth.list_profiles(ctx.uid)]
    return {"profiles": rows, "active": ctx.profile_id}


@app.post("/api/profiles")
async def create_profile(req: ProfileRequest | None = None,
                         ctx: Ctx = Depends(build_ctx)):
    """A fresh, empty session in the sidebar (optionally named) and open it.
    Accepts a bodyless POST (the UI sends `POST /api/profiles` with no JSON) so
    the + New button always works even if the client sends no name."""
    name = ((req.name if req else "") or "New session").strip()
    pid = auth.create_profile(ctx.uid, name)
    auth.set_session_profile(ctx.token, pid)
    return {"ok": True, "id": pid, "name": name}


@app.post("/api/profiles/activate")
async def activate_profile(req: ProfileIdRequest, ctx: Ctx = Depends(build_ctx)):
    """Switch to a saved session: current choices are snapshotted back into
    the open profile first, then the target's resume/skills are loaded."""
    live = ctx.cfg
    ctx.save(skills=live.skills, roles=live.target_roles,
             cities=live.preferred_cities, remote_only=live.remote_only,
             resume_text=live.resume_text, github_url=live.github_url,
             work_modes=live.work_modes, regions=live.regions,
             countries=live.countries)
    # Refuse an id this account does not own rather than silently landing on
    # its default - otherwise probing ids would leak which sessions exist.
    if not auth.get_profile(ctx.uid, req.id):
        return JSONResponse({"ok": False, "error": "No such session"},
                            status_code=404)
    auth.set_session_profile(ctx.token, req.id)
    snap = _profile_for(ctx.uid, req.id)
    try:
        bundle = await _bundle_for(ctx.uid, req.id, snap)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    cfg = bundle.cfg
    _log({"event": "profile_loaded", "user": ctx.user["email"],
          "profile": req.id, "skills": len(cfg.skills),
          "work_modes": cfg.work_modes, "regions": cfg.regions,
          "countries": cfg.countries})
    return {"ok": True, "profile_id": req.id, "target_role": cfg.target_role,
            "skills": cfg.skills, "roles": list(cfg.target_roles),
            "cities": cfg.preferred_cities, "remote_only": cfg.remote_only,
            "work_modes": cfg.work_modes, "regions": cfg.regions,
            "countries": cfg.countries,
            "github_url": cfg.github_url or ""}


@app.post("/api/profiles/delete")
async def delete_profile(req: ProfileIdRequest, ctx: Ctx = Depends(build_ctx)):
    """Remove a saved session. The default one can't be deleted; deleting the
    open one falls back to the default."""
    if req.id == "default":
        return JSONResponse(
            {"ok": False, "error": "The Default session can't be deleted."},
            status_code=400)
    if not auth.delete_profile(ctx.uid, req.id):
        return JSONResponse({"ok": False, "error": "No such session"},
                            status_code=404)
    _drop_bundle(ctx.uid, req.id)
    if ctx.profile_id == req.id:
        auth.set_session_profile(ctx.token, "default")
    return {"ok": True}


@app.post("/api/reset")
async def reset_everything(ctx: Ctx = Depends(build_ctx)):
    """Wipe the account back to a genuinely empty state.

    Every non-default session is deleted, the default one is rebuilt blank
    (no CV text, no skills, no roles, no cities, no GitHub link, no filters)
    and the session cookie is pointed back at it. The account, its email and
    its password are untouched - this is a reset of the job-search data, not
    of the login.

    Also cleared, because the UI still shows them afterwards:
    - the cached agent bundle for every session, not just the default one.
      The bundles were dropped by uid+id only for the default here, so a
      deleted session's compiled graph survived the reset.
    - the in-memory run history (RUNS), which is what the "previous runs" list
      is built from. Without this the sidebar still listed every past search.
    - this account's scraper "already shown" memory, so the first hunt after a
      reset is fresh instead of an immediate page of repeats.

    NOT cleared: the dated jsonl files in logs/. Those are the server's audit
    trail, they are shared between accounts on the same day, and a "reset my
    dashboard" click is not a request to rewrite someone else's log entries.
    """
    removed = 0
    for prof in auth.list_profiles(ctx.uid):
        pid = prof.get("id")
        if pid and pid != "default" and auth.delete_profile(ctx.uid, pid):
            removed += 1
    blank = auth.blank_profile()
    blank["id"] = "default"
    auth.save_profile(ctx.uid, "default", blank)
    cache = _bundle_cache()
    for key in [k for k in cache if k[0] == ctx.uid]:
        cache.pop(key, None)
    RUNS.pop(ctx.uid, None)
    auth.set_session_profile(ctx.token, "default")
    SKILL_CACHE.clear()
    # This account's already-shown postings, for the reason documented on
    # forget_scope: kept, the next hunt has nothing fresh left to show and
    # falls back to repeats, which reads as a broken search rather than a reset.
    try:
        import mcp_server_indeed_scraper as _scraper_mod
        forgotten = _scraper_mod.forget_scope(f"u{ctx.uid}")
    except Exception:
        forgotten = 0
    _log({"event": "reset", "user": ctx.user.get("email"),
          "sessions_removed": removed, "seen_links_forgotten": forgotten})
    return {"ok": True, "removed": removed, "history_cleared": True,
            "seen_links_forgotten": forgotten}


def _ocr_payload(res: cv_ocr.OcrResult) -> dict:
    """OCR outcome in the shape the dashboard and the upload log both want."""
    return {"used": res.ok, "attempted": True, "pages": res.pages,
            "seconds": res.seconds, "confidence": res.mean_conf,
            "dropped_lines": res.dropped_lines, "truncated": res.truncated,
            "warnings": list(res.warnings), "email": res.email,
            "phone": res.phone, "detail": res.detail}


def _recover_text_by_ocr(data: bytes, filename: str, text: str) -> tuple[str, dict]:
    """Second attempt at an upload that came back with no readable text.

    A CV that was scanned, photographed or re-saved as page images has no text
    layer for pypdf to find, and that upload used to fail outright. OCR costs a
    few seconds, so it only runs when the extraction was genuinely too thin to be
    a document and the container is one OCR can read.
    """
    if not cv_ocr.needs_ocr(text) or not cv_ocr.supported(filename):
        return text, {}
    res = cv_ocr.ocr_document(data, filename)
    if not res.ok:
        return text, _ocr_payload(res)
    recovered = res.text
    if filename.lower().endswith(".pdf"):
        # A scan can still declare hyperlinks, and pypdf is the only reader
        # that sees them, so keep them ahead of the recognised text.
        try:
            from pypdf import PdfReader
            links = _pdf_link_targets(PdfReader(io.BytesIO(data)))
            if links:
                recovered = recovered + "\n" + "\n".join(links)
        except Exception:
            pass
    return recovered, _ocr_payload(res)


@app.post("/api/resume")
async def upload_resume(file: UploadFile = File(...),
                        ctx: Ctx = Depends(build_ctx)):
    """Crunch a resume/CV into skill keywords for it, suggest roles for it,
    infer the user's location/remote preference, and feed all of it to the
    agent.

    Every upload starts its OWN fresh session (exactly like the sidebar
    "+ New" session): the sidebar keeps every previous CV/session untouched
    with its own skills/roles/cities, and this new CV opens a brand-new blank
    session fed by exactly this file — none of the old cross-session merge /
    accumulate logic applies anymore.

    A CV with no text layer (a scan, or a photo of a page) is read by OCR rather
    than rejected; `ocr` in the response reports what that cost and what the
    recogniser was unsure about."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False, "error": (
            f"That file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB — "
            "please upload a trimmed CV.")}
    text = _extract_text(data, file.filename or "")
    ocr_info: dict = {}
    if cv_ocr.needs_ocr(text) and cv_ocr.supported(file.filename or ""):
        # OCR runs for seconds per page, so it goes to a worker thread; leaving
        # it inline would stall the event loop and freeze every open dashboard
        # connection for the duration.
        text, ocr_info = await run_in_threadpool(
            _recover_text_by_ocr, data, file.filename or "", text)
    if not text.strip():
        error = "Could not read any text from that file."
        if ocr_info and not ocr_info.get("used"):
            error += " " + (ocr_info.get("detail") or "OCR found nothing.")
        elif not ocr_info:
            error += (" If it is a scan or a photo, upload it as a PDF or PNG "
                      "so it can be read by OCR.")
        return {"ok": False, "error": error, "ocr": ocr_info}

    new_skills = await extract_skills(text, ctx.cfg.model, ctx.cfg.num_ctx)
    if not new_skills:
        return {"ok": False, "error": "No skills could be extracted.",
                "skills": list(ctx.cfg.skills),
                "preview": text[:300], "ocr": ocr_info}

    # Every upload = its OWN fresh session: the previous sessions keep their
    # CV/skills/roles untouched in the sidebar, and this new CV starts a blank
    # session (a clean slate fed by exactly this file — no cross-session carry-
    # over, no stale "Hyderabad" resume_data asset). None of the old merge /
    # accumulate logic applies anymore.
    # The role names go through untouched. They used to be rewritten to
    # "<role> / entry-level, remote", which then became the search query and
    # the chip label, so the user could never pick that exact role from the
    # list without it being changed behind their back.
    suggestions = suggest_roles(new_skills)
    inferred = infer_location(text)
    # Retired: the old build hard-set "remote, outside India" here, so simply
    # uploading a CV silently narrowed every later search. Nothing is set now —
    # geography is only narrowed by what the user picks.
    remote_only = False

    # Cities are also a filter, and this is the user's OWN CV, so seeding its
    # city is legitimate context. Roles are NOT seeded: the CV produces
    # suggestions the user can accept, not filters that are already applied.
    cities = ([inferred["city"]] if inferred.get("city") else [])

    # Pull the GitHub (and other profile) link out of the CV text so the LaTeX
    # resume header carries the real one instead of a hardcoded copy.
    detected_links = detect_links(text)
    github_url = detected_links.get("github") or ""

    # Build a brand-new profile for this upload (skills/cities come from
    # THIS CV alone — nothing is merged in from other sessions) and open it,
    # exactly like the sidebar "+ New" session would, then hand the CV to it.
    try:
        name = resume_generator.extract_contact(text)["name"] or "New session"
    except Exception:
        name = "New session"
    pid = auth.create_profile(ctx.uid, name, {
        "resume_text": text, "github_url": github_url,
        "skills": list(new_skills), "cities": cities,
        # EMPTY on purpose. A role chip is a live search filter, and the user
        # asked for suggestions they can accept — not filters applied for them.
        # The drawer shows the suggestions and one click adds the role.
        "roles": [],
        # Persisted so switching back to this session restores the suggestion
        # chips (renderSuggestions reads cfg.role_suggestions).
        "role_suggestions": [s["role"] for s in suggestions],
        "work_modes": list(filters.DEFAULT_WORK_MODES),
        "regions": [],
        # Deliberately EMPTY, not the detected country: a real country chip is
        # an active filter, and seeding one would make a brand-new session
        # return nothing on its first search. The detected country is returned
        # as a one-click suggestion instead (see `inferred` below).
        "countries": [],
        "remote_only": remote_only,
    })

    auth.set_session_profile(ctx.token, pid)
    _drop_bundle(ctx.uid, pid)

    _log({"event": "resume_upload", "user": ctx.user["email"],
          "file": file.filename, "chars": len(text),
          "ocr": ocr_info,
          "skills_added": len(new_skills), "skills_total": len(new_skills),
          "suggested_roles": [s["role"] for s in suggestions],
          "roles_applied": 0,
          "inferred": inferred, "cities": cities,
          "links": detected_links, "profile": pid})
    return {"ok": True, "skills": list(new_skills), "suggestions": suggestions,
            "inferred": inferred, "target_role": "",
            "cities": cities, "preview": text[:400], "chars": len(text),
            "skills_added": len(new_skills), "github": github_url,
            "links": detected_links, "profile_id": pid, "ocr": ocr_info}


@app.post("/api/resume/detect")
async def detect_resume_github(ctx: Ctx = Depends(build_ctx)):
    """Pull profile links out of the ACTIVE session's resume/CV.

    The *Detect* buttons in the Settings drawer call this: it re-runs
    detect_links() over the resume text we already hold, so a user can grab
    their GitHub / LinkedIn / portfolio link at any time — even without
    re-uploading the file.

    It reads the stored record rather than ctx.cfg: the agent bundle is cached
    per (account, profile), so a CV written after the bundle was built is not in
    it yet, and Detect would miss the session the user is looking at.

    It scans ONLY the active session. It used to fall back to the account's
    other saved CVs, on the reasoning that a link belongs to the person rather
    than the session. That is why a CV with no GitHub still ended up with one:
    the answer was copied into cfg.github_url, which outranks the CV in the
    generated resume, so an unrelated older CV's URL shipped in every document
    produced afterwards. A blank answer is the honest one - the user can paste
    a link, or open the CV that has it.
    """
    text = ""
    for rec in auth.list_profiles(ctx.uid):
        if rec.get("id") == ctx.profile_id:
            text = (rec.get("resume_text") or "").strip()
            break
    links = detect_links(text) if text else {}
    return {"ok": True,
            "github_url": links.get("github") or "",
            "linkedin": links.get("linkedin") or "",
            "portfolio": links.get("portfolio") or "",
            "scanned": ctx.profile_id if text else ""}


@app.get("/api/rag/docs")
async def rag_docs(ctx: Ctx = Depends(build_ctx)):
    """This account's document library - the corpus retrieval reads from."""
    return {"ok": True, "docs": await asyncio.to_thread(
        rag_store.list_documents, ctx.uid)}


@app.post("/api/rag/docs")
async def rag_doc_upload(file: UploadFile = File(...),
                         ctx: Ctx = Depends(build_ctx)):
    """Add a document to the library (a CV, a project write-up, a certificate).

    Text is extracted, and a scan goes through OCR, on exactly the same path as
    a resume upload - one reader for the whole app, so a photographed
    certificate behaves like a photographed CV.

    Storing a document does NOT change the open profile: the library is what
    retrieval may draw on, while the profile's CV is still what the resume is
    rendered from. Uploading a second CV here adds a source to match against
    rather than silently replacing the one being edited.
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False, "error": (
            f"That file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB — "
            "please upload a trimmed file.")}
    filename = file.filename or "document"
    text = _extract_text(data, filename)
    ocr_info: dict = {}
    if cv_ocr.needs_ocr(text) and cv_ocr.supported(filename):
        text, ocr_info = await run_in_threadpool(
            _recover_text_by_ocr, data, filename, text)
    if not text.strip():
        error = "Could not read any text from that file."
        if ocr_info and not ocr_info.get("used"):
            error += " " + (ocr_info.get("detail") or "OCR found nothing.")
        return {"ok": False, "error": error, "ocr": ocr_info}

    # The document id is derived from the name, so re-uploading the same file
    # updates it in place instead of filling the library with duplicates.
    doc_id = re.sub(r"[^a-z0-9]+", "-", filename.lower()).strip("-")[:80] \
        or "document"
    try:
        stored = await asyncio.to_thread(
            rag_store.add_document, ctx.uid, doc_id, filename, text)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    _log({"event": "rag_doc_added", "user": ctx.user["email"], "doc": doc_id,
          "chars": stored["chars"], "ocr": bool(ocr_info.get("used"))})
    return {"ok": True, "doc": stored, "ocr": ocr_info,
            "docs": await asyncio.to_thread(rag_store.list_documents, ctx.uid)}


@app.delete("/api/rag/docs/{doc_id}")
async def rag_doc_delete(doc_id: str, ctx: Ctx = Depends(build_ctx)):
    """Remove one document. Scoped to the account, so another user's id is a
    miss rather than a delete."""
    removed = await asyncio.to_thread(
        rag_store.delete_document, ctx.uid, doc_id)
    if not removed:
        return {"ok": False, "error": "No such document."}
    return {"ok": True, "docs": await asyncio.to_thread(
        rag_store.list_documents, ctx.uid)}


@app.post("/api/resume/gen")
async def generate_resume(req: ResumeGenRequest, ctx: Ctx = Depends(build_ctx)):
    """Build a job-tailored LaTeX resume for a single listing.

    Tailoring is driven by the posting text: pasted JD first, and only if the
    user pasted nothing do we scrape a snippet from the link (best-effort - most
    boards 403). That text is then used twice: to rank the skill groups, and as
    the BM25 query over the candidate's own document library, so the bullets
    that lead the resume are the ones the posting actually asks for. Retrieval
    returns stored text verbatim; nothing is generated or paraphrased.
    """
    job = {"title": req.title, "company": req.company, "location": req.location,
           "link": req.link, "source": req.source}
    try:
        pasted = (req.jd_text or "").strip()
        if pasted:
            desc, jd_origin = pasted, "pasted"
        else:
            desc = await asyncio.to_thread(resume_generator.desc_snippet, req.link)
            jd_origin = "link" if desc else "none"
        cfg = ctx.cfg
        github_url = resume_generator._normalize_url(
            req.github_url or getattr(cfg, "github_url", ""))
        # The query is the JD plus the title, so a thin JD (or none) still
        # steers selection instead of falling back to document order.
        query = f"{desc} {job['title']} {job['company']}".strip()
        hits = await asyncio.to_thread(rag_store.retrieve, ctx.uid, query)
        # With no CV on this profile, the best-matching uploaded document
        # becomes the resume's content source instead of the canonical
        # placeholder - otherwise the library is ignored for exactly the
        # accounts that built it.
        source_text = ""
        if not (cfg.resume_text or "").strip() and hits:
            source_text = await asyncio.to_thread(
                rag_store.document_text, ctx.uid, hits[0][0].doc_id)
        tex, fname = await asyncio.to_thread(
            resume_generator.build_resume, cfg, job, desc, github_url,
            rag_hits=hits, rag_source_text=source_text)
        preview = await asyncio.to_thread(resume_generator.tex_to_html, tex)
    except Exception as exc:
        _log({"event": "resume_gen_error", "user": ctx.user["email"],
              "detail": str(exc)}, kind="errors")
        return {"ok": False, "error": str(exc)}
    _log({"event": "resume_generated", "user": ctx.user["email"], "job": job,
          "desc_chars": len(desc), "jd_origin": jd_origin,
          "rag_hits": len(hits), "rag_docs": len({c.doc_id for c, _ in hits}),
          "github_url": github_url})
    return {"ok": True, "filename": fname, "tex": tex, "preview": preview,
            "desc_fetched": bool(desc), "jd_origin": jd_origin,
            "rag_hits": len(hits),
            "rag_sources": sorted({c.source_ref for c, _ in hits}),
            "github_url": github_url}


@app.get("/api/logs")
async def get_logs(ctx: Ctx = Depends(build_ctx)):
    """History for this account only.

    The on-disk JSONL is shared by everyone, so error lines are filtered to the
    ones this account wrote (records carry the account's address).
    """
    mine = {"user": ctx.user["email"]}
    errors = [e for e in _recent_errors(kind="errors", n=200) if e.get("user") == mine["user"]]
    return {"runs": _user_runs_all(ctx.uid)[:50], "errors": errors[:20]}


def _recent_errors(kind: str, n: int) -> list[dict]:
    """Tail the JSONL logs as a plain list (read-only, safe).

    Scans the last LOG_SCAN_DAYS daily files, newest first, rather than only
    today's — the README promises the History drawer survives a restart, but
    reading just today's file meant every error from yesterday vanished
    overnight. Stops as soon as it has enough records.
    """
    now = time.time()
    out: list[dict] = []
    for day in range(LOG_SCAN_DAYS):
        stamp = time.strftime("%Y%m%d", time.localtime(now - day * 86400))
        path = os.path.join(LOG_DIR, f"{kind}-{stamp}.jsonl")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        out = out + rows          # walking newest -> oldest, so append
        if len(out) >= n:
            break
    return out[-n:]


# --------------------------------------------------------------------------
# Run (SSE)
# --------------------------------------------------------------------------

async def _run_stream(message: str, run_id: str, t0: float, bundle, uid: int,
                      profile_id: str, user_email: str):
    """Stream one hunt. The bundle is passed in rather than read from
    app.state, so a run always uses the config of the account that started it
    even if they switch sessions mid-stream."""
    skills = bundle.cfg.skills
    yield _frame({"type": "status", "stage": "starting",
                  "message": "Agent starting…"})
    _log({"run": run_id, "event": "start", "user": user_email,
          "message": message,
          "skills": skills, "model": bundle.cfg.model,
          "remote_only": bundle.cfg.remote_only,
          "work_modes": bundle.cfg.work_modes, "regions": bundle.cfg.regions,
          "profile": profile_id})

    payload = {"messages": [system_message(bundle.cfg),
                            HumanMessage(content=message)]}
    summary = {"run": run_id, "message": message, "tools": [],
               "token_chars": 0, "duration_s": None,
               "status": "ok", "error": None, "jobs": 0}
    # SUMMARY_MODE=template (the default) builds the reply in code and returns
    # a complete AIMessage, which LangGraph reports through "updates" only --
    # never as a "messages" chunk. Track which message ids already streamed so
    # the same text is not sent twice when the LLM summary mode is used.
    streamed_ids: set = set()
    try:
        async for mode, data in bundle.app.astream(
            payload, stream_mode=["messages", "updates"],
            config={"recursion_limit": 60},
        ):
            if mode == "messages":
                chunk, meta = data
                if isinstance(chunk, AIMessageChunk):
                    cid = getattr(chunk, "id", None)
                    if cid:
                        streamed_ids.add(cid)
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
                            # The node's answer text. In llm mode the chunks
                            # already arrived via "messages"; in template mode
                            # this is the only place the reply exists.
                            mid = getattr(msg, "id", None)
                            if mid and mid in streamed_ids:
                                continue
                            text = _text_content(getattr(msg, "content", ""))
                            if text:
                                summary["token_chars"] += len(text)
                                yield _frame({"type": "token", "text": text})
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

        if not summary["token_chars"]:
            # Never leave the user staring at an empty bubble: explain what the
            # run actually did instead of failing silently.
            note = ("I could not build a summary for that search.\n\n"
                    f"Job boards searched: {', '.join(summary['tools']) or 'none'}\n"
                    f"Listings found: {summary['jobs']}\n\n"
                    "Check the session filters (roles, keywords, countries) in "
                    "Settings, or paste your resume in a fresh session and try again.")
            summary["token_chars"] += len(note)
            yield _frame({"type": "token", "text": note})

        summary["duration_s"] = round(time.time() - t0, 1)
        yield _frame({"type": "status", "stage": "done"})
        _log({"run": run_id, "event": "done", **summary})
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = str(exc)[:500]
        summary["duration_s"] = round(time.time() - t0, 1)
        detail = traceback.format_exc()
        _log({"run": run_id, "event": "error", "user": user_email, **summary},
             kind="errors")
        _log({"run": run_id, "event": "error", "exception": detail})
        yield _frame({"type": "error", "message": str(exc), "run": run_id})
    finally:
        _remember_run(summary, uid)


@app.post("/api/run")
async def run(req: RunRequest, ctx: Ctx = Depends(build_ctx)):
    run_id = uuid.uuid4().hex[:8]
    return StreamingResponse(
        _run_stream(req.message, run_id, time.time(), ctx.bundle, ctx.uid,
                    ctx.profile_id, ctx.user["email"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def already_serving(host: str, port: int) -> str | None:
    """Explain an occupied port, or return None when it is free to bind.

    Binding an address that is taken raises WinError 10048 on Windows and
    "address already in use" on Linux, and neither message says which program
    holds it nor that the real answer is usually "open the window that is
    already up". Checking first turns a stack trace into one line of advice.
    """
    import socket
    import urllib.request

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Deliberately NOT setting SO_REUSEADDR: on Windows that flag lets a
        # bind succeed against a live port, which would hide the very conflict
        # this exists to detect.
        try:
            sock.bind((host, port))
        except OSError:
            pass
        else:
            return None
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=2) as resp:
            body = resp.read(200_000).decode("utf-8", "replace")
    except Exception:
        return (f"Port {port} is held by another program, so JobScope did not "
                f"start. Find it with:\n"
                f"  Get-NetTCPConnection -LocalPort {port} "
                f"-State Listen | Select-Object OwningProcess\n"
                f"or start JobScope on a different port: --port {port + 1}")
    if "JobScope" in body:
        # ASCII only: this prints to a Windows console, whose default code page
        # (cp1252) turns an em-dash into a replacement glyph.
        return (f"JobScope is already running at http://{host}:{port}/ - "
                f"nothing to start.\nOpen that address, or stop the running "
                f"instance first:\n"
                f"  Stop-Process -Id (Get-NetTCPConnection -LocalPort {port} "
                f"-State Listen).OwningProcess\n"
                f"Code edits to dashboard.py / filters.py need that restart; "
                f"static/index.html does not (it is re-read per request).")
    return (f"Port {port} is serving something that is not JobScope, so it "
            f"did not start. Try --port {port + 1}.")


if __name__ == "__main__":
    import uvicorn

    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1]) if "--port" in args else 8000
    # HOST defaults to loopback. Set HOST=0.0.0.0 to serve other people on the
    # LAN - only do that behind a real TLS terminator, because the session
    # cookie is marked Secure only when the request itself is https.
    host = os.environ.get("HOST", "127.0.0.1")
    # reload=True re-imports the module on every edit, so a change to
    # dashboard.py / filters.py is picked up without hunting for the running
    # process. Off unless asked for: it restarts the app mid-run, which would
    # abort an in-flight search.
    reload = "--reload" in args

    if not reload:                      # the reloader binds the port itself
        busy = already_serving(host, port)
        if busy:
            print(busy)
            raise SystemExit(0 if "already running" in busy else 1)

    uvicorn.run("dashboard:app", host=host, port=port, reload=reload)