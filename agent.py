"""
agent.py — shared job-hunting agent.

Single source of truth for the LangGraph wiring that was previously
copy-pasted across step4-step7. Every consumer calls create_agent():

  - step7_scraper_with_fallback.py  (CLI demo)
  - ui.py                            (Chainlit web UI)
  - a scheduled digest job           (future)

What's improved vs the old per-step graphs:
  - THE FALLBACK IS NOW A GRAPH EDGE, not a prompt instruction.
    A small deterministic router inspects scrape_indeed_jobs' output for
    blocked/error markers; if it finds one, it routes to a "fallback" node
    that pulls the same data from Gmail job-alert emails instead. No
    reliance on the LLM reading prose instructions correctly.
- Filter rules live in code (SENIORITY_FILTER / EXPERIENCE_FILTER), not
     in the prompt, and are applied to fallback data before the LLM sees it.
   - Fresh job hunts START in PARALLEL: one deterministic step fans out to
     the board scraper AND both Gmail alert senders at once, skipping the
     LLM tool-decision round-trip (the slowest inference on local CPUs) and
     the sequential boards-then-fallback wait. A "prep" node then picks the
     winning source (boards if they returned jobs, else Gmail) in code.
   - Config/env is centralized in load_config() so the role, model, context
     window, and search parameters can't drift apart across files.
"""

import os
import re
import sys
import json
import asyncio
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, HumanMessage, RemoveMessage,
                                     SystemMessage, ToolMessage)
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import filters
from filters import (FALLBACK_SENDERS, JOBS_MARKER, boards_location,
                     detect_board, keep_job)

load_dotenv()

# If search_job_boards' output contains any of these, treat it as blocked
# and fall back to Gmail. IMPORTANT: "blocked" alone is NOT a marker — a
# multi-board search can fail on some boards while still returning jobs.
# Only an all-failed run carries "All boards were blocked".
BLOCKED_MARKERS = (
    "all boards were blocked",
    "captcha",
    "http 403",
    "fall back",
    "no job links",
    "couldn't extract",
    "request failed",
    "js-rendered",
)

# Spelling slips that show up constantly in uploaded CVs and in the role
# suggestions built from them. Left uncorrected they reach the board as a search
# term that matches nothing ("Java Devloper" finds no Java roles). Only
# unambiguous, single-word slips are listed; genuine title words are never
# rewritten, and a word missing from this map is always passed through as-is.
TYPO_FIXES = {
    "devloper": "developer",
    "devlopper": "developer",
    "develuper": "developer",
    "phyton": "python",
    "pythn": "python",
    "pyhton": "python",
    "javasript": "javascript",
    "javascrpit": "javascript",
    "reaact": "react",
    "reacct": "react",
    "postgre": "postgres",
    "datasceince": "data science",
    "machien": "machine",
    "analayst": "analyst",
    "anaylst": "analyst",
    "enginerr": "engineer",
    "managment": "management",
    "administraion": "administration",
    "fullstack": "full stack",
}

# Separators that can divide a role string into a title plus trailing notes.
# A bare hyphen is excluded: "Full-Stack" and "Work-From-Home" are single words,
# and a spaced " - " is handled by the same pattern.
ROLE_SEP_RE = re.compile(r"\s*[/,;|]\s*|\s+[-(]\s*|\s*-\s+")

# "(backend)", "(Remote - US only)", "[contract]" — a note to us, not to a board.
ROLE_ASIDE_RE = re.compile(r"[\(\[\{][^()\[\]{}]*[\)\]\}]")

# Words that describe a constraint on the job rather than the job itself. A
# trailing segment built only from these is a note ("/ entry-level, remote"),
# not a job title, and is dropped before the query reaches a board.
ROLE_QUALIFIER_WORDS = {
    "remote", "onsite", "on-site", "hybrid", "wfh", "anywhere", "worldwide",
    "full-time", "fulltime", "part-time", "contract", "contractual",
    "temporary", "temp", "permanent", "freelance", "intern", "internship",
    "trainee", "apprentice", "graduate", "graduates", "fresher", "freshers",
    "entry", "entry-level", "junior", "senior", "lead", "principal", "staff",
    "head", "chief", "director", "manager", "executive", "vp",
    "immediate", "urgent", "asap", "hiring", "openings", "multiple",
    "experience", "experienced", "years", "yrs", "y.o.", "no", "zero",
    "plus", "nice", "have", "has", "with", "good", "exposure", "knowledge",
    "and", "or", "of", "in", "at", "for", "to", "level", "based",
}

# Messages that read like a job hunt trigger the deterministic parallel
# collect below — we skip the LLM's "which tool should I call?" round-trip
# (on a local CPU that single inference is the slowest part of a run).
HUNT_HINT = re.compile(
    r"\b(hunt|jobs?|find|search|remote|position|vacanc|role|skills?|keyword|"
    r"listing|postings?|apply)\b", re.IGNORECASE)


def is_hunt_request(text: str) -> bool:
    """True when a message should trigger the deterministic job-source fan-out.

    A bare board name counts. Typing just "linkedin" names the source the user
    wants, and it matched no verb in HUNT_HINT, so the LLM decided which tool to
    call instead - llama3.1 reached for the Gmail alert tool and the user got
    no job cards at all.
    """
    return bool(HUNT_HINT.search(text or "")) or detect_board(text) is not None

SYSTEM_PROMPT = (
    "You are a job-hunting assistant, matching listings to the target role "
    "the user specifies. You have tools to search several job sources.\n"
    "Geography is decided by the user's own filters only. Never drop or add a "
    "listing based on where the user lives — apply exactly the countries and "
    "work types configured for the session, and if none are set, search "
    "everywhere.\n"
    "Always answer with a clean markdown list: source, job title, "
    "company/location, and application link. Every listed job MUST include "
    "its full https application link exactly as given.\n"
    "Strictly discard any listing that requires 5+ years of experience or "
    "carries a senior/lead/manager/principal/staff title - those do not "
    "match an entry-level target.\n"
    "When judging a listing, check its required skills against the user's "
    "skill keywords. Prefer roles whose requirements overlap strongly with "
    "the user's skills, and say which of the user's skills map to each role."
)


@dataclass
class AgentConfig:
    model: str = "llama3.1"
    num_ctx: int = 8192
    target_role: str = "Junior Data Analyst / entry-level, 0-2 years experience"
    days_back: int = 60
    max_results: int = 10
    indeed_query: str = "Junior Data Analyst"
    # All the roles the user is open to (multi-select in Settings). target_role
    # stays the PRIMARY one that focuses the board query; the full list is used
    # for matching and context so every chosen role gets coverage.
    target_roles: list[str] = field(default_factory=list)
    indeed_location: str = "Remote"
    indeed_domain: str = "www.indeed.com"
    # Skill keywords (typically extracted from the user's resume/CV). When
    # present they become the SEARCH terms and the match criteria — the hunt
    # is driven by what the user can actually do, not just the role title.
    skills: list[str] = field(default_factory=list)
    resume_text: str = ""
    # Remote / geography constraints. Legacy: the old build shipped with this
    # hard-ON, which silently dropped every India-tied posting. It is now OFF
    # everywhere and the settings UI no longer exposes it, so a search only
    # narrows by geography when the user narrows it themselves.
    remote_only: bool = False
    # Work arrangement chips the user picked (any of remote / wfh / hybrid /
    # onsite). Empty = no arrangement filtering.
    work_modes: list[str] = field(default_factory=list)
    # Retired region chips. Kept on the dataclass so old saved profiles still
    # load, but nothing in the UI writes it and an empty list means no
    # geography narrowing — the boards' own worldwide behaviour.
    regions: list[str] = field(default_factory=list)
    # Country picker (see filters.COUNTRIES). The only geography filter the
    # user drives: ["Germany"] keeps only listings in Germany. [] = every
    # country allowed.
    countries: list[str] = field(default_factory=list)
    exclude_locations: list[str] = field(default_factory=list)
    # Cities the user is open to working in/from (chosen in Settings; seeded
    # from the CV city on first upload). They set the boards' location when an
    # on-site/hybrid work type is chosen; with no work type set they only
    # contextualize the hunt + the resume's contact block.
    preferred_cities: list[str] = field(default_factory=list)
    # Role titles suggested from the resume's skills (dashboard displays them).
    role_suggestions: list[str] = field(default_factory=list)
    # GitHub link used in the LaTeX resume header. Auto-filled from the
    # uploaded resume when one is found; the user can also set it per session
    # (Settings drawer / resume modal). Empty -> the generator's own default.
    github_url: str = ""
    # "template" (default) builds the final summary in code — instant. "llm"
    # runs the local Ollama model instead (slower but more conversational).
    summary_mode: str = "template"
    # Identifies the account doing the hunt, passed to the board scraper so
    # its "already shown" memory is per person. Left empty the memory is
    # shared, which is what a bare CLI run gets. Never surfaced to the model.
    seen_scope: str = ""


def load_config() -> AgentConfig:
    """Build AgentConfig from .env, with sane defaults."""
    return AgentConfig(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
        target_role=os.environ.get(
            "TARGET_ROLE",
            "Junior Data Analyst / entry-level, 0-2 years experience",
        ),
        days_back=int(os.environ.get("JOB_DAYS_BACK", "60")),
        max_results=int(os.environ.get("JOB_MAX_RESULTS", "10")),
        summary_mode=os.environ.get("AGENT_SUMMARY_MODE", "template"),
    )


class AgentState(MessagesState):
    # extra bookkeeping key (optional): set once the fallback has run, so the
    # router can't send us into an infinite scrape-blocked -> fallback loop.
    fallback_used: bool


def _text_content(content) -> str:
    """Normalize an LLM content value to plain text.

    MCP tools return ToolMessage content as a *list of blocks*
    ([{'type': 'text', 'text': '...'}, ...]), not a plain string. Any code
    that inspects or previews tool output must go through this, or it hits
    AttributeError ('list' object has no attribute 'lower'/'replace'...).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("name") or ""
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _scrape_blocked(state: AgentState) -> bool:
    """Look at the most recent tool output: was it the scraper, and blocked?"""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            continue
        name = (getattr(msg, "name", "") or "").lower()
        if "scrape" not in name and "job_boards" not in name:
            return False  # the last tool that ran wasn't the board scraper
        content = _text_content(msg.content).lower()
        return any(marker in content for marker in BLOCKED_MARKERS)
    return False


def _filter_listings(text: str, cfg: AgentConfig | None = None) -> str:
    """Drop lines that look senior, or India-tied (remote-only constraint).
    Code rules, not prose. The rules themselves live in filters.py so the
    scraper and this fallback can't drift apart."""
    out = []
    remote_only = bool(cfg and cfg.remote_only)
    for line in text.splitlines():
        if filters.is_senior_or_experienced(line):
            continue
        if remote_only and not filters.outside_india(line):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _link_block(jobs: list[dict]) -> str:
    """One compact line per listing (title — company — location — link). This is
    what the summary LLM reads instead of the raw multi-KB scraper/email dumps —
    shrinking the prompt is the single biggest speed lever on local CPUs."""
    return "\n".join(
        f"- {j.get('title', 'Job')} — {j.get('company', '') or j.get('source', '')}"
        f" — {j.get('location', '')} — {j['link']}"
        for j in jobs)


def _summary_payload(jobs: list[dict], note: str,
                     blocked_reason: str = "") -> str:
    """Self-contained, compact payload handed to the summary LLM (and to the
    dashboard's flashcard deck via the ###JOBS_JSON### block).

    blocked_reason is carried through so the deck can say WHY a restricted
    search came back empty. Without it the UI only knows the list is empty and
    cannot tell the user that the board they named refused to answer.
    """
    content = note + "\n\n" + (_link_block(jobs) if jobs else "(no matching listings found)")
    content += "\n\n" + JOBS_MARKER + "\n" + json.dumps(
        {"sources": {"collected": len(jobs)}, "total": len(jobs),
         "jobs": jobs, "blocked_reasons": [blocked_reason] if blocked_reason else []},
        ensure_ascii=False)
    return content


def _drop_bulky(messages) -> list:
    """RemoveMessage entries that delete just the bulky intermediate messages —
    raw ToolMessages and the empty placeholder collect call — while KEEPING the
    system prompt and the user's original ask. The summary LLM then only pays
    attention over a small, self-contained context (system + user + payload)."""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            if m.id:
                out.append(RemoveMessage(id=m.id))
        elif isinstance(m, AIMessage) and not getattr(m, "content", None):
            if m.id:
                out.append(RemoveMessage(id=m.id))
    return out


def _template_summary(content: str) -> str:
    """Deterministic, instant summary built in code (SUMMARY_MODE=template, the
    default). Avoids a 2-4 minute local-LLM inference on CPU: every job title,
    company, location and Apply link is already in the payload's JOBS_JSON."""
    jobs = []
    marker = content.find(JOBS_MARKER)
    if marker >= 0:
        try:
            blob = json.loads(content[marker + len(JOBS_MARKER):].strip())
            jobs = blob.get("jobs", []) or []
        except (json.JSONDecodeError, ValueError):
            jobs = []
    if not jobs:
        return content.split(JOBS_MARKER, 1)[0].strip()
    lines = ["Here are the latest matching jobs:", ""]
    for j in jobs:
        title = j.get("title") or "Job"
        company = j.get("company") or ""
        location = j.get("location") or ""
        link = j.get("link") or ""
        row = f"- **{title}**"
        if company:
            row += f" at {company}"
        if location:
            row += f" ({location})"
        if link:
            row += f" — apply: {link}"
        lines.append(row)
    lines += ["", f"{len(jobs)} matching role(s) listed above; every Apply link "
                  f"is included. Source(s): {sorted({j.get('source') or 'web' for j in jobs}) or 'dashboard'}."]
    return "\n".join(lines)


def _keep_job(job: dict, cfg: AgentConfig) -> bool:
    """Structured-listing filter, applied to EVERY source before the deck is
    built. Delegates to filters.keep_job so boards and Gmail are held to the
    same bar (board results used to skip this entirely).

    Empty work_modes/countries are passed as None ("don't filter")
    rather than [], which filters.py treats as "nothing selected". Regions are
    never forwarded — the region filter was retired."""
    if cfg is None:
        return keep_job(job)
    return keep_job(job,
                    remote_only=bool(cfg.remote_only),
                    work_modes=cfg.work_modes or None,
                    countries=cfg.countries or None)


def _extract_struct_from(res_text: str) -> list[dict]:
    """Pull the ###JOBS_JSON### block out of an MCP tool's text output."""
    if JOBS_MARKER not in res_text:
        return []
    try:
        blob = json.loads(res_text.split(JOBS_MARKER, 1)[1].strip())
        return blob.get("jobs", []) or []
    except (json.JSONDecodeError, ValueError):
        return []


def _iter_payloads(text: str):
    """Yield each parsed ###JOBS_JSON### payload found in a tool's output."""
    for chunk in text.split(JOBS_MARKER)[1:]:
        try:
            yield json.loads(chunk.split("###", 1)[0].strip())
        except (json.JSONDecodeError, ValueError):
            continue


def _blocked_marker(text: str) -> str:
    """Why a board scrape came back empty, or "" when it didn't look blocked.

    An empty result is not the same thing as a blocked one: a board can answer
    with zero cards for the query, time out, or get walled. Only the second and
    third are worth telling the user about, because only those are worth
    retrying differently.

    The scraper already reports this authoritatively in its
    ###JOBS_JSON### payload ("blocked": [...], "blocked_reasons": {...}), so
    read that first. The prose markers are only a fallback for a tool that
    raised instead of returning a payload - matching them against real output
    misfires, because a healthy "0 listings" scrape says "nothing matched this
    query" and used to read as a block.
    """
    for payload in _iter_payloads(text):
        blocked = payload.get("blocked") or []
        if blocked:
            reasons = payload.get("blocked_reasons") or {}
            first = reasons.get(blocked[0]) if isinstance(reasons, dict) else None
            return str(first or ", ".join(map(str, blocked)))
    low = text.lower()
    for marker in BLOCKED_MARKERS:
        if marker in low:
            return marker
    return ""


def _silent_senders(texts: list[str]) -> list[str]:
    """Gmail alert senders that had nothing at all in the window.

    search_job_emails says "No emails found from '<sender>' ..." when the mailbox
    has no alert from that sender, which is the single most useful fact when the
    fallback comes back empty: it means the user's alerts aren't arriving (or the
    sender is wrong), not that the listings were filtered away.
    """
    out = []
    for text in texts:
        for addr in re.findall(r"No emails found from '([^']+)'", text):
            if addr not in out:
                out.append(addr)
    return out


def _fallback_note(marker: str, silent: list[str], gmail_total: int,
                   gmail_kept: int) -> tuple[str, str]:
    """The note for "no board listings, possibly some from Gmail".

    Returns (note, blocked_reason). The rule this exists to enforce: never
    announce fallback data that isn't there. The old wording said "instead here
    is fallback data from Gmail job-alert emails" and then printed "(no matching
    listings found)" on the very next line, which reads as a contradiction and
    leaves the user with no idea what to change.
    """
    if gmail_kept:
        lead = ("Job boards were blocked, so " if marker else
                "The job boards came back empty, so ")
        return (
            lead + "here is fallback data from Gmail job-alert emails, "
            "already filtered for seniority and India-tied roles. Summarize "
            "the remaining listings for the target role and keep every "
            "https link exactly as given.", marker)

    why = []
    if marker:
        why.append(f"the boards were blocked ({marker})")
    if silent:
        why.append("no job-alert email arrived from " + ", ".join(silent))
    elif gmail_total:
        why.append(f"the {gmail_total} Gmail listing(s) were all filtered out "
                   "as senior or India-tied")
    if not why:
        why.append("neither the boards nor the Gmail job alerts returned "
                   "anything")
    return (
        "Nothing came back for this search: " + "; ".join(why) + ". Say "
        "plainly that there are no matching listings right now. Do not invent "
        "listings or links. Suggest waiting for the next job-alert email, "
        "widening the seniority or location filters, or searching all boards "
        "again.", marker)


def _fallback_node_note(silent: list[str], gmail_total: int) -> str:
    """Note for the legacy blocked->Gmail-fallback path when it is also empty."""
    if silent:
        why = "no job-alert email arrived from " + ", ".join(silent)
    elif gmail_total:
        why = (f"the {gmail_total} listing(s) the alerts contained were all "
               "filtered out as senior or India-tied")
    else:
        why = "the job-alert emails returned nothing usable"
    return ("The job boards were blocked and the Gmail fallback is empty too: "
            f"{why}. Say plainly that there are no matching listings right now. "
            "Do not invent listings or links. Suggest waiting for the next "
            "job-alert email or retrying the boards later.")


def _find_tool(tools, name_fragment: str):
    """Return the first discovered MCP tool whose name contains the given
    fragment (e.g. 'search_job_emails'), else raise a clear error."""
    for t in tools:
        if name_fragment in t.name:
            return t
    raise RuntimeError(f"tool containing {name_fragment!r} not found in MCP tools")


def system_message(cfg: AgentConfig) -> SystemMessage:
    extra = ""
    if cfg.skills:
        extra += (
            "\nThe user's skill keywords (from their resume): "
            + ", ".join(cfg.skills)
            + ". Search with these terms where practical and use them as the "
            "match criteria."
        )
    modes = filters.normalize_work_modes(cfg.work_modes)
    if modes:
        extra += ("\nWork arrangement wanted: "
                  + ", ".join(filters.WORK_MODES[m] for m in modes)
                  + ". Only offer roles matching one of these — the user has "
                    "explicitly filtered the rest out.")
    countries = filters.normalize_countries(cfg.countries)
    if countries:
        extra += ("\nCountry wanted: " + ", ".join(countries)
                  + ". The country filter is narrower than the region filter — "
                    "offer only roles actually located in one of these countries.")
    roles = [r for r in (cfg.target_roles or [cfg.target_role]) if r]
    if roles:
        wanted = f"\nTarget roles: {', '.join(roles)}"
    else:
        # No role chosen yet. Say so instead of sending an empty list, so the
        # agent asks what the user is after rather than inventing one.
        wanted = ("\nTarget roles: none set yet. If the user's message does not "
                  "say what role they want, ask them instead of searching.")
    return SystemMessage(content=f"{SYSTEM_PROMPT}{extra}{wanted}")


def board_tool_args(cfg: AgentConfig, human_text: str = "") -> dict:
    """Arguments for the search_job_boards tool call.

    Pure and side-effect free so the mapping from config -> tool arguments is
    testable without Ollama or a live scrape.

    work_modes/regions are omitted entirely when unset, rather than sent as
    empty lists: the scraper treats a missing filter as "don't filter", and
    leaving them out keeps the call byte-identical to the pre-feature agent.
    """
    args = {"query": search_query_from(cfg),
            "location": boards_location(cfg.work_modes, cfg.preferred_cities,
                                        cfg.remote_only, cfg.countries),
            "remote_only": cfg.remote_only}
    if cfg.work_modes:
        args["work_modes"] = list(cfg.work_modes)
    if cfg.countries:
        args["countries"] = list(cfg.countries)
    # The scraper's "already shown" memory is per account, so one person's
    # hunt does not turn a listing into a repeat for everybody else. Sent
    # only when known: the tool defaults to a shared bucket, which is the
    # right answer for a bare CLI run with no account behind it.
    if cfg.seen_scope:
        args["seen_scope"] = cfg.seen_scope
    # cfg.regions is intentionally NOT forwarded: the region filter was retired,
    # and silently honouring a stale one saved in an older profile would keep
    # narrowing a search the user can no longer see or edit.
    requested = detect_board(human_text)
    if requested:
        args["board"] = requested
    return args


def _is_role_qualifier(segment: str) -> bool:
    """True when a segment is only a constraint/level note, not a job title.

    Decided word by word so a real title is never discarded:
    "Senior Software Engineer" is a title even though "senior" is a qualifier
    word, because it has content words left over.
    """
    words = [w.strip(".,;:*").lower() for w in segment.split()]
    # Bare numbers carry no title meaning: "2+ years experience" is a
    # requirement, not a job type, so "2+" must not make the segment look
    # like a real title.
    words = [w for w in words if w and not any(ch.isdigit() for ch in w)]
    if not words:
        return True
    return all(w in ROLE_QUALIFIER_WORDS for w in words)


def _board_role_phrase(role: str) -> str:
    """Reduce a suggested/annotated role down to something a board can match.

    The role fields are filled in by the CV extractor and the suggestion chips,
    so they arrive as a title with notes stuck on the end rather than as a job
    title:
      "Python Developer / entry-level, remote"      -> "Python Developer"
      "Full Stack Developer, Intern"                 -> "Full Stack Developer"
      "Senior Software Engineer (backend)"           -> "Senior Software Engineer"
      "Machine Learning Engineer - Remote"           -> "Machine Learning Engineer"

    Boards AND every term in the query, so a trailing qualifier left in place
    either empties the result set or forces one exact spelling. Anything in
    brackets is a note to us, and a trailing segment made only of qualifier
    words is a constraint, not a title — both are dropped from the right.
    Separators *inside* a title are kept, because "DevOps/Cloud Engineer" and
    "C# / .NET Developer" are alternate spellings of one role, not a title
    followed by a note.

    Finally a few common CV typo/OCR slips are corrected so the query still
    reaches real postings. Words absent from TYPO_FIXES are passed through
    untouched: guessing at unfamiliar titles would lose more than it gains.
    """
    role = (role or "").strip()
    if not role:
        return ""

    # Bracketed asides are always notes, never part of the searchable title.
    without_asides = ROLE_ASIDE_RE.sub(" ", role)
    segments = [s.strip() for s in ROLE_SEP_RE.split(without_asides)]
    segments = [s for s in segments if s]

    kept = list(segments)
    while len(kept) > 1 and _is_role_qualifier(kept[-1]):
        kept.pop()

    phrase = " ".join(kept).strip()
    if not phrase:
        # Every segment looked like a qualifier ("Remote, Hybrid"). Better to
        # search the original text than to return an empty query.
        phrase = without_asides.strip() or role

    fixed, taken = [], set()
    for w in phrase.split():
        corrected = TYPO_FIXES.get(w.lower())
        if corrected is not None:
            if w.isupper():
                corrected = corrected.upper()      # JAVA DEVLOPER -> JAVA DEVELOPER
            elif w[:1].isupper():
                # Keep the user's capitalisation: "Java Devloper" should read
                # "Java Developer", not "Java developer".
                corrected = corrected[:1].upper() + corrected[1:]
        else:
            corrected = w
        # "Python Phyton Developer" corrects to a doubled "Python Python", and
        # a repeated term narrows a board search for no reason.
        key = corrected.lower()
        if key in taken:
            continue
        taken.add(key)
        fixed.append(corrected)
    return " ".join(fixed).strip()


def search_query_from(cfg: AgentConfig) -> str:
    """Keyword-driven search query: the PRIMARY role plus at most TWO skills.

    The whole skills list is used for matching listings, but stuffing five
    skills into the board query ("python sql pandas excel power bi Junior Data
    Analyst") returns worse results than the role alone — boards AND the terms
    loosely, and then rank by any single matching word. So we keep the role and
    add only the two most *distinctive* skills: multi-word ones first ("power
    bi", "machine learning"), since a one-word skill like "excel" narrows a
    search far less than it costs.

    The role itself is often an annotated phrase rather than a job title — the
    UI suggests "Python Developer / entry-level, remote", and the CV seeds
    "Java Devloper". Handed to a board verbatim that whole string becomes one
    search term, so nothing matches: no posting is titled
    "Python Developer / entry-level, remote". The qualifier is a note to us, not
    to the board, so it is split off here and only the actual title part is
    searched. The full phrase is still used for the agent's own matching.
    """
    raw_role = (cfg.target_roles[0] if cfg.target_roles else cfg.target_role).strip()
    role = _board_role_phrase(raw_role)
    terms = [role] if role else []
    seen = raw_role.lower()

    candidates = []
    for raw in (cfg.skills or [])[:10]:
        s = (raw or "").strip()
        low = s.lower()
        if len(s) < 3 or low in seen or any(low in p or p in low for p in candidates):
            continue
        candidates.append(low)

    candidates.sort(key=lambda s: (-len(s.split()), -len(s)))
    for s in candidates[:2]:
        if all(s not in t.lower() and t.lower() not in s for t in terms):
            terms.append(s)
    # Never hand the scraper an empty query, even for a half-configured cfg.
    # A session with no roles at all is a real state (a brand-new account has
    # none until the user picks one), so fall back to a neutral query rather
    # than the baked-in "Junior Data Analyst" example role.
    return " ".join(terms).strip() or cfg.target_role or "software developer"


def default_task_messages(cfg: AgentConfig, human_text: str = ""):
    query = search_query_from(cfg)
    skill_hint = (
        f" Your skill keywords: {', '.join(cfg.skills)}." if cfg.skills else ""
    )
    remote = "remote_only=True (remote, outside India)" if cfg.remote_only else "remote_only=False"
    geo = [c for c in (filters.normalize_countries(cfg.countries) or []) if c]
    # A named board has to reach the model too, not just the deterministic
    # fast path. Without this, "only linkedin" was silently dropped whenever
    # the LLM chose the tool itself, and the user got every board's results
    # despite asking for one.
    only = detect_board(human_text)
    board_hint = (
        f", board={only!r} — search THAT board ONLY. Do not call the tool "
        f"again for another source, and do not merge in listings from "
        f"elsewhere: the user asked for this one platform."
        if only else
        " (no board was named, so every board is in play)"
    )
    user = (
        f"Find jobs matching: '{cfg.target_role}'.{skill_hint}\n"
        f"Start by calling search_job_boards with query={query!r}, "
        f"location={boards_location(cfg.work_modes, cfg.preferred_cities, cfg.remote_only, cfg.countries)!r}, "
        f"{remote}"
        + (f", countries={geo}" if geo else " (no country filter — search worldwide)")
        + board_hint
        + ".\n"
        "Then give me a clean markdown list of the matches: source, job "
        "title, company/location, application link — and for each, note "
        "which of my skills the role needs."
    )
    return [system_message(cfg), ("user", user)]


class MCPRuntime:
    """A discovered MCP client + its tools.

    Building this costs ~3.4s: get_tools() launches all three stdio servers,
    completes a ListTools handshake, then tears them back down. The tool set
    never changes at runtime, so long-lived callers (the dashboard) build the
    runtime ONCE and pass it to every create_agent() call instead of paying
    that cost again on each settings save or profile switch.
    """
    __slots__ = ("client", "tools")

    def __init__(self, client, tools):
        self.client = client
        self.tools = tools


async def create_runtime() -> MCPRuntime:
    """Launch the MCP servers once and discover their tools."""
    # NOTE: stdio transport does NOT inherit os.environ into the subprocess.
    # Secrets must be passed explicitly per-server via "env", as in step6.
    # The "command" MUST be sys.executable: a bare "python" resolves to
    # whatever's first on PATH (may be a different interpreter with no deps).
    python_exe = sys.executable
    client = MultiServerMCPClient({
        "github": {
            "command": python_exe,
            "args": ["mcp_server_github.py"],
            "transport": "stdio",
            "env": {
                "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
                "GITHUB_USERNAME": os.environ.get("GITHUB_USERNAME", ""),
            },
        },
        "gmail": {
            "command": python_exe,
            "args": ["mcp_server_gmail.py"],
            "transport": "stdio",
            "env": {
                "GMAIL_ADDRESS": os.environ.get("GMAIL_ADDRESS", ""),
                "GMAIL_APP_PASSWORD": os.environ.get("GMAIL_APP_PASSWORD", ""),
            },
        },
        "indeed_scraper": {
            "command": python_exe,
            "args": ["mcp_server_indeed_scraper.py"],
            "transport": "stdio",
        },
    })

    tools = await client.get_tools()
    return MCPRuntime(client, tools)


def _last_human_text(state) -> str:
    """The user's own words for this hunt, i.e. the newest HumanMessage.

    The graph is seeded with templated messages that already mention the
    target role and skills, so this is the only place a "only linkedin" style
    instruction is still recognisable as the user's own words.
    """
    return next(
        (_text_content(getattr(m, "content", "")) for m in reversed(state["messages"])
         if isinstance(m, HumanMessage)),
        "",
    )


def requested_board(state) -> str:
    """The single board this hunt is restricted to, or "" for "any board".

    "Only linkedin" means only LinkedIn. That has to hold for every source,
    not just the scraper: the Gmail fallback reads job-alert emails from
    several boards at once, so falling back to it after the named board came
    back empty answered a different question than the one that was asked.

    detect_board returns None when nothing matched; normalised to "" so callers
    only ever have to test truthiness.
    """
    return detect_board(_last_human_text(state)) or ""


# "search the gmail alerts" / "just email me the matches" / "check my inbox".
# Deliberately narrow: a bare "remote jobs" must not be read as a request for
# Gmail, and "job alert" has to win over the word "job" alone.
GMAIL_SOURCE = re.compile(
    r"\b(gmail|gmial|"
    r"e-?mail alerts?|job alerts?|my inbox|inbox jobs?|alerts? from|"
    r"e-?mail me|send me (?:the )?(?:e-?mails?|alerts?|links?|matches?)|"
    r"check (?:my )?(?:mail|inbox)|(?:from|in) my (?:mail|inbox|e-?mails?))\b",
    re.IGNORECASE,
)


def requested_source(state) -> str:
    """Which source this run should search: "gmail", "boards" or "" for either.

    One source per run, chosen by the user rather than fanned out. The previous
    behaviour fired the board scraper AND both Gmail alert senders on every
    hunt, so the user saw three searches they never asked for and had no way to
    say "just the boards". Worse, the Gmail results were computed even when the
    boards answered, then thrown away by the prep node - paid for and never
    shown.

    Returns "" when the message names neither source, which the caller treats as
    "boards": a job hunt is a board search unless Gmail is explicitly requested.
    """
    text = _last_human_text(state)
    if GMAIL_SOURCE.search(text or ""):
        return "gmail"
    return "boards"


async def create_agent(cfg: AgentConfig | None = None,
                       runtime: MCPRuntime | None = None) -> "AgentBundle":
    """Compile the LangGraph agent for `cfg`.

    Pass an existing `runtime` (see MCPRuntime) to reuse already-discovered MCP
    tools — this only rebuilds the LLM bindings and the graph, which is what
    actually depends on the config. Omit it (first call) to build one.
    """
    cfg = cfg or load_config()
    runtime = runtime or await create_runtime()
    tools = runtime.tools

    llm = ChatOllama(model=cfg.model, num_ctx=cfg.num_ctx).bind_tools(tools)
    # The final summary never calls tools — binding tool schemas slows every
    # generation (long function definitions on a small context), so summarize
    # with a plain, unbound instance of the same model.
    llm_summary = ChatOllama(model=cfg.model, num_ctx=min(cfg.num_ctx, 4096))

    def _collect_message(state: AgentState) -> AIMessage:
        """Deterministic first step of a hunt: emit ONE AIMessage whose
        ONE tool call, for ONE source. Zero LLM inference, so results arrive as
        fast as the single network call instead of an LLM deciding first.

        This used to emit the board scraper AND both Gmail alert senders as
        parallel tool_calls on every hunt. That produced the "two job search
        emails" the user saw, searched sources nobody asked for, and computed
        the Gmail results even when the boards answered - the prep node then
        discarded them, so the slowest of the three calls set the wall-clock for
        output that was thrown away. One source per run, chosen by
        requested_source(), and if the user named a board ("only linkedin") that
        board is passed through so the deck comes from it alone."""
        calls = []
        source = requested_source(state)
        if source == "gmail":
            try:
                gmail = _find_tool(tools, "search_job_emails")
                # ONE sender per run. FALLBACK_SENDERS holds two, and querying
                # both in parallel is exactly what produced the "two job search
                # emails" the user reported. Ask again to check the other one.
                calls.append({
                    "id": "g0", "type": "tool_call", "name": gmail.name,
                    "args": {"sender": FALLBACK_SENDERS[0],
                             "days_back": cfg.days_back,
                             "max_results": cfg.max_results},
                })
            except RuntimeError:
                pass  # gmail server missing
            return AIMessage(content="", tool_calls=calls)
        try:
            board = _find_tool(tools, "search_job_boards")
            calls.append({
                "id": "b0", "type": "tool_call", "name": board.name,
                "args": board_tool_args(cfg, _last_human_text(state)),
            })
        except RuntimeError:
            pass  # scraper server missing — boards search will be skipped
        return AIMessage(content="", tool_calls=calls)

    def agent_node(state: AgentState) -> dict:
        """Emit the graph's next message. On a fresh hunt we skip the LLM and
        fan out to every job source at once (see _collect_message); otherwise
        the LLM runs — the summary step uses the unbound, faster instance, or
        in SUMMARY_MODE=template a deterministic in-code summary (no LLM)."""
        has_tool_msgs = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not has_tool_msgs and not state.get("fallback_used") and any(
                is_hunt_request(_text_content(getattr(m, "content", m)))
                for m in state["messages"]):
            return {"messages": [_collect_message(state)]}
        if not (has_tool_msgs or state.get("fallback_used")):
            return {"messages": [llm.invoke(state["messages"])]}
        if cfg.summary_mode != "llm":
            payload_content = ""
            for m in reversed(state["messages"]):
                if not isinstance(m, (HumanMessage, ToolMessage)):
                    continue
                txt = _text_content(getattr(m, "content", ""))
                if JOBS_MARKER in txt:
                    payload_content = txt
                    break
            return {"messages": [AIMessage(
                content=_template_summary(payload_content))]}
        return {"messages": [llm_summary.invoke(state["messages"])]}

    def route_after_tools(state: AgentState) -> str:
        """Deterministic router after a tools step.

        If the parallel-collect path ran (Gmail results are already in state),
        always go to "prep" — it picks the winning source in code. Otherwise
        keep the legacy behavior: blocked boards -> fallback, else summarize.
        A named board never takes the fallback edge: there were no Gmail calls
        to collect, so "fallback" would fire one now and return another
        board's listings."""
        if state.get("fallback_used"):
            return "agent"
        for m in state["messages"]:
            if isinstance(m, ToolMessage):
                name = (getattr(m, "name", "") or "").lower()
                if "email" in name or "gmail" in name:
                    return "prep"
        if requested_board(state):
            return "prep"
        return "fallback" if _scrape_blocked(state) else "agent"

    async def prep_node(state: AgentState) -> dict:
        """Parallel-collect path: boards AND Gmail both ran in the same tools
        step, so nothing is re-fetched here. Picks the winning source (boards
        if they returned jobs, else Gmail), filters in code, and hands the
        summary LLM a *compact* payload — the bulky raw tool messages are
        removed from state so the local LLM isn't re-reading ~8k tokens."""
        board_jobs, gmail_jobs, gmail_texts = [], [], []
        board_text = ""
        for m in state["messages"]:
            if not isinstance(m, ToolMessage):
                continue
            name = (getattr(m, "name", "") or "").lower()
            txt = _text_content(m.content)
            if "job_boards" in name or "scrape" in name:
                board_jobs = _extract_struct_from(txt)
                board_text += "\n" + txt
            elif "email" in name or "gmail" in name:
                gmail_texts.append(txt)
                gmail_jobs.extend(_extract_struct_from(txt))

        # EVERY source goes through the same filter. Board results used to
        # skip this and go straight into the deck, so boards and Gmail were
        # held to different bars.
        boards_kept = [j for j in board_jobs if _keep_job(j, cfg)]
        gmail_kept = list({j["link"]: j for j in gmail_jobs
                           if _keep_job(j, cfg) and j.get("link")}.values())

        if boards_kept:
            note = ("Job boards responded with listings. Summarize the matches "
                    "for the target role and keep every https link exactly as "
                    "given.")
            content = _summary_payload(boards_kept, note)
        elif requested_board(state):
            # A board was named and it produced nothing usable. Gmail alerts
            # cover several boards at once, so showing them here would answer
            # a different question than the one that was asked - the user said
            # one board and would get listings from the others. Report the
            # reason instead.
            wanted = requested_board(state)
            blocked = bool(board_jobs)
            why = ("every listing it returned was filtered out "
                   "(seniority or location)" if blocked else
                   "it did not respond or has no listings for this search")
            note = (f"The user asked for {wanted} only, and {wanted} {why}. "
                    "Do not use listings from any other source. Say plainly "
                    "that there is nothing from "
                    f"{wanted} right now and suggest loosening the filters "
                    "or searching all boards.")
            content = _summary_payload([], note, blocked_reason=f"{wanted}: {why}")
        else:
            # Boards came back with nothing usable. Say WHY, rather than always
            # blaming a block: an empty result and a walled board need different
            # advice, and "blocked" was being asserted whenever the parse came
            # back empty.
            note, marker = _fallback_note(
                _blocked_marker(board_text), _silent_senders(gmail_texts),
                len(gmail_jobs), len(gmail_kept))
            content = _summary_payload(gmail_kept, note,
                                       blocked_reason=marker)
        return {"messages": [*_drop_bulky(state["messages"]),
                             HumanMessage(content=content)],
                "fallback_used": True}

    async def fallback_node(state: AgentState) -> dict:
        gmail_tool = _find_tool(tools, "search_job_emails")
        parts = []
        structured = []
        # Both alert senders in parallel: separate IMAP connections, so the
        # two searches compete instead of serializing their connect+login.
        results = await asyncio.gather(
            *[asyncio.wait_for(
                gmail_tool.ainvoke({
                    "sender": sender,
                    "days_back": cfg.days_back,
                    "max_results": cfg.max_results,
                }),
                timeout=60,
            ) for sender in FALLBACK_SENDERS],
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):  # keep the run alive, report to the LLM
                parts.append(f"search_job_emails failed: {res}")
                continue
            # MCP tools return content as a list of blocks, not a str — normalize.
            res_text = _text_content(res)
            parts.append(res_text)
            # Keep the machine-readable listings (with links) too, so the
            # final report carries a working Apply button per job.
            structured.extend(_extract_struct_from(res_text))

        kept = list({j["link"]: j for j in structured if _keep_job(j, cfg)}.values())
        if kept:
            note = ("The direct job-board search was blocked, so this is fallback "
                    "job data pulled from Gmail job-alert emails instead. It has "
                    "already been filtered for seniority and India-tied roles. "
                    "Summarize the remaining listings for the target role, and "
                    "keep the https links exactly as given.")
        else:
            # Reaching here means the boards really were blocked, but the Gmail
            # fallback has nothing either. Say that plainly instead of
            # announcing fallback data and then printing "(no matching
            # listings found)".
            note = _fallback_node_note(_silent_senders(parts), len(structured))
        return {"messages": [*_drop_bulky(state["messages"]),
                             HumanMessage(content=_summary_payload(
                                 kept, note, blocked_reason="boards blocked"))],
                "fallback_used": True}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("fallback", fallback_node)
    graph.add_node("prep", prep_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    # tools -> router -> (summarize | legacy Gmail fallback | parallel prep)
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "fallback": "fallback", "prep": "prep"},
    )
    graph.add_edge("fallback", "agent")
    graph.add_edge("prep", "agent")

    return AgentBundle(graph.compile(), runtime, cfg)


class AgentBundle:
    """Compiled app + the MCP runtime that produced it.

    The runtime is kept so a caller can rebuild the agent with a new config
    WITHOUT rediscovering tools (see MCPRuntime). No long-lived subprocess is
    held: the adapters open a session per tool call and close it afterwards.
    """

    def __init__(self, app, runtime, cfg: AgentConfig):
        self.app = app
        self.runtime = runtime
        self.client = runtime.client
        self.tools = runtime.tools
        self.cfg = cfg

    async def reconfigure(self, cfg: AgentConfig) -> "AgentBundle":
        """Rebuild the graph for a new config, reusing the discovered tools."""
        return await create_agent(cfg, runtime=self.runtime)


if __name__ == "__main__":
    async def _selfcheck() -> None:
        """Smoke test: create the agent (spawns MCP servers), list the
        discovered tools, then rebuild with a second config to prove the
        runtime is genuinely reusable — run as `python agent.py`."""
        import time

        print("[selfcheck] building agent (launches MCP subprocesses)...")
        t0 = time.perf_counter()
        bundle = await create_agent()
        print(f"[selfcheck] discovered {len(bundle.tools)} tool(s) in "
              f"{time.perf_counter() - t0:.1f}s:")
        for t in bundle.tools:
            print(f"  - {t.name}")

        from dataclasses import replace as _replace
        t1 = time.perf_counter()
        again = await bundle.reconfigure(_replace(bundle.cfg, num_ctx=4096))
        print(f"[selfcheck] reconfigure with cached tools: "
              f"{time.perf_counter() - t1:.2f}s (no server relaunch)")
        assert len(again.tools) == len(bundle.tools)
        print("[selfcheck] OK. Run step7 or `python dashboard.py` for the real run.")

    asyncio.run(_selfcheck())