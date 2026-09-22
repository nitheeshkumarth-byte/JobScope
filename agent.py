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

load_dotenv()

# Marks a machine-readable JSON block of parsed job listings. Tool servers
# (scraper, gmail) append it to their output; the fallback node and the
# dashboard parse it so every listing gets a working Apply link/card.
JOBS_MARKER = "###JOBS_JSON###"

# Job-alert senders the fallback path queries (same as step6).
GOOGLE_CAREERS_SENDER = "careers-noreply@google.com"
INDEED_ALERT_SENDER = "donotreply@match.indeed.com"
FALLBACK_SENDERS = (GOOGLE_CAREERS_SENDER, INDEED_ALERT_SENDER)

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

# Code-level filter rules applied to fallback job data.
SENIORITY_FILTER = re.compile(
    r"senior|lead\b|manager|principal|staff|architect|director|head of",
    re.IGNORECASE,
)

# Plain-language board names a user might type ("only linkedin", "just
# indeed"). When one is detected in the request, the boards tool is told to
# scrape ONLY that platform — one board asked for means one board in the deck.
BOARD_ALIASES = {
    "Indeed": r"\bindeed\b",
    "LinkedIn": r"\blinked\s*in\b|\blinkedin\b",
    "Naukri": r"\bnaukri\b",
    "Glassdoor": r"\bglassdoor\b",
    "Foundit": r"\bfoundit\b|\bmonster\b",
    "Internshala": r"\binternshala\b",
    "WeWorkRemotely": r"\bwework(?:remotely)?\b|\bwe\s+work\s+remote(?:ly)?\b|\bwwr\b",
    "Remotive": r"\bremotive\b",
    "Arbeitnow": r"\barbeitnow\b",
}


def _detect_board(text: str) -> str | None:
    """First board name mentioned in free-form user text, canonicalized to the
    scraper's BOARDS keys (None when no board was explicitly requested)."""
    for canonical, pattern in BOARD_ALIASES.items():
        if re.search(pattern, text, re.I):
            return canonical
    return None
EXPERIENCE_FILTER = re.compile(r"\b(?:[5-9]|\d{2,})\+\s*(?:years|yrs)\b", re.IGNORECASE)

# Messages that read like a job hunt trigger the deterministic parallel
# collect below — we skip the LLM's "which tool should I call?" round-trip
# (on a local CPU that single inference is the slowest part of a run).
HUNT_HINT = re.compile(
    r"\b(hunt|jobs?|find|search|remote|position|vacanc|role|skills?|keyword|"
    r"listing|postings?|apply)\b", re.IGNORECASE)

# The user lives in India and only wants remote roles that are NOT India-tied.
# Used to drop India-based postings from any source (scraper + Gmail fallback).
INDIA_TOKENS = re.compile(
    r"\b(india|indian|hyderabad|bangalore|bengaluru|mumbai|pune|delhi|noida|"
    r"gurugram|gurgaon|chennai|kolkata|ahmedabad|coimbatore|indore|jaipur|"
    r"lucknow|kerala|karnataka|maharashtra|telangana|tamil ?nadu|uttar ?pradesh|"
    r"rajasthan|gujarat|bihar|punjab|west ?bengal|andhra|remote-india|"
    r"work from home india)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a job-hunting assistant, matching listings to the target role "
    "the user specifies. You have tools to search several job sources.\n"
    "The user lives in India and is ONLY looking for REMOTE jobs that are "
    "NOT based in India — exclude India-tied postings (location fields such "
    "as Bangalore, Pune, 'Remote - India', etc.).\n"
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
    # Remote / geography constraints (inferred from the CV).
    remote_only: bool = True
    exclude_locations: list[str] = field(default_factory=list)
    # Cities the user is open to working in/from (chosen in Settings; seeded
    # from the CV city on first upload). When remote_only is off they set the
    # boards' location; with remote_only on the location stays "Remote" and
    # these cities just contextualize the hunt + the resume's contact block.
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
    Code rules, not prose."""
    out = []
    for line in text.splitlines():
        if SENIORITY_FILTER.search(line) or EXPERIENCE_FILTER.search(line):
            continue
        if cfg and cfg.remote_only and INDIA_TOKENS.search(line):
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


def _summary_payload(jobs: list[dict], note: str) -> str:
    """Self-contained, compact payload handed to the summary LLM (and to the
    dashboard's flashcard deck via the ###JOBS_JSON### block)."""
    content = note + "\n\n" + (_link_block(jobs) if jobs else "(no matching listings found)")
    content += "\n\n" + JOBS_MARKER + "\n" + json.dumps(
        {"sources": {"collected": len(jobs)}, "total": len(jobs),
         "jobs": jobs}, ensure_ascii=False)
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
        source = j.get("source") or "job source"
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
    """Structured-listing twin of _filter_listings: drop senior/experience/
    India-tied roles and anything without a usable http link."""
    hay = f"{job.get('title', '')} {job.get('company', '')} {job.get('location', '')}"
    if SENIORITY_FILTER.search(hay) or EXPERIENCE_FILTER.search(hay):
        return False
    if cfg.remote_only and INDIA_TOKENS.search(job.get("location", "") or ""):
        return False
    link = job.get("link", "") or ""
    return link.startswith(("http://", "https://"))


def _extract_struct_from(res_text: str) -> list[dict]:
    """Pull the ###JOBS_JSON### block out of an MCP tool's text output."""
    if JOBS_MARKER not in res_text:
        return []
    try:
        blob = json.loads(res_text.split(JOBS_MARKER, 1)[1].strip())
        return blob.get("jobs", []) or []
    except (json.JSONDecodeError, ValueError):
        return []


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
    if cfg.remote_only:
        extra += "\nConstraint: only remote roles NOT based in India."
    roles = cfg.target_roles or [cfg.target_role]
    return SystemMessage(
        content=f"{SYSTEM_PROMPT}{extra}\nTarget roles: {', '.join(r for r in roles if r)}"
    )


def search_query_from(cfg: AgentConfig) -> str:
    """Keyword-driven search query: the PRIMARY role plus the strongest skill
    keywords (the whole explicit 'Technical Skills' list is used for matching;
    the query keeps the top few so board searches stay focused)."""
    terms: list[str] = []
    if cfg.skills:
        terms.extend(cfg.skills[:5])
    role = (cfg.target_roles[0] if cfg.target_roles else cfg.target_role).strip()
    if role and role.lower() not in " ".join(terms).lower():
        terms.insert(0, role)
    return " ".join(terms) or cfg.target_role


def default_task_messages(cfg: AgentConfig):
    query = search_query_from(cfg)
    skill_hint = (
        f" Your skill keywords: {', '.join(cfg.skills)}." if cfg.skills else ""
    )
    remote = "remote_only=True (remote, outside India)" if cfg.remote_only else "remote_only=False"
    user = (
        f"Find jobs matching: '{cfg.target_role}'.{skill_hint}\n"
        f"Start by calling search_job_boards with query={query!r}, "
        f"location={cfg.indeed_location!r}, {remote}.\n"
        "Then give me a clean markdown list of the matches: source, job "
        "title, company/location, application link — and for each, note "
        "which of my skills the role needs."
    )
    return [system_message(cfg), ("user", user)]


async def create_agent(cfg: AgentConfig | None = None) -> "AgentBundle":
    cfg = cfg or load_config()

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
    llm = ChatOllama(model=cfg.model, num_ctx=cfg.num_ctx).bind_tools(tools)
    # The final summary never calls tools — binding tool schemas slows every
    # generation (long function definitions on a small context), so summarize
    # with a plain, unbound instance of the same model.
    llm_summary = ChatOllama(model=cfg.model, num_ctx=min(cfg.num_ctx, 4096))

    def _collect_message(state: AgentState) -> AIMessage:
        """Deterministic first step of a hunt: emit ONE AIMessage whose
        parallel tool_calls hit the board scraper AND every Gmail alert sender
        simultaneously. Zero LLM inference, so results arrive as fast as the
        slowest network call instead of an LLM deciding + a sequential fallback.
        If the user named a specific board ("only linkedin"), that board is
        passed to the scraper so the deck comes from it alone."""
        calls = []
        try:
            board = _find_tool(tools, "search_job_boards")
            human = next(
                (_text_content(getattr(m, "content", "")) for m in reversed(state["messages"])
                 if isinstance(m, HumanMessage)),
                "",
            )
            args = {"query": search_query_from(cfg),
                    "location": cfg.indeed_location,
                    "remote_only": cfg.remote_only}
            requested = _detect_board(human)
            if requested:
                args["board"] = requested
            calls.append({
                "id": "c0", "type": "tool_call", "name": board.name,
                "args": args,
            })
        except RuntimeError:
            pass  # scraper server missing — boards search will be skipped
        try:
            gmail = _find_tool(tools, "search_job_emails")
            for i, sender in enumerate(FALLBACK_SENDERS):
                calls.append({
                    "id": f"c{i + 1}", "type": "tool_call", "name": gmail.name,
                    "args": {"sender": sender, "days_back": cfg.days_back,
                             "max_results": cfg.max_results},
                })
        except RuntimeError:
            pass  # gmail server missing — fallback will be skipped
        return AIMessage(content="", tool_calls=calls)

    def agent_node(state: AgentState) -> dict:
        """Emit the graph's next message. On a fresh hunt we skip the LLM and
        fan out to every job source at once (see _collect_message); otherwise
        the LLM runs — the summary step uses the unbound, faster instance, or
        in SUMMARY_MODE=template a deterministic in-code summary (no LLM)."""
        has_tool_msgs = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not has_tool_msgs and not state.get("fallback_used") and any(
                HUNT_HINT.search(_text_content(getattr(m, "content", m)))
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
        keep the legacy behavior: blocked boards -> fallback, else summarize."""
        if state.get("fallback_used"):
            return "agent"
        for m in state["messages"]:
            if isinstance(m, ToolMessage):
                name = (getattr(m, "name", "") or "").lower()
                if "email" in name or "gmail" in name:
                    return "prep"
        return "fallback" if _scrape_blocked(state) else "agent"

    async def prep_node(state: AgentState) -> dict:
        """Parallel-collect path: boards AND Gmail both ran in the same tools
        step, so nothing is re-fetched here. Picks the winning source (boards
        if they returned jobs, else Gmail), filters in code, and hands the
        summary LLM a *compact* payload — the bulky raw tool messages are
        removed from state so the local LLM isn't re-reading ~8k tokens."""
        board_text, board_jobs = "", []
        kept = []
        for m in state["messages"]:
            if not isinstance(m, ToolMessage):
                continue
            name = (getattr(m, "name", "") or "").lower()
            txt = _text_content(m.content)
            if "job_boards" in name or "scrape" in name:
                board_text, board_jobs = txt, _extract_struct_from(txt)
            elif "email" in name or "gmail" in name:
                kept.extend(j for j in _extract_struct_from(txt) if _keep_job(j, cfg))
        kept = list({j["link"]: j for j in kept}.values())  # dedupe by link

        if board_jobs:
            note = ("Job boards responded with listings. Summarize the matches "
                    "for the target role and keep every https link exactly as "
                    "given.")
            content = _summary_payload(board_jobs, note)
        else:
            note = ("The job-board search was blocked; instead here is fallback "
                    "data from Gmail job-alert emails, already filtered for "
                    "seniority and India-tied roles. Summarize the remaining "
                    "listings for the target role and keep every https link "
                    "exactly as given.")
            content = _summary_payload(kept, note)
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
        note = ("The direct job-board search was blocked, so this is fallback job "
                "data pulled from Gmail job-alert emails instead. It has already "
                "been filtered for seniority and India-tied roles. Summarize the "
                "remaining listings for the target role, and keep the https links "
                "exactly as given.")
        return {"messages": [*_drop_bulky(state["messages"]),
                             HumanMessage(content=_summary_payload(kept, note))],
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

    return AgentBundle(graph.compile(), client, tools, cfg)


class AgentBundle:
    """Compiled app + the MCP client config that produced it.

    Sessions are created per tool call by the adapters, so there is no
    long-lived connection to tear down; keep the client only if you need
    to re-derive tools later.
    """

    def __init__(self, app, client, tools, cfg: AgentConfig):
        """Bundle of the compiled graph + the MCP client/tools/config that
        built it (kept together so callers can inspect or re-detect tools)."""
        self.app = app
        self.client = client
        self.tools = tools
        self.cfg = cfg


if __name__ == "__main__":
    import asyncio

    async def _selfcheck() -> None:
        """Smoke test: create the agent (spawns MCP servers) and list the
        tools it discovered — run as `python agent.py`."""
        print("[selfcheck] building agent (launches MCP subprocesses)...")
        bundle = await create_agent()
        print(f"[selfcheck] discovered {len(bundle.tools)} tool(s):")
        for t in bundle.tools:
            print(f"  - {t.name}")
        print("[selfcheck] OK. Run step7 or `chainlit run ui.py` for the real run.")

    asyncio.run(_selfcheck())