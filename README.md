# Job Hunter Agent

A local, LLM-driven agent that hunts **remote jobs outside India** across
multiple job boards and checks them against **your skills**. Upload a resume/CV
in the dashboard and the extracted skill keywords (Python, SQL, Pandas, Power
BI, …) are used to **suggest the role you should target** and become the actual
search terms — matching job requirements against what you can actually do.
Built as a 7-step learning progression ending in a **LangGraph** agent that
talks to three **MCP** tool servers through local **Ollama**.

> **Documentation:** a full user guide in LaTeX lives at
> [`docs/JobScope_Guide.tex`](docs/JobScope_Guide.tex) (compile with
> `pdflatex` — architecture, graph flow, filters, MCP servers, API, setup).

## Architecture

```
                ┌───────────── Chainlit UI (ui.py) ─────────────┐
                │            or CLI (step7...py)                │
                └──────────────────┬────────────────────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │     agent.py      │  ← shared graph (every consumer)
                         └─────────┬─────────┘
        MultiServerMCPClient       │       LangGraph: START→agent→tools→…
       ┌───────────────────────────┼────────────────────────────┐
       ▼                           ▼                            ▼
┌───────┴────────┐      ┌──────────┴─────────┐      ┌───────────┴──────────┐
│  mcp_server_   │      │  mcp_server_gmail  │      │  mcp_server_indeed_  │
│  github.py     │      │  .py               │      │  scraper.py          │
│  (stdio)       │      │  (stdio, IMAP)     │      │  (stdio)             │
└────────────────┘      └────────────────────┘      └──────────────────────┘
   list_github_repos       search_job_emails           scrape_indeed_jobs
   get_repo_readme         (Google Careers +            (experimental,
                           Indeed alerts via            may be bot-blocked)
                           app password)
```

- Every consumer (CLI, web UI, future scheduled digest) calls the same
  `create_agent()` from `agent.py` — no duplicated graph wiring.
- MCP tools are **discovered at runtime**; the client knows nothing about the
  server internals.
- The LLM runs locally via Ollama (`llama3.1` or `qwen2.5`), free and offline.

## Workflow

The fallback and filtering decisions are **graph logic, not prompt luck**:
```
START
  │
  ▼
agent ──► (hunt message?) ──► NO  ──► LLM decides tools (as before)
  │                              │
  │ YES ─ deterministic: ONE step fans out IN PARALLEL to:
  │         · search_job_boards(query, location=Remote, remote_only=True)
  │         · search_job_emails(Google Careers)   ⌐  Gmail alerts,
  │         · search_job_emails(Indeed alerts)    ⌐  fetched concurrently
  │
  ▼
tools ──► router (code) ──► prep node picks the winning source:
  │             · boards returned jobs  → use boards, drop Gmail
  │             · all boards blocked    → use filtered Gmail listings
  ▼
agent ──► summarize (template: in code, instant)
  │              · AGENT_SUMMARY_MODE=llm → local model instead (minutes)
  ▼
END
```

1. A fresh job hunt skips the slow LLM "which tool?" round-trip: one
   deterministic step fires the multi-board scraper and both Gmail alert
   senders **in parallel** (India-tied postings are dropped before the LLM
   sees anything).
2. Blocked boards are reported per-source; a just-failed board is cached for
   90s so a repeat hunt doesn't re-hit the wall.
3. A `prep` node picks the winner in code — boards if they returned any
   listings, otherwise the filtered Gmail fallback — and hands the summary a
   compact payload; the bulky raw tool output is dropped from state so the
   model (when used) reads far fewer tokens.
4. The final summary is **template-based by default** (`AGENT_SUMMARY_MODE=
   template`): a deterministic in-code markdown answer with every title,
   company, location and Apply link — *instant*. `AGENT_SUMMARY_MODE=llm`
   swaps in the unbound local model (nicer prose, but a 2-4 minute CPU
   inference). The dashboard turns the structured listings into an
   **Apply-able flashcard deck** as soon as `prep` emits them.

## Project files

| File | Purpose |
|---|---|
| `step1_basics.py` … `step7_scraper_with_fallback.py` | 7-step tutorial: graph mechanics → agent loop → GitHub tool → MCP client → multi-server → Gmail → full agent with fallback |
| `agent.py` | **Shared agent** — config, MCP client, deterministic fallback router, seniority filter |
| `dashboard.py` | **FastAPI backend** for the dashboard (SSE streaming + config API) |
| `ui.py` | Chainlit web UI (streams the agent live) |
| `mcp_server_github.py` | MCP server: GitHub repos + READMEs |
| `mcp_server_gmail.py` | MCP server: Gmail job alerts via IMAP + app password |
| `mcp_server_indeed_scraper.py` | MCP server: multi-board scraper (`search_job_boards`), remote+outside-India filter, structured JSON feed |
| `resume_generator.py` | Job-tailored LaTeX resume builder (`\heading`/`\subheading` ATS template) + HTML preview converter |
| `resume_data.py` | Canonical resume content (name/contact/skills/experience/education) the generator draws from |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for `.env` (secrets) |

## Dashboard UI

The graphical dashboard (`static/index.html`) connects to FastAPI over SSE:
it streams LLM tokens into the chat, shows every MCP tool call as an
expandable card, and animates a **circuit-style pipeline** (Agent → Job boards
→ Router → Gmail → Report) — hover a node to see that step's log. It also:

- **Search by your resume.** Upload your CV (.pdf/.txt/.md) in Settings — the
  skills are extracted locally (LLM-first, lexicon fallback) into editable
  keyword chips, and matched against a role rubric to **suggest roles** (e.g.
  skills `sql python power bi tableau` → "Data Analyst"). Your location is
  inferred from the CV, and **remote + outside-India** is enforced.
- **Flashcard deck + Apply buttons.** When the scraper returns listing data,
  jobs appear as a smooth deck of flashcards (page-turn / slide animation,
  arrows, dots, swipe, keyboard) — each card has a big **Apply now** button
  that opens the exact job posting in a new tab. Even the Gmail fallback now
  feeds the deck: alert emails are parsed into structured listings (title,
  company, location, link) so Gmail-based results still get Apply buttons.
- **LaTeX resume per job.** Each flashcard has a **LaTeX resume** button that
  fetches the posting's description (best-effort), reorders your canonical
  skill groups to match, and generates an ATS-friendly `\section*`-based
  resume in your exact template (`resume_generator.py` + `resume_data.py`) —
  with a live HTML preview tab, **Copy LaTeX**, and **Download .tex**.
- **Audit everything.** Every run and every error is appended to
  `logs/runs-*.jsonl` / `logs/errors.jsonl`. The History drawer shows past
  runs with durations, tool calls and errors, with one-click **Retry** and
  **Restart agent** actions when something fails.
- Soft light theme (with a dark toggle), error banner with Retry/Restart, and
  a live status/model pill up top.

```
┌───────────────┐   SSE:      ┌────────────────────────┐
│  Browser UI   │ ─────────►  │  dashboard.py (FastAPI) │
│  static/      │ ◄─────────  └──────────┬─────────────┘
│  index.html   │  token/tool/fallback   │ create_agent()
└───────────────┘        events          ▼
                                   agent.py ☩ MCP servers
```

### Viewing logs

Runs and errors are written as JSONL every time, even across restarts:

```powershell
Get-Content logs\runs-2026-09-18.jsonl   # every run: start, done, tokens, tools
Get-Content logs\errors.jsonl            # failures with full tracebacks
```

In the UI: open **History**, expand a run to see tools/duration/error, and hit
**Retry** to re-run exactly what failed.

## Setup

```powershell
# 1. Create + activate the virtualenv (once)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (on Windows use the venv interpreter explicitly)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
#    (adds FastAPI/uvicorn via chainlit, and pypdf for resume parsing)

# 3. Configure secrets — copy the template and fill in YOUR values
Copy-Item .env.example .env
#    GITHUB_TOKEN   -> https://github.com/settings/tokens (no scopes for public reads)
#    GMAIL_ADDRESS / GMAIL_APP_PASSWORD -> https://myaccount.google.com/apppasswords
#    (requires 2-Step Verification on the Google account)

# 4. Install + start Ollama, pull a model
#    https://ollama.com/download  (auto-starts as a background service)
ollama pull llama3.1
```

> Never commit `.env` — it holds real tokens. `.gitignore` already excludes it.

## Launch commands

| What | Command |
|---|---|
| **Dashboard UI** | `python dashboard.py` — opens `http://localhost:8000` |
| Dashboard (alt port) | `python dashboard.py --port 9000` or `uvicorn dashboard:app --port 9000` |
| **Web UI (Chainlit)** | `chainlit run ui.py` — opens `http://localhost:8000` |
| **CLI run (full hunt + fallback)** | `python step7_scraper_with_fallback.py` |
| Agent self-check (launches MCP servers, lists tools) | `python agent.py` |
| Run individual tutorial step | `python step1_basics.py` … `python step6_gmail_source.py` |
| Start an MCP server manually (debug) | `python mcp_server_github.py` |

Use `.venv\Scripts\…` if you don't activate the venv in the current shell:

```powershell
.\.venv\Scripts\python.exe agent.py
.\.venv\Scripts\chainlit.exe run ui.py
```

**Prerequisite for the actual run:** Ollama must be running (`http://localhost:11434`)
with the configured model pulled, and `.env` must contain valid GitHub/Gmail values.

## Configuration (`.env`)

| Variable | Meaning | Default |
|---|---|---|
| `OLLAMA_MODEL` | Ollama model name | `llama3.1` |
| `OLLAMA_NUM_CTX` | Model context window (tokens) | `8192` |
| `TARGET_ROLE` | The role the agent matches listings to | `Junior Data Analyst / entry-level, 0-2 years experience` |
| `RESUME_EXTRACT_MODE` | `auto` (lexicon-first, instant; LLM only if nothing found), `fast` (lexicon only), `llm` (always the slow local model) | `auto` |
| `AGENT_SUMMARY_MODE` | `template` builds the final answer in code (instant); `llm` uses the local model (nicer prose, minutes on CPU) | `template` |
| `JOB_DAYS_BACK` | Email search window (days) | `60` |
| `JOB_MAX_RESULTS` | Emails to read per sender | `10` |
| `GITHUB_TOKEN` | GitHub PAT (public reads, no scopes) | — |
| `GITHUB_USERNAME` | GitHub username for repo tools | — |
| `GMAIL_ADDRESS` | Gmail address for IMAP | — |
| `GMAIL_APP_PASSWORD` | Gmail app password (spaces stripped) | — |

## Requirements & "connectors"

**No external connectors are needed.** Everything talks to local processes:

| Connector | What it is | Required for |
|---|---|---|
| `Ollama` (~/tools) | Local LLM server, stdio-free `http://localhost:11434` | Every run — must be running |
| MCP servers | Local `mcp_server_*.py` subprocesses (stdio transport) | Every run — auto-launched by `agent.py` via `sys.executable` |
| Gmail IMAP | Plain network call (`imap.gmail.com:993`) | The fallback path |
| Indeed | Plain HTTPS scrape (often 403-blocked) | The primary path |
| `pypdf` | Local resume/PDF text extraction | Resume upload |

The only open machine ports are the dashboard (`8000`) and Ollama (`11434`);
MCP servers are stdio pipes, not network endpoints.

Python requirements: `fastapi`, `uvicorn`, `langchain`, `langgraph`,
`langchain-ollama`, `python-multipart`, `pypdf` — all in `requirements.txt`
(chainlit pulls most of them in).

## Current limitations

- **Most boards are bot-protected from plain HTTP.** Indeed, Glassdoor,
  LinkedIn and Internshala answer 403/CAPTCHA, Naukri and Foundit refuse the
  connection outright — but WeWorkRemotely serves static HTML and usually
  gets through, so the board path works and the Gmail fallback is the backup,
  not the only way. (A browser-based scraper for the other six would help;
  that's a larger change.)
- Scrapers return page 1 only.
- **Gmail IMAP** is a blunt search (`FROM sender`) — filtering happens later in
  the agent, and Google's own alerts are loose matches.
- **Single-turn** in the UI (each message runs fresh with system instructions).
- Skills come from your resume; listings are judged against them, but a
  boards search that's fully blocked falls back to Gmail's own alert senders.
- `INDEED_ACCESS_TOKEN` (official remote MCP server) is **not wired up** — it
  needs its own OAuth flow; see `step5_two_mcp_servers.py`.

## Roadmap

- [x] Graphical dashboard UI (FastAPI + SSE, live pipeline + token streaming)
- [x] Shared `agent.py` + deterministic fallback router
- [x] Resume/CV → skill keywords → keyword-driven search + **role suggestions**
- [x] Remote + outside-India filtering (skills/rubric, location inference)
- [x] Multi-board search (Indeed · LinkedIn · Naukri · Glassdoor · Foundit · Internshala · WeWorkRemotely)
- [x] Flashcard job deck with **Apply buttons** + circuit pipeline UI
- [x] Run + error logging (`logs/*.jsonl`) and History drawer with Retry/Restart
- [x] Resume tailoring: per-job LaTeX resume (ATS template) + live preview, Copy & Download
- [ ] Compile the generated `.tex` to PDF (needs a TeX engine) or docx
- [ ] Step 8: scheduled daily digest (Windows Task Scheduler) + JSON result persistence
- [ ] Browser-driven scraping (Playwright) to defeat the board bot-walls
- [ ] Multi-turn conversation + persisted chat history in the UI
- [ ] Scraper pagination + more robust card parsing