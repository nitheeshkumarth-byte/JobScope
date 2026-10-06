# Job Hunter Agent

A local, LLM-driven agent that hunts **remote jobs** across
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
   (profile from the       Indeed alerts via            may be bot-blocked)
    CV's own link,          app password,
    token optional)         then answered from a
                            local SQLite cache)
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
  │         · search_job_boards(query, location=Remote, work_modes/countries)
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
   senders **in parallel** (senior/experience listings are dropped before the
   LLM sees anything).
2. Blocked boards are reported per-source; a just-failed board is cached for
   90s so a repeat hunt doesn't re-hit the wall. The scraper's "already shown"
   memory is keyed **per account** (`seen_scope`), so a posting one person has
   seen still leads a different person's hunt as fresh; each account's bucket
   is bounded separately and at most 16 are kept in memory.
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
| `agent.py` | **Shared agent** — config, MCP client/runtime, deterministic fallback router, seniority filter |
| `filters.py` | **The listing rules, defined once** — seniority/experience, work-type (remote/WFH/hybrid/on-site), the searchable country list, board aliases, `keep_job()`; shared by the agent and the scraper so the two can't drift. The India/region rules are still here but inert — both filters were retired |
| `dashboard.py` | **FastAPI backend** for the dashboard (SSE streaming + config API), plus the per-account session layer and the machine-to-machine `POST /api/screen` scoring endpoint |
| `auth.py` | Accounts, scrypt passwords, hashed session cookies, one-time email tokens, per-user CV profiles (SQLite) |
| `mailer.py` | Outbound confirmation/reset mail over `smtplib`, stdlib only |
| `ui.py` | Chainlit web UI (streams the agent live) |
| `mcp_server_github.py` | MCP server: GitHub repos + READMEs |
| `mcp_server_gmail.py` | MCP server: Gmail job alerts via IMAP + app password. Syncs into `gmail_cache.db` once, then searches locally |
| `mcp_server_indeed_scraper.py` | MCP server: multi-board scraper (`search_job_boards`), work-type + country filtering, structured JSON feed |
| `resume_generator.py` | Job-tailored LaTeX resume builder (`\heading`/`\subheading` ATS template) + HTML preview converter |
| `github_projects.py` | Reads the candidate's **own** GitHub profile (handle taken from the CV link) and picks JD-matching repositories, evidence-checked against each README |
| `ats_score.py` | Explainable 100-point ATS-readiness estimate over the rendered resume — contact, sections, structure, dates, JD keywords, evidence |
| `gmail_index.py` | Local SQLite index of synced job-alert mail, so every search after the first is offline and instant |
| `resume_data.py` | Canonical resume content (name/contact/skills/experience/education) the generator draws from |
| `n8n/workflows/resume-screening.json` | n8n workflow: upload a resume + JD → `/api/screen` → Ollama agent explains Skills/Experience/Missing/Verdict → JSON answer |
| `n8n/import.py` | Imports that workflow into n8n (Public API with `N8N_API_KEY`, Docker CLI fallback), creating the token/Ollama credentials |
| `requirements.txt` | Python dependencies |
| `requirements-dev.txt` | Test deps (pytest) on top of `requirements.txt` |
| `tests/` | pytest suite for the pure logic — filters, skill extraction, query building, payload parsing — plus the accounts, mailer and HTTP auth routes |
| `.env.example` | Template for `.env` (secrets) |

## Tests

The scrapers and the graph need live HTTP and a local model, but every rule
that decides whether a listing reaches you is a pure function and is covered:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q     # 1138 tests
```

`tests/` is mostly regression tests for bugs that actually shipped — the
seniority regex matching *company* names, `remote-india` cancelling itself out,
`GLOBAL_DEADLINE` reporting "timed out" and then blocking anyway, and a
duplicated `_internshala` definition silently shadowing the first. Add a case
when you touch `filters.py`.

## Dashboard UI

The graphical dashboard (`static/index.html`) connects to FastAPI over SSE:
it streams LLM tokens into the chat, shows every MCP tool call as an
expandable card, and animates a **circuit-style pipeline** (Agent → Job boards
→ Router → Gmail → Report) — hover a node to see that step's log. It also:

- **Search by your resume.** Upload your CV (.pdf/.txt/.md) in Settings — the
  skill keywords come straight from your **"Technical Skills"** block and are
  only keywords that literally appear in the CV (no invented extras; the free
  local parse catches every entry of the column — colons, pipe-separated
  table rows, bullets). Uploading another CV **adds its keywords to the set**
  instead of replacing them (deduped). You pick the **target role(s)** as
  multi-select chips — click any suggested role or type your own, keep as many
  as you like (the first is primary for the query, all are matched) — and the
  **cities** you're open to (also multi-select). The CV only seeds these on
  the first upload. They drive the board search query + location, the match
  criteria, and are matched against a role rubric to **suggest roles** (e.g.
  skills `sql python power bi tableau` → "Data Analyst").
- **Filter by work type and country.** Multi-select chip rows in Settings, each
  with its own Clear all:
  - **Work type** — Remote · Work From Home · Hybrid · On-site / Office.
    Ticking On-site or Hybrid reuses your first **city** as the boards'
    location instead of searching the literal string `Remote`.
  - **Countries** — a searchable picker over ~97 countries (type `germ`). A
    country chip keeps only listings actually located in that country. An
    explicitly named country always beats an ambiguous city name, so
    `Paris, Ontario, Canada` is Canada and `New Mexico, USA` is not Mexico.
    `Remote, Worldwide` listings pass any country filter — they are the easiest
    to apply to. When you pick countries but no city and the search isn't
    remote, the first country alphabetically becomes the location the boards are
    pointed at.

  Both are **optional**: select nothing and nothing is filtered, which is exactly
  the old behaviour. Both are saved per session and restored on restart. Every
  group in the drawer has a **Clear all** button, and one that is already empty
  is disabled rather than pretending there is something to undo. Save & restart
  reports failures inline (no more silent `failed to fetch`), and pages are
  served `no-store` so the UI always matches the backend.
- **Retired: the Region chips and the "remote outside India" checkbox.** Both
  narrowed a search for reasons that were not visible on screen, and both were
  applied by default: the old build hard-set *remote, outside India*, so simply
  uploading a CV silently discarded every India-tied posting. Geography is now
  driven **only** by the country chips you pick. `AgentConfig.regions` and
  `remote_only` still exist so old saved profiles load, but nothing forwards
  them to the scraper and nothing sets them — a stale value left in an existing
  profile cannot quietly narrow a search you can no longer see or edit. Loading
  a profile forces both off rather than reading them back, which is what kept
  the India rule alive for anyone whose profile predated the change.
- **Job-role dropdown.** The drawer offers the full job-role catalog
  (`JOB_ROLE_OPTIONS` in `dashboard.py`, 66 entries) in a custom combobox. It
  replaces the original `<select>` because the OS paints a native popup: there
  was nowhere to put a transition, and no way to filter a 66-item list. It now
  has a filter box, arrow-key navigation, a ✓ on roles already added, and a
  staggered reveal. It is still a *suggestion list*: picking a row only stages
  the value, and the role becomes part of your search when you press **Add
  role**. The same rule applies to CV uploads — a CV fills the suggestion
  chips, never the active role filters, so the first search in a new session is
  not decided for you.
- **Sign-in screen.** The gate is a designed surface rather than a plain form:
  a drifting aurora and masked grid behind a glass panel with a gradient
  hairline, an animated mark, a staggered field-by-field entrance, a scrolling
  board ticker, a shake on error, a ticked confirmation state, and a
  submit-button sheen. The theme can be switched before signing in, and both
  toggles stay in step. Every loop and the list stagger are switched off under
  `prefers-reduced-motion`.
- **Name one board and you get that board only.** Asking for *"only linkedin"*
  passes `board=LinkedIn` to the scraper **and** suppresses the Gmail alert
  fallback, because those emails cover several boards at once — falling back
  after the board you named came back empty answers a question you did not ask.
  The deck then says why it is empty (no response / everything filtered out)
  rather than filling the space with other sources' listings.
- **Each run starts from a clean deck.** The previous run's cards and
  `state.deck` are removed when a new hunt begins, so a search aimed at a
  different board never leaves the last run's results on screen looking like
  this run's answer. (This used to call `renderDeck([])`, which was a no-op: a
  bare array has no `.jobs`, so it returned immediately and cleared nothing.)
- **Reset everything.** The sidebar's **Reset everything** button wipes the
  account's job-search data in one confirmed step: every non-default session is
  deleted, the default is rebuilt blank (no CV, skills, roles, cities, GitHub
  link or filters), the cached agent bundles and the run history go too. Your
  login is untouched. The dated `logs/*.jsonl` files are *not* rewritten — they
  are the server's audit trail and are shared between accounts.
- **Flashcard deck + Apply buttons.** When the scraper returns listing data,
  jobs appear as a smooth deck of flashcards (page-turn / slide animation,
  arrows, dots, swipe, keyboard) — each card has a big **Apply now** button
  that opens the exact job posting in a new tab. Even the Gmail fallback now
  feeds the deck: alert emails are parsed into structured listings (title,
  company, location, link) so Gmail-based results still get Apply buttons.
- **LaTeX resume per job.** Each flashcard has a **LaTeX resume** button that
  fetches the posting's description (best-effort) and generates an
  ATS-friendly `\section*`-based resume in your exact template
  (`resume_generator.py` + `resume_data.py`) — with a live HTML preview tab,
  **Copy LaTeX**, and **Download .tex**.

  **An uploaded CV is the content; the template is only structure.** The
  objective, skills, experience, projects, education, certifications and
  languages are all parsed out of the file you uploaded, and the skill groups
  are reordered to match the posting without being rewritten or filtered. This
  used not to be true: the canonical experience/projects/education were emitted
  verbatim and the CV only ever moved the header, so uploading a different CV
  produced a document that still described the canonical resume. A section the
  CV does not have is left out entirely — including its heading — rather than
  backfilled from `resume_data`, and with no CV uploaded at all `resume_data`
  is used so a fresh account still gets a complete document. The one exception
  is **Objective**: if your CV has no summary of its own, one is written from
  the parts of the document that do exist — the role being targeted, your own
  skill lines, your most recent experience line and your education — and never
  from anything the CV does not say, so a generated paragraph obeys the same
  no-invention rule as a quoted one.

  **No canonical links in a CV-driven resume.** GitHub is resolved as
  explicit per-request argument → the CV's own link → your saved setting, and
  if none exist the entry is dropped. The *Detect* button reads the **active**
  session's CV only. Both were fixed because the old behaviour stamped somebody
  else's profile into every generated resume: Detect used to scan your other
  saved CVs and copy the answer into the active profile's `github_url`, which
  the generator read *ahead* of the CV, so an unrelated URL overrode the file
  being rendered. LinkedIn and the portfolio link follow the same rule now.

  **Projects come from the CV's own GitHub link** (checkbox in the resume
  modal). The handle is scraped from the uploaded document, never from a
  configured user, so two accounts on one deployment read two different
  profiles; `GITHUB_TOKEN` is optional and only raises the API rate limit.
  Repositories are scored **against the posting's own vocabulary** — a
  technology the JD never mentions scores nothing — then the shortlist has its
  READMEs fetched and each project line has to be supported by that README
  before it can be printed. A repository the CV already lists is never added a
  second time, and a repo whose name means nothing to the posting is left out
  entirely. Forks and archived repositories are skipped.

  **An ATS-readiness estimate, not an official score.** Six components over the
  rendered document — contact 12, sections 14, structure 16, dates 16, JD
  keywords 26, evidence 16 — each with the specific reason it lost points, plus
  the posting's terms the resume does not use yet. Add only what you have
  actually done: the list is a prompt, not permission. The recurring findings
  are the usual ones — no contact line, tables or multi-column layout, dates
  without months, keywords that appear only in the skills list, and bullets
  with no number in them.
- **Audit everything.** Every run and every error is appended to
  `logs/runs-*.jsonl` / `logs/errors.jsonl`. The History drawer shows past
  runs with durations, tool calls and errors, with one-click **Retry** and
  **Restart agent** actions when something fails.
- Soft light theme (with a dark toggle), error banner with Retry/Restart, and
  a live status/model pill up top.
- **The sign-in screen carries the runtime architecture as a backdrop**
  (`static/gate-architecture.svg`). It is a CSS *mask*, not an embedded page, so
  the line work takes the app's accent colour in both themes from one small
  asset; the full 760 KB Archify document was deliberately not embedded because
  it brings its own toolbar and theme state with it. It sits under the card at
  `pointer-events: none`, so it cannot intercept a click, and both its drift and
  the SVG's internal dash-flow stop for `prefers-reduced-motion` and while the
  tab is in the background.

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

## n8n resume screening

`POST /api/screen` exists so tools outside the dashboard can use the exact
scorer the resume modal uses: CV in, one explainable `ats` score out (score,
band, breakdown, hits, missing, issues), authenticated with `SCREEN_TOKEN` in
the `X-JobScope-Token` header and with no session — n8n cannot hold a
browser cookie. It accepts `multipart/form-data` (`resume` file field) or
`application/json` (`resume_base64`), both with optional `jd_text` /
`job_url`.

The bundled workflow in `n8n/` wires that into an agent:

```
Webhook → Pack request → Screen resume → Screening ok?
        ├─ yes → AI Agent (Ollama) → Shape analysis → 200 JSON
        └─ no  → Screening failed → 4xx/5xx JSON
```

The agent receives the scorer's JSON and must reply with exactly four lines —
`Skills:`, `Experience:`, `Missing:`, `Verdict:` — which `Shape analysis`
parses into `analysis`; a model that ignores the contract still lands under
`analysis.verdict` instead of breaking the response. The score itself is
never the model's job: `note` in the response says which side computed it.

```powershell
.\.venv\Scripts\python.exe n8n\import.py     # needs N8N_API_KEY in .env,
                                             # otherwise falls back to Docker CLI
curl -F resume=@cv.pdf -F "jd_text=We need a Python and Kubernetes engineer" `
     http://localhost:5678/webhook/resume-screening
```

Two credentials carry the secrets (`JobScope Screen Token` = Header Auth with
`X-JobScope-Token`, `JobScope Ollama` = `http://host.docker.internal:11434`),
so no token is stored in the workflow JSON. Full setup, response examples and
troubleshooting: [`n8n/README.md`](n8n/README.md).

## Setup

```powershell
# 1. Create + activate the virtualenv (once)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Fetch the pinned Scrapling checkout.
#    requirements.txt installs it with `-e ./Scrapling-main[fetchers]`, so this
#    has to happen BEFORE step 3 or pip cannot resolve that requirement. It is
#    a third-party repository, so it is gitignored rather than committed.
git clone --depth 1 --branch v0.4.15 https://github.com/D4Vinci/Scrapling Scrapling-main

# 3. Install dependencies (on Windows use the venv interpreter explicitly)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
#    (adds FastAPI/uvicorn via chainlit, and pypdf for resume parsing)

# 4. Chromium, for the JS-only boards and as the escalation path
.\.venv\Scripts\python.exe -m playwright install chromium

# 5. Configure secrets — copy the template and fill in YOUR values
Copy-Item .env.example .env
#    GITHUB_TOKEN   -> https://github.com/settings/tokens (no scopes for public reads)
#    GMAIL_ADDRESS / GMAIL_APP_PASSWORD -> https://myaccount.google.com/apppasswords
#    (requires 2-Step Verification on the Google account)

# 6. Install + start Ollama, pull a model
#    https://ollama.com/download  (auto-starts as a background service)
ollama pull llama3.1
```

> Never commit `.env` — it holds real tokens. `.gitignore` already excludes it.

## Launch commands

| What | Command |
|---|---|
| **Dashboard UI** | `python dashboard.py` — opens `http://localhost:8000` |
| Dashboard, auto-restarting on edit | `python dashboard.py --reload` |
| Dashboard (alt port) | `python dashboard.py --port 9000` or `uvicorn dashboard:app --port 9000` |
| **Web UI (Chainlit)** | `chainlit run ui.py` — opens `http://localhost:8000` |
| **CLI run (full hunt + fallback)** | `python step7_scraper_with_fallback.py` |
| Agent self-check (launches MCP servers, lists tools) | `python agent.py` |
| Run individual tutorial step | `python step1_basics.py` … `python step6_gmail_source.py` |
| Start an MCP server manually (debug) | `python mcp_server_github.py` |

### "only one usage of each socket address" / port 8000 already in use

The dashboard is **already running** — that message means a second copy tried to
bind the port the first one still holds. `python dashboard.py` now says so
instead of raising, and prints the exact command to stop the running copy:

```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
```

Or just leave it running and use <http://127.0.0.1:8000> — nothing needs
restarting for a **frontend** edit, because `static/index.html` is re-read on
every request. **Backend** edits (`dashboard.py`, `filters.py`, `auth.py`) do
need the restart, or `python dashboard.py --reload` while developing.

Two ways to run it without either problem: use `--reload` in development, or
pick another port with `--port 9000` if something unrelated already owns 8000.
Note that the dashboard and the Chainlit UI both default to 8000, so those two
cannot run at once.

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
| `RESUME_EXTRACT_MODE` | `strict` (only keywords literally in the CV — default), `auto` (also adds known-skill lexicon terms), `fast` (lexicon only), `llm` (always the slow local model) | `strict` |
| `AGENT_SUMMARY_MODE` | `template` builds the final answer in code (instant); `llm` uses the local model (nicer prose, minutes on CPU) | `template` |
| `JOB_DAYS_BACK` | Email search window (days) | `60` |
| `JOB_MAX_RESULTS` | Emails to read per sender | `10` |
| `GITHUB_TOKEN` | Optional. Public repos are read without it; a PAT only raises the hourly rate limit | — |
| `GMAIL_ADDRESS` | Gmail address for IMAP | — |
| `GMAIL_APP_PASSWORD` | Gmail app password (spaces stripped) | — |
| `SMTP_HOST` | Mail server for confirmation + reset links. Unset ⇒ no mail is sent | — |
| `SMTP_PORT` | `587` for STARTTLS, `465` for implicit TLS | `587` |
| `SMTP_USER` / `SMTP_PASSWORD` | Mail login. Both blank ⇒ no `AUTH`, for an open relay | — |
| `SMTP_FROM` | Envelope + `From:` address | `SMTP_USER` |
| `SMTP_TLS` | `1` upgrades with STARTTLS before sending credentials | `1` |
| `APP_URL` | Absolute base URL for emailed links, when behind a proxy/tunnel | from the request |
| `HOST` | Interface to bind | `127.0.0.1` |
| `AUTH_DEV_LINKS` | Return an unsent link in the response instead of mailing it | off |
| `SCREEN_TOKEN` | Shared secret for `POST /api/screen`. Unset ⇒ that endpoint answers 503 for everyone (a fresh checkout never exposes an open scanner) | — |
| `N8N_API_KEY` | Optional. n8n Public API key; when set, `n8n/import.py` imports and activates the workflow through the API | — |
| `N8N_URL` | Where n8n listens, for `n8n/import.py` | `http://localhost:5678` |
| `N8N_OLLAMA_BASE_URL` | Ollama URL *as the n8n container sees it*; only needed if `host.docker.internal` is wrong for your setup | `OLLAMA_URL` with `localhost` → `host.docker.internal` |

### Choosing a model

`OLLAMA_MODEL` is free text — any model `ollama pull` has installed works, and
the dashboard's model pill shows which one a run used. What the model actually
does here is narrow: the *first* step of a hunt is deterministic (the tool calls
are built in code, `agent.py:_collect_message`) and with the default
`AGENT_SUMMARY_MODE=template` the summary is built in code too. The model only
sees text when you switch the summary to `llm`, or the skill extractor to
`RESUME_EXTRACT_MODE=llm`.

That makes the practical trade-off different from the usual one:

| Model | Best when |
|---|---|
| `llama3.1` (default) | You want a known-good general model already installed. |
| `qwen2.5` | Same job, often a little steadier at following the "keep every https link exactly as given" instruction — which is the one thing the summary must not get wrong. |
| Anything smaller (3B and under) | Fine in `template` mode, where it does almost nothing; expect drift in `llm` summary mode. |

Whichever you pick, keep `OLLAMA_NUM_CTX` at `8192` or higher — the summary is
handed the whole listing payload, and a small context is the most common cause
of links being dropped from a summary. On CPU, expect minutes per run in `llm`
mode; `template` mode finishes in seconds regardless of model.

## Accounts

The dashboard is multi-user. Each person signs in with their own email address
and password, and their CV sessions, settings, agent config and run history are
theirs alone.

**First run.** With no accounts yet, the sign-in screen asks you to create one.
That first account becomes the *owner*, and it adopts every session from an
older single-user `profiles.json`, so an existing install keeps its history. The
old file is copied, never modified, and stays where it is.

**Storage.** One SQLite file, `users.db`, beside the source (gitignored). It holds
accounts, the CV profiles, login sessions and the one-time email links.

- Passwords: `hashlib.scrypt`, per-password random salt, parameters stored with
  the hash so the cost can be raised later without invalidating anyone. No
  third-party password library, matching the rest of the project.
- Session cookies: a random 32-byte token in the browser, only its SHA-256
  digest in the database, so a leaked file cannot be replayed as a login. The
  cookie is `HttpOnly` and `SameSite=Lax`, and marked `Secure` when the request
  arrived over HTTPS. Each session records which CV profile that browser has
  open.
- The agent graph is built per (account, session) and cached under that key, so
  two signed-in people never share a search context.

**Confirming an address.** With `SMTP_HOST` set, a new account gets a
confirmation link by mail and the app stays closed (`403`) until it is followed.
Without it, the server confirms the address itself and shows the link in the
browser — convenient locally, and not something to rely on for a shared server.

**Resetting a password.** `POST /api/auth/forgot` mails a link that works once
and expires in 2 hours. Setting a new password drops every existing session, so
a cookie captured beforehand stops working. The response is identical for known
and unknown addresses, so the endpoint cannot be used to discover who has an
account. Setting `AUTH_DEV_LINKS=1` returns the unsent link instead of mailing
it, which is only safe on a machine you control.

**Serving it to other people.** Keep `HOST=127.0.0.1` for local use. To share it
on a LAN, set `HOST=0.0.0.0` and put it behind real TLS: over plain HTTP the
session cookie is not marked `Secure`, so it would travel in the clear.

## Requirements & "connectors"

**No external connectors are needed.** Everything talks to local processes:

| Connector | What it is | Required for |
|---|---|---|
| `Ollama` (~/tools) | Local LLM server, stdio-free `http://localhost:11434` | Every run — must be running |
| MCP servers | Local `mcp_server_*.py` subprocesses (stdio transport) | Every run — auto-launched by `agent.py` via `sys.executable` |
| Gmail IMAP | Plain network call (`imap.gmail.com:993`) | The fallback path |
| [Scrapling](https://github.com/D4Vinci/Scrapling) (vendored in `Scrapling-main/`) | TLS/browser impersonation + a real-Chromium path | The primary path — see [Job boards](#job-boards) |
| Chromium (Playwright) | `python -m playwright install chromium` | Only for the JS-only boards, and as escalation |
| `pypdf` | Local resume/PDF text extraction | Resume upload |

The only open machine ports are the dashboard (`8000`) and Ollama (`11434`);
MCP servers are stdio pipes, not network endpoints.

Python requirements: `fastapi`, `uvicorn`, `langchain`, `langgraph`,
`langchain-ollama`, `python-multipart`, `pypdf`, `scrapling[fetchers]` — all in
`requirements.txt` (chainlit pulls most of them in). Scrapling is installed from
the vendored tree (`-e ./Scrapling-main[fetchers]`) so the pinned 0.4.15 is used.

## Job boards

Collection goes through `boards.py`, which wraps Scrapling. Two fetch paths:

- **Impersonated HTTP** (the default) — `curl_cffi` replays a real Chrome TLS
  and HTTP/2 fingerprint. No browser, ~1s per board. This is what actually
  clears the walls; a `User-Agent` header on its own does not, because the
  block is at the TLS layer.
- **Real Chromium** (`StealthyFetcher`) — reserved for JS-only boards and for
  escalation when the cheap path returns nothing parseable. Correct, but slow
  (~20–45s), so it is never the first thing tried.

Measured live (2026-10-02), all-boards hunt in **16.8s**:

| Board | Default hunt | Result |
|---|---|---|
| Indeed | yes | 12 listings |
| LinkedIn | yes | 30 listings |
| Internshala | yes | 48 listings |
| WeWorkRemotely | yes | 10 listings (filtered by the query) |
| Remotive | yes | 9 listings (keyless JSON) |
| Arbeitnow | yes | 8 listings (keyless JSON) |
| Naukri | no | renders nothing even in a real browser — needs a signed-in account. Cost a 21s browser escalation for a guaranteed zero, so it is not probed by default |
| Glassdoor | no | works when named (10 listings), but duplicates what the others already cover |
| Foundit | no | needs a browser and 403s about half the time, so it is not worth the wait on every hunt |

Naukri, Glassdoor and Foundit are still reachable by name — `board="Glassdoor"`
probes Glassdoor and nothing else. They are listed as `[skipped]` in the hunt
summary so a missing result reads as a deliberate choice rather than a failure.


## Reading a scanned CV (OCR)

A CV is normally a PDF with a real text layer, and `pypdf` reads that in
milliseconds. A CV that was **scanned, photographed, or re-saved as page
images** has no text layer at all, and that upload used to fail with *"Could
not read any text from that file."*

`cv_ocr.py` is the second attempt for exactly that case, and only that case —
if the first extraction came back with fewer than 200 letters, and the file is a
PDF or an image, the page pixels are rendered with PyMuPDF and read with
RapidOCR (PaddleOCR's models compiled to ONNX) through `onnxruntime`.

Everything runs on the CPU. There is no torch, no GPU, and no system binary to
install. On an i7-8650U with no GPU a page reads in **about 4–5 seconds** at
**99.5% mean confidence**, and the contact line comes back character-exact.

Why not a vision-language model. Measured on this machine, Ollama generates
**2.53 tok/s** on a text-only model, so a CV page would take minutes once the
image itself is encoded. Worse, a VLM asked to *read* a page will confidently
invent a plausible email address or phone number — and that field is what every
application from this profile then uses. RapidOCR reads exact characters, and
where it is unsure it drops the line and says so rather than guessing.

Because a misread digit is invisible after the fact, the dashboard shows the
recognised text and any warnings underneath the upload, instead of trusting the
read silently.

Configuration, all optional (`.env`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `JOBSCOPE_OCR` | `auto` | `off` disables it entirely |
| `JOBSCOPE_OCR_MAX_PAGES` | `4` | pages read before stopping |
| `JOBSCOPE_OCR_DPI` | `150` | render resolution |
| `JOBSCOPE_OCR_SECONDS` | `45` | wall-clock budget per upload |
| `JOBSCOPE_OCR_MIN_CONF` | `0.55` | lines below this are dropped |

The page cap and the time budget exist because this runs inside a request — a
40-page scan must not hold a worker open for minutes. Past either limit it
returns what it has and tells you so.

> `PyMuPDF` is AGPL-3.0 / dual-licensed. That is fine for a locally run tool. If
> you ever host this as a service, swap the renderer for `pypdfium2` (BSD-3) —
> only `cv_ocr._read_pdf()` touches it.

## Current limitations

- **Naukri needs an account.** It answers with a JavaScript shell and renders
  nothing even in a real Chromium session, so no amount of stealth fixes it —
  it wants a signed-in session. It is not probed by default for that reason.
  Every other board now returns real listings (see [Job boards](#job-boards));
  the earlier "Indeed is 403-blocked" note no longer holds, because the block
  was at the TLS layer and impersonation clears it.
- Scrapers return page 1 only.
- **Gmail IMAP** is a blunt search (`FROM sender`) — filtering happens later in
  the agent, and Google's own alerts are loose matches.
- **OCR can drop a line.** Measured at 3.15% character error on a synthetic
  scan, and the whole of that error was one short all-caps heading the detector
  skipped; accuracy did not change between 150 and 300 dpi. Skill headings are
  the most likely thing to be lost, so the upload shows what was recognised for
  you to check.
- **Single-turn** in the UI (each message runs fresh with system instructions).
- Skills come from your resume; listings are judged against them. If a named
  board comes back empty the hunt says so rather than quietly substituting
  Gmail, because that would answer a question you did not ask.
- `INDEED_ACCESS_TOKEN` (official remote MCP server) is **not wired up** — it
  needs its own OAuth flow; see `step5_two_mcp_servers.py`.

## Roadmap

- [x] Graphical dashboard UI (FastAPI + SSE, live pipeline + token streaming)
- [x] Shared `agent.py` + deterministic fallback router
- [x] Resume/CV → skill keywords → keyword-driven search + **role suggestions**
- [x] Location inference from the CV (offered as a one-click chip, never auto-applied)
- [x] Work-type (remote / WFH / hybrid / on-site) + country filters, incl. local job search
- [x] Job-role dropdown (filterable animated combobox, opt-in — never applied without an explicit add)
- [x] Designed sign-in screen: animated backdrop, glass panel, staggered entrance, reduced-motion aware
- [x] Per-account "already seen" memory, so one person's hunt doesn't starve another's
- [x] Blank starting session on every login; old CVs preserved in the sidebar
- [x] Multi-board search (Indeed · LinkedIn · Naukri · Glassdoor · Foundit · Internshala · WeWorkRemotely · Remotive · Arbeitnow)
- [x] Flashcard job deck with **Apply buttons** + circuit pipeline UI
- [x] Run + error logging (`logs/*.jsonl`) and History drawer with Retry/Restart
- [x] Resume tailoring: per-job LaTeX resume (ATS template) + live preview, Copy & Download
- [ ] Compile the generated `.tex` to PDF (needs a TeX engine) or docx
- [ ] Step 8: scheduled daily digest (Windows Task Scheduler) + JSON result persistence
- [x] Browser-driven scraping (Scrapling `StealthyFetcher` / Chromium) to defeat
      the board bot-walls — now the escalation path behind the cheap
      impersonated request
- [ ] Multi-turn conversation + persisted chat history in the UI
- [ ] Scraper pagination + more robust card parsing