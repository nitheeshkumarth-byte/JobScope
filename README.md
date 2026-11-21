# JobScope — AI-Powered Job Hunting Assistant

[Demo Video](files%20-%20Antigravity%20IDE%20-%20JobScope_Research.tex%202026-10-12%2018-44-37.mp4)

**JobScope** is a locally-run, full-stack job-hunting platform that searches job boards in parallel, falls back to Gmail alerts when blocked, and tailors ATS-friendly LaTeX resumes via a LangGraph agent with MCP tools and a real-time dashboard. Everything runs on your machine with Ollama.

---

## Table of Contents

1. [What It Does](#-what-it-does)
2. [Architecture Overview](#-architecture-overview)
3. [Quick Start](#-quick-start)
4. [Project Structure](#-project-structure)
5. [The LangGraph Agent](#-the-langgraph-agent)
6. [Job Boards & Scraping](#-job-boards--scraping)
7. [Gmail Fallback Pipeline](#-gmail-fallback-pipeline)
8. [Resume Pipeline](#-resume-pipeline)
9. [Filtering Rules](#-filtering-rules)
10. [HTTP API Reference](#-http-api-reference)
11. [SSE Event Protocol](#-sse-event-protocol)
12. [MCP Tool Servers](#-mcp-tool-servers)
13. [Authentication & Sessions](#-authentication--sessions)
14. [Configuration Reference](#-configuration-reference)
15. [Data Model](#-data-model)
16. [Tutorial Progression (step1–step7)](#-tutorial-progression-step1step7)
17. [Testing](#-testing)
18. [Monitoring & Logs](#-monitoring--logs)
19. [Troubleshooting](#-troubleshooting)
20. [Key Design Decisions](#-key-design-decisions)
21. [Known Limitations & Roadmap](#-known-limitations--roadmap)
22. [Contributing · License · Acknowledgments](#-contributing--license--acknowledgments)

---

## What It Does

| Capability | Description |
|---|---|
| **Multi-board search** | Queries **9 job boards** concurrently (Indeed, LinkedIn, Naukri, Glassdoor, Foundit, Internshala, WeWorkRemotely, Remotive, Arbeitnow) under a 40-second hard deadline with per-board blocked-status reporting |
| **Smart fallback** | When boards are bot-walled (403/CAPTCHA/JS-shell), a **deterministic graph edge** — not a prompt — routes the run to Gmail job-alert emails (Indeed, LinkedIn, Google Careers, WeWorkRemotely senders) |
| **Local Gmail cache** | Gmail is synced once via the **Gmail API (OAuth 2.0)** into a local SQLite cache (`gmail_cache.db`); every query answers instantly from the cache with sender-diversity ranking |
| **ATS-friendly resume generation** | Parses your uploaded CV, BM25-ranks your own bullets against the job description, reorders (never invents) content, and emits a compilable LaTeX resume + HTML preview |
| **BM25 retrieval (RAG)** | Okapi BM25 over your CV and document library — no embeddings, no hallucination, every hit explainable ("this bullet contains all three JD terms") |
| **Explainable ATS scoring** | 100-point score across 6 weighted components (contact, sections, structure, dates, keywords, evidence), each with actionable issue notes |
| **Role suggestions** | Infers target roles from your CV's skills against a ~60-role rubric (data, backend, DevOps, mobile, security, …) |
| **GitHub project matching** | Pulls your public repos, ranks them against the JD, and merges the top matches into the resume's Projects section |
| **Real-time dashboard** | FastAPI + SSE streaming UI: circuit-style pipeline animation, flashcard job deck with **Apply** and **LaTeX resume** buttons, sessions sidebar, settings/history drawers |
| **Multi-account auth** | Email+password auth (scrypt), email verification, password reset, idle/absolute session timeouts, profiles/sessions stored in SQLite |
| **Human-in-the-loop wall solving** | Optional visible-browser mode (`JOBSCOPE_MANUAL_WALL=on`) that lets *you* solve reCAPTCHA/hCaptcha; the cleared page is then parsed automatically |
| **Tier-2 OCR** | Image-only CVs are recovered with PyMuPDF rendering + RapidOCR (CPU, ~3.15% measured char error rate) when text-layer extraction yields < 200 characters |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACES                                   │
├─────────────────────────────┬──────────────────────────────────────────────┤
│  FastAPI Dashboard          │  Chainlit Chat (ui.py)                       │
│  (dashboard.py +            │  - Streaming tokens & tool steps             │
│   static/index.html)        │  - Starter prompts                           │
│  - Auth gate, sessions,     │                                              │
│    profiles, settings       │                                              │
│  - SSE event stream         │                                              │
│  - Flashcard deck + resume  │                                              │
│    modal, ATS panel         │                                              │
└──────────────┬──────────────┴──────────────────┬───────────────────────────┘
               │  POST /api/run (SSE)           │  bundle.app.astream
               ▼                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  AGENT CORE — LangGraph StateGraph (agent.py)               │
│                                                                             │
│   START ─► ┌────────┐  tools_condition  ┌────────┐  route_after_tools      │
│            │ agent  │──────────────────►│ tools  │───┐                      │
│            └────────┘  (AIMessage with  └────────┘   │                      │
│                 ▲         tool_calls)                ├─► "prep"    ──────┐  │
│                 │                                    ├─► "fallback"──┐   │  │
│                 │                                    └─► "agent"     │   │  │
│                 │                                               │   │   │
│            ┌────┴─────┐   ┌───────────┐                         ▼   ▼   │
│            │  prep    │◄──┤  fallback │ (legacy path)      (both loop   │
│            │ (winner  │   └───────────┘                       back to    │
│            │  in code)│                                         "agent") │
│            └──────────┘                                                   │
└──────────────┬────────────────────────────────────────┬───────────────────┘
               │ MCP tool calls (stdio subprocesses)    │ prompt (Ollama)
               ▼                                        ▼
┌──────────────────────────────────┐    ┌───────────────────────────────────┐
│  MCP SERVERS                     │    │  Ollama LLM                       │
│  • mcp_server_indeed_scraper.py  │    │  localhost:11434 (default         │
│    → search_job_boards (9 boards)│    │  llama3.1, configurable)          │
│  • mcp_server_gmail.py           │    └───────────────────────────────────┘
│    → sync_gmail_cache            │
│    → search_job_emails (OAuth)   │    ┌───────────────────────────────────┐
│  • mcp_server_github.py          │    │  RESUME PIPELINE                  │
│    → list_github_repos           │    │  1. cv_ocr / pypdf text extract   │
│    → get_repo_readme             │    │  2. rag_store BM25 vs JD          │
└──────────────┬───────────────────┘    │  3. resume_generator build_resume │
               │                        │  4. tex_to_html preview           │
               ▼                        │  5. ats_score score_resume        │
┌──────────────────────────────────┐    └───────────────────────────────────┘
│  EXTERNAL                        │
│  • Job boards (HTTPS / browser)  │         ┌────────────────────────────┐
│  • Gmail API (OAuth 2.0)         │         │  SQLite: users.db,         │
│  • GitHub REST API               │         │  gmail_cache.db, logs/*.jl │
└──────────────────────────────────┘         └────────────────────────────┘
```

**Invariants that shape the design:**

- The dashboard and the chat CLI share the **same compiled graph** — `create_agent()` is the single source of truth.
- MCP servers are **stdio subprocesses** (not network services). Only ports **8000** (dashboard) and **11434** (Ollama) are open locally.
- Every job source ends its output with a machine-readable `###JOBS_JSON###` payload, which the dashboard turns into flashcards and the template summariser parses.
- Filtering, routing, and resume tailoring are **deterministic code**; the LLM is used only where judgement genuinely helps (and can be disabled entirely via `AGENT_SUMMARY_MODE=template`).

---

## Quick Start

### Prerequisites

- **Python 3.14** (the version used by the Windows installer and CI)
- **[Ollama](https://ollama.com/download)** running locally with a model pulled (default `llama3.1`)
- A display is only needed for optional manual board verification
- Optional: **Gmail OAuth credentials** (client secret) for the job-alert fallback
- Optional: **GitHub PAT** (public reads work without it; token raises the 60 req/h rate limit)

### Installation

From a checkout of JobScope, run this in PowerShell on Windows:

```powershell
.\install-windows.ps1
```

Add `-InstallBrowser` if you want Chromium for browser-based job boards. The script creates a virtual environment and copies `.env.example` only when `.env` is absent.

If an older virtual environment still contains Chainlit or reports dependency conflicts, rename `.venv` before running the installer so it can build a fresh environment.

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
# Optional browser support
python -m playwright install chromium
```

Start Ollama and run `ollama pull llama3.1` to use model features. For development, install `requirements-dev.txt` in the same environment.

### Gmail OAuth setup (fallback + alert sync)

```bash
# One-shot interactive login; writes gmail_oauth_token.json
python gmail_login.py
```

`gmail_login.py` opens a browser consent flow against `client_secret.json` (or the file named by `GMAIL_OAUTH_CREDENTIALS`) with the `gmail.readonly` scope. It writes the token to `GMAIL_OAUTH_TOKEN_FILE`, which defaults to `gmail_oauth_token.json`. JobScope refreshes this token as needed and can show cached alerts when a sync fails.

### Google sign-in and Gmail for multiple users

1. In [Google Auth Platform](https://console.cloud.google.com/auth/overview), create a **Web application** OAuth client. For a student pilot with personal Google accounts, use an **External** audience with publishing status **Testing** and add the accounts that will connect Gmail or Forms. Personal-mode projects require a separate project for multiple users. Enable the Gmail and Forms APIs for the project you use for those integrations.
2. Register the exact dashboard callback. For example, use `http://127.0.0.1:8001/api/google/callback` when the portal is opened at `http://127.0.0.1:8001`. Use the corresponding HTTPS callback and set `APP_URL` for a hosted instance.
3. Download the Web client JSON to `client_secret.web.json` and set `GOOGLE_WEB_OAUTH_CREDENTIALS=client_secret.web.json` in `.env`. Keep this file private. Restart the dashboard after changing configuration.
4. Set `JOBSCOPE_GOOGLE_TOKEN_KEY` to a persistent Fernet key before enabling per-user Gmail. Generate it with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` and keep it private. Changing the key requires users to reconnect Gmail.

Existing candidates and invited recruiters sign in as usual and select **Link Google sign-in** once. A first Google signup creates a candidate account. Recruiter and company access still comes from JobScope. Administrators continue to use their existing verified-account login.

Candidates select **Connect Gmail** separately and choose which inbox to authorize. JobScope reads job-alert senders, stores each connection's encrypted refresh token, and saves parsed listings only for that connection. Connecting or switching accounts starts a fresh cache without copying the shared local cache. **Disconnect Gmail** removes the saved token and listings from JobScope. Revoke the grant from the Google account to remove the application's authorization there. The `gmail_login.py` desktop connection remains available as a local option. The job-search agent's Gmail fallback uses that local connection only when the current user is authorized for it and has no per-user connection.

Recruiters connect **Google Forms** in their own portal. The Forms callback remains `/api/recruiter/integrations/google-forms/callback`. To use a Web client for Forms, register that callback for the same dashboard origin and configure `GOOGLE_OAUTH_CREDENTIALS` and, if needed, `GOOGLE_FORMS_REDIRECT_URI`. Recruiters should reconnect Forms after changing its OAuth client because their existing refresh tokens belong to the original client.

Google's External testing grants for Gmail and Forms expire after seven days. Review [Google's Gmail scope requirements](https://developers.google.com/workspace/gmail/api/auth/scopes) before hosting Gmail imports publicly.

### Run the dashboard

On Windows, use `.\.venv\Scripts\python.exe dashboard.py`. On Linux or macOS, run these commands in the activated environment:

```bash
python dashboard.py                 # http://127.0.0.1:8000
python dashboard.py --port 9000     # alternate port
```

In the recruiter portal, changing a candidate application to **Shortlisted**, **Offer released**, or **Rejected** sends a candidate email through the configured SMTP server. A rejection requires a candidate-facing reason, which also appears in **My applications**. If SMTP delivery fails, the application status is saved and the recruiter can select **Retry email**.

The settings in [`.env.example`](.env.example) enable private use on this computer. If you already have an `.env`, set `HOST=127.0.0.1` and add `JOBSCOPE_LOCAL_MODE=1` for local signup and the local Gmail connection. To use Gmail with a candidate account, set `JOBSCOPE_GMAIL_USER` to that account's verified JobScope email. The first signup through a loopback connection creates the owner. Sign in at `/admin` with that account's email and password.

To create the first owner from the terminal, run `python dashboard.py --create-owner` using the project's virtual environment. It prompts for the password and imports any legacy profiles. Existing owners can continue using their own accounts.

The admin portal accepts verified owner and super-admin accounts. A new super admin, a newly promoted administrator, or an account with a changed email must verify its address. Sign in from the main dashboard and request a verification email if needed.

### Publish the free portfolio demo

1. Review the sample files under `site/` for personal information.
2. In GitHub's repository settings, open **Pages** and set **Source** to **GitHub Actions**.
3. Push the changes to `main`, then run **Deploy JobScope demo** from the **Actions** tab. The run displays the published URL. Later changes under `site/` deploy automatically.

The workflow uploads `site/`, which replays sample responses in the browser. Keep the full dashboard and any real CVs, SQLite databases, and Gmail credentials on your machine. The test workflow runs on pull requests and on pushes to `main`.

For a separate FastAPI pilot, set `JOBSCOPE_LOCAL_MODE=0`, configure `APP_URL` to your HTTPS origin, and create the owner at the terminal before inviting testers. Public signup requires SMTP and successful email verification. Without SMTP, the owner can create verified standard accounts from `/admin`. Gmail and manual browser verification are local integrations. The built-in rate limits are per process, so use a single worker for a small pilot. Use synthetic data on hosts with ephemeral storage.

### Double-click launcher (Windows)

```bash
python -m pip install pyinstaller
python -m PyInstaller --onefile --noconsole --name JobScope launcher.py
```

Copy `dist/JobScope.exe` to the project directory. `JobScope.exe` starts `dashboard.py` through `.venv\Scripts\pythonw.exe`, waits for it to answer, then opens the browser. It needs the virtualenv and `dashboard.py` beside it; server output goes to `logs/launcher.log`, and `HOST` is forced to `127.0.0.1` so a double-click never exposes the app to the LAN.

### Run the Chainlit chat UI

Activate a separate Python 3.13 virtual environment for this optional interface.

```bash
python -m pip install -r requirements-chainlit.txt
chainlit run ui.py
```

### Run a one-off CLI hunt

```bash
python step7_scraper_with_fallback.py "Find Junior Python Developer jobs remote"
```

### Agent self-check

```bash
python agent.py    # builds the graph, lists discovered MCP tools, rebuilds with a smaller ctx to prove runtime reuse
```

---

## Project Structure

```
jobscope/
├── agent.py                          # LangGraph agent core: AgentConfig, MCP client, graph, router
├── dashboard.py                      # FastAPI backend: auth, SSE runs, resume API, JSONL logging
├── ui.py                             # Chainlit chat frontend (alternative to the dashboard)
├── launcher.py                       # double-click launcher (starts pythonw dashboard.py, opens the browser)
├── JobScope.exe                      # built launcher (git-ignored; build it as shown in Quick Start)
├── install-windows.ps1               # Windows setup helper
│
├── filters.py                        # SINGLE SOURCE OF TRUTH for listing rules (seniority,
│                                     #   experience, India-tied, work modes, regions, countries,
│                                     #   board aliases, role synonyms, alert senders)
│
├── boards.py                         # Scrapling-based board parsers: TLS impersonation (Fetcher)
│                                     #   + browser escalation (StealthyFetcher/patchright)
├── mcp_server_indeed_scraper.py      # MCP tool: search_job_boards (9 boards, deadline, seen-memory)
├── mcp_server_gmail.py               # MCP tools: sync_gmail_cache, search_job_emails (Gmail API/OAuth)
# mcp_server_gmail_oauth_fixed.py   # LEGACY superseded Gmail server (currently has a SyntaxError — do not import)
├── mcp_server_github.py              # MCP tools: list_github_repos, get_repo_readme
│
├── gmail_login.py                    # one-shot Gmail OAuth consent → gmail_oauth_token.json
├── gmail_index.py                    # SQLite cache of job-alert mail (messages + listings + sync state)
├── manual_verify.py                  # human-in-the-loop bot-wall solving (visible Chromium)
├── job_desc.py                       # background job-description fetch + 24 h cache
│
├── resume_generator.py               # ATS LaTeX resume builder + tex→HTML preview
├── resume_data.py                    # canonical resume content (fallback when no CV uploaded)
├── rag_store.py                      # Okapi BM25 document library + retrieval
├── ats_score.py                      # explainable 100-point ATS score
├── cv_ocr.py                         # Tier-2 OCR (PyMuPDF + RapidOCR) for image-only CVs
├── github_projects.py                # JD-relevant repo matching + README summarisation
├── auth.py                           # SQLite auth: users, sessions, tokens, profiles, RAG docs, JD cache
├── mailer.py                         # SMTP sender (verify / reset / welcome emails)
├── google_forms_api.py               # Google Forms → Applications intake (per-recruiter OAuth, form creation, Drive CV import, sync)
│
├── step1_basics.py … step7_scraper_with_fallback.py   # seven-step tutorial progression
│
├── tests/                            # pytest suite (25 modules)
├── static/                           # dashboard SPA: index.html (CSS+JS single file), SVG assets
├── site/                             # static showcase + scrubbed demo (Cloudflare Pages)
├── docs/                             # LaTeX guides, recruiter patch, captured HTML
│   ├── captures/                     # captured recruiter portal HTML
│   └── scratch/                      # scratch snippets, not used by the app
├── logs/                             # JSONL run/error logs + launcher.log (auto-created)
│
├── users.db                          # SQLite: accounts, sessions, profiles, RAG docs, JD cache
├── gmail_cache.db                    # SQLite: synced job-alert messages & extracted listings
├── profiles.json                     # legacy profile store (migrated into SQLite on first login)
├── client_secret.json                # Gmail OAuth client secret (git-ignored)
├── gmail_oauth_token.json            # Gmail OAuth token (git-ignored)
├── .env                              # secrets/config (git-ignored)
├── requirements.txt / requirements-dev.txt
└── pytest.ini / conftest.py          # test config + personal-data isolation fixture
```

---

## The LangGraph Agent

### State

`AgentState(MessagesState)` with one extra key: `fallback_used: bool` — set once the fallback/prep path has run, preventing infinite `scrape-blocked → fallback → scrape-blocked` loops.

### Nodes

| Node | Role |
|---|---|
| `agent` | On a fresh hunt **skips the LLM entirely**: emits one `AIMessage` whose parallel `tool_calls` fan out to `search_job_boards` (or `search_job_emails` when the user asked for mail) — zero inference, instant dispatch. Afterwards the final answer is built **in code by default** (`AGENT_SUMMARY_MODE=template`); with `llm` mode an unbound Ollama instance on a trimmed context writes it |
| `tools` | Prebuilt `ToolNode` executes the MCP tools; parallel tool calls run concurrently |
| `prep` | Picks the winning source **in code** — boards if they returned any listings, else the filtered Gmail fallback — then hands the summariser one self-contained payload (with the `###JOBS_JSON###` feed for flashcards) |
| `fallback` | Legacy sequential path for LLM-decided runs where the scraper was blocked; pulls Gmail alert senders in parallel (`asyncio.wait_for`, 120 s) |

### Edges

```
START -> agent -> (tools_condition)
                    "tools"   -> tools
                    "__end__" -> END
tools -> route_after_tools
          "prep"     (parallel collect ran — Gmail results already in state)
          "fallback" (legacy: scraper blocked, no Gmail collected yet)
          "agent"    (summarise normally)
prep     -> agent
fallback -> agent
```

`route_after_tools` logic, in order:

1. `fallback_used` already set → `"agent"` (summarise, stop looping).
2. Any Gmail/email `ToolMessage` present in state → `"prep"` (the parallel-collect path).
3. The user named a single board (`requested_board(state)`) → `"prep"` (never legacy-fallback, since Gmail wasn't collected).
4. The most recent scraper `ToolMessage` contains a **blocked marker** → legacy `"fallback"`.
5. Else → `"agent"` to write the final answer.

### Blocked markers

A multi-board run can fail on *some* boards yet still return jobs, so the bare word `"blocked"` is **not** a trigger. The markers that are (`agent.py`):

```
"all boards were blocked", "captcha", "http 403", "fall back",
"no job links", "couldn't extract", "request failed", "js-rendered"
```

Blocked *boards* are reported per-source from the structured `###JOBS_JSON###` payload (`blocked`, `blocked_reasons`), which takes precedence over prose matching.

### Query construction

`search_query_from(cfg)` builds the board query:

- Role = first of `target_roles` else `target_role`, reduced by `_board_role_phrase()` (strips bracketed asides, splits on `/ , ; |`, drops pure-qualifier trailing segments, applies typo fixes like `devloper→developer`, dedupes).
- Up to **2** skills from the CV are appended (multi-word skills first), skipping short tokens, substrings of the role, and near-duplicates.
- Never empty — falls back to `cfg.target_role or "software developer"`.

`board_tool_args(cfg, human_text)` then adds `location` (from `filters.boards_location`), `remote_only`, normalized `work_modes`/`countries`, the per-account `seen_scope`, and `board` when a single board was named.

### MCP runtime

`create_runtime()` builds one `MultiServerMCPClient` with **three stdio servers**, each launched with `sys.executable` (never bare `python`):

| Server key | Script | Env forwarded |
|---|---|---|
| `github` | `mcp_server_github.py` | `GITHUB_TOKEN` only if set |
| `gmail` | `mcp_server_gmail.py` | **all** `GMAIL_*` vars (covers account 2, cache DB override, OAuth file paths) |
| `indeed_scraper` | `mcp_server_indeed_scraper.py` | none |

Tool discovery costs ~3.4 s, so the dashboard caches the runtime for the app's lifetime and caches compiled agent bundles per `(user_id, profile_id)`.

---

## Job Boards & Scraping

### The nine boards

| Board | Strategy | In default sweep? |
|---|---|---|
| Indeed | HTTP TLS impersonation (curl_cffi, Chrome JA3) | yes |
| LinkedIn | HTTP impersonation (`f_WT=2` remote filter) | yes |
| Naukri | Browser (JS shell over plain HTTP) | opt-in (`board="Naukri"`) |
| Glassdoor | HTTP impersonation | opt-in |
| Foundit | Browser (403 over plain HTTP) | opt-in |
| Internshala | HTTP impersonation (work-from-home internships page) | yes |
| WeWorkRemotely | HTTP + relevance re-rank (board ignores query params) | yes |
| Remotive | Keyless JSON API (`/api/remote-jobs`) | yes |
| Arbeitnow | Keyless JSON API (`/api/job-board-api`) | yes |

### Two-layer fetch (`boards.py`)

1. **Cheap path (default):** Scrapling `Fetcher` with `curl_cffi` TLS/JA3 impersonation, pinned Chrome UA, `stealthy_headers=True`, ~1 s per board.
2. **Browser path:** `StealthyFetcher` (patchright + Chromium) with Cloudflare solving, ad/resource blocking, `network_idle` for search pages — ~15–45 s; used for JS-only boards and as one-shot escalation on `BoardWall` / `ThrottledOrUnparsed` / zero-row results.

### Blocked detection

- `BoardWall` — outright refusal (HTTP 403/429, ≥400 status, or challenge page).
- `ThrottledOrUnparsed` — page loaded but yielded no cards (escalation helps here).
- `_looks_walled()` matches nine phrases (`"verify you are a human"`, `"unusual traffic…"`, `"cf-browser-verification"`, `"pardon the interruption"`, …) against the **rendered body text only**, so an embedded reCAPTCHA site key on a healthy page is not flagged.

### Multi-board sweep (`mcp_server_indeed_scraper.py`)

- One `ThreadPoolExecutor` worker per board; **`GLOBAL_DEADLINE = 40 s`** caps the entire hunt; stragglers report `blocked (timed out)` and the pool is shut down with `cancel_futures=True`.
- Failed boards are cached for **90 s** (`BLOCK_CACHE_SECONDS`) so a retry storm doesn't re-attack a wall.
- Thin results (< 12 listings) trigger one **sibling-role widening** sweep over healthy boards only, reported as `[ALSO]`.
- Dedupe key: `(title.lower(), link)`.
- **Per-account "already seen" memory:** up to 400 links per scope (16 LRU scopes); `seen_scope="u{user_id}"` is set by the dashboard so repeat hunts return fresh cards; repeats are still included but marked (`[NOTE] … already shown`).
- Output: human-readable per-board status lines + `###JOBS_JSON###` payload with `sources`, `source_detail`, `blocked`, `blocked_reasons`, `no_matches`, `widened_to`, `total`, `fresh`, `repeats`, `jobs[]` (`source,title,company,location,salary,link[,desc]`).

### Manual wall solving (`manual_verify.py`)

TLS impersonation handles plain 403s and `solve_cloudflare` handles Cloudflare interstitials, but reCAPTCHA/hCaptcha checkboxes are *designed* to be unsolvable by machines. With `JOBSCOPE_MANUAL_WALL=on`, a blocked board opens a **visible** Chromium window (persistent `.browser_profile` so logins survive), you solve the challenge, and the cleared page is parsed by the board's own parser. Exposed as async jobs (`starting → waiting → done|blocked|failed`, 4-min default budget) and reachable from the dashboard via **Verify** buttons (`POST /api/verify-board`, polled at 1.5 s).

---

## Gmail Fallback Pipeline

Reading Gmail uses the **Gmail API over OAuth 2.0** (not IMAP):

1. **Consent once:** `python gmail_login.py` → `gmail_oauth_token.json` (scope `gmail.readonly`).
2. **Sync:** `sync_gmail_cache(days_back, account)` lists messages per alert sender (`from:"{sender}" after:{YYYY/MM/DD}`), fetches raw RFC822, parses them (`parse_message`), and upserts into `gmail_cache.db`. HTTP 403/429 are retried with exponential backoff (4 tries). Default syncs **both** configured mailboxes.
3. **Search:** `search_job_emails(...)` answers from the local cache — auto-resyncing when stale (> 1 h) — over-fetching per account and then applying `diverse_listings()` (round-robin by sender, so one LinkedIn digest can't fill the deck).
4. **Parsing:** footer tracking blocks are cut, per-line junk URLs dropped, known tracking params stripped, each email split into structured listings around "view job"/"apply" links, and the single best job URL ranked (direct posting > job path > known-employer host > click-tracker > search page). Bodies are capped at 1200 chars, titles at 90.

**Alert senders** (from `filters.py`): Indeed job-alert + match, LinkedIn jobs-noreply + digest, WeWorkRemotely hello@m, Google Careers. LinkedIn *notification* senders (messages/updates) are explicitly excluded as noise.

The cache (`gmail_index.py`) is append-only SQLite (WAL): `mail_messages` (one row per message), `mail_listings` (one row per extracted listing, ordinal PK), `mail_sync_state` (per-account sync bookkeeping). Override the file with `GMAIL_CACHE_DB`.

---

## Resume Pipeline

Every flashcard's **LaTeX resume** button calls `POST /api/resume/gen`:

1. **Job description resolution** (priority order): pasted JD → JD carried in the job payload → fresh `job_desc` cache → `desc_snippet(link)` quick fetch. A background browser fetch (`job_desc.ensure`, 2 Chromium workers, 24 h TTL, URL-normalized cache shared across accounts) always runs for the link; the modal polls `GET /api/resume/desc` and offers **Regenerate** when the full text lands.
2. **Retrieval:** `rag_store.retrieve(uid, f"{desc} {title} {company}")` — Okapi BM25 (k1=1.5, b=0.75) over 260-char bullet/paragraph chunks of your CV + document library. If the open profile has no CV, the best-matching uploaded document is borrowed wholesale.
3. **Tailoring (`resume_generator.build_resume`):**
   - The CV's **own skill groups** are reordered by posting-term hits (`_rank_skill_groups`, 49 posting terms) — never added to or deleted.
   - Experience/Project **bullets are reordered within each entry** by BM25 against the JD — whole line spans move, chronological entry order never changes, and nothing is rewritten.
   - The Objective is quoted from the CV summary (or synthesised strictly from CV lines) plus a "Targeting the X role at Y" line seeded by the top matched skills.
   - Optional GitHub merge: `github_projects.find_projects()` ranks your public repos against the JD, fetches only matching READMEs (24 h disk cache), and prepends projects the CV doesn't already list — each bullet must share ≥1 tech term with its own README or it is dropped.
   - Output: ATS-friendly 10 pt article (`\section*` headings, single column, no tables, accent `1F4E79`) as `Resume_<company>_<role>.tex`, plus a regex-based HTML preview so the browser shows the resume **without a TeX install**.
4. **ATS scoring:** `ats_score.score_resume(preview, f"{desc} {title}")` returns `{total, band, breakdown, issues, hits, missing}` — components: contact 12, sections 14, structure 16, dates 16, keywords 26 (sqrt-damped coverage), evidence 16. Bands: ≥85 strong, ≥70 good, ≥55 fair, else weak.

**CV intake** (`POST /api/resume`): PDF (pypdf) / DOCX / RTF / text extraction; if fewer than 200 alpha chars are found, Tier-2 OCR (PyMuPDF render + RapidOCR, CPU ONNX) recovers the text (measured 3.15% char error rate at 99.5% mean confidence on a 150-dpi A4 scan). `extract_skills` in `strict` mode takes **only** keywords literally present in the CV (Technical Skills block if present); `suggest_roles` scores ~60 roles; `infer_location` resolves the city and seeds work-mode defaults. Each upload creates a fresh profile.

---

## Filtering Rules

All filtering is **deterministic code** in `filters.py`, shared by the scraper and the agent so they cannot drift:

| Rule | Implementation |
|---|---|
| **Seniority** | `SENIORITY_FILTER` drops titles matching senior/lead/principal/staff/architect/director/manager/vp/chief/head-of — matched against **title + location only**, never the company name |
| **Experience** | `EXPERIENCE_FILTER` drops ≥5-year requirements (`5+ years`, `5-8 yrs`, `min 5 years`…) with lookbehinds that spare `less than 5 years`, `up to 5`, and `0-2 years` |
| **Geography** | `INDIA_TOKENS` (Indian cities/states + "Remote-India") excludes India-tied listings when the remote rule applies; genuine WFH is exempt, but India-scoped remote (`remote-india`, `wfh-india`) is not |
| **Work modes** | Chips remote / wfh / hybrid / onsite; `listing_work_modes()` reads only explicit statements (WFH ⇒ {wfh, remote}; hybrid ⇒ {hybrid} only); unstated arrangements are kept at search time |
| **Countries / regions** | ~84 canonical countries with aliases and per-region membership; blank locations are dropped when a country filter is active; `worldwide` matches everything |
| **Links** | `keep_job()` requires a valid `http(s)` link so every card is Apply-able |

The system prompt mirrors these rules so the LLM re-rationalises them in prose — but the graph never relies on the LLM to enforce them.

---

## HTTP API Reference

All routes require the `js_session` cookie unless marked *public*. Responses carry `Cache-Control: no-store`.

### Authentication

| Method & path | Purpose |
|---|---|
| `GET /api/auth/state` | Bootstrap: first-run flag, auth/verify state, SMTP configured?, idle countdown |
| `POST /api/auth/register` | `{email,password}` — issues a session, sends verification when SMTP is configured, and offers owner bootstrap on loopback in local mode |
| `POST /api/auth/resend` | Re-issue verify link (session required; no account enumeration) |
| `GET /api/auth/verify?token=` | Consume single-use verify token → mark verified → welcome mail |
| `POST /api/auth/login` | `{email,password}` — identical error for unknown vs wrong password |
| `POST /api/auth/logout` | Destroy session, clear cookie |
| `GET /api/auth/session` | Read-only idle countdown |
| `POST /api/auth/extend` | +5 min idle allowance (cannot revive expired sessions) |
| `POST /api/auth/identity` | `{nickname?,avatar?}` — 12 gradient avatars |
| `GET /api/auth/avatars` | Avatar catalogue |
| `POST /api/auth/forgot` | Always `{ok:true}`; mails reset link (2 h, single-use) |
| `POST /api/auth/reset` | `{token,password}` — sets password, kills all sessions |

### Application

| Method & path | Purpose |
|---|---|
| `GET /` | Serve the SPA |
| `GET /api/boards` | *Public* board list + defaults (for the sign-in ticker) |
| `GET /api/config` | Full agent config + role/city/work-mode/region/country options + discovered tool names |
| `POST /api/config` | Rebuild MCP runtime + agent with new settings; persists to the active profile |
| `GET /api/profiles` · `POST /api/profiles` | List / create session profiles |
| `POST /api/profiles/activate` · `POST /api/profiles/delete` | Swap / delete (default profile undeletable) |
| `POST /api/reset` | Wipe all non-default profiles, runs, RAG docs, seen-memory for the account |
| `POST /api/run` | **SSE stream** of a full agent run |
| `GET /api/logs` | Recent runs (≤50) + errors (≤20) for the account |

### Recruiter CRM — phase 1 (`candidates_api.py`)

Candidates are owned by the recruiter who added them; sourced pipeline entries are owned by their company. Another company's candidate or entry returns `404`, never `403`.

| Method & path | Purpose |
|---|---|
| `GET /api/candidates?q=&skill=&status=&tag=&limit=&offset=` | Filter the caller's candidates; `skill` matches skills extracted from the uploaded resume |
| `GET /api/candidates/{id}` | Candidate details + notes (newest first) + tags + active resume (with extracted skills) + sourced pipeline rows |
| `DELETE /api/candidates/{id}` | Delete the candidate and cascade notes, tags, resume row, stored file and company pipeline entries |
| `POST /api/candidates/{id}/notes` | `{note}` — author and UTC timestamp recorded, shown newest first |
| `POST /api/candidates/{id}/tags` | `{tag, remove?}` — add or remove one tag |
| `POST /api/candidates/{id}/resume` | Multipart PDF/DOCX (≤ 5 MB, validated by file signature) → parsed text + skills; image-only PDFs fall back to the Tier-2 OCR; replaces the previous resume |
| `GET /api/candidates/{id}/resume` | Authenticated download; served as `Content-Disposition: attachment` from `data/resumes/` |
| `GET /api/recruiter/pipeline` | Sourced `pipeline_entries` for the caller's company (board load) |
| `POST /api/recruiter/pipeline` | `{job_id, candidate_id, status?}` — hand-add a candidate to a job; repeat updates instead of duplicating |
| `POST /api/recruiter/pipeline/{id}/status` | Move a sourced entry to another application stage (status whitelist enforced) |
| `GET /api/recruiter/screening?job_id=&limit=&offset=` | Score every company candidate resume against the job's posting with the existing `ats_score.score_resume`: ATS total, band, matched and missing keywords, best first. Another company's job returns `404` |

The Resume Screening page uses this route; it does not re-implement matching. Scoring is the same code the resume generator and `/api/screen` use.

### Google Forms applications (`google_forms_api.py`)

Candidates can apply through a Google Form instead of the portal. Each recruiter connects their own Google account once, links a form to one of their jobs, and syncs responses; every response becomes a normal `applications` row, so it appears in Applications, Resume Screening and the pipeline like any other application. The respondent gets an unverified candidate account keyed by their email (an unusable password hash, `verified_at` NULL), so a later sign-up with that address claims the row. Re-syncing never duplicates a response and never overwrites a status the recruiter set. The answers in the form's linked Google Sheet are the same responses read through the Forms API, so there is no separate Sheets pull to configure.

Every job created while Google is connected also gets its own published `Application — <job title>` form in the recruiter's Drive, linked to that job automatically (name, email, phone and a "why this role" question). The Forms API cannot create file-upload questions: add one **File upload** question to the form once in the Google Forms UI, and each uploaded CV is downloaded from the respondent's Drive into `data/resumes/` on sync — the same store the portal upload uses, so it is parsed, OCR'd when scanned, and scored in Resume Screening. The `application` is kept even if a CV download fails (bad file type, revoked access), and the count is reported separately.

Setup: enable the **Google Forms API** and **Google Drive API** in the same Google Cloud project as `client_secret.json`, add the callback URL above as an authorised redirect URI, then use **Google Forms → Connect Google** in the recruiter portal. The integration asks for `forms.responses.readonly`, `forms.body` (create/publish forms) and `drive.readonly` (download uploaded CVs); the refresh token is stored per recruiter in `google_oauth`. When `GOOGLE_FORMS_REDIRECT_URI` is unset the callback is derived from the host/port the portal is actually served on, so `localhost` and `127.0.0.1` both work; that exact URI is stored with the `state` and repeated at the token exchange, so a browser returning on another host/port cannot break the exchange. A refused exchange returns Google's own error code (for example `invalid_grant`) instead of a generic message.

| Method & path | Purpose |
|---|---|
| `GET /api/recruiter/integrations/google-forms` | Connection state, connected Google address and the company's linked forms |
| `GET /api/recruiter/integrations/google-forms/start` | Consent URL for this recruiter's own Google account (single-use `state` + PKCE verifier stored together, 10-minute TTL) |
| `GET /api/recruiter/integrations/google-forms/callback` | OAuth callback; resolves the recruiter from `state` (a missing session cookie on another host does not 401), replays the PKCE verifier and redirect URI stored with that state, stores the refresh token and returns to `/recruiter#google-forms` |
| `DELETE /api/recruiter/integrations/google-forms` | Forget this recruiter's token; linked forms stay, syncs stop until reconnected |
| `POST /api/recruiter/integrations/google-forms/links` | `{form, job_id}` — `form` is a Forms URL or id; the form must be readable by the connected account and the job must belong to the caller's company |
| `DELETE /api/recruiter/integrations/google-forms/links/{id}` | Unlink a form (another company's link returns `404`) |
| `POST /api/recruiter/integrations/google-forms/links/{id}/sync` | Pull responses, insert new applications and import uploaded CVs; returns `{added, seen, skipped, duplicates, no_email, resumes}` |

`POST /api/recruiter/jobs` also returns `form: {form_id, form_title, responder_uri}` (or `null` when Google is not connected / form creation failed), so the portal can point the recruiter straight at the new form. The submitted answers are kept as a Q&A transcript in `applications.cover_letter`. A response with no usable email address is skipped and counted in `no_email` rather than inventing an account — a form that never asks for an address imports nothing, so enable **Settings → Responses → Collect email addresses** (or add an Email question) and have the respondent submit again.

Scheduling an interview (also when a candidate-proposed slot is confirmed) emails the candidate the job title, time in UTC, duration, meeting link/location and instructions, using the address and name they applied with — for a form respondent, exactly what Google Forms collected. Both routes return `mail: {sent, error}`, and a delivery problem never undoes the booking.

### Resume & RAG

| Method & path | Purpose |
|---|---|
| `POST /api/resume` | Multipart CV (≤ 8 MB) → text/OCR → skills, role suggestions, location, links; creates + activates a new profile |
| `POST /api/resume/detect` | Re-detect GitHub links from the active profile's stored CV |
| `POST /api/resume/gen` | Build tailored LaTeX resume + HTML preview + ATS score for one listing |
| `GET /api/resume/desc?link=` | Poll background JD fetch (`ready`/`pending`/`empty`) |
| `GET /api/rag/docs` · `POST /api/rag/docs` · `DELETE /api/rag/docs/{id}` | Document library CRUD |
| `POST /api/screen` | *Header `X-JobScope-Token` vs `SCREEN_TOKEN`* — machine-to-machine ATS screening (multipart or base64 JSON) |

### Gmail & boards

| Method & path | Purpose |
|---|---|
| `POST /api/gmail/sync` | Sync job-alert mail into the local cache |
| `GET /api/gmail/listings` | Diverse cached listings (accounts pooled) |
| `POST /api/verify-board` · `GET /api/verify-board/{job_id}` | Start / poll a manual bot-wall solve |

---

## SSE Event Protocol

`POST /api/run` responds `text/event-stream`; each frame is `data: {json}\n\n`:

| `type` | Payload highlights |
|---|---|
| `status` | `stage: starting \| done` + message |
| `token` | Streamed LLM token chunks (template mode emits the full answer once) |
| `tool_start` | `{name, args}` — lights the pipeline node |
| `tool_end` | `{name, output}` (output capped at 900 chars) |
| `jobs` | `{payload}` — parsed `###JOBS_JSON###` deck → flashcards |
| `fallback` | "Job boards were blocked — using Gmail job alerts." |
| `error` | `{message, run}` — banner with Retry / Restart |

Run records (start/done/error, durations, tool lists, job counts) are appended to `logs/runs-YYYYMMDD.jsonl`; failures also land in `logs/errors-YYYYMMDD.jsonl` with tracebacks.

---

## MCP Tool Servers

Each server declares tools with `@mcp.tool()` and runs `mcp.run(transport="stdio")`. The agent's client spawns them with `sys.executable`.

| Server | Tools | Notes |
|---|---|---|
| `mcp_server_indeed_scraper.py` | `search_job_boards(query, location="Remote", remote_only=True, board=None, work_modes=None, regions=None, countries=None, seen_scope="")` | 9 boards, 40 s deadline, per-source blocked status, seen-memory, `###JOBS_JSON###` output |
| `mcp_server_gmail.py` | `sync_gmail_cache(days_back=3650, account=0)` · `search_job_emails(sender="", days_back=60, max_results=10, account=0, query="", resync_if_stale=True)` | Gmail API over OAuth; answers from local cache |
| `mcp_server_github.py` | `list_github_repos(github_url_or_username)` · `get_repo_readme(github_url_or_username, repo_name)` | Public API (60 req/h anonymous); PAT optional; READMEs capped at 3000 chars |

---

## Authentication & Sessions

- **Storage:** stdlib SQLite (`users.db`), no ORM; WAL-friendly connection factory with `foreign_keys=ON`.
- **Passwords:** `hashlib.scrypt` (N=2¹⁴, r=8, p=1, 32-byte key, 16-byte salt), stored as `scrypt$n$r$p$salt$hash`; verification is constant-time; failed logins burn time against a dummy hash.
- **Sessions:** 32-byte `token_urlsafe` tokens; only the SHA-256 digest is stored. Cookie `js_session`. **Idle timeout 25 min** (touch on each authed request, extend +5 min) and **absolute 30-day expiry**.
- **Email tokens:** single-use verify (24 h) and reset (2 h) tokens, digest-only storage; previous unused tokens of the same purpose are retired on re-issue.
- **Profiles:** per-user JSON documents (resume text, skills, roles, cities, work modes, countries, GitHub link) keyed `(user_id, profile_id)`; the *active* profile lives on the session row, so different browsers can sit on different profiles.
- **First-run bootstrap:** Use `python dashboard.py --create-owner`, or register on loopback with `JOBSCOPE_LOCAL_MODE=1`. The owner inherits legacy `profiles.json` data and uses a verified account session in `/admin`.

---

## Configuration Reference

| Variable | Meaning | Default |
|---|---|---|
| `OLLAMA_URL` | Ollama endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name | `llama3.1` |
| `OLLAMA_NUM_CTX` | Context window (tokens) | `8192` |
| `AGENT_SUMMARY_MODE` | Final answer: `template` (instant, in-code) \| `llm` | `template` |
| `RESUME_EXTRACT_MODE` | Skill extraction: `strict` (CV-only keywords) \| `auto` \| `fast` \| `llm` | `strict` |
| `TARGET_ROLE` | Default target role | Junior Data Analyst… |
| `JOB_DAYS_BACK` | Email search window (days) | `60` |
| `JOB_MAX_RESULTS` | Emails per sender | `10` |
| `GITHUB_TOKEN` / `GITHUB_USERNAME` | GitHub PAT / handle | — |
| `GMAIL_OAUTH_CREDENTIALS` / `GMAIL_OAUTH_TOKEN_FILE` | OAuth client secret / token paths | `client_secret.json` / `gmail_oauth_token.json` |
| `GMAIL_CACHE_DB` | Mail cache location | `gmail_cache.db` |
| `SMTP_HOST/PORT/USER/PASSWORD/FROM/TLS/TIMEOUT` | Outbound mail (verify/reset/welcome) | port 587, TLS on, 15 s |
| `JOBSCOPE_LOCAL_MODE` | Allow local bootstrap and private integrations on loopback | off unless enabled in `.env` |
| `AUTH_DEV_LINKS` | Return unsent verification/reset links in explicit local mode | off |
| `HOST` / `APP_URL` | Bind address / public base URL | `127.0.0.1` / `http://localhost:8000` |
| `SCREEN_TOKEN` | Enables `POST /api/screen` | disabled |
| `JOBSCOPE_MANUAL_WALL` | `auto` (on iff display) \| `on` \| `off` | `auto` |
| `JOBSCOPE_MANUAL_SECONDS` | Manual-solve budget | `240` |
| `JOBSCOPE_OCR` / `_MAX_PAGES` / `_DPI` / `_SECONDS` / `_MIN_CONF` | OCR tier-2 behaviour | `auto` / 4 / 150 / 45 / 0.55 |
| `N8N_URL` / `N8N_API_KEY` / `N8N_OLLAMA_BASE_URL` | Optional n8n workflow hooks | — |

`.env` is never committed — `.env.example` is the template.

---

## Data Model

### `users.db` (auth + app)

| Table | Key | Contents |
|---|---|---|
| `users` | `id` | email (unique, NOCASE), scrypt hash, created/verified timestamps, `is_owner`, nickname, avatar |
| `sessions` | `token_hash` | user FK (CASCADE), created/expires, `active_profile`, `last_seen` |
| `email_tokens` | `token_hash` | user FK, purpose (`verify`/`reset`), created/expires/used |
| `profiles` | `(user_id, id)` | name, created_at, `is_default`, JSON `data` (resume text, skills, roles, cities, modes, countries, GitHub) |
| `rag_documents` | `(user_id, id)` | name, kind (`cv`/other), full text (≤ 400 000 chars), created_at |
| `job_desc_cache` | `url` | text (≤ 6000 chars), source (`http`/`http+browser`/`none`), `fetched_at` — deliberately **not** user-scoped (posting text is public) |
| `candidate_resumes` | `id` | user FK + candidate id, original filename, random `stored_name` under `data/resumes/`, size, extracted text, created_at — one active row per candidate |
| `candidate_tags` | `(user_id, candidate_id, tag)` | recruiter's free-form tags on their own candidates |
| `candidate_notes` | `id` | user FK + candidate id, author display name, note (≤ 5000 chars), UTC created_at |
| `pipeline_entries` | `id` | company FK + job FK + candidate id, stage status, created/updated; unique per `(company, job, candidate)` — hand-added candidates that have no user account |

### `gmail_cache.db`

| Table | Key | Contents |
|---|---|---|
| `mail_messages` | `msg_id` | account, sender, subject, body (≤ 1200 chars), sent_at, raw metadata |
| `mail_listings` | `(msg_id, ordinal)` | extracted title/company/location/salary/link rows |
| `mail_sync_state` | `account` | last sync timestamp, counts, error |

---

## Tutorial Progression (step1–step7)

The `step*.py` files grow the concept from a bare state machine to the full agent:

| Step | File | Teaches |
|---|---|---|
| 1 | `step1_basics.py` | LangGraph fundamentals: `TypedDict` state, three plain nodes, `START`/`END` — no LLM, no keys |
| 2 | `step2_agent_with_tool.py` | Real agent loop: `@tool`, `MessagesState`, `bind_tools`, `ToolNode`, `tools_condition` + Ollama health check |
| 3 | `step3_github_tool.py` | Decorated tools hitting the GitHub REST API with a Bearer PAT |
| 4 | `step4_mcp_client.py` | Moving tools into a standalone MCP server; runtime discovery via `MultiServerMCPClient` |
| 5 | `step5_two_mcp_servers.py` | Two servers at once — local stdio GitHub + Indeed's remote `streamable_http` server (needs OAuth token) |
| 6 | `step6_gmail_source.py` | Gmail as a job source (tutorial path uses IMAP + app password with explicit env forwarding) |
| 7 | `step7_scraper_with_fallback.py` | Thin CLI over the shared `agent.py` graph — deterministic fallback wiring |

---

## Testing

```bash
pytest                          # full suite (network stubbed; ~minutes)
pytest -m "not slow"            # skip real-OCR / live-network tests
pytest tests/test_filters.py -v
pytest tests/test_rag_store.py -v
pytest tests/test_resume.py -v
pytest tests/test_scraper.py -v
pytest tests/test_agent.py -v
```

- `pytest.ini` pins `testpaths = tests` so a bare run never recurses into an optional local `Scrapling-main/` checkout, and registers the `slow` marker.
- `conftest.py` has an autouse fixture that redirects `users.db` and `profiles.json` into a temp directory — **no test ever touches real account data**.
- 22 test modules cover: agent routing, auth, auth routes, boards, dashboard, CV OCR, filters, GitHub + ATS, Gmail index, job descriptions, mailer, manual verify, n8n workflow, RAG routes/resume/store, resume builder, role suggestion, scraper, screen route, session identity.

---

## Monitoring & Logs

Every run and error is appended as JSONL under `logs/`:

- `logs/runs-YYYYMMDD.jsonl` — run id, user, message, skills, model, work modes, tools used, token chars, duration, job counts, status.
- `logs/errors-YYYYMMDD.jsonl` — exceptions with full tracebacks.

Plus structured events for `config_updated`, `resume_upload`, `resume_generated`, `login_failed`, `password_reset`, `reset`, `screen_resume`, and more. The dashboard's History drawer reads the last 7 days (`LOG_SCAN_DAYS`); in-memory run history is capped at 60 per account.

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| **All boards blocked (403/CAPTCHA)** | Expected from datacentre IPs. The router falls back to Gmail automatically; the deck still populates from parsed alert emails. For interactive solving set `JOBSCOPE_MANUAL_WALL=on` and use the Verify button |
| **Gmail sync fails / 403** | Re-run `python gmail_login.py` to refresh OAuth consent; check the `gmail.readonly` scope and that `client_secret.json` exists |
| **Runs are slow** | Pure-CPU Ollama inference. Use a smaller model, raise GPU support in Ollama, or keep `AGENT_SUMMARY_MODE=template` (default) which skips the summariser LLM entirely |
| **`watchfiles` fails to install** | Harmless (no Rust toolchain); uvicorn works without it |
| **Resume preview looks empty** | The CV didn't parse. Check the upload preview text; scanned PDFs need the OCR tier (`pymupdf` + `rapidocr` installed) |
| **Port 8000 busy** | `python dashboard.py --port 9000`, or the app prints a friendly "already serving" probe result |
| **Tests hit my real data** | They can't — the autouse `isolate_personal_data` fixture sandboxes `users.db` / `profiles.json` |

---

## Key Design Decisions

1. **Deterministic routing over LLM prompts.** The boards→Gmail fallback is a *graph edge*, not prose. A router inspects tool output for blocked markers and routes in code — eliminating the #1 failure mode of prompt-driven agents: the LLM misreading instructions under a small context window.
2. **Parallel collection, code-based selection.** A fresh hunt fans out to board scraping *and* Gmail simultaneously (one `AIMessage`, two tool calls); the `prep` node picks the winner in code. This removes the sequential wait and the LLM tool-decision round-trip (the slowest part on local CPU).
3. **Filters live in code, not prompts.** `filters.py` is imported by both the scraper and the agent; the rules cannot drift, and the LLM never decides geography or seniority.
4. **BM25 over embeddings.** At CV scale (hundreds of chunks) Okapi BM25 is competitive with dense retrievers, needs no model download, runs on Python 3.14 where torch wheels are unstable, and every score is explainable.
5. **Resume generation = structure, not invention.** The generator only *reorders* your own bullets and skills to match the JD. It never invents skills, rewrites bullets, or adds content you didn't write — the only defence against a resume that lies.
6. **Fetch once, query locally.** Gmail is synced into SQLite instead of re-searching the API per query — faster, immune to sender-address drift, and kinder to rate limits.
7. **Cheap-first, escalate-on-failure scraping.** TLS impersonation (~1 s) before a headless browser (~15–45 s), with a hard global deadline so a stuck board can never hang a run.
8. **Honest CAPTCHA handling.** Rather than pretending to solve captchas, the system opens a visible browser for the *human* and reuses the cleared session via a persistent profile.
9. **Explainable scoring.** A score that cannot be argued with is worse than no score — every ATS component returns specific, actionable reasons.
10. **Local-first privacy.** LLM inference, mail cache, resume data, and logs all stay on the machine; the only network calls are to the job boards, Gmail/GitHub APIs, and (optionally) SMTP.

---

## Known Limitations & Roadmap

### Known limitations

| Area | Status |
|---|---|
| Indeed / LinkedIn / Naukri / Glassdoor / Foundit | Frequently bot-wall plain HTTP clients; Naukri/Foundit need the browser path and Naukri usually wants a signed-in session (opt-in via `board=`) |
| Pagination | Scrapers return page 1 only |
| Gmail queries | Cache-based search is term-matching, not semantic; Google Careers alerts are loose matches |
| `mcp_server_gmail_oauth_fixed.py` | Legacy superseded variant; contains a SyntaxError and is intentionally not imported |
| PDF text layer | Pipes/tabs from LaTeX tables are flattened; glued camelCase words are not repaired |
| Embeddings | Not used (BM25 only) — could be added for semantic matching at scale |
| Multi-language | English-only tokenisation/stemming |
| Chat UI | Chainlit path is single-turn per message (each message re-runs with system context) |
| Remote Indeed MCP | `INDEED_ACCESS_TOKEN` remote server (step5) needs its own OAuth flow and is not wired into the agent |

### Roadmap

- ✅ Graphical dashboard; shared `agent.py`; resume → skills → role suggestions; remote/outside-India filtering; multi-board search; flashcard deck with Apply + LaTeX buttons; run/error logging with History drawer; auth + profiles; Gmail OAuth cache; RAG document library; ATS scoring; manual wall solving; OCR tier.
- ⬜ Scheduled daily digest (OS scheduler / n8n) with JSON persistence.
- ⬜ Browser-driven scraping as the default for the walled big-6 boards.
- ⬜ Per-job LaTeX → PDF compilation service.
- ⬜ Multi-turn conversation + persisted chat history.
- ⬜ Scraper pagination and more robust card parsing.
- ⬜ Optional dense re-ranking on top of BM25.

---

## Contributing

1. Fork the repo and create a feature branch.
2. Follow the existing style — filters and rules belong in `filters.py`; new job sources belong behind MCP tools.
3. Run `pytest` (and `pytest -m "not slow"` while iterating) — all tests must pass.
4. Keep personal secrets out of commits (`.env`, `client_secret.json`, `gmail_oauth_token.json`, `users.db`, `gmail_cache.db` are git-ignored).
5. Submit a PR with a clear description of the behaviour change.

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — the state machine behind the agent
- [Scrapling](https://github.com/D4Vinci/Scrapling) — TLS impersonation + stealthy browser fallback
- [Ollama](https://ollama.ai) — local LLM inference
- [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) — dashboard backend
- [Chainlit](https://chainlit.io) — chat UI
- [MCP](https://modelcontextprotocol.io) — the tool protocol binding it all together
- [RapidOCR](https://github.com/RapidAI/RapidOCR) / [PyMuPDF](https://pymupdf.readthedocs.io) — the OCR tier
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) — the retrieval function in `rag_store.py`

---

*Related documents:* [`docs/JobScope_Documentation.tex`](docs/JobScope_Documentation.tex) (full technical reference, LaTeX) · [`docs/JobScope_Research.tex`](docs/JobScope_Research.tex) (research paper, LaTeX) · [`docs/JobScope_Update_2026-11.tex`](docs/JobScope_Update_2026-11.tex) (latest update, LaTeX)
