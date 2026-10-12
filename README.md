# JobScope — AI-Powered Job Hunting Assistant

An intelligent job-hunting platform that searches multiple job boards, parses Gmail job alerts, generates tailored LaTeX resumes, and provides a real-time dashboard — all running locally with Ollama LLMs.

## 🎯 What It Does

**JobScope** automates the job search pipeline for technical candidates:

| Capability | Description |
|------------|-------------|
| **Multi-board search** | Queries 9 job boards (Indeed, LinkedIn, Naukri, Glassdoor, Foundit, Internshala, WeWorkRemotely, Remotive, Arbeitnow) in parallel |
| **Smart fallback** | When boards are blocked, automatically falls back to Gmail job-alert emails (Indeed, LinkedIn, Google Careers, WeWorkRemotely) |
| **Deterministic routing** | Graph-based edge routing (not LLM prompts) decides between board results and Gmail fallback |
| **ATS-friendly resume generation** | Builds job-tailored LaTeX resumes from your CV + job posting, compiles on Overleaf/plain pdflatex |
| **BM25 retrieval** | Ranks your own CV bullets against the job description — no hallucination, only your words |
| **Role suggestions** | Infers target roles from your uploaded CV's skills (60+ roles across data, backend, DevOps, mobile, security, etc.) |
| **Real-time dashboard** | FastAPI + SSE streaming UI with flashcard job cards, profile management, and run history |

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                                   │
├──────────────────────────┬──────────────────────────────────────────────┤
│  Chainlit Chat (ui.py)   │  FastAPI Dashboard (dashboard.py)            │
│  - Streaming tokens      │  - SSE event stream                          │
│  - Tool step rendering   │  - Flashcard deck with Apply buttons         │
│  - Starter prompts       │  - Profile/session management                │
└───────────┬──────────────┴────────────────┬────────────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT CORE (agent.py)                                │
│  LangGraph state machine with deterministic routing:                    │
│  ┌──────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐              │
│  │  agent   │───▶│ tools   │───▶│ router  │───▶│ prep/  │              │
│  │  node    │    │ (MCP)   │    │ (code)  │    │ fallback│              │
│  └──────────┘    └─────────┘    └─────────┘    └────────┘              │
│       ▲                                              │                  │
│       └──────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
            │                               │
            ▼                               ▼
┌────────────────────────┐    ┌──────────────────────────────────────┐
│  MCP SERVERS (stdio)   │    │         RESUME PIPELINE              │
├────────────────────────┤    ├──────────────────────────────────────┤
│ • mcp_server_github    │    │ 1. Parse uploaded CV into sections   │
│ • mcp_server_gmail     │    │ 2. BM25 rank bullets vs JD (rag_store)│
│ • mcp_server_indeed    │    │ 3. Reorder skills/experience         │
│    (wraps boards.py)   │    │ 4. Inject into ATS LaTeX template    │
│                        │    │ 5. Output .tex + HTML preview        │
└────────────────────────┘    └──────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) running locally with a model (default: `llama3.1`)
- Gmail app passwords for job-alert fallback (optional but recommended)
- GitHub token for repo matching (optional)

### Installation
```bash
git clone <this-repo>
cd jobscope
pip install -r requirements.txt
cp .env.example .env  # configure your keys
```

### Environment Variables (`.env`)
```bash
# LLM
OLLAMA_MODEL=llama3.1
OLLAMA_NUM_CTX=8192

# Job search
TARGET_ROLE="Junior Data Analyst / entry-level, 0-2 years experience"
JOB_DAYS_BACK=60
JOB_MAX_RESULTS=10

# Gmail (up to 3 accounts)
GMAIL_ADDRESS=you@gmail.com
GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx
GMAIL_SENDERS=donotreply@jobalert.indeed.com,jobs-noreply@linkedin.com
# GMAIL_ADDRESS_2, GMAIL_APP_PASSWORD_2, GMAIL_SENDERS_2 for second account

# GitHub (optional, for repo matching)
GITHUB_TOKEN=ghp_xxxxxxxxxxxx

# Manual bot-wall solving (optional)
JOBSCOPE_MANUAL_WALL=on
```

### Run the Dashboard (Recommended)
```bash
uvicorn dashboard:app --reload --port 8000
# Open http://localhost:8000
```

### Run the Chainlit Chat UI
```bash
chainlit run ui.py
```

### Run a One-off CLI Hunt
```bash
python step7_scraper_with_fallback.py "Find Junior Python Developer jobs remote"
```

---

## 📁 Project Structure

```
jobscope/
├── agent.py                    # LangGraph agent (core orchestration)
├── dashboard.py                # FastAPI backend + static UI
├── ui.py                       # Chainlit chat interface
├── filters.py                  # Shared job-listing rules (seniority, India, work modes, regions, countries)
├── resume_generator.py         # LaTeX resume builder + HTML preview
├── resume_data.py              # Canonical resume content (fallback when no CV uploaded)
├── rag_store.py                # BM25 document library + retrieval
├── boards.py                   # Scrapling-based board parsers (HTTP impersonation + browser fallback)
├── mcp_server_indeed_scraper.py # MCP tool: search_job_boards
├── mcp_server_gmail.py         # MCP tool: search_job_emails (IMAP + local cache)
├── mcp_server_github.py        # MCP tool: fetch_user_repos + match against JD
├── auth.py                     # SQLite auth + profiles + sessions
├── ats_score.py                # ATS scoring against a job description
├── github_projects.py          # GitHub repo README matching
├── job_desc.py                 # Job description fetch + parsing
├── cv_ocr.py                   # PDF/DOCX text extraction
├── mailer.py                   # Email digest sender
├── tests/                      # pytest suite (filters, scraper, rag, resume, etc.)
├── static/                     # Dashboard frontend (index.html, app.js, styles.css)
├── logs/                       # JSONL run/error logs (auto-created)
└── profiles.json               # Legacy profile store (migrated to SQLite on first login)
```

---

## 🔧 Key Design Decisions

### 1. Deterministic Routing Over LLM Prompts
The fallback from job boards to Gmail is a **graph edge**, not a prompt instruction. A router function inspects tool output for block markers (`"all boards were blocked"`, `"captcha"`, `"http 403"`, etc.) and routes to the fallback node. This eliminates the #1 failure mode of the old design: the LLM misreading prose instructions.

### 2. Parallel Collection, Code-Based Selection
On a fresh hunt, the agent fans out to **board scraper AND Gmail alerts simultaneously** (one AIMessage with two tool_calls). A `prep` node then picks the winner in code: boards if they returned listings, else Gmail. This avoids the sequential boards→fallback wait and the LLM tool-decision round-trip (slowest part on local CPU).

### 3. Filters Live in Code, Not Prompts
`filters.py` is the single source of truth for:
- Seniority/experience filtering (title + location only, never company name)
- India-tied location dropping (exempting genuine WFH)
- Work arrangement modes (remote / wfh / hybrid / onsite)
- Region and country filters
Both the board scraper and the agent's post-filter import from here — they cannot drift apart.

### 4. BM25 Over Embeddings
`rag_store.py` uses Okapi BM25 for retrieval over your uploaded documents. At CV-scale (hundreds of chunks), BM25 is competitive with dense retrievers, needs no model download, works on Python 3.14 where torch wheels are unstable, and every score is explainable ("this chunk contains all three JD terms").

### 5. Resume Generation = Structure, Not Invention
`resume_generator.py` parses your uploaded CV into sections and **only reorders** bullets/skills to match the job description. It never invents skills, rewrites bullets, or adds content you didn't write. The LaTeX template is ATS-friendly: 10pt article, standard `\section*` headings, single column, no tables.

### 6. Per-Account "Already Seen" Memory
The board scraper remembers which links each account has already been shown (`seen_scope="u{user_id}"`). This prevents repeat hunts from returning the same top-20 cards. Buckets are bounded (400 entries) and LRU-evicted per account.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Specific suites
pytest tests/test_filters.py -v
pytest tests/test_rag_store.py -v
pytest tests/test_resume.py -v
pytest tests/test_scraper.py -v
```

---

## 📊 Monitoring & Logs

Every run and error is appended as JSONL under `logs/`:
- `logs/runs-YYYYMMDD.jsonl` — successful runs with timings, sources, result counts
- `logs/errors-YYYYMMDD.jsonl` — exceptions with tracebacks

The dashboard's History drawer reads the last 7 days by default (`LOG_SCAN_DAYS`).

---

## 🛣 Roadmap / Known Limitations

| Area | Status |
|------|--------|
| Naukri / Foundit / Glassdoor | Require browser (Chromium via patchright); opt-in via `board=` param |
| LinkedIn notifications | Filtered out (noise); only `jobs-noreply@linkedin.com` and `jobalerts-noreply@linkedin.com` are synced |
| PDF upload text layer | Pipes/tabs from LaTeX tables are flattened; camelCase glued words not repaired |
| Embeddings | Not used (BM25 only); could be added for semantic matching at scale |
| Multi-language | English-only tokenization/stemming |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Run `pytest` and ensure all tests pass
4. Submit a PR with a clear description

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the state machine
- [Scrapling](https://github.com/D4Vinci/Scrapling) for TLS impersonation + browser fallback
- [Ollama](https://ollama.ai) for local LLM inference
- [FastAPI](https://fastapi.tiangolo.com) for the dashboard backend
- [Chainlit](https://chainlit.io) for the chat UI