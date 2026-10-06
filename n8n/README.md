# n8n resume screening

An n8n workflow that turns JobScope's resume scorer into an endpoint other
tools can call: upload a resume + paste a job description, get back an
explainable match score with an AI-written Skills / Experience / Missing /
Verdict summary.

```
Webhook → Pack request → Screen resume (HTTP) → Screening ok?
        ├─ yes → AI Agent (Ollama) → Shape analysis → Respond analysis 200
        └─ no  → Screening failed → Respond failed (4xx/5xx)
```

Two rules the workflow is built around:

- **The score is always computed by JobScope.** The HTTP node calls
  `POST /api/screen`, which runs `ats_score.score_resume` — the same call the
  dashboard's resume modal makes. The AI Agent receives that JSON and only
  explains it; it may not recompute, contradict or invent numbers.
- **The shared secret lives in a credential, not in the workflow JSON.**
  The token travels as the `X-JobScope-Token` header, stored in an n8n
  Header Auth credential that only someone with n8n access can read.

## What you need

| Thing | Why |
|---|---|
| JobScope running locally (`python -m uvicorn dashboard:app --port 8000`) | the workflow posts to `http://host.docker.internal:8000/api/screen` — and because that address is the *host's* IP, JobScope must bind beyond loopback: set `HOST=0.0.0.0` in `.env` (the dashboard still sits behind its login; `/api/screen` still needs the token) |
| `SCREEN_TOKEN` in `.env` | the endpoint answers `503` for everyone while it is unset, so a fresh checkout never exposes an open scanner |
| n8n (Docker is fine) | the workflow itself |
| Ollama + a model (default `llama3.1`) | the AI Agent's chat model |

## Import it

```powershell
# With N8N_API_KEY in .env (or the environment) - creates the two
# credentials, imports and activates the workflow, no restart:
.\.venv\Scripts\python.exe n8n\import.py

# Validate the prepared workflow without touching n8n:
.\.venv\Scripts\python.exe n8n\import.py --dry-run

# Without an API key - imports through the n8n CLI in the running
# container and RESTARTS it (CLI imports are only visible after a restart):
$env:N8N_API_KEY = ""  # or just leave it out of .env
.\.venv\Scripts\python.exe n8n\import.py
```

The script reads `.env` (`SCREEN_TOKEN`, `N8N_URL`, `N8N_API_KEY`,
`OLLAMA_MODEL`, `OLLAMA_URL`) and never prints any of the secret values.

### The two credentials

`n8n/import.py` creates both; if you import by hand, create them in the n8n
UI and pick them on the two nodes that show a red triangle:

| Credential | Type | Value |
|---|---|---|
| `JobScope Screen Token` | Header Auth | Name `X-JobScope-Token`, Value = `SCREEN_TOKEN` from `.env` |
| `JobScope Ollama` | Ollama | Base URL `http://host.docker.internal:11434` |

`OLLAMA_URL` in `.env` is `http://127.0.0.1:11434`, which points at the
container itself once n8n runs in Docker — that is why the import script
rewrites it to `host.docker.internal` (override with `N8N_OLLAMA_BASE_URL` if
you run n8n elsewhere).

## Call it

With the workflow **Active**:

```powershell
curl -F resume=@cv.pdf `
     -F "jd_text=We need a Python and Kubernetes engineer" `
     http://localhost:5678/webhook/resume-screening
```

- `resume` — the file (pdf/txt/md; scanned PDFs go through OCR like the rest
  of the app, so first response can take a few seconds)
- `jd_text` — pasted job description (wins over `job_url`)
- `job_url` — posting link instead: read from the 24-hour description cache;
  on a cold cache the answer says `jd_origin: "pending"` and you re-post

Response:

```json
{
  "ok": true,
  "ats": { "total": 62.4, "band": "fair", "breakdown": [...],
           "hits": ["python"], "missing": ["kubernetes"],
           "issues": [...], "keyword_total": 40 },
  "jd_origin": "pasted", "jd_chars": 71, "resume_chars": 1420,
  "ocr": {},
  "analysis": { "skills": "...", "experience": "...",
                "missing": "...", "verdict": "..." },
  "note": "ats.total was computed by JobScope ats_score.score_resume; the AI Agent only explains it."
}
```

Failures come back as `{ "ok": false, "error": "..." }` with the status the
endpoint used (`401` bad token, `413` too large, `422` no readable text,
`503` no `SCREEN_TOKEN` configured).

### Calling `/api/screen` directly (without n8n)

The endpoint accepts two shapes — useful for scripts:

```powershell
# multipart, like a browser form
curl -H "X-JobScope-Token: $env:SCREEN_TOKEN" -F resume=@cv.pdf `
     -F "jd_text=..." http://127.0.0.1:8000/api/screen

# application/json with the file base64 (what the n8n HTTP node sends)
curl -H "X-JobScope-Token: $env:SCREEN_TOKEN" -H "Content-Type: application/json" `
     -d '{\"resume_base64\":\"...\",\"filename\":\"cv.pdf\",\"jd_text\":\"...\"}' `
     http://127.0.0.1:8000/api/screen
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `503 SCREEN_TOKEN is not configured` | `SCREEN_TOKEN` missing from `.env` — restart JobScope after adding it |
| `401 missing or invalid X-JobScope-Token` | the Header Auth credential value does not match `.env` |
| Workflow error "connect ECONNREFUSED host.docker.internal:8000" | JobScope not running, or started with `--host 127.0.0.1` behind a firewall that blocks the container (bind with `HOST=0.0.0.0` for Docker Desktop) |
| "model ... not found" from Ollama | `ollama pull llama3.1` (or change the model on the *Ollama Chat Model* node / `OLLAMA_MODEL` in `.env`) |
| Agent replies with one blob instead of four lines | lower temperature is already set; nudge the system message, and `Shape analysis` still falls back to the raw text under `analysis.verdict` |
| Webhook says "workflow not found"/404 | the workflow is not Active — flip it on in the n8n UI |

Files: `workflows/resume-screening.json` (the workflow),
`import.py` (the importer). Tests: `tests/test_n8n_workflow.py`.
