"""Import the JobScope resume-screening workflow into a local n8n.

Two ways in, chosen automatically:

* `N8N_API_KEY` is set in the environment or `.env` -> the n8n Public API.
  The two credentials are created there (or reused if they already exist),
  their ids are patched into the workflow, then the workflow is created (or
  updated in place when one with the same name exists) and activated. No
  container restart, nothing else on the machine is touched.
* No API key -> the running `n8n` Docker container via the n8n CLI
  (`n8n import:credentials`, `n8n import:workflow`, `n8n update:workflow`).
  CLI imports only become visible after a restart, so this path restarts the
  container and says so loudly.

Either way it prints the webhook URL and a curl example. Secrets are read,
never printed: the screen token ends up inside an n8n Header Auth credential,
not in the workflow JSON.

Usage (from the repository root, with the project interpreter):

    python n8n/import.py            # API path if N8N_API_KEY is set
    python n8n/import.py --dry-run  # prepare and validate only, touch nothing
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_FILE = Path(__file__).resolve().parent / "workflows" / "resume-screening.json"

# Fixed ids so a re-run updates/recognises the same credentials instead of
# leaking a new pair every time. The API caps ids at 16 [a-zA-Z0-9_-] chars.
SCREEN_CRED_ID = "jobscope-token"
OLLAMA_CRED_ID = "jobscope-ollama"
SCREEN_CRED_NAME = "JobScope Screen Token"
OLLAMA_CRED_NAME = "JobScope Ollama"
WORKFLOW_NAME = "JobScope Resume Screening"


def load_env() -> dict[str, str]:
    """Environment plus `.env`, environment variables winning.

    Only KEY=VALUE lines are read; comments and blank lines are skipped, and
    nothing is echoed back out.
    """
    values: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    import os
    values.update(os.environ)
    return values


def api_call(method: str, url: str, key: str, payload: dict | None = None,
             timeout: int = 15) -> requests.Response:
    return requests.request(method, url, json=payload, timeout=timeout,
                            headers={"X-N8N-API-KEY": key,
                                     "Content-Type": "application/json"})


def ensure_credential(base: str, key: str, name: str, cred_type: str,
                      data: dict) -> tuple[str, str] | None:
    """Return (id, name) for an existing or newly created credential."""
    try:
        listed = api_call("GET", f"{base}/api/v1/credentials?limit=250", key)
        listed.raise_for_status()
        for row in listed.json().get("data", []):
            if row.get("name") == name and row.get("type") == cred_type:
                return str(row["id"]), name
    except (requests.RequestException, ValueError, KeyError):
        pass
    try:
        created = api_call("POST", f"{base}/api/v1/credentials", key,
                           {"name": name, "type": cred_type, "data": data})
        created.raise_for_status()
        body = created.json()
        return str(body["id"]), name
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  ! could not create credential {name!r}: "
              f"{getattr(exc, 'response', None) and exc.response.text[:200] or exc}")
        return None


def ollama_base_url(env: dict[str, str]) -> str:
    """Where n8n's container can reach Ollama.

    `.env` stores `http://127.0.0.1:11434`, which is the container itself, so
    the loopback address is swapped for Docker Desktop's host alias.
    """
    override = env.get("N8N_OLLAMA_BASE_URL")
    if override:
        return override
    raw = env.get("OLLAMA_URL", "http://127.0.0.1:11434")
    return raw.replace("127.0.0.1", "host.docker.internal").replace(
        "localhost", "host.docker.internal")


def prepare_workflow(env: dict[str, str],
                     screen_cred: tuple[str, str] | None,
                     ollama_cred: tuple[str, str] | None) -> dict:
    """Patch placeholders (credential ids, model) into a fresh copy."""
    workflow = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
    model = env.get("OLLAMA_MODEL", "llama3.1")
    for node in workflow["nodes"]:
        creds = node.get("credentials") or {}
        if "httpHeaderAuth" in creds and screen_cred:
            creds["httpHeaderAuth"] = {"id": screen_cred[0],
                                       "name": screen_cred[1]}
        if "ollamaApi" in creds and ollama_cred:
            creds["ollamaApi"] = {"id": ollama_cred[0], "name": ollama_cred[1]}
        if node.get("type") == "@n8n/n8n-nodes-langchain.lmChatOllama":
            node.setdefault("parameters", {})["model"] = model
    return workflow


def import_via_api(base: str, key: str, env: dict[str, str],
                   workflow: dict) -> str | None:
    print(f"Using the n8n Public API at {base}")
    screen = ensure_credential(
        base, key, SCREEN_CRED_NAME, "httpHeaderAuth",
        {"name": "X-JobScope-Token", "value": env.get("SCREEN_TOKEN", "")})
    if not env.get("SCREEN_TOKEN"):
        print("  ! SCREEN_TOKEN is not set - create it first, the endpoint "
              "answers 503 without it.")
    ollama = ensure_credential(base, key, OLLAMA_CRED_NAME, "ollamaApi",
                               {"baseUrl": ollama_base_url(env)})
    if screen is None or ollama is None:
        print("  ! credentials could not be created; the workflow will be "
              "imported with placeholder ids (set them in the n8n UI).")
    workflow = prepare_workflow(env, screen, ollama)

    body = {"name": workflow["name"], "nodes": workflow["nodes"],
            "connections": workflow["connections"],
            "settings": workflow.get("settings", {})}
    listed = api_call("GET", f"{base}/api/v1/workflows?limit=250", key)
    listed.raise_for_status()
    wf_id = next((str(row["id"]) for row in listed.json().get("data", [])
                  if row.get("name") == workflow["name"]), None)
    if wf_id:
        updated = api_call("PUT", f"{base}/api/v1/workflows/{wf_id}", key, body)
        updated.raise_for_status()
        print(f"  updated workflow id {wf_id}")
    else:
        created = api_call("POST", f"{base}/api/v1/workflows", key, body)
        created.raise_for_status()
        wf_id = str(created.json()["id"])
        print(f"  created workflow id {wf_id}")

    for path in (f"/api/v1/workflows/{wf_id}/activate",):
        try:
            r = api_call("POST", f"{base}{path}", key)
            if r.status_code < 400:
                print("  activated")
                return wf_id
        except requests.RequestException:
            pass
    print("  ! no activate endpoint on this n8n version - flip the workflow "
          "to Active in the UI, or re-run without N8N_API_KEY to use the "
          "container path (which activates by CLI).")
    return wf_id


def docker_container() -> str | None:
    if not shutil.which("docker"):
        return None
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    for line in out.splitlines():
        name, _, image = line.partition("\t")
        if "n8n" in image.lower() or name.lower() == "n8n":
            return name.strip()
    return None


def run_n8n_cli(container: str, args: list[str],
                local_file: Path | None = None) -> subprocess.CompletedProcess:
    """Run `n8n <args>` in the container, optionally copying a file to /tmp."""
    if local_file is not None:
        container_path = f"/tmp/{local_file.name}"
        copied = subprocess.run(
            ["docker", "cp", str(local_file), f"{container}:{container_path}"],
            capture_output=True, text=True, timeout=60)
        if copied.returncode:
            raise RuntimeError(f"docker cp failed: {copied.stderr.strip()[:200]}")
        args = [*(a for a in args), f"--input={container_path}"]
    return subprocess.run(["docker", "exec", container, "n8n", *args],
                          capture_output=True, text=True, timeout=120)


def import_via_docker(env: dict[str, str], workflow: dict) -> None:
    container = docker_container()
    if not container:
        print("No n8n container found. Import by hand: open the n8n UI -> "
              "Workflows -> Import from File and pick "
              f"{WORKFLOW_FILE}, then create the two credentials listed in "
              "n8n/README.md.")
        return
    print(f"Using the n8n CLI inside container {container!r} "
          "(this path restarts the container so the import becomes visible)")
    workflow = prepare_workflow(env, (SCREEN_CRED_ID, SCREEN_CRED_NAME),
                                (OLLAMA_CRED_ID, OLLAMA_CRED_NAME))
    creds = [
        {"id": SCREEN_CRED_ID, "name": SCREEN_CRED_NAME,
         "type": "httpHeaderAuth",
         "data": {"name": "X-JobScope-Token",
                  "value": env.get("SCREEN_TOKEN", "")}},
        {"id": OLLAMA_CRED_ID, "name": OLLAMA_CRED_NAME,
         "type": "ollamaApi", "data": {"baseUrl": ollama_base_url(env)}},
    ]
    wf_path = Path(tempfile.gettempdir()) / "jobscope-resume-screening.json"
    cr_path = Path(tempfile.gettempdir()) / "jobscope-n8n-credentials.json"
    wf_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    cr_path.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    try:
        res = run_n8n_cli(container, ["import:credentials"], local_file=cr_path)
        if res.returncode:
            print(f"  ! credentials import: {res.stderr.strip()[:300]} "
                  "(an existing pair is fine)")
        res = run_n8n_cli(container, ["import:workflow"], local_file=wf_path)
        if res.returncode:
            print(f"  ! workflow import: {res.stderr.strip()[:300]}")
            return
        print("  imported; activating")
        listed = run_n8n_cli(container, ["list:workflow", "--all"])
        wf_id = None
        for line in listed.stdout.splitlines():
            if WORKFLOW_NAME in line:
                m = re.match(r"\s*(\d+)", line)
                if m:
                    wf_id = m.group(1)
                break
        if wf_id:
            act = run_n8n_cli(container,
                              ["update:workflow", f"--id={wf_id}",
                               "--active=true"])
            if act.returncode:
                print(f"  ! activation: {act.stderr.strip()[:200]}")
            else:
                print(f"  workflow {wf_id} activated")
        else:
            print("  ! could not read the workflow id - activate it in the UI")
        print("  restarting the container ...")
        subprocess.run(["docker", "restart", container], check=True,
                       capture_output=True, timeout=120)
        health = env.get("N8N_URL", "http://localhost:5678").rstrip("/")
        for _ in range(30):
            try:
                if requests.get(f"{health}/healthz", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        print("  n8n is back up")
    finally:
        wf_path.unlink(missing_ok=True)
        cr_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="prepare and validate the workflow, change nothing")
    args = parser.parse_args()

    env = load_env()
    base = env.get("N8N_URL", "http://localhost:5678").rstrip("/")
    key = env.get("N8N_API_KEY", "")

    # Placeholders stay until the chosen path knows the real credential ids.
    workflow = prepare_workflow(env, None, None)
    json.dumps(workflow)  # fails loudly if anything above broke the structure
    model = next(n["parameters"]["model"] for n in workflow["nodes"]
                 if n["type"] == "@n8n/n8n-nodes-langchain.lmChatOllama")
    print(f"Prepared {WORKFLOW_FILE.name}: {len(workflow['nodes'])} nodes, "
          f"model {model!r}")

    if args.dry_run:
        print("dry run - nothing was sent to n8n")
        return 0

    try:
        ping = requests.get(f"{base}/healthz", timeout=5)
        ping.raise_for_status()
    except requests.RequestException as exc:
        print(f"n8n is not reachable at {base} ({exc}). Start it first.")
        return 1

    try:
        if key:
            import_via_api(base, key, env, workflow)
        else:
            print("N8N_API_KEY is not set - falling back to the Docker path "
                  "(add the key to .env for the API path).")
            import_via_docker(env, workflow)
    except requests.RequestException as exc:
        body = getattr(exc, "response", None)
        print(f"API call failed: {exc}"
              + (f" - {body.text[:300]}" if body is not None else ""))
        return 1

    webhook = f"{base}/webhook/resume-screening"
    print("\nNext:")
    print("  1. Start JobScope: python -m uvicorn dashboard:app --port 8000")
    print("  2. Make sure the workflow is Active in the n8n UI and the two "
          "credentials (JobScope Screen Token, JobScope Ollama) are set.")
    print("  3. Screen a resume:")
    print(f'     curl -F resume=@cv.pdf -F jd_text="We need a Python and '
          f'Kubernetes engineer" {webhook}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
