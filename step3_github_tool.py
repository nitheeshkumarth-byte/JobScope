"""
STEP 3: A real tool — fetch your GitHub repos + READMEs.

Why a token: GitHub allows only 60 unauthenticated requests/hour per IP.
With a token (even a basic one, no special scopes needed for public repo
reads) that jumps to 5,000/hour. Get one at:
https://github.com/settings/tokens -> "Generate new token (classic)"
-> no scopes needed if you're only reading public repos.

Requires:
  export GITHUB_TOKEN=ghp_...
  export GITHUB_USERNAME=nitheeshkumarth-byte
  Ollama running locally (ollama pull llama3.1)

Run: python step3_github_tool.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ, if present
import requests
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "nitheeshkumarth-byte")


def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# --- REAL TOOL 1: list repos ---
@tool
def list_github_repos() -> str:
    """List the user's public GitHub repositories, most recently updated first.
    Returns repo name, primary language, and last updated date for each."""
    url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    resp = requests.get(url, headers=_gh_headers(), params={"per_page": 100, "sort": "updated"})
    resp.raise_for_status()
    repos = resp.json()

    non_fork = [r for r in repos if not r["fork"]]
    lines = [
        f"- {r['name']} (language: {r.get('language') or 'unknown'}, "
        f"updated: {r['updated_at'][:10]})"
        for r in non_fork
    ]
    return "\n".join(lines) if lines else "No public non-fork repos found."


# --- REAL TOOL 2: get a specific repo's README ---
@tool
def get_repo_readme(repo_name: str) -> str:
    """Fetch the README content for a specific GitHub repo by name.
    Use this after list_github_repos to inspect a repo's tech stack and description."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/readme"
    resp = requests.get(url, headers=_gh_headers())
    if resp.status_code == 404:
        return f"No README found for repo '{repo_name}'."
    resp.raise_for_status()
    import base64
    content = base64.b64decode(resp.json()["content"]).decode("utf-8", errors="ignore")
    return content[:3000]  # cap length so we don't blow the context window


tools = [list_github_repos, get_repo_readme]
llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.1")).bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()


if __name__ == "__main__":
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
    except Exception:
        print("Ollama doesn't seem to be running. Install it (https://ollama.com/download), "
              "then run: ollama pull llama3.1")
        raise SystemExit(1)

    result = app.invoke({
        "messages": [(
            "user",
            "Look at my GitHub repos, pick the 2 most relevant to a "
            "'Junior Data Analyst' role, and summarize what tech each uses "
            "based on their READMEs."
        )]
    })

    print("\n--- CONVERSATION ---")
    for msg in result["messages"]:
        role = msg.type if hasattr(msg, "type") else "user"
        print(f"[{role}] {msg.content}\n")
