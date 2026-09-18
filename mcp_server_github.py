"""
MCP SERVER: exposes GitHub repo/README tools over the MCP protocol.

This is the same logic as step3's @tool functions, but now packaged
as a standalone MCP server. The key difference: this file doesn't
know or care that LangGraph will be the client. Any MCP-compatible
client (LangGraph, Claude Desktop, another agent framework) could
talk to this exact same server with zero changes here.

This runs over stdio (the client launches it as a subprocess and
talks to it over stdin/stdout) — the simplest MCP transport, good
for local dev. Later, if you deployed this remotely, you'd switch
to the SSE/HTTP transport instead, and only the transport line
would change, not the tool definitions.

You do NOT run this file directly in normal use — the MCP client
(step4) launches it automatically as a subprocess.
"""

import os
import base64
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("github-tools")

GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "nitheeshkumarth-byte")


def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# The @mcp.tool() decorator is MCP's equivalent of LangChain's @tool —
# same idea (docstring = what the LLM reads to decide when to call it),
# different protocol underneath.
@mcp.tool()
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


@mcp.tool()
def get_repo_readme(repo_name: str) -> str:
    """Fetch the README content for a specific GitHub repo by name.
    Use this after list_github_repos to inspect a repo's tech stack and description."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/readme"
    resp = requests.get(url, headers=_gh_headers())
    if resp.status_code == 404:
        return f"No README found for repo '{repo_name}'."
    resp.raise_for_status()
    content = base64.b64decode(resp.json()["content"]).decode("utf-8", errors="ignore")
    return content[:3000]


if __name__ == "__main__":
    mcp.run(transport="stdio")
