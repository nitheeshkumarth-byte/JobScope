"""
MCP SERVER: exposes GitHub repo/README tools over the MCP protocol.

This is the same logic as step3's @tool functions, but now packaged
as a standalone MCP server. The key difference: this file doesn't
know or care that LangGraph will be the client. Any MCP-compatible
client (LangGraph, Claude Desktop, another agent framework) could
talk to this exact same server with zero changes here.

This runs over stdio (the client launches it as a subprocess and
talks to it over stdin/stdout) - the simplest MCP transport, good
for local dev. Later, if you deployed this remotely, you'd switch
to the SSE/HTTP transport instead, and only the transport line
would change, not the tool definitions.

You do NOT run this file directly in normal use - the MCP client
(step4) launches it automatically as a subprocess.

There is no configured user here. The repository owner is an ARGUMENT,
supplied from the GitHub link on the candidate's own CV, because a
one-process deployment serves many accounts and a hardcoded handle served
one of them to everybody. GITHUB_TOKEN is optional: public profiles work
without it, and setting it only raises the hourly rate limit.
"""

import os
import base64
import requests
from mcp.server.fastmcp import FastMCP

import github_projects

mcp = FastMCP("github-tools")


def _gh_headers() -> dict:
    """Public repos need no credentials. A token, when present, is only a
    rate-limit upgrade, so an unset or blank one is not an error."""
    headers = {"Accept": "application/vnd.github+json"}
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _handle(value: str) -> str:
    """Pull a username out of whatever the caller passed.

    Accepts a bare handle, an @handle, or any GitHub profile/URL text, so
    the caller can hand over the CV line verbatim instead of parsing it.
    Returns "" when the text carries no usable handle - which is a normal
    outcome for a CV without a GitHub link, not an error.
    """
    return github_projects.github_username(value or "")


@mcp.tool()
def list_github_repos(github_url_or_username: str) -> str:
    """List the PUBLIC repositories of the GitHub profile named in the CV.

    Pass the GitHub link exactly as it appears on the CV (or a bare handle).
    Forks and archived repositories are omitted: they are somebody else's
    work. Returns repo name, language, and last update date per line.
    """
    user = _handle(github_url_or_username)
    if not user:
        return ("No GitHub profile found in that text. Pass the GitHub link "
                "from the CV, e.g. https://github.com/some-user.")
    url = f"https://api.github.com/users/{user}/repos"
    resp = requests.get(url, headers=_gh_headers(),
                        params={"per_page": 100, "sort": "updated"}, timeout=20)
    resp.raise_for_status()
    repos = [r for r in resp.json() if not r.get("fork")]
    lines = [
        f"- {r['name']} (language: {r.get('language') or 'unknown'}, "
        f"updated: {r['updated_at'][:10]})"
        for r in repos
    ]
    return "\n".join(lines) if lines else f"No public non-fork repos on github.com/{user}."


@mcp.tool()
def get_repo_readme(github_url_or_username: str, repo_name: str) -> str:
    """Fetch the README of one repository for stack and purpose.

    Call after list_github_repos. Returns the README text, capped, so a
    generated summary is based on what the repository actually says.
    """
    user = _handle(github_url_or_username)
    if not user:
        return ("No GitHub profile found in that text. Pass the GitHub link "
                "from the CV, e.g. https://github.com/some-user.")
    url = f"https://api.github.com/repos/{user}/{repo_name}/readme"
    resp = requests.get(url, headers=_gh_headers(), timeout=20)
    if resp.status_code == 404:
        return f"No README found for repo '{repo_name}'."
    resp.raise_for_status()
    content = base64.b64decode(resp.json()["content"]).decode("utf-8", errors="ignore")
    return content[:3000]


if __name__ == "__main__":
    mcp.run(transport="stdio")
