"""
STEP 6 (REVISED): Gmail as a job-listing SOURCE, via IMAP + app password.

No OAuth needed anymore — mcp_server_gmail.py connects with your
Gmail address + app password over IMAP, which is a much lower-setup
path than the Google Cloud OAuth flow we originally sketched. This
now follows the EXACT same pattern as GitHub in step 4: a local
stdio MCP server, launched as a subprocess, tools discovered
automatically.

This is based on REAL data pulled from a real inbox during earlier
testing — not a hypothetical:
  - Real sender: careers-noreply@google.com (Google Careers digest)
  - Real structure: raw HTML body, job titles in <h5><a> tags,
    location/recency in the following <span>, requirements in <ul>
  - Real problem: Google's own alert matching is loose — a saved
    search for "Information Technology Apprenticeship" returned
    senior roles like "Technical Lead Manager" needing 8+ years.
    So the parsing node must FILTER by actual fit, not just extract
    everything present.

Env vars required (.env, never in chat):
  GMAIL_ADDRESS=youraddress@gmail.com
  GMAIL_APP_PASSWORD=your16charapppassword

Also requires: Ollama running locally (ollama pull llama3.1)

Run: python step6_gmail_source.py
"""

import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # IMPORTANT: MCP's stdio transport does NOT inherit your full
    # environment into the subprocess by default — only a safe minimal
    # set (PATH, HOME, etc.). Custom vars like GMAIL_ADDRESS or
    # GITHUB_TOKEN must be passed explicitly via "env" below, or the
    # subprocess won't see them even though load_dotenv() loaded them
    # into THIS process's os.environ.
    client = MultiServerMCPClient({
        "github": {
            "command": "python",
            "args": ["mcp_server_github.py"],
            "transport": "stdio",
            "env": {
                "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
                "GITHUB_USERNAME": os.environ.get("GITHUB_USERNAME", ""),
            },
        },
        "gmail": {
            "command": "python",
            "args": ["mcp_server_gmail.py"],
            "transport": "stdio",
            "env": {
                "GMAIL_ADDRESS": os.environ.get("GMAIL_ADDRESS", ""),
                "GMAIL_APP_PASSWORD": os.environ.get("GMAIL_APP_PASSWORD", ""),
            },
        },
    })

    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s):")
    for t in tools:
        print(f"  - {t.name}")

    # num_ctx: Ollama's default context window (often 2048-4096) is too
    # small for a conversation with multiple tool calls returning real
    # email data. Too small a context silently drops content, which is
    # a likely cause of hallucinated answers — the model "fills in" what
    # it can no longer see. 8192 gives real headroom for this workload.
    llm = ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        num_ctx=8192,
    ).bind_tools(tools)

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

    # The prompt does the "parsing" instruction inline — telling the
    # agent the real email structures and how to filter noise, based
    # on what we found testing this live against a real inbox.
    #
    # Two sources now: Google Careers digest emails (known HTML
    # structure, documented below) and Indeed alert emails from
    # donotreply@match.indeed.com (structure not pre-documented here —
    # the agent inspects the raw HTML itself and extracts what it can,
    # same tool, different sender, called twice).
    target_role = "Junior Data Analyst / entry-level, 0-2 years experience"
    result = await app.ainvoke({
        "messages": [(
            "user",
            "Call search_job_emails TWICE: once with sender "
            "'careers-noreply@google.com', once with sender "
            "'donotreply@match.indeed.com', both for the last 60 days. "
            "\n\nFor the Google Careers results: each email body is raw "
            "HTML with job listings inside <h5><a href='...'>Title</a></h5> "
            "tags, followed by a location/recency span and a <ul> of "
            "requirements. "
            "\n\nFor the Indeed results: inspect the raw HTML yourself "
            "and extract job title, company, location, and application "
            "link as best you can — the structure may differ from "
            "Google's. "
            "\n\nFrom BOTH sources combined, keep ONLY listings matching: "
            f"'{target_role}' — discard anything requiring 5+ years "
            "experience or a senior/lead/manager title. For each match, "
            "tell me: source (Google Careers or Indeed), job title, "
            "company/location, and the application link."
        )]
    })

    print("\n--- CONVERSATION ---")
    for msg in result["messages"]:
        role = msg.type if hasattr(msg, "type") else "user"
        print(f"[{role}] {msg.content}\n")


if __name__ == "__main__":
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
    except Exception:
        print("Ollama doesn't seem to be running. Install it (https://ollama.com/download), "
              "then run: ollama pull llama3.1")
        raise SystemExit(1)
    if not os.environ.get("GMAIL_ADDRESS") or not os.environ.get("GMAIL_APP_PASSWORD"):
        print("Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD first (in .env).")
        raise SystemExit(1)
    asyncio.run(main())
