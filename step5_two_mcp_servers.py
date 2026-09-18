"""
STEP 5: Adding Indeed as a SECOND MCP server alongside GitHub.

IMPORTANT — read this before the code:

The Indeed MCP server (mcp.indeed.com) is a REMOTE server, not a local
subprocess like our GitHub one. Remote MCP servers over HTTP/SSE almost
always require OAuth: your script needs to open a browser, let you log
into Indeed, and receive back an access token before it can call any
tool. That handshake is genuinely more setup than a stdio server —
it's not just a config change.

What Claude.ai's "connector" does for you: it already ran that OAuth
flow and holds the resulting token on your behalf, tied to your Claude
account. That's WHY I could call Indeed's search_jobs directly in our
chat above without you doing anything — but that token is scoped to
Claude.ai's connector session, not something your standalone Python
script can reuse.

To use Indeed's real job-search API from your own script, you have
two realistic paths:
  1. Implement the OAuth flow yourself (register an OAuth client with
     Indeed if they support third-party registration, handle the
     browser redirect + token exchange) — this is real engineering
     work, separate from LangGraph/MCP itself.
  2. Check if Indeed offers a simpler API key-based option for
     developers (their publisher/partner APIs have historically been
     invite-only — worth checking their current developer docs before
     assuming this path is open).

Below is the CODE SHAPE for once you have a working token — notice
it's almost identical to the GitHub client, just a different
transport and auth header. This is the point: MCP standardizes the
CLIENT side. The auth story is still on you, per-server.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ, if present
import asyncio
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient({
        # our existing local GitHub server — unchanged
        "github": {
            "command": "python",
            "args": ["mcp_server_github.py"],
            "transport": "stdio",
        },
        # Indeed — REMOTE server, needs a real OAuth access token.
        # This entry will only work once INDEED_ACCESS_TOKEN is a
        # valid token you obtained yourself (see notes above).
        "indeed": {
            "url": "https://mcp.indeed.com/claude/mcp",
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {os.environ.get('INDEED_ACCESS_TOKEN', '')}"
            },
        },
    })

    # tools from BOTH servers come back in one flat list — the agent
    # doesn't know or care which server a tool came from
    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s) total:")
    for t in tools:
        print(f"  - {t.name}")

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

    result = await app.ainvoke({
        "messages": [(
            "user",
            "Search for Junior Data Analyst jobs in Hyderabad, then look at "
            "my GitHub repos and tell me which 2 projects best match what "
            "these jobs are asking for."
        )]
    })

    print("\n--- CONVERSATION ---")
    for msg in result["messages"]:
        role = msg.type if hasattr(msg, "type") else "user"
        print(f"[{role}] {msg.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
