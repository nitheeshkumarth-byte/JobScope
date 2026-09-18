"""
STEP 4: LangGraph consuming tools from a REAL MCP server.

Compare this file to step3_github_tool.py:
  - step3: we hand-wrote @tool functions with the GitHub logic inline
  - step4: the GitHub logic lives in mcp_server_github.py, a standalone
           MCP server. This file just CONNECTS to it and gets tools
           back automatically via MultiServerMCPClient.

The payoff: mcp_server_github.py could be swapped for someone else's
already-built MCP server (Slack, Postgres, filesystem, etc.) and this
file's graph code wouldn't need to change AT ALL — you'd just add
another entry to the `client` config below.

Requires:
  export GITHUB_TOKEN=ghp_...
  export GITHUB_USERNAME=nitheeshkumarth-byte
  Ollama running locally (ollama pull llama3.1)

Run: python step4_mcp_client.py
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
    # This config says: "launch mcp_server_github.py as a subprocess,
    # talk to it over stdio, and treat whatever tools it exposes as
    # available tools." You could add a second server here (e.g. a
    # real Indeed MCP server) and both sets of tools would show up
    # together, automatically namespaced.
    client = MultiServerMCPClient({
        "github": {
            "command": "python",
            "args": ["mcp_server_github.py"],
            "transport": "stdio",
        }
    })

    # This is the key line: tools are DISCOVERED from the server at
    # runtime, not hand-written. If you add a tool to the server file
    # later, it shows up here automatically — no changes needed here.
    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s) from MCP server:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:80]}")

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
            "Look at my GitHub repos, pick the 2 most relevant to a "
            "'Junior Data Analyst' role, and summarize what tech each uses "
            "based on their READMEs."
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
    asyncio.run(main())
