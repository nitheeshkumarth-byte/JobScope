"""
STEP 2: A real agent node with tool-calling.

New concepts vs step1:
  - MessagesState        : the standard LangGraph state shape for chat/agent loops
  - bind_tools()          : gives the LLM a menu of tools it CAN choose to call
  - ToolNode              : a node that actually executes whichever tool the LLM picked
  - tools_condition        : a conditional edge — routes to "tools" if the LLM
                             asked for a tool call, otherwise routes to END
  - the agent <-> tools LOOP: this is what makes it agentic instead of linear

We use ONE dummy tool here (fake job search) so you can see the loop
working before we swap in real MCP tools in step 3.

Requires: Ollama running locally with a tool-calling model pulled.
  Install: https://ollama.com/download
  Then:    ollama pull llama3.1
Run: python step2_agent_with_tool.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ, if present
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.tools import tool


# --- 1. A DUMMY TOOL (standing in for a real MCP tool, e.g. Indeed search) ---
# The @tool decorator turns a normal Python function into something the LLM
# can "see" and choose to call. The docstring is NOT just documentation —
# the LLM reads it to decide when to use this tool.
@tool
def search_jobs(role: str, location: str = "remote") -> str:
    """Search for job openings matching a role and location.
    Returns a short list of fake job postings for demo purposes."""
    print(f"  >> [tool executed] search_jobs(role={role!r}, location={location!r})")
    return (
        f"Found 2 fake postings for '{role}' in {location}:\n"
        f"1. Junior Data Analyst at Acme Corp\n"
        f"2. Data Analyst Intern at Globex Inc"
    )


tools = [search_jobs]

# --- 2. THE LLM, WITH TOOLS BOUND TO IT ---
# bind_tools() doesn't make the LLM call the tool automatically — it just
# tells the model "these functions exist, you can request one if you want".
# The model itself decides, based on the conversation, whether to respond
# directly or ask for a tool call.
llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.1")).bind_tools(tools)


# --- 3. THE AGENT NODE ---
# MessagesState is a built-in state shape: {"messages": [...]}
# Each node appends to that list rather than replacing it.
def agent_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# --- 4. GRAPH WITH A LOOP ---
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

# tools_condition inspects the LLM's last message:
#   - if it contains a tool call -> go to "tools"
#   - otherwise                  -> go to END
graph.add_conditional_edges("agent", tools_condition)

# after the tool runs, go BACK to the agent so it can read the tool's
# result and decide what to do next (answer, or call another tool)
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
        "messages": [
            ("user", "Find me junior data analyst jobs, remote only.")
        ]
    })

    print("\n--- CONVERSATION ---")
    for msg in result["messages"]:
        role = msg.type if hasattr(msg, "type") else "user"
        print(f"[{role}] {msg.content}")
