"""
STEP 1: LangGraph basics — no LLM, no MCP, no API key needed.

Goal: understand the three core building blocks before we add any
real intelligence or tools:
  1. State   — a shared dict that flows through the graph
  2. Nodes   — plain Python functions that read/update state
  3. Edges   — the wiring that decides what runs next

Run this file directly: python step1_basics.py
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. STATE
# This is the schema of data that gets passed between nodes.
# Every node receives the current state and returns a dict of
# updates to merge into it.
class JobAgentState(TypedDict):
    job_title: str
    step_log: list[str]   # we'll append to this so we can see the flow


# 2. NODES
# Each node is just a function: (state) -> dict of updates.
# No LLM here yet — we're faking the "search" and "tailor" steps
# so you can see the graph mechanics in isolation.

def search_jobs_node(state: JobAgentState) -> dict:
    print(f"[search_jobs] looking for: {state['job_title']}")
    return {
        "step_log": state["step_log"] + ["searched for jobs"]
    }


def tailor_resume_node(state: JobAgentState) -> dict:
    print(f"[tailor_resume] tailoring resume for: {state['job_title']}")
    return {
        "step_log": state["step_log"] + ["tailored resume"]
    }


def review_node(state: JobAgentState) -> dict:
    print(f"[review] steps completed so far: {state['step_log']}")
    return {
        "step_log": state["step_log"] + ["reviewed"]
    }


# 3. GRAPH — wire the nodes together
graph = StateGraph(JobAgentState)

graph.add_node("search_jobs", search_jobs_node)
graph.add_node("tailor_resume", tailor_resume_node)
graph.add_node("review", review_node)

# START and END are special markers LangGraph provides.
graph.add_edge(START, "search_jobs")
graph.add_edge("search_jobs", "tailor_resume")
graph.add_edge("tailor_resume", "review")
graph.add_edge("review", END)

app = graph.compile()


if __name__ == "__main__":
    initial_state = {"job_title": "Junior Data Analyst", "step_log": []}
    final_state = app.invoke(initial_state)

    print("\n--- FINAL STATE ---")
    print(final_state)
