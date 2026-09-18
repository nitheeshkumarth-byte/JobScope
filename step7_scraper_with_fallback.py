"""
STEP 7 (REVISED): same three MCP servers, but the graph now lives in agent.py.

What changed vs the old step7:
  - The fallback decision (Indeed scraper blocked -> use Gmail alerts) is
    now a DETERMINISTIC graph edge inside agent.py, not a sentence in the
    prompt that the LLM is trusted to follow.
  - The graph wiring is shared via create_agent(); this file is just a CLI
    entry point plus the message-formatting code we always reused.
  - Filtering for senior/seniority is code-level on the fallback path.

Run: python step7_scraper_with_fallback.py
"""

import asyncio

from agent import create_agent, default_task_messages, load_config


async def main() -> None:
    """CLI entry point: load config, create the shared agent (this also
    boots the MCP server subprocesses), run one full job-hunt pass, and
    print the whole conversation transcript to stdout."""
    cfg = load_config()
    bundle = await create_agent(cfg)

    result = await bundle.app.ainvoke({"messages": default_task_messages(cfg)})

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