"""
ui.py — Chainlit web UI for the shared job-hunting agent (agent.py).

Run:    chainlit run ui.py
(uses the same .env as everything else)

The graph runs with astream(stream_mode="messages") so you can *watch*
the agent work: each LLM token streams into the chat, tool calls from the
MCP servers appear as steps, and a "fallback" step shows when the Indeed
scraper was blocked and the run switched to Gmail alerts.

Configuration (target role, model, etc.) lives in .env / agent.py —
this file stays minimal by design.
"""

import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from agent import create_agent, default_task_messages, load_config

TOOL_OUTPUT_PREVIEW = 600


@cl.set_starters
async def starters():
    """One-click example prompts shown in the Chainlit launcher
    (decorated by @cl.set_starters so they render as clickable chips)."""
    return [
        cl.Starter(
            label="Default job hunt",
            message="Find me Junior Data Analyst jobs in Hyderabad.",
        ),
        cl.Starter(
            label="Jobs + best GitHub match",
            message=(
                "Find Junior Data Analyst jobs, then tell me which two of "
                "my GitHub repos best match what these jobs ask for."
            ),
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    """New session: build the shared agent (spawns both MCP servers) and
    stash it plus the parsed config on the Chainlit user session."""
    config = load_config()
    bundle = await create_agent(config)
    cl.user_session.set("bundle", bundle)
    cl.user_session.set("config", config)
    await cl.Message(
        content=f"Ready. Target: **{config.target_role}**\n\n"
        "Try a starter below, or just ask for jobs in your own words."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Main chat handler: stream the LangGraph run token-by-token into a
    Chainlit message, render MCP tool results as collapsible steps, and
    surface a dedicated step whenever the deterministic fallback fires."""
    bundle = cl.user_session.get("bundle")
    config = cl.user_session.get("config")
    if bundle is None:
        await cl.Message(content="Agent not ready — refresh the page.").send()
        return

    # same system prompt + target role that the CLI uses
    system = default_task_messages(config)[0]
    user_task = SystemMessage(content=message.content)
    payload = {"messages": [system, user_task]}

    root_step = cl.Step(name="job-agent", type="run")
    await root_step.start()

    answer = cl.Message(content="")
    await answer.send()

    try:
        async for chunk, metadata in bundle.app.astream(
            payload,
            stream_mode="messages",
            config={"recursion_limit": 60},
        ):
            node = metadata.get("langgraph_node", "")

            # LLM output tokens -> stream into the chat message
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                await answer.stream_token(chunk.content)

            # Tool execution result -> show as a collapsible step
            elif isinstance(chunk, ToolMessage):
                step = cl.Step(
                    name=f"tool: {chunk.name or 'mcp'}",
                    type="tool",
                    parent_id=root_step.id,
                )
                await step.start()
                step.output = str(chunk.content)[:TOOL_OUTPUT_PREVIEW]
                await step.end()

            # Deterministic source-selection fired (parallel collect): boards
            # won OR the scraper was blocked and Gmail alerts were used in
            # place. Shows _one_ step so the user sees the decision.
            elif isinstance(chunk, HumanMessage) and node in ("prep", "fallback"):
                used_gmail = not str(chunk.content).startswith(
                    "Job boards responded with listings")
                step = cl.Step(name="source", type="run", parent_id=root_step.id)
                await step.start()
                step.output = (
                    "Boards were blocked -> pulled job alerts from Gmail "
                    "instead (parallel fallback)."
                    if used_gmail else
                    "Job boards responded -> summarizing their listings."
                )
                await step.end()
    finally:
        await root_step.end()

    await answer.update()