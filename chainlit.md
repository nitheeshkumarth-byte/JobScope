# Job Hunter Agent

A LangGraph agent that finds **Junior Data Analyst** jobs and matches them to your GitHub projects.

## How it works
1. Tries scraping **Indeed** directly (`in.indeed.com`).
2. If Indeed blocks the scraper, it **automatically falls back** to your **Gmail** job alerts (Google Careers + Indeed alert emails) — handled by a graph edge in `agent.py`, not the prompt.
3. Filters out senior/lead/5+ years listings, then summarizes matches.

## Try it
Ask for jobs in your own words (e.g. *"Find me a Junior Data Analyst role in Hyderabad"*), or use the starters. Each tool call is shown as a step while the agent works.