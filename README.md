# AI Job Market Analysis Crew

A multi-agent system (built with [CrewAI](https://www.crewai.com/)) that automates the kind of exploratory analysis I originally did by hand for my [AI & Data Job Market Tableau dashboard](../ai-data-job-market-analysis) — turning a raw CSV of 1,500 job postings into a written executive summary, with three specialized agents each handling one stage of the pipeline.

**Live dashboard for the same dataset:** [Tableau Public link]
**Portfolio write-up:** [Your portfolio site link]

---

## Why I built this

I'd already analyzed this dataset manually in Tableau. This project asks a different question: *could an agent crew do the first pass of that same analysis — cleaning, analyzing, and writing up findings — with a human reviewing the output rather than producing it from scratch?* It's a small, honest test of where agentic workflows are actually useful right now, not a toy demo.

## Architecture

```
Data Validator Agent  →  Insight Analyst Agent  →  Report Writer Agent
  (checks data           (extracts findings           (writes the final
   quality)                using real computed          executive summary)
                           statistics via tools)
```

Three agents, run sequentially, each with one job:

| Agent | Responsibility | Tools |
|---|---|---|
| **Data Validator** | Checks row/column counts, missing values, duplicates, salary range. Gives a go/no-go before analysis proceeds. | `check_data_quality` |
| **Insight Analyst** | Identifies the 5 most important, evidence-backed patterns across skills, role type, and geography. | `get_skill_insights`, `get_role_comparison`, `get_geo_insights` |
| **Report Writer** | Synthesizes the analyst's findings into a clean, business-readable markdown report. | none — works only from the analyst's output |

## The key design decision: grounded tools, not free-form reasoning

The most important choice in this project isn't the agent roles — it's that **every number in the final report comes from a real pandas computation, not from the LLM's own reasoning.**

Each agent tool (in `tools/data_tools.py`) is a plain Python function that loads the CSV, computes real statistics (average salary by skill, YoY growth, role-type comparisons, geography breakdowns), and returns the result as text. The LLM's job is only ever to *interpret and communicate* these numbers — never to estimate or recall them from context. This matters because LLMs are unreliable at exact arithmetic over large datasets; asking one to "remember" salary figures across 1,500 rows invites hallucination. Grounding every fact in a tool call closes that gap.

This also means the Report Writer agent never touches the raw CSV directly — it only receives the Analyst's already-verified findings via CrewAI's task `context` chaining, so errors can't compound silently between stages.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your free Groq API key (console.groq.com -> API Keys)
```

Place the dataset at `data/ai_jobs_market_2025_2026.csv` (same file used in the [Tableau project](../ai-data-job-market-analysis) — see that repo for the original Kaggle source).

Run:
```bash
python main.py
```

The final report is saved to `output/report.md`, and the full agent transcript (each agent's reasoning and tool calls) prints to the console.

## Using a different LLM provider

This project uses **Groq** by default (free tier, no credit card, plenty fast for a project this size). To switch to OpenAI or Anthropic instead, edit the `llm` definition at the top of `agents.py`:

```python
# OpenAI
llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic
llm = LLM(model="anthropic/claude-sonnet-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
```

Then set the corresponding key in your `.env` file instead of `GROQ_API_KEY`.

## Project structure

```
ai-job-market-crew/
├── agents.py           # Agent role/goal/backstory definitions
├── tasks.py             # Task definitions, chained via context
├── main.py               # Entry point — assembles and runs the crew
├── tools/
│   └── data_tools.py    # Pandas-based tools agents call for real numbers
├── data/
│   └── ai_jobs_market_2025_2026.csv   # (not included — see Setup)
├── output/
│   └── report.md         # Generated on run
├── requirements.txt
└── .env.example
```

## What I'd improve next

- Add a fourth agent (or a human-in-the-loop step) to fact-check the final report's language against the analyst's raw figures before saving
- Swap `Process.sequential` for `Process.hierarchical` with a manager agent, and compare output quality
- Add a lightweight eval: re-run the crew 5x and check how consistent the "5 insights" selection is across runs

## Tools

Python · CrewAI · pandas · OpenAI / Anthropic API
