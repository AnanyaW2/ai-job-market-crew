"""
Agent definitions for the AI Job Market Analysis Crew.

Three agents, each with a narrow, well-defined responsibility:
  1. Data Validator  — checks the dataset is safe to analyze
  2. Insight Analyst  — extracts and interprets the key patterns
  3. Report Writer    — turns findings into a polished executive summary

Keeping agents narrowly scoped (rather than one agent doing everything)
mirrors good software design: each agent is easy to reason about, test,
and swap out independently.

This project uses Groq (free tier, no credit card) as the LLM provider
by default. CrewAI supports Groq natively via the "groq/" model prefix,
using the GROQ_API_KEY environment variable — see .env.example.
"""

import os
from crewai import Agent, LLM
from tools.data_tools import (
    check_data_quality,
    get_skill_insights,
    get_role_comparison,
    get_geo_insights,
)

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

data_validator = Agent(
    role="Data Quality Validator",
    goal=(
        "Confirm the AI job market dataset is clean and safe to analyze. "
        "Flag any missing values, duplicates, or anomalies before the "
        "analysis stage begins."
    ),
    backstory=(
        "You are a meticulous data engineer who has seen too many reports "
        "built on bad data. You never let an analysis proceed until you've "
        "personally verified the dataset's integrity."
    ),
    tools=[check_data_quality],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

insight_analyst = Agent(
    role="Senior Data Analyst",
    goal=(
        "Identify the most important, evidence-backed insights in the AI "
        "job market dataset — covering skills, salary, role type, and "
        "geography — using only the real computed statistics available "
        "through your tools."
    ),
    backstory=(
        "You are a data analyst with a strong instinct for what matters "
        "and what's noise. You never state a number you haven't pulled "
        "from your tools, and you always note sample size when it's "
        "relevant to how much a finding should be trusted."
    ),
    tools=[get_skill_insights, get_role_comparison, get_geo_insights],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

report_writer = Agent(
    role="Report Writer",
    goal=(
        "Turn the analyst's findings into a clear, well-structured "
        "executive summary that a non-technical hiring manager or job "
        "seeker could read in under three minutes."
    ),
    backstory=(
        "You are a business writer who specializes in translating dense "
        "analysis into decision-ready language, without dumbing down or "
        "distorting the underlying numbers."
    ),
    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)
