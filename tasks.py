"""
Task definitions for the AI Job Market Analysis Crew.

Tasks run sequentially: validation -> analysis -> report writing, with
each task's output passed as context into the next (via CrewAI's
`context` parameter). This means the Report Writer never sees the raw
data directly — it only ever works from the Analyst's already-verified
findings, which keeps the final report grounded.
"""

from crewai import Task
from agents import data_validator, insight_analyst, report_writer

validate_task = Task(
    description=(
        "Run a full data quality check on the AI job market dataset. "
        "Report the row/column counts, any missing values, duplicate "
        "rows, and the overall salary range. Explicitly state whether "
        "the dataset is safe to proceed with analysis."
    ),
    expected_output=(
        "A short data quality report (5-8 lines) covering row/column "
        "counts, missing values, duplicates, salary range, and a clear "
        "go/no-go statement on proceeding to analysis."
    ),
    agent=data_validator,
)

analyze_task = Task(
    description=(
        "Using the validated dataset, identify the 5 most important, "
        "evidence-backed insights across four areas: (1) which skills "
        "command the highest salaries and fastest growth, (2) how "
        "AI-native roles compare to traditional roles by experience "
        "level, (3) which countries pay best and offer the most remote "
        "work, and (4) any single most surprising or non-obvious pattern "
        "you found. Use your tools for every number you cite — do not "
        "estimate or guess."
    ),
    expected_output=(
        "A numbered list of exactly 5 insights, each 2-3 sentences, each "
        "citing specific figures (dollar amounts, percentages, sample "
        "sizes) pulled directly from your tools."
    ),
    agent=insight_analyst,
    context=[validate_task],
)

report_task = Task(
    description=(
        "Write a polished executive summary report based on the "
        "analyst's 5 insights. Structure it with a one-paragraph "
        "overview, then a clearly headed section per insight, and a "
        "short closing paragraph on what this means for someone job "
        "hunting in AI/data roles right now. Write in professional but "
        "accessible business language — no jargon without explanation. "
        "Format the whole thing in clean markdown."
    ),
    expected_output=(
        "A complete markdown report, 500-700 words, with a title, "
        "overview paragraph, headed sections per insight, and a closing "
        "'What This Means For Your Job Search' section."
    ),
    agent=report_writer,
    context=[analyze_task],
    output_file="output/report.md",
)
