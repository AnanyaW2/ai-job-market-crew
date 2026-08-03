"""
Custom tools for the AI Job Market Analysis Crew.

Design principle: agents should never be asked to "remember" or estimate
numbers from a 1,500-row dataset out of their own reasoning. Every figure
that ends up in the final report is computed here, in real pandas code,
and handed to the agents as grounded input. The LLM's job is to interpret
and communicate these numbers, not invent them.
"""

import pandas as pd
from crewai.tools import tool

DATA_PATH = "data/ai_jobs_market_2025_2026.csv"


@tool("Dataset Quality Check")
def check_data_quality(note: str = "") -> str:
    """
    Loads the AI job market dataset and returns a data quality report:
    row/column counts, missing values, duplicate rows, and basic dtype info.
    Use this before any analysis to confirm the dataset is safe to use.
    The 'note' parameter is unused — pass an empty string.
    """
    df = pd.read_csv(DATA_PATH)

    report = []
    report.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    report.append(f"Duplicate rows: {df.duplicated().sum()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        report.append("Missing values: none")
    else:
        report.append(f"Missing values by column: {missing.to_dict()}")

    if "annual_salary_usd" in df.columns:
        report.append(
            f"Salary range: ${df['annual_salary_usd'].min():,.0f} - "
            f"${df['annual_salary_usd'].max():,.0f}, "
            f"mean ${df['annual_salary_usd'].mean():,.0f}"
        )

    return "\n".join(report)


@tool("Top Paying and Fastest Growing Skills")
def get_skill_insights(note: str = "") -> str:
    """
    Parses the pipe-delimited required_skills column and returns the top
    10 skills by average salary, and the top 10 by average YoY demand
    growth, each with sample size so low-count skills can be judged
    appropriately. The 'note' parameter is unused — pass an empty string.
    """
    df = pd.read_csv(DATA_PATH)
    exploded = df.assign(skill=df["required_skills"].str.split("|")).explode("skill")
    exploded["skill"] = exploded["skill"].str.strip()

    by_salary = (
        exploded.groupby("skill")
        .agg(avg_salary=("annual_salary_usd", "mean"), count=("skill", "count"))
        .query("count >= 20")
        .sort_values("avg_salary", ascending=False)
        .head(10)
    )

    by_growth = (
        exploded.groupby("skill")
        .agg(avg_growth=("demand_growth_yoy_pct", "mean"), count=("skill", "count"))
        .query("count >= 20")
        .sort_values("avg_growth", ascending=False)
        .head(10)
    )

    lines = ["TOP 10 SKILLS BY AVERAGE SALARY (min. 20 postings):"]
    for skill, row in by_salary.iterrows():
        lines.append(f"  {skill}: ${row['avg_salary']:,.0f} (n={int(row['count'])})")

    lines.append("\nTOP 10 SKILLS BY YOY DEMAND GROWTH (min. 20 postings):")
    for skill, row in by_growth.iterrows():
        lines.append(f"  {skill}: {row['avg_growth']:.1f}% growth (n={int(row['count'])})")

    return "\n".join(lines)


@tool("AI-Native vs Traditional Role Comparison")
def get_role_comparison(note: str = "") -> str:
    """
    Compares average salary for AI-native roles (LLM Engineer, RAG Engineer,
    Prompt Engineer, etc.) versus traditional roles (Data Scientist, ML
    Engineer, etc.) at each experience level. The 'note' parameter is
    unused — pass an empty string.
    """
    df = pd.read_csv(DATA_PATH)
    df["role_type"] = df["is_llm_role"].map({1: "AI-Native", 0: "Traditional"})

    summary = (
        df.groupby(["experience_level", "role_type"])["annual_salary_usd"]
        .mean()
        .round(0)
        .unstack()
    )

    order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
    summary = summary.reindex([lvl for lvl in order if lvl in summary.index])

    lines = ["AVERAGE SALARY BY EXPERIENCE LEVEL AND ROLE TYPE:"]
    for level, row in summary.iterrows():
        ai_val = row.get("AI-Native", float("nan"))
        trad_val = row.get("Traditional", float("nan"))
        gap = (ai_val - trad_val) / trad_val * 100 if pd.notna(ai_val) and pd.notna(trad_val) else None
        gap_str = f", AI-native premium: {gap:.1f}%" if gap is not None else ""
        lines.append(f"  {level}: AI-Native ${ai_val:,.0f} vs Traditional ${trad_val:,.0f}{gap_str}")

    return "\n".join(lines)


@tool("Geography and Remote Work Insights")
def get_geo_insights(note: str = "") -> str:
    """
    Returns average salary by country and the percentage of fully-remote
    postings by country, for the top countries by posting volume. The
    'note' parameter is unused — pass an empty string.
    """
    df = pd.read_csv(DATA_PATH)

    top_countries = df["country"].value_counts().head(8).index.tolist()
    sub = df[df["country"].isin(top_countries)]

    salary_by_country = sub.groupby("country")["annual_salary_usd"].mean().sort_values(ascending=False)

    remote_pct = (
        df[df["remote_work"] == "Fully Remote"]
        .groupby("country")
        .size()
        / df.groupby("country").size()
        * 100
    ).dropna().sort_values(ascending=False)

    lines = ["AVERAGE SALARY BY COUNTRY (top countries by posting volume):"]
    for country, val in salary_by_country.items():
        lines.append(f"  {country}: ${val:,.0f}")

    lines.append("\n% OF POSTINGS THAT ARE FULLY REMOTE, BY COUNTRY:")
    for country, val in remote_pct.head(8).items():
        lines.append(f"  {country}: {val:.1f}%")

    return "\n".join(lines)