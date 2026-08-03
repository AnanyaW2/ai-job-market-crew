"""
AI Job Market Analysis Crew — entry point.

Run with:  python main.py

Requires an API key for whichever LLM provider you configure (Groq by
default, free tier) set as an environment variable — see README.md and
.env.example for setup.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Workaround for a known CrewAI bug (github.com/crewAIInc/crewAI/issues/5886):
# CrewAI injects an Anthropic-only "cache_breakpoint" property into messages
# for all providers, which Groq's API rejects. This neutralizes that
# injection so non-Anthropic providers like Groq work correctly.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Crew, Process
from agents import data_validator, insight_analyst, report_writer
from tasks import validate_task, analyze_task, report_task


def main():
    os.makedirs("output", exist_ok=True)

    crew = Crew(
        agents=[data_validator, insight_analyst, report_writer],
        tasks=[validate_task, analyze_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("CREW FINISHED. Final report saved to output/report.md")
    print("=" * 60 + "\n")
    print(result)


if __name__ == "__main__":
    main()
