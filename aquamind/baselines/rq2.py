"""RQ2 multi-turn investigation baselines (proposal Section 10).

Modular architecture: delegates to dedicated baseline modules:
  - react_agent.py: standard tool-using ReAct LLM agent
  - plain_llm.py: zero-tool one-shot LLM
  - fixed_rules.py: heuristic rule-based investigation agent (no LLM, ADR-011)
  - crew_practice.py: current municipal utility crew practice
"""
from __future__ import annotations

from .crew_practice import CrewPractice
from .fixed_rules import FixedRules
from .plain_llm import PlainLLM
from .react_agent import ReActAgent, resolve_place, format_sensor_summary, STEP_BUDGET


def registry(llm_factory) -> dict:
    """The RQ2 baselines by name.

    Args: llm_factory = makes the LLM client.
    Returns: {'fixed_rules', 'crew_practice', 'plain_llm', 'react_agent'} -> method objects.
    """
    return {
        "fixed_rules": FixedRules(),
        "crew_practice": CrewPractice(),
        "plain_llm": PlainLLM(llm_factory),
        "react_agent": ReActAgent(llm_factory),
    }


__all__ = [
    "ReActAgent",
    "PlainLLM",
    "FixedRules",
    "CrewPractice",
    "registry",
    "resolve_place",
    "format_sensor_summary",
    "STEP_BUDGET",
]
