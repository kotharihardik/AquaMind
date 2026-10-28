"""Proposal Section 11 Ablation 1: Sensors only (No Citizen Reports).

The citizen text is never read (null reader) AND the leak map ignores messages,
so neither the deterministic code nor the LLM agents see what a complaint says (evaluates RQ1).
"""
from __future__ import annotations

from typing import Any, Callable
from ..agent.crew import Crew
from ..methods import ToolboxMethod


def build_no_reports(llm_factory: Callable[[], Any] | None, reader: str = "rules") -> dict[str, ToolboxMethod]:
    if llm_factory is None:
        return {}
    H = {"hypothesis_mode": True, "reply_confirms": False, "use_messages": False}
    methods = {}
    for name in ("ablation_no_reports", "agentix_crew_no_reports"):
        methods[name] = ToolboxMethod(
            name,
            lambda l: Crew(l),
            reader="none",
            llm_factory=llm_factory,
            search_kw=H,
            use_planner=False,
            min_checks_close=3,
        )
    return methods
