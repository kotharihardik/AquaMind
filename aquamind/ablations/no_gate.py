"""Proposal Section 11 Ablation 4: No Safety Gate (RQ2 Safety).

The safety code gate is disabled: a dig proposal requires neither P(success) >= 0.80 (0.35 as a last resort)
nor a field check that supports the spot (a lit step-test section or a pinpointed leak noise). Only the two-dig limit and the
operator approval token remain.
"""
from __future__ import annotations

from typing import Any, Callable
from ..agent.crew import Crew
from ..methods import ToolboxMethod


def build_no_gate(llm_factory: Callable[[], Any] | None, reader: str = "rules") -> dict[str, ToolboxMethod]:
    if llm_factory is None:
        return {}
    H = {"hypothesis_mode": True, "reply_confirms": False}
    methods = {}
    for name in ("ablation_no_gate", "agentix_crew_no_gate"):
        methods[name] = ToolboxMethod(
            name,
            lambda l: Crew(l),
            reader=reader,
            llm_factory=llm_factory,
            search_kw=H,
            use_planner=False,
            min_checks_close=3,
            use_gate=False,
        )
    return methods
