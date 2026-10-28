"""Proposal Section 11 Ablation 2: No Temporary Sensor Tool.

The agent cannot collect new flow/pressure data. In AgentiX the crew's new-measurement tool
is the night step test (the temporary logger is a baseline-only tool); this ablation removes step tests while acoustic listening stays.
"""
from __future__ import annotations

from typing import Any, Callable
from ..agent.crew import Crew
from ..methods import ToolboxMethod


def build_no_temp_sensor(llm_factory: Callable[[], Any] | None, reader: str = "rules") -> dict[str, ToolboxMethod]:
    if llm_factory is None:
        return {}
    H = {"hypothesis_mode": True, "reply_confirms": False}
    methods = {}
    for name in ("ablation_no_temp_sensor", "agentix_crew_no_temp_sensor"):
        methods[name] = ToolboxMethod(
            name,
            lambda l: Crew(l),
            reader=reader,
            llm_factory=llm_factory,
            search_kw=H,
            use_planner=False,
            min_checks_close=3,
            allow_step=False,
        )
    return methods
