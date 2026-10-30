"""Proposal Section 11 Ablation 3: No RAG (Utility Rules & History Removed).

No utility SOP rules and no repair or complaint history are retrieved into the agent prompts.
"""
from __future__ import annotations

from typing import Any, Callable
from ..agent.crew import Crew
from ..methods import ToolboxMethod


def build_no_rag(llm_factory: Callable[[], Any] | None, reader: str = "rules") -> dict[str, ToolboxMethod]:
    if llm_factory is None:
        return {}
    H = {"hypothesis_mode": True, "reply_confirms": False}
    methods = {}
    for name in ("ablation_no_rag", "agentix_crew_no_rag"):
        methods[name] = ToolboxMethod(
            name,
            lambda l: Crew(l, sop=False),
            reader=reader,
            llm_factory=llm_factory,
            search_kw=H,
            use_planner=False,
            min_checks_close=3,
            use_rag=False,
        )
    return methods
