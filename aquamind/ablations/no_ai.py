"""Proposal Section 10/11 Ablation 6 / Baseline: Complete System with AI Removed (ADR-011).

Pure deterministic code policy: uses deterministic rules for all operations,
including citizen report extraction (null/heuristic reader), hydraulic heuristics, and dispatch.
"""
from __future__ import annotations

from typing import Any, Callable
from ..agent.code_policy import CodePolicy
from ..methods import ToolboxMethod


def build_no_ai(llm_factory: Callable[[], Any] | None = None, reader: str = "rules") -> dict[str, ToolboxMethod]:
    """Build the no-AI ablation: pure code, no LLM and no reading of message text.

    Args: llm_factory = unused (kept for the same signature); reader = unused (this ablation reads nothing).
    Returns: {method name: ToolboxMethod} for 'ablation_no_ai' and its alias 'agentix_code'.
    """
    H = {"hypothesis_mode": True, "reply_confirms": False, "use_messages": False}
    no_ai = dict(reader="none", search_kw=H, use_planner=False, min_checks_close=3)
    methods = {}
    for name in ("ablation_no_ai", "agentix_code"):
        methods[name] = ToolboxMethod(name, lambda _: CodePolicy(), **no_ai)
    return methods
