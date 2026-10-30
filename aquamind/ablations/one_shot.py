"""Proposal Section 11 Ablation 5: One-Shot Reasoning (No Step-by-Step Investigation Loop).

The Advocate inspects the board once, asks its callers questions, and formulates hypotheses.
The case is then answered immediately from initial evidence: dig the best spot if P(leak) >= 0.5,
else close the case. Safety gate is off because collecting field evidence requires an investigation loop.
"""
from __future__ import annotations

from typing import Any, Callable
from ..methods import ToolboxMethod


class OneShotBrain:
    """No step-by-step investigation (proposal Section 11). The Advocate looks once: it reads the board, asks its callers its own questions and names
    its hypotheses (one LLM call). The case is then answered at once, from what is known: dig the best spot if a leak is believed (P(leak) >= 0.5,
    code's number), else close. It runs with the gate off: the field checks that open the gate are what a loop collects."""
    BELIEF = 0.5

    def __init__(self, llm=None):
        from ..agent.crew import Crew
        self.crew = Crew(llm, skeptic=False, strategist=False, auditor=False, sop=False)

    @property
    def turns(self):
        return self.crew.turns

    def run(self, tb):
        self.crew.forensic(tb, "one shot: read the evidence, ask the people, name your hypotheses; the case is answered right after this")
        j, p = tb.board.best_dig()
        self.crew.turns.append({
            "turn": len(self.crew.turns),
            "role": "one_shot",
            "spot": tb.describe_node(j),
            "p_success": round(p, 2),
            "p_leak": round(tb.board.p_leak, 2),
        })
        if tb.board.p_leak >= self.BELIEF:
            tb.propose_dig("one-shot answer")
        if not (tb.done or tb.env.done):
            tb.close("hand_over" if tb.board.p_leak >= self.BELIEF else "no_leak", "one-shot answer", force=True)


def build_one_shot(llm_factory: Callable[[], Any] | None, reader: str = "rules") -> dict[str, ToolboxMethod]:
    if llm_factory is None:
        return {}
    H = {"hypothesis_mode": True, "reply_confirms": False}
    methods = {}
    for name in ("ablation_one_shot", "agentix_crew_one_shot"):
        methods[name] = ToolboxMethod(
            name,
            lambda l: OneShotBrain(l),
            reader=reader,
            llm_factory=llm_factory,
            search_kw=H,
            use_planner=False,
            min_checks_close=3,
            use_gate=False,
        )
    return methods
