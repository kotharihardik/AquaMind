"""Crew-role ablations (ADR-014/015): one switch per role, so the agentic core can be shown to be indispensable.

Not part of the six proposal ablations (no_reports, no_temp_sensor, no_rag, no_gate, one_shot, no_ai); they answer a different question:
what does each of the four agents add inside the crew?
- no_forensic:      no own questions, no hypotheses, no records, no triangulation; the fixed heuristic names where to look
- no_skeptic:       the Advocate's case is not attacked with the physical data
- no_strategist:    the rule dispatcher picks the field checks instead of the Strategist
- no_auditor:       no last refutation before a dig and no SOP paperwork
- no_sop:           the Auditor does not receive the SOP clauses (sop.md)
- no_triangulation: no cross-witness triangulation over elevation
"""
from __future__ import annotations

from typing import Any, Callable

from ..agent.crew import Crew
from ..methods import ToolboxMethod

ROLE_SWITCHES = {
    "no_forensic": {"forensic": False},
    "no_skeptic": {"skeptic": False},
    "no_strategist": {"strategist": False},
    "no_auditor": {"auditor": False},
    "no_sop": {"sop": False},
    "no_triangulation": {"triangulation": False},
}

ROLE_DESCRIPTIONS = {
    "no_forensic": "No Forensic Advocate: no own questions or hypotheses; the fixed heuristic names where to look",
    "no_skeptic": "No Critical Skeptic: the Advocate's case is not attacked",
    "no_strategist": "No Hydraulic Strategist: the rule dispatcher picks the field checks",
    "no_auditor": "No Safety & SOP Auditor: no last refutation, no SOP paperwork",
    "no_sop": "The Auditor does not receive the SOP clauses",
    "no_triangulation": "No cross-witness triangulation over elevation",
}


def build_roles(llm_factory: Callable[[], Any] | None, reader: str = "rules") -> dict[str, ToolboxMethod]:
    if llm_factory is None:
        return {}
    H = {"hypothesis_mode": True, "reply_confirms": False}
    methods = {}
    for short, switch in ROLE_SWITCHES.items():
        for name in (f"ablation_{short}", f"agentix_crew_{short}"):
            methods[name] = ToolboxMethod(name, lambda l, sw=switch: Crew(l, **sw), reader=reader, llm_factory=llm_factory,
                                          search_kw=H, use_planner=False, min_checks_close=3)
    return methods
