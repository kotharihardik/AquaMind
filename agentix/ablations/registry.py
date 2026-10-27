"""Centralized Registry for Proposal Section 11 Ablations.

Each ablation is implemented in its own modular file:
- no_reports.py: Sensors only (no citizen complaint reading, RQ1)
- no_temp_sensor.py: No new flow/pressure measurement (night step tests disabled)
- no_rag.py: No SOP rules and no repair/complaint history
- no_gate.py: Safety gate disabled (no P(success) threshold, RQ2 safety)
- one_shot.py: No investigation loop (one-shot Advocate evaluation)
- no_ai.py: Complete system with all AI removed (pure deterministic code, ADR-017)
- roles.py: one switch per crew role (Forensic, Skeptic, Strategist, Auditor, SOP, triangulation); category "role_ablation"
"""
from __future__ import annotations

from typing import Any, Callable

from .no_reports import build_no_reports
from .no_temp_sensor import build_no_temp_sensor
from .no_rag import build_no_rag
from .no_gate import build_no_gate
from .one_shot import build_one_shot, OneShotBrain
from .no_ai import build_no_ai
from .roles import ROLE_DESCRIPTIONS, build_roles

_ALL_K = ["spread_k3", "spread_k5", "spread_k10"]


def _meta(short: str, desc: str, section: str, *, llm: bool = True, cat: str = "proposal_ablation", ks=None) -> dict:
    return {
        "category": cat,
        "description": desc,
        "needs_llm": llm,
        "proposal_section": section,
        "sensor_configs": ks or ["spread_k5"],
        "alias": short,
    }


ABLATION_METADATA = {
    "ablation_no_reports": _meta(
        "agentix_crew_no_reports",
        "Sensors only: citizen text is never read by code or agents (Proposal Section 11 - RQ1)",
        "11 (Ablation: No citizen reports)",
        ks=_ALL_K,
    ),
    "ablation_no_temp_sensor": _meta(
        "agentix_crew_no_temp_sensor",
        "No new flow/pressure measurement: night step tests disabled (the crew's temporary-measurement tool); listening stays",
        "11 (Ablation: No temporary-sensor tool)",
    ),
    "ablation_no_rag": _meta(
        "agentix_crew_no_rag",
        "RAG disabled: no utility rules (SOP) and no complaint/repair history",
        "11 (Ablation: No RAG)",
    ),
    "ablation_no_gate": _meta(
        "agentix_crew_no_gate",
        "Safety code gate disabled: a dig needs neither P(success) nor field support (operator token stays; RQ2 safety)",
        "11 (Ablation: No safety gate)",
    ),
    "ablation_one_shot": _meta(
        "agentix_crew_one_shot",
        "One-shot: one Advocate look, then dig the best spot if the leak is believed, else close; no investigation loop",
        "11 (Ablation: No step-by-step reasoning)",
    ),
    "ablation_no_ai": _meta(
        "agentix_code",
        "Complete system with all AI removed, including language understanding (pure code, ADR-017)",
        "10/11 (No-AI Baseline / Ablation, ADR-017)",
        llm=False,
        ks=_ALL_K,
    ),
}

for _short, _desc in ROLE_DESCRIPTIONS.items():
    ABLATION_METADATA[f"ablation_{_short}"] = _meta(f"agentix_crew_{_short}", _desc, "ADR-015 (crew role)", cat="role_ablation")


def get_ablation_registry(llm_factory: Callable[[], Any] | None = None, reader: str = "rules") -> dict[str, Any]:
    """Instantiate and return all ablation methods from modular files.
    Every ablation is registered under both its `ablation_*` and its `agentix_crew_*` alias.
    """
    out: dict[str, Any] = {}
    # Non-LLM ablations
    out.update(build_no_ai(llm_factory, reader=reader))

    # LLM ablations
    if llm_factory is not None:
        out.update(build_no_reports(llm_factory, reader=reader))
        out.update(build_no_temp_sensor(llm_factory, reader=reader))
        out.update(build_no_rag(llm_factory, reader=reader))
        out.update(build_no_gate(llm_factory, reader=reader))
        out.update(build_one_shot(llm_factory, reader=reader))
        out.update(build_roles(llm_factory, reader=reader))

    return out


def is_llm_ablation(name: str) -> bool:
    """Return True if the ablation requires an LLM."""
    meta = ABLATION_METADATA.get(name)
    if meta:
        return meta["needs_llm"]
    for k, v in ABLATION_METADATA.items():
        if v.get("alias") == name:
            return v["needs_llm"]
    return True
