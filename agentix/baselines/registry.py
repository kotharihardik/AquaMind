"""Centralized Registry for all Evaluation Baselines.

Categorized into:
1. PROPOSAL BASELINES (Proposal Section 10):
   - lila_style: LILA (Daniel et al. 2022), two-stage re-implementation: sensor-to-sensor regression, then model-based localisation (sensors only, RQ1).
   - sensitivity_match: Classic sensitivity correlation without messages (sensor-only, RQ1).
   - report_hotspot: Spatial centroid of readable citizen messages (no physics, RQ1).
   - react_agent: Standard tool-using ReAct LLM agent (same tools, step budget 15, raw text, no hypothesis board, no code gate, RQ2).
   - plain_llm: Zero-tool one-shot LLM (sensor summary + raw messages as text, answers once: dig or no dig, RQ2).
   - fixed_rules: Heuristic rule-based investigation agent (no LLM, tests what LLM adds, ADR-017, RQ2).

2. SUPPLEMENTARY / REAL-WORLD BASELINES (Not strictly in proposal core table, clearly separated):
   - classical_direct_inversion / classical_bruteforce: Real-world EPANET brute-force sensitivity inversion across 782 junctions with uncalibrated roughness and demand noise.
   - crew_practice: Current municipal utility crew practice (listen where complaints point, walk outward).
"""
from __future__ import annotations

from typing import Any, Callable

# Metadata mapping for all baselines
BASELINE_METADATA = {
    # ------------------ Core Proposal Baselines (Section 10) ------------------
    "sensitivity_match": {
        "category": "proposal_baseline_rq1",
        "description": "Correlate pressure drop vector with precomputed leak fingerprints (sensors only, no reports)",
        "needs_llm": False,
        "in_proposal": True,
        "proposal_section": "10 (RQ1)",
        "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
    },
    "report_hotspot": {
        "category": "proposal_baseline_rq1",
        "description": "Report hotspot / spatial centroid of citizen reports (no hydraulics)",
        "needs_llm": False,
        "in_proposal": True,
        "proposal_section": "10 (RQ1)",
        "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
    },
    "react_agent": {
        "category": "proposal_baseline_rq2",
        "description": "Standard ReAct tool-using LLM agent (same tools, step budget 15, no hypothesis board, no code gate)",
        "needs_llm": True,
        "in_proposal": True,
        "proposal_section": "10 (RQ2)",
        "sensor_configs": ["spread_k5"],
    },
    "plain_llm": {
        "category": "proposal_baseline_rq2",
        "description": "Plain LLM without tools: reads sensor summary + messages as text, decides dig/no-dig in one shot",
        "needs_llm": True,
        "in_proposal": True,
        "proposal_section": "10 (RQ2)",
        "sensor_configs": ["spread_k5"],
    },
    "fixed_rules": {
        "category": "proposal_baseline_rq2",
        "description": "Fixed heuristic rules: place one temporary sensor at max disagreement, then dig best candidate (no LLM)",
        "needs_llm": False,
        "in_proposal": True,
        "proposal_section": "10 (RQ2)",
        "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
    },

    # ------------------ Supplementary Baselines ------------------
    "classical_bruteforce": {
        "category": "supplementary_baseline",
        "description": "Practical model-based brute force: EPANET sensitivity inversion across 782 junctions with uncalibrated roughness and demand noise, AWWA M36 acoustic dispatch, zero valve cuts (ADR-022)",
        "needs_llm": False,
        "in_proposal": False,
        "proposal_section": "Supplementary (Practical Academic Inversion / ADR-022)",
        "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
    },
    "classical_direct_inversion": {
        "category": "supplementary_baseline",
        "description": "Alias for classical_bruteforce (practical sensitivity inversion)",
        "needs_llm": False,
        "in_proposal": False,
        "proposal_section": "Supplementary (Practical Academic Inversion / ADR-022)",
        "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
    },
    "lila_style": {
        "category": "proposal_baseline_rq1",
        "description": "LILA (Daniel et al. 2022) re-implemented in two stages: sensor-to-sensor regression residuals, then model-based localisation over all junctions (sensors only)",
        "needs_llm": False,
        "in_proposal": True,
        "proposal_section": "10 (RQ1)",
        "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
    },
    "crew_practice": {
        "category": "supplementary_baseline",
        "description": "Municipal crews' current practice: acoustic survey at citizen complaints, walk outward",
        "needs_llm": False,
        "in_proposal": False,
        "proposal_section": "Supplementary (Utility Status Quo)",
        "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
    },
}


def get_baseline_registry(llm_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Instantiate and return all baseline methods that can run cases."""
    from .bruteforce import ClassicalBruteforce, ClassicalDirectInversion
    from .crew_practice import CrewPractice
    from .fixed_rules import FixedRules
    from .plain_llm import PlainLLM
    from .react_agent import ReActAgent

    baselines: dict[str, Any] = {
        "classical_direct_inversion": ClassicalDirectInversion(),
        "classical_bruteforce": ClassicalBruteforce(),
        "fixed_rules": FixedRules(),
        "crew_practice": CrewPractice(),
    }
    if llm_factory is not None:
        baselines["plain_llm"] = PlainLLM(llm_factory)
        baselines["react_agent"] = ReActAgent(llm_factory)
    return baselines


def is_llm_baseline(name: str) -> bool:
    """Return True if the baseline requires an LLM."""
    meta = BASELINE_METADATA.get(name)
    return meta["needs_llm"] if meta else False
