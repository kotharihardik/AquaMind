"""RQ1 one-look methods (proposal Section 10): where would each method send the crew from the first look?

"agentix_search*" is the proposal's component 2 as code (Section 7: a leak at each of the 782 junctions simulated once, every junction scored
against the sensors, plus the report score): the first look of the leak search WITHOUT the LLM agents. It is the RQ1 "our system" because RQ1 asks
what citizen reports add to simulation, and the investigation crew (hypothesis-driven, ADR-016, needs an LLM) is evaluated in RQ2. The 782-junction
table is therefore used here and by sensitivity_match / lila_style (they need it by definition), and by no agentic method.

Modular architecture: delegates to dedicated baseline modules:
  - sensitivity_matching.py
  - report_hotspot.py
  - lila.py
"""
from __future__ import annotations

import numpy as np

from .lila import predict_lila_style
from .report_hotspot import predict_report_hotspot
from .sensitivity_matching import predict_sensitivity_match


def one_look(world, case, reads, method: str) -> tuple[int | None, np.ndarray | None]:
    if method in ("agentix_search", "agentix_no_reports", "agentix_search_brown"):
        s = world.search(use_messages=method != "agentix_no_reports", use_footprint=method == "agentix_search_brown")
        b = s.start(case, reads)
        return b.best_dig()[0], b.post
    if method == "report_hotspot":
        return predict_report_hotspot(world, case, reads)
    if method == "sensitivity_match":
        return predict_sensitivity_match(world, case)
    if method == "lila_style":
        return predict_lila_style(world, case)
    raise ValueError(method)


METHODS = ["agentix_search", "agentix_search_brown", "agentix_no_reports", "sensitivity_match", "report_hotspot", "lila_style"]
