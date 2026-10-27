"""Ablations: System and tool ablations from Proposal Section 11."""
from .registry import ABLATION_METADATA, get_ablation_registry, is_llm_ablation
from .no_reports import build_no_reports
from .no_temp_sensor import build_no_temp_sensor
from .no_rag import build_no_rag
from .no_gate import build_no_gate
from .one_shot import build_one_shot, OneShotBrain
from .no_ai import build_no_ai
from .roles import build_roles

__all__ = [
    "ABLATION_METADATA",
    "get_ablation_registry",
    "is_llm_ablation",
    "build_no_reports",
    "build_no_temp_sensor",
    "build_no_rag",
    "build_no_gate",
    "build_one_shot",
    "OneShotBrain",
    "build_no_ai",
    "build_roles",
]

