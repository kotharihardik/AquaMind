"""Baselines: RQ1 one-look methods and RQ2 agents / rules / current practice (proposal Section 10)."""
from .registry import BASELINE_METADATA, get_baseline_registry, is_llm_baseline

__all__ = ["BASELINE_METADATA", "get_baseline_registry", "is_llm_baseline"]
