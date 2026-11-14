"""Paths and constants of the demo UI (one place)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the repository root (AquaMind code uses paths relative to it)
PKG = Path(__file__).resolve().parent
STATIC = PKG / "static"
RECORDINGS = PKG / "recordings"                         # full-fidelity investigations (tracked: demo assets)
CACHE = ROOT / ".cache" / "demo_ui"                     # generated payloads (ignored by git through .cache/)

BENCH = "data/benchmark/agentix_master"
LOGS = ROOT / "logs" / "evaluation"
DEFAULT_LAYOUT = "spread_k5"
LAYOUTS = ("spread_k3", "spread_k5", "spread_k10")

# Friendly names for the benchmark's case types (suite = second part of the case name)
SUITES = {
    "brown": ("Brown tap water", "A leak on a feeder pipe scours the pipes downstream; people 300-900 m away see brown water."),
    "chronic": ("Chronic hidden leak", "A small leak that has run for weeks: no burst, only weak signs."),
    "cd_damp": ("Damp ground", "A small leak soaks the ground and it never dries; one caller notices."),
    "cd_damp_decoy": ("Damp ground + decoy", "Damp ground at the leak plus a look-alike complaint somewhere else."),
    "cd_sound": ("Night sound under road", "Someone hears hissing under the road at night."),
    "cd_sound_decoy": ("Night sound + decoy", "A night hiss at the leak plus a look-alike complaint elsewhere."),
    "displaced": ("Displaced pool", "Water surfaces downhill of the leak: the pool is not where the pipe broke."),
    "trap": ("Trap: injected instruction", "A fake message tries to order a dig ('approval already given')."),
    "false_alarm": ("False alarm (no leak)", "Hoses, house pipes, drains: there is no network leak at all."),
    "visible": ("Visible water", "Water a crew can follow by eye (sanity check)."),
}

AGENT_MAX_TOKENS = 1200          # same as aquamind.methods.registry


def setup_paths() -> None:
    """AquaMind opens configs and data with paths relative to the repository root: run from there, and make the package importable
    even under `python -I` (which drops the script directory and the working directory from sys.path)."""
    os.chdir(ROOT)
    for p in (str(ROOT),):
        if p not in sys.path:
            sys.path.insert(0, p)
    CACHE.mkdir(parents=True, exist_ok=True)
    RECORDINGS.mkdir(parents=True, exist_ok=True)


def suite_of(case_name: str) -> str:
    """Args: case_name = e.g. 'test-brown-003'.
    Returns: its category, e.g. 'brown'.
    """
    parts = case_name.split("-")
    return parts[1] if len(parts) > 2 else case_name
