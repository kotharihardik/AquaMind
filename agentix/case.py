"""The ONLY way a method reads a case: public data until the case opened, restricted to one sensor layout.

Hidden files (truth) are read only by the environment (agentix.sim.field.CaseEnv) and the scorer.
Never show internal case names or paths to an LLM: use the opaque public `case_id`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class PublicCase:
    case_id: str                # opaque public id
    name: str                   # internal folder name (harness use only, never shown to an LLM)
    trigger: str                # citizen_message | scada_alarm
    start_datetime: str
    case_open_s: float
    horizon_s: float
    history_days: int
    time_s: np.ndarray
    pressure: np.ndarray        # [T, k] layout sensors only
    pressure_nodes: list[str]
    flow: np.ndarray            # [T, 3] m3/h: two zone inlets + tank pump
    tank_level: np.ndarray
    reports: list[dict]         # raw messages (UNTRUSTED text)
    trigger_report: str | None = None      # the message the control room opened this case on (None: SCADA alarm)
    works: list | None = None              # the utility's own works log up to the opening (hydrant flushing, valve operations)


def load_public_case(bench_dir: str, name: str, layout: list[str], text_variant: str = "realistic") -> PublicCase:
    d = f"{bench_dir}/cases/{name}"
    case = json.load(open(f"{d}/case.json"))
    if text_variant == "realistic" and os.path.exists(f"{d}/messages_realistic.json"):
        alt = json.load(open(f"{d}/messages_realistic.json"))
        case["reports"] = [{**r, "text": alt.get(r["report_id"], r["text"])} for r in case["reports"]]
    obs = np.load(f"{d}/obs.npz")
    nodes = list(obs["pressure_nodes"])
    cols = [nodes.index(n) for n in layout]
    return PublicCase(case_id=case["case_id"], name=name, trigger=case["trigger"], start_datetime=case["start_datetime"],
                      case_open_s=case["case_open_s"], horizon_s=case["horizon_s"], history_days=case["history_days"],
                      time_s=obs["time_s"], pressure=obs["pressure"][:, cols], pressure_nodes=list(layout),
                      flow=obs["flow"], tank_level=obs["tank_level"], reports=case["reports"],
                      trigger_report=case.get("trigger_report"), works=case.get("works") or [])


def load_layouts(bench_dir: str) -> dict[str, list[str]]:
    return json.load(open(f"{bench_dir}/layouts.json"))
