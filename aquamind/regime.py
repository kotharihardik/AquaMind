"""Stress regimes (ADR-012): the SAME benchmark cases replayed under harder conditions. Set with `scripts/run_eval.py --regime a,b`
(environment variable AGENTIX_REGIME). Applied only by the harness (aquamind/eval/runner.py), never visible to a method.

  survey          no complaint and no alarm opened this case: the utility is surveying a zone and does not know whether a leak exists
                  (trigger 'survey', no messages ever, nobody to call back)
  plastic         acoustic tools are poor (plastic pipes, noisy streets): listening hears much less. The agent knows its tools' quality
  plastic_blind   the same, but the agent still believes its listening works as in a metal network (tool model mis-specified)
  drift           one pressure sensor drifts by 1-2 m over the last 3 days (an uncalibrated / failing sensor)
  outage          telemetry has been stale (flat-lined) for the last 36 h on every sensor and flow meter
"""
from __future__ import annotations

import os
import zlib

import numpy as np


def active() -> set[str]:
    return {x for x in os.environ.get("AGENTIX_REGIME", "").split(",") if x}


def apply(case, env, name: str) -> None:
    reg = active()
    if not reg:
        return
    rng = np.random.default_rng(zlib.crc32(f"{name}|regime".encode()))
    if "survey" in reg:
        case.reports, case.trigger, case.trigger_report = [], "survey", None
        env.hid["reports"] = []
    if reg & {"plastic", "plastic_blind"}:
        c = env.cfg["listen"]
        c["p_detect_1lps"], c["p_detect_max"], c["far_penalty"] = 0.12, 0.45, 0.7            # truth: sound travels badly
    t = np.asarray(case.time_s, float)
    if "drift" in reg and case.pressure.shape[1]:
        P = case.pressure.copy()
        j = int(rng.integers(P.shape[1]))
        ramp = np.clip((t - (case.case_open_s - 72 * 3600)) / (72 * 3600), 0, 1) * rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 2.0)
        P[:, j] += ramp
        case.pressure = P
    if "outage" in reg:
        stale = t > case.case_open_s - 36 * 3600
        if stale.any() and (~stale).any():
            i = int(np.where(~stale)[0][-1])
            case.pressure = case.pressure.copy(); case.flow = case.flow.copy(); lvl = np.asarray(case.tank_level).copy()
            case.pressure[stale] = case.pressure[i]; case.flow[stale] = case.flow[i]
            lvl[stale] = lvl[i]
            case.tank_level = lvl


def tools_for_agent() -> None:
    """Called once per process: the agent's own beliefs about its tools follow the regime (except 'plastic_blind')."""
    if "plastic" in active():
        from .search import tool_models
        tool_models.LISTEN_P1, tool_models.LISTEN_PMAX = 0.15, 0.5
