"""Hydraulic truth: insert a leak into the truth model and simulate it with EPANET (via WNTR).

Leaks are inserted the way BattLeDIM did it: split the pipe and put the leak on a new node.
  abrupt    -> EPANET emitter q = Cd * A * sqrt(2 g h) (pressure-dependent orifice), switched on at the start time
  incipient -> a demand that ramps up to the orifice flow at the local pressure
The truth model is never the inference model (no inverse crime): methods use the nominal L-TOWN.inp.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass

import numpy as np
import wntr

G, CD = 9.81, 0.75


@dataclass
class LeakSpec:
    pipe: str
    frac: float            # position along the pipe (0..1)
    diameter_m: float      # orifice diameter
    kind: str              # abrupt | incipient
    start_s: int
    ramp_s: int = 0        # incipient: time to reach full size

    def to_dict(self):
        return asdict(self)


def orifice_coeff(diameter_m: float) -> float:
    """Emitter coefficient (m3/s per sqrt(m)) of an orifice."""
    return CD * np.pi * (diameter_m / 2) ** 2 * np.sqrt(2 * G)


def orifice_flow_lps(diameter_m: float, pressure_m: float) -> float:
    return 1000 * orifice_coeff(diameter_m) * np.sqrt(max(pressure_m, 0.0))


def insert_leak(wn, leak: LeakSpec, n_steps: int, step_s: int, local_pressure_m: float) -> tuple[str, str | None]:
    """Split the pipe and attach the leak. Returns (leak node, emitter tap node or None)."""
    node = f"{leak.pipe}_leaknode"
    wntr.morph.split_pipe(wn, leak.pipe, f"{leak.pipe}_B", node, split_at_point=leak.frac, return_copy=False)
    j = wn.get_node(node)
    coeff = orifice_coeff(leak.diameter_m)
    if leak.kind == "abrupt":
        # emitter on a tap node behind a 1 m pipe that opens at the start time (exact pressure-driven orifice)
        tap = f"{leak.pipe}_leaktap"
        wn.add_junction(tap, base_demand=0.0, elevation=j.elevation, coordinates=j.coordinates)
        wn.add_pipe(f"{leak.pipe}_leakpipe", node, tap, length=1.0, diameter=0.5, roughness=140, initial_status="Closed")
        wn.get_node(tap).emitter_coefficient = coeff
        act = wntr.network.controls.ControlAction(wn.get_link(f"{leak.pipe}_leakpipe"), "status",
                                                  wntr.network.base.LinkStatus.Open)
        cond = wntr.network.controls.SimTimeCondition(wn, "=", leak.start_s)
        wn.add_control(f"leakstart_{leak.pipe}", wntr.network.controls.Control(cond, act))
        return node, tap
    q_full = coeff * np.sqrt(max(local_pressure_m, 1.0))
    t = np.arange(n_steps) * step_s
    ramp = np.clip((t - leak.start_s) / max(leak.ramp_s, step_s), 0, 1)
    ramp[t < leak.start_s] = 0
    wn.add_pattern(f"LEAK_{leak.pipe}"[:31], ramp.tolist())
    j.add_demand(q_full, f"LEAK_{leak.pipe}"[:31], "leak")
    return node, None


def simulate(wn):
    """Run EPANET; its scratch files (~50 MB per 14-day run) are deleted right after the results are read."""
    d = tempfile.mkdtemp(prefix="agx_")
    try:
        return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(d, "sim"))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def summarize(res, junctions: list[str], sensor_nodes: list[str], flow_links: list[str], leak_node: str | None,
              tap_node: str | None) -> dict:
    """Keep only what a case needs (small files)."""
    p = res.node["pressure"]
    idx = p.index.values
    day = (idx // 86400).astype(int)
    hour = (idx % 86400) / 3600
    night = (hour >= 2) & (hour < 4)
    out = {"time_s": idx.astype(np.int64),
           "sensor_pressure": p[sensor_nodes].values.astype(np.float32),
           "link_flow": (res.link["flowrate"][flow_links].values * 3600).astype(np.float32),     # m3/h
           "night_mean_pressure": p[junctions][night].groupby(day[night]).mean().values.astype(np.float32)}
    if leak_node is not None:
        d = res.node["demand"]
        out["leak_lps"] = (d[tap_node if tap_node in d.columns else leak_node].values * 1000).astype(np.float32)
    else:
        out["leak_lps"] = np.zeros(len(idx), np.float32)
    return out
