"""Truth-side network faults that are NOT leaks but move the SCADA data like one (ADR-017, benchmark v6 "open world").

A utility's model is never the real network: valves are left shut after works, pressure-reducing valves drift, instruments lose their
calibration, and water is drawn at night without a meter. Each fault starts after the leak-free history (so the analytics see a CHANGE)
and stays until someone finds it. Families:
  closed_valve   an isolation valve on a loop pipe left closed after works (KWR: 0.7-1 % of valves are not in the recorded position)
  prv_drift      a pressure-reducing valve's outlet setting drifts down (or up) by 2-6 m
  sensor_bias    one pressure sensor's calibration jumps by 0.3-1.0 m (damaged, re-installed at another height)
  meter_bias     one zone-inlet flow meter over-reads by 2-6 L/s
  night_draw     unauthorised night draw from a hydrant, 5-15 L/s for 1.5-3 h every night (tanker filling)
The methods' inference model never contains a fault; only the field inspections (`CaseEnv.inspect`) reveal one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
from wntr.network.base import LinkStatus
from wntr.network.controls import Control, ControlAction, SimTimeCondition

FAMILIES = ("closed_valve", "prv_drift", "sensor_bias", "meter_bias", "night_draw")
INLET_METERS = ("p227", "p235")


@dataclass
class FaultSpec:
    family: str
    element: str              # pipe, PRV, sensor node, meter link, or hydrant junction
    value: float              # setting change (m), bias (m or L/s), draw (L/s); 0 for a closed valve
    start_s: int
    draw_h: tuple = ()        # night_draw: (start hour, duration h)

    def to_dict(self):
        return asdict(self)


def loop_pipes(wn, min_diam_m: float = 0.1) -> list[str]:
    """Pipes that can be shut without cutting anyone off (not a bridge of the network graph)."""
    g = nx.Graph()
    for name, link in wn.links():
        g.add_edge(link.start_node_name, link.end_node_name, name=name)
    bridges = {g.edges[e]["name"] for e in nx.bridges(g)}
    return [p for p in wn.pipe_name_list if p not in bridges and wn.get_link(p).diameter >= min_diam_m]


def draw_sites(wn, min_diam_m: float = 0.1) -> list[str]:
    """Junctions on mains of at least 100 mm (where hydrants are)."""
    out = set()
    for p in wn.pipe_name_list:
        link = wn.get_link(p)
        if link.diameter >= min_diam_m:
            out |= {link.start_node_name, link.end_node_name}
    return sorted(n for n in out if n in wn.junction_name_list)


def sample(rng, family: str, wn, onset_s: int, sensors: list[str]) -> FaultSpec:
    if family == "closed_valve":
        return FaultSpec(family, str(rng.choice(loop_pipes(wn))), 0.0, onset_s)
    if family == "prv_drift":
        dh = float(rng.uniform(2.0, 6.0)) * (-1 if rng.random() < 0.8 else 1)
        return FaultSpec(family, str(rng.choice(["PRV-1", "PRV-2", "PRV-3"])), round(dh, 2), onset_s)
    if family == "sensor_bias":
        b = float(rng.uniform(0.3, 1.0)) * (-1 if rng.random() < 0.6 else 1)
        return FaultSpec(family, str(rng.choice(sensors)), round(b, 2), onset_s)
    if family == "meter_bias":
        return FaultSpec(family, str(rng.choice(INLET_METERS)), round(float(rng.uniform(2.0, 6.0)), 2), onset_s)
    if family == "night_draw":
        return FaultSpec(family, str(rng.choice(draw_sites(wn))), round(float(rng.uniform(5.0, 15.0)), 1), onset_s,
                         draw_h=(round(float(rng.uniform(22.0, 24.5)), 2), round(float(rng.uniform(1.5, 3.0)), 2)))
    raise ValueError(family)


def draw_on(t_s: np.ndarray, f: FaultSpec) -> np.ndarray:
    """1 while the night draw runs (every night from the onset), else 0."""
    h0, dur = f.draw_h
    t = np.asarray(t_s, float)
    since = (t - h0 * 3600) % 86400                     # seconds since the draw's start hour
    return ((t >= f.start_s) & (since < dur * 3600)).astype(float)


def apply_hydraulic(wn, f: FaultSpec, n_steps: int, step_s: int) -> None:
    """Faults that change the water itself (valve, PRV, draw). Instrument faults are applied to the observations."""
    cond = SimTimeCondition(wn, "=", f.start_s)
    if f.family == "closed_valve":
        wn.add_control(f"fault_{f.element}", Control(cond, ControlAction(wn.get_link(f.element), "status", LinkStatus.Closed)))
    elif f.family == "prv_drift":
        v = wn.get_link(f.element)
        wn.add_control(f"fault_{f.element}", Control(cond, ControlAction(v, "setting", float(v.initial_setting) + f.value)))
    elif f.family == "night_draw":
        t = np.arange(n_steps) * step_s
        wn.add_pattern("FAULT_DRAW", draw_on(t, f).tolist())
        wn.get_node(f.element).add_demand(f.value / 1000.0, "FAULT_DRAW", "draw")


def apply_observation(f: FaultSpec, t_s: np.ndarray, p_obs: np.ndarray, sensor_nodes: list[str], f_obs: np.ndarray,
                      flow_links: list[str]) -> None:
    """Instrument faults: the readings are wrong from the onset (in place)."""
    on = np.asarray(t_s) >= f.start_s
    if f.family == "sensor_bias" and f.element in sensor_nodes:
        p_obs[on, sensor_nodes.index(f.element)] += f.value
    elif f.family == "meter_bias" and f.element in flow_links:
        f_obs[on, flow_links.index(f.element)] += f.value * 3.6          # L/s -> m3/h


def describe(f: FaultSpec) -> str:
    return {"closed_valve": f"the isolation valve on pipe {f.element} was left closed",
            "prv_drift": f"{f.element} outlet setting drifted by {f.value:+.1f} m",
            "sensor_bias": f"pressure sensor {f.element} reads {f.value:+.2f} m wrong",
            "meter_bias": f"inlet flow meter {f.element} over-reads by {f.value:.1f} L/s",
            "night_draw": f"{f.value:.0f} L/s drawn at night from the hydrant at {f.element}"}[f.family]


def visible(res_fault, res_base, sensors: list[str], t_from: float, min_m: float = 0.15) -> float:
    """Largest night-mean pressure change at the candidate sensor spots caused by the fault (m), for case screening."""
    p1, p0 = res_fault.node["pressure"], res_base.node["pressure"]
    t = p1.index.values
    night = (t >= t_from) & (((t % 86400) / 3600 >= 1) & ((t % 86400) / 3600 < 5))
    return float(np.abs((p1[sensors].values - p0[sensors].values)[night].mean(axis=0)).max()) if night.any() else 0.0


__all__ = ["FAMILIES", "FaultSpec", "sample", "apply_hydraulic", "apply_observation", "describe", "draw_on", "loop_pipes", "draw_sites",
           "visible", "INLET_METERS"]
