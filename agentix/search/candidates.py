"""Where to look when NO agent formulates hypotheses: the fixed fallback of the no-AI ablation (ADR-016).

A dispatcher without an investigator does the obvious: look near the places people named, near the sensors that dropped most, and inside any
section a step test lit up. Nothing else: no reasoning about downhill flow, look-alike causes, records or what the sensors imply further away.
"""
from __future__ import annotations

import numpy as np

COMPLAINT_RADIUS_M = 250.0
SENSOR_RADIUS_M = 300.0


def heuristic(tb) -> list[int]:
    b, net = tb.board, tb.net
    js: list[int] = []
    for m in b.messages.values():
        if m.usable():
            js += list(np.where(net.pipe_distance[int(np.argmax(m.w))] <= COMPLAINT_RADIUS_M)[0])
    ev = b.pressure_evidence
    for k in [k for k in np.argsort(-ev.z)[:2] if ev.z[k] > 1.0]:
        s = net.index[tb.case.pressure_nodes[k]]
        js += list(np.where(net.pipe_distance[s] <= SENSOR_RADIUS_M)[0])
    for c in b.checks:
        if c["kind"] == "step" and c.get("support") is not None:
            js += list(np.where(c["support"])[0])
    return [int(j) for j in dict.fromkeys(js)]
