"""RQ1 one-look methods (proposal Section 10): where would each method send the crew from the first look?

All see the same inputs: the layout's sensors, the zone flows, the tank, and the messages read by the same reader.
  agentix_search     our leak search: physics fingerprints of all 782 junctions + messages (downhill, false alarms)
  agentix_no_reports the same without messages (ablation)
  sensitivity_match  classic: correlate the pressure change with the leak fingerprints (no messages)
  report_hotspot     the centre of the readable messages (no physics)
  lila_style         LILA stage-1 idea (Daniel et al. 2022): predict each sensor from the others on leak-free days;
                     the sensor that drops most below its prediction marks the leak (re-implemented from the paper,
                     the official code is GPL-3.0)
"""
from __future__ import annotations

import numpy as np

from ..search.pressure import _tod


def one_look(world, case, reads, method: str) -> tuple[int | None, np.ndarray | None]:
    net, ls = world.town.net, world.search()
    cover = (net.pipe_distance <= 90).astype(np.float32)
    if method in ("agentix_search", "agentix_no_reports", "agentix_search_brown"):
        s = world.search(use_messages=method != "agentix_no_reports", use_footprint=method == "agentix_search_brown")
        b = s.start(case, reads)
        return b.best_dig()[0], b.post
    if method == "report_hotspot":
        ws = [w for r in reads.values() if r["category"] == "leak" and not r["contains_instructions"]
              and (w := ls.rep.place_weights(r["places"])) is not None]
        if not ws:
            return None, None
        dens = np.sum(ws, axis=0) / len(ws)
        return int(np.argmax(cover @ dens)), dens
    t = case.time_s.astype(float)
    base = t < case.history_days * 86400
    post = t > case.case_open_s - 6 * 3600
    if method == "sensitivity_match":
        prof = np.column_stack([_tod(t, case.pressure[:, s], base) for s in range(case.pressure.shape[1])])
        drop = -(case.pressure - prof)[post].mean(0)
        g = ls.fp.g_night[:, [net.index[n] for n in case.pressure_nodes]]
        cos = (g @ drop) / (np.linalg.norm(g, axis=1) * np.linalg.norm(drop) + 1e-12)
        return int(np.argmax(cos)), np.exp(20 * cos)
    if method == "lila_style":
        P, Q = case.pressure, (case.flow[:, 0] + case.flow[:, 1] - case.flow[:, 2]) / 3.6
        k = P.shape[1]
        score = np.zeros(k)
        for j in range(k):
            errs = []
            for i in range(k):
                if i != j:
                    X = np.column_stack([np.ones_like(Q), P[:, i], Q])
                    coef, *_ = np.linalg.lstsq(X[base], P[base, j], rcond=None)
                    e = P[:, j] - X @ coef
                    errs.append(-e[post].mean() / (e[base].std() + 1e-6))
            score[j] = np.mean(errs)
        return net.index[case.pressure_nodes[int(np.argmax(score))]], None
    raise ValueError(method)


METHODS = ["agentix_search", "agentix_search_brown", "agentix_no_reports", "sensitivity_match", "report_hotspot", "lila_style"]
