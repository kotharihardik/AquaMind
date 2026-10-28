"""RQ1 Baseline: Report Hot Spot (Proposal Section 10).

Dig at the centre of the report cluster / spatial centroid.
Approximates what municipal utilities do by hand, ignoring hydraulics and downhill water flow.
"""
from __future__ import annotations

import numpy as np


def predict_report_hotspot(world, case, reads) -> tuple[int | None, np.ndarray | None]:
    """Calculate the spatial centroid of readable citizen leak reports."""
    net, ls = world.town.net, world.search()
    cover = (net.pipe_distance <= 90).astype(np.float32)

    ws = [w for r in reads.values() if r.get("category") == "leak" and not r.get("contains_instructions")
          and (w := ls.rep.place_weights(r.get("places"))) is not None]
    if not ws:
        return None, None
    dens = np.sum(ws, axis=0) / len(ws)
    return int(np.argmax(cover @ dens)), dens
