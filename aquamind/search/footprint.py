"""Hydraulic footprints of a leak: where would a leak of size q at junction j scour the pipes? (ADR-010, code only)

Sediment on pipe walls is mobilised when the flow is clearly faster than the pipe is used to (shear grows with velocity squared) or
reverses. A leak therefore turns the water brown on the pipes that FEED it, often hundreds of metres away, never at the leak itself.
Footprints come from the NOMINAL model (scripts/build_footprints.py: 782 junctions x 3 leak sizes x 905 pipes). A complaint of brown
water at a place is a weak, area-level observation: P(complaint at place | leak at j) = b + kappa * households * exposure_j(place),
b = the town's usual rate of brown-water complaints (rusty house plumbing, works).  The leak's size comes from the zone-inflow rise.
"""
from __future__ import annotations

import numpy as np


class FootprintModel:
    def __init__(self, path: str, n_junctions: int, margin: float = 0.2, scale: float = 0.3):
        z = np.load(path)
        self.sizes = z["sizes_lps"].astype(float)
        ratio, rev, ends = z["ratio"].astype(np.float32), z["reverse"], z["pipe_ends"]
        m = np.maximum(np.clip((ratio / (1.0 + margin) - 1.0) / scale, 0, 1), rev.astype(np.float32))      # velocity > (1+margin) x usual
        # exposure of every junction = the largest mobilisation among the pipes attached to it: [J leak sites, Q sizes, J places]
        E = np.zeros(m.shape[:2] + (n_junctions,), np.float32)
        for k in (0, 1):
            ok = ends[:, k] >= 0
            for pi in np.where(ok)[0]:
                n = ends[pi, k]
                E[:, :, n] = np.maximum(E[:, :, n], m[:, :, pi])
        self.E = E

    def exposure(self, q_hat: float, q_sd_rel: float = 0.6) -> np.ndarray:
        """[J leak sites, J places] expected exposure, averaged over the plausible leak sizes (log-normal weights around q_hat)."""
        q = float(np.clip(q_hat, self.sizes[0] * 0.6, self.sizes[-1] * 1.5))
        w = np.exp(-0.5 * ((np.log(self.sizes) - np.log(q)) / q_sd_rel) ** 2)
        w /= w.sum()
        return np.tensordot(self.E, w, axes=([1], [0])).astype(np.float32)
