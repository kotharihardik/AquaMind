"""On-demand hydraulic physics for HYPOTHESISED leak locations only (ADR-016).

There is no precomputed table of all 782 leak fingerprints any more in the agentic system, its ablations or the baselines. When an agent (or the
no-AI fallback rule) names a candidate junction, this tool runs the utility's NOMINAL EPANET model with a 3 L/s leak at that junction and returns the
pressure change it predicts at every node, for the same night and day windows the sensor evidence uses. Results are memoised inside the worker process
(a compute cache: the leak map still only uses the junctions that were hypothesised in the case at hand).
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import wntr
from scipy.special import logsumexp

from .pressure import Q_GRID, PressureEvidence

Q_REF = 3.0


class PhysicsTool:
    def __init__(self, cfg: dict, junctions: list[str]):
        self.cfg, self.junctions = cfg, junctions
        self.windows = cfg["signatures"]["windows_h"]
        self._memo: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.n_sims = 0
        base = self._sim(None)
        self.base = base
        self.p_ref = base["night"]

    def _model(self):
        wn = wntr.network.WaterNetworkModel(self.cfg["nominal_inp"])
        wn.options.time.duration = int(self.cfg["signatures"]["duration_h"] * 3600)
        wn.options.time.report_timestep = 900
        wn.options.time.pattern_start = int(self.cfg["signatures"]["weekday_offset_days"] * 86400)
        return wn

    def _sim(self, node: str | None, q_lps: float = Q_REF) -> dict:
        wn = self._model()
        if node is not None:
            wn.get_node(node).emitter_coefficient = (q_lps / 1000.0) / np.sqrt(max(float(self.p_ref[self.junctions.index(node)]), 1.0))
        d = tempfile.mkdtemp(prefix="agx_phys_")
        try:
            res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(d, "s"))
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.n_sims += node is not None
        p = res.node["pressure"]
        idx = p.index.values
        out = {}
        for name, (h0, h1) in self.windows.items():
            m = (idx >= (24 + h0) * 3600) & (idx < (24 + h1) * 3600)
            out[name] = p[m].mean()[self.junctions].values.astype(np.float32)
        return out

    def fingerprint(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """-> (night drop per L/s, day drop per L/s) at every junction for a leak at junction index j (simulated on demand)."""
        if j not in self._memo:
            r = self._sim(self.junctions[j])
            self._memo[j] = ((self.base["night"] - r["night"]) / Q_REF, (self.base["day"] - r["day"]) / Q_REF)
        return self._memo[j]

    def pressure_loglik(self, ev: PressureEvidence, cands: list[int], sensor_idx: list[int], rel_err: float = 0.25, floor_m: float = 0.01,
                        q_prior_rel: float = 0.5) -> tuple[np.ndarray, float]:
        """The same likelihood as pressure.pressure_loglik, but only for the candidate junctions (rows simulated on demand)."""
        gn = np.array([self.fingerprint(j)[0][sensor_idx] for j in cands]) if cands else np.zeros((0, len(sensor_idx)))
        gd = np.array([self.fingerprint(j)[1][sensor_idx] for j in cands]) if cands else np.zeros((0, len(sensor_idx)))
        g = ev.night_frac * gn + (1 - ev.night_frac) * gd
        h = g - ev.beta[None, :]
        mu = Q_GRID[None, :, None] * h[:, None, :]
        var = ev.drop_var[None, None, :] + (rel_err * mu) ** 2 + floor_m ** 2
        ll = -0.5 * np.sum((ev.drop_m[None, None, :] - mu) ** 2 / var + np.log(2 * np.pi * var), axis=2)
        sd = np.sqrt(ev.q_sd ** 2 + (q_prior_rel * max(ev.q_hat, 0.5)) ** 2)
        lp = -0.5 * ((Q_GRID - max(ev.q_hat, 0.0)) / sd) ** 2 + np.log(np.gradient(np.log(Q_GRID)) * Q_GRID)
        lp -= logsumexp(lp)
        var0 = ev.drop_var + floor_m ** 2
        ll0 = float(-0.5 * np.sum(ev.drop_m ** 2 / var0 + np.log(2 * np.pi * var0)))
        return (logsumexp(ll + lp[None, :], axis=1) if len(cands) else np.zeros(0)), ll0
