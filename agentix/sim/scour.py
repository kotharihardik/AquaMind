"""TRUTH side of brown water (ADR-010): which junctions' water is scoured by a leak, and when?

Sediment on pipe walls is mobilised when the flow in a pipe is clearly faster than the pipe has been used to (shear ~ velocity squared),
or when it reverses. "Used to" = the largest velocity in the leak-free history AND in the same days without the leak (normal demand
drift must not count as leak effect). Computed from the truth simulation (official Real model), never from the nominal model.
"""
from __future__ import annotations

import numpy as np


class ScourMap:
    def __init__(self, footprint_path: str):
        z = np.load(footprint_path)
        self.pipes = [str(p) for p in z["pipes"]]
        self.ends = z["pipe_ends"]
        self.ratio_lib = z["ratio"]; self.rev_lib = z["reverse"]; self.sizes = z["sizes_lps"]
        self.n_junctions = int(self.ends.max()) + 1 if self.ends.size else 0

    def to_nodes(self, m_pipe: np.ndarray, t_pipe: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        m = np.zeros(n, np.float32); t = np.full(n, np.inf)
        for k in (0, 1):
            for pi in np.where(self.ends[:, k] >= 0)[0]:
                j = self.ends[pi, k]
                if m_pipe[pi] > m[j]:
                    m[j] = m_pipe[pi]
                if m_pipe[pi] > 0:
                    t[j] = min(t[j], t_pipe[pi])
        return m, t

    def truth(self, res, res0, onset_s: float, hist_end_s: float, margin: float, scale: float = 0.3, n_junctions: int = 782):
        """res = truth run with the leak, res0 = same days without it -> (mobilisation [J] in 0..1, first time [J] in s)."""
        V, V0 = res.link["velocity"].abs(), res0.link["velocity"].abs()
        F, F0 = res.link["flowrate"], res0.link["flowrate"]
        t = V.index.values
        hist = t < hist_end_s
        after = (t >= onset_s + 1800) & (t <= onset_s + 72 * 3600)       # the loose layer is used up within the first nights
        step = int(np.median(np.diff(t[:5])))
        m_pipe = np.zeros(len(self.pipes), np.float32); t_pipe = np.full(len(self.pipes), np.inf)
        for i, p in enumerate(self.pipes):
            v = V[p].values
            if p + "_B" in V.columns:
                v = np.maximum(v, V[p + "_B"].values)
            cond = max(V0[p].values[hist].max(), V0[p].values[after].max(), 1e-4)
            over = after & (v > cond * (1 + margin))
            if over.any():
                m_pipe[i] = float(np.clip((v[after].max() / (cond * (1 + margin)) - 1.0) / scale, 0, 1))
                t_pipe[i] = float(t[over][0])
            f, f0 = F[p].values, F0[p].values
            flip = after & (np.abs(f0) > 1e-4) & (f * f0 < 0)              # flow (> 0.1 L/s) runs the other way than it would have
            if flip.sum() * step >= 3 * 3600:
                m_pipe[i] = 1.0
                t_pipe[i] = min(t_pipe[i], float(t[flip][0]))
        return self.to_nodes(m_pipe, t_pipe, n_junctions)

    def flush(self, node: int, rng, margin: float = 0.2, scale: float = 0.3) -> np.ndarray:
        """Scouring by a hydrant flush at `node` (about 8-12 L/s for an hour), from the library with per-pipe noise: [J]."""
        r = self.ratio_lib[node, -1].astype(np.float32) * np.exp(rng.normal(0, 0.12, len(self.pipes)))
        m = np.maximum(np.clip((r / (1 + margin) - 1.0) / scale, 0, 1), self.rev_lib[node, -1].astype(np.float32))
        return self.to_nodes(m, np.zeros(len(m)), self.n_junctions)[0]
