"""RQ1 Baseline: Sensitivity Matching (Proposal Section 10).

Classic method: match observed pressure changes to precomputed leak sensitivity patterns.
Uses only pressure sensors (zero citizen messages).
"""
from __future__ import annotations

import numpy as np

from ..search.pressure import _tod


def predict_sensitivity_match(world, case) -> tuple[int | None, np.ndarray | None]:
    """Correlate pressure change with night leak sensitivity fingerprints."""
    net, ls = world.town.net, world.search()
    t = case.time_s.astype(float)
    base = t < case.history_days * 86400
    post = t > case.case_open_s - 6 * 3600

    prof = np.column_stack([_tod(t, case.pressure[:, s], base) for s in range(case.pressure.shape[1])])
    drop = -(case.pressure - prof)[post].mean(0)
    g = ls.fp.g_night[:, [net.index[n] for n in case.pressure_nodes]]
    cos = (g @ drop) / (np.linalg.norm(g, axis=1) * np.linalg.norm(drop) + 1e-12)
    return int(np.argmax(cos)), np.exp(20 * cos)
