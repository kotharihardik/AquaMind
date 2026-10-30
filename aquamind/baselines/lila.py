"""RQ1 Baseline: LILA (Daniel et al. 2022), re-implemented in its two stages (3rd place at BattLeDIM 2020).

LILA is a *sequential, pressure-based* method for a network with pressure sensors and a hydraulic model:
  Stage 1, data-driven leakage identification: every sensor is predicted from the OTHER sensors by a regression trained on leak-free data.
           The residual (what the sensor reads below its prediction) is the leak signal; common demand effects cancel in it.
  Stage 2, model-based localisation: a leak is simulated at every candidate junction on the hydraulic model; the SAME sensor-to-sensor
           regression is applied to the simulated pressure changes, and the junction whose simulated residual pattern best matches the
           observed residual pattern is the leak.
The original code is not available offline, so this is a faithful two-stage re-implementation on our data and our nominal model (uncalibrated
roughness included, as for every method). Earlier versions of this file stopped after stage 1 and returned the SENSOR with the largest residual,
which is not a localisation; stage 2 below is what makes it a localisation method.

It ignores citizen reports completely (that is the point of the RQ1 comparison).
"""
from __future__ import annotations

import numpy as np

RIDGE = 1e-3          # tiny ridge on standardised features: with 3 sensors the regressors are nearly collinear


def predict_lila_style(world, case) -> tuple[int | None, np.ndarray | None]:
    net, ls = world.town.net, world.search()
    t = case.time_s.astype(float)
    base = t < case.history_days * 86400                      # leak-free history
    post = t > case.case_open_s - 6 * 3600                    # the window to explain
    P = np.asarray(case.pressure, dtype=float)                # [T, k]
    k = P.shape[1]
    if k < 2:
        return None, None
    sidx = [net.index[n] for n in case.pressure_nodes]

    # ---- stage 1: sensor-to-sensor regression, one model per sensor, trained on leak-free data
    B = np.zeros((k, k))                                      # B[j, i]: weight of sensor i in the prediction of sensor j
    obs = np.zeros(k)
    scale = np.ones(k)
    for j in range(k):
        others = [i for i in range(k) if i != j]
        X = P[:, others]
        mu, sd = X[base].mean(0), X[base].std(0) + 1e-9
        Z = np.column_stack([np.ones(len(P)), (X - mu) / sd])
        A = Z[base].T @ Z[base] + RIDGE * np.diag([0.0] + [1.0] * len(others))
        coef = np.linalg.solve(A, Z[base].T @ P[base, j])
        B[j, others] = coef[1:] / sd                          # back to metres per metre
        e = P[:, j] - Z @ coef                                # residual: reading minus prediction
        obs[j] = -float(e[post].mean())                       # a leak pulls the reading BELOW its prediction
        scale[j] = 1.0 / (float(e[base].std()) + 1e-6)        # whiten: noisy sensors count less (same weights on both sides)

    # ---- stage 2: model-based localisation. Simulated drop per L/s at the sensors, passed through the SAME regression.
    g = ls.fp.g_night[:, sidx]                                # [J, k] night pressure drop per unit leak flow at each sensor
    sim = g - g @ B.T                                         # simulated residual pattern of a leak at every junction
    o, S = obs * scale, sim * scale
    cos = (S @ o) / (np.linalg.norm(S, axis=1) * np.linalg.norm(o) + 1e-12)
    return int(np.argmax(cos)), np.exp(20 * cos)
