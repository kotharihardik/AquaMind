"""Pressure evidence: sensor series -> how well a leak at each of the 782 junctions explains them (code only).

1. Demand normalisation per sensor, fitted on the leak-free history days:
       p_s(t) ~ a + b Q(t) + c Q(t)^2 + d H_tank(t) + e Q_pump(t)
   Normal water use moves pressure far more than a small leak, so it is removed first (LILA uses the same idea).
2. Evidence = mean residual drop per sensor over a recent window, with an honest variance.
3. Leak size prior from the rise of the NET zone inflow (inlets minus the tank-filling pump) against its
   time-of-day profile. (The old engine forgot to subtract the pump, whose on/off cycles swamped the leak.)
4. Likelihood: drop_s = q * (g_js - beta_s) + noise, with g the nominal fingerprint per L/s and beta the share the
   demand regression already explained away; q is integrated over a grid with the size prior. A relative model error
   term covers the nominal-vs-real mismatch; the result is tempered (tau) because it is still over-confident.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

Q_GRID = np.geomspace(0.3, 15.0, 36)


@dataclass
class PressureEvidence:
    drop_m: np.ndarray          # [k] mean residual pressure drop per sensor (positive = went down)
    drop_var: np.ndarray        # [k]
    beta: np.ndarray            # [k] pressure sensitivity to total inflow (m per L/s)
    q_hat: float                # leak size estimate from the net inflow rise (L/s)
    q_sd: float
    night_frac: float           # share of the window in night hours (choose the fingerprint mix)
    z: np.ndarray               # [k] drop / sd per sensor (for people: "which sensors dropped")


def _tod(t, x, mask, step_s=900):
    slot = ((t % 86400) // step_s).astype(int)
    prof = np.array([x[mask & (slot == s)].mean() if (mask & (slot == s)).any() else np.nan
                     for s in range(86400 // step_s)])
    prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
    return prof[slot]


def build_evidence(time_s, pressure, flow, tank, now_s, history_days: float = 7.0, window_h: float = 6.0,
                   corr: float = 6.0) -> PressureEvidence:
    """pressure [T,k] m; flow [T,3] m3/h (two zone inlets + the tank pump); tank [T] or [T,1] m."""
    t = np.asarray(time_s, float)
    Qin = (flow[:, 0] + flow[:, 1]) / 3.6
    Qp = flow[:, 2] / 3.6 if flow.shape[1] > 2 else np.zeros_like(Qin)
    H = np.asarray(tank).reshape(len(t), -1)[:, 0] if np.size(tank) else np.zeros_like(Qin)
    base = t < history_days * 86400
    X = np.column_stack([np.ones_like(Qin), Qin, Qin ** 2, H, Qp])
    coef, *_ = np.linalg.lstsq(X[base], pressure[base], rcond=None)
    resid = pressure - X @ coef
    sig = resid[base].std(axis=0) + 1e-4
    hour = (t % 86400) / 3600
    night = (hour >= 1) & (hour < 5)
    Qn = Qin[base & night].mean() if (base & night).any() else Qin[base].mean()
    beta = -(coef[1] + 2 * coef[2] * Qn)
    Qnet = Qin - Qp
    eQ = Qnet - _tod(t, Qnet, base)
    sigQ = eQ[base].std() + 1e-3
    win = (t > now_s - window_h * 3600) & (t <= now_s)
    n_eff = max(win.sum() / corr, 1.0)
    drop = -resid[win].mean(axis=0)
    var = sig ** 2 / n_eff
    return PressureEvidence(drop_m=drop, drop_var=var, beta=beta, q_hat=float(eQ[win].mean()),
                            q_sd=float(np.sqrt(sigQ ** 2 / n_eff)), night_frac=float(night[win].mean()),
                            z=drop / np.sqrt(var))


class Fingerprints:
    """Nominal leak fingerprints per L/s: g[j, s] = pressure drop at junction s for a leak at junction j."""

    def __init__(self, path: str):
        z = np.load(path)
        i3 = list(z["sizes_lps"]).index(3.0)
        self.g_night = z["dp_night"][i3] / 3.0
        self.g_day = z["dp_day"][i3] / 3.0
        self.base_p_night = z["base_p_night"]


def pressure_loglik(ev: PressureEvidence, fp: Fingerprints, sensor_idx: list[int], rel_err: float = 0.25,
                    floor_m: float = 0.01, q_prior_rel: float = 0.5) -> tuple[np.ndarray, float]:
    """-> (log-likelihood per junction, log-likelihood of 'no leak')."""
    g = ev.night_frac * fp.g_night[:, sensor_idx] + (1 - ev.night_frac) * fp.g_day[:, sensor_idx]
    h = g - ev.beta[None, :]
    mu = Q_GRID[None, :, None] * h[:, None, :]
    var = ev.drop_var[None, None, :] + (rel_err * mu) ** 2 + floor_m ** 2
    ll = -0.5 * np.sum((ev.drop_m[None, None, :] - mu) ** 2 / var + np.log(2 * np.pi * var), axis=2)
    sd = np.sqrt(ev.q_sd ** 2 + (q_prior_rel * max(ev.q_hat, 0.5)) ** 2)
    lp = -0.5 * ((Q_GRID - max(ev.q_hat, 0.0)) / sd) ** 2 + np.log(np.gradient(np.log(Q_GRID)) * Q_GRID)
    lp -= logsumexp(lp)
    var0 = ev.drop_var + floor_m ** 2
    ll0 = float(-0.5 * np.sum(ev.drop_m ** 2 / var0 + np.log(2 * np.pi * var0)))
    return logsumexp(ll + lp[None, :], axis=1), ll0


def zone_c_balance(time_s, flow, tank, now_s, tank_area_m2: float, history_days: float = 7.0) -> tuple[float, float]:
    """Night flow balance of the tank-fed zone C (a district-meter check): demand = pump inflow - tank filling.
    -> (excess of the last night (01-05 h) over the history nights, its sd) in L/s. A leak in zone C raises it;
    pressure sensors in zone A barely see such a leak."""
    t = np.asarray(time_s, float)
    if flow.shape[1] < 3 or not np.size(tank):
        return 0.0, 1e3
    H = np.asarray(tank).reshape(len(t), -1)[:, 0]
    C = flow[:, 2] / 3.6 - tank_area_m2 * np.gradient(H, t) * 1000
    hour, day = (t % 86400) / 3600, (t // 86400).astype(int)
    night = (hour >= 1) & (hour < 5)
    hist = np.array([C[night & (day == d)].mean() for d in range(int(history_days)) if (night & (day == d)).any()])
    last = night & (t > now_s - 24 * 3600) & (t >= history_days * 86400)
    if not last.any() or len(hist) < 3:
        return 0.0, 1e3
    return float(C[last].mean() - hist.mean()), float(hist.std() + 0.4)


def zone_c_absolute(time_s, flow, tank, now_s, tank_area_m2: float, base_lps: float, sd_lps: float) -> tuple[float, float]:
    """ABSOLUTE night-flow check of the tank-fed zone C (district-meter practice): last night's (01-05 h) demand = pump inflow - tank filling,
    minus what the utility has recorded for this zone when it was last audited leak-free. Unlike zone_c_balance it does not compare with the
    history, so a leak that has run for months (already in the history) is seen too. -> (excess in L/s, its sd)."""
    t = np.asarray(time_s, float)
    if flow.shape[1] < 3 or not np.size(tank):
        return 0.0, 1e3
    H = np.asarray(tank).reshape(len(t), -1)[:, 0]
    C = flow[:, 2] / 3.6 - tank_area_m2 * np.gradient(H, t) * 1000
    hour = (t % 86400) / 3600
    last = (hour >= 1) & (hour < 5) & (t > now_s - 30 * 3600) & (t <= now_s)
    if not last.any():
        return 0.0, 1e3
    return float(C[last].mean() - base_lps), float(sd_lps)

