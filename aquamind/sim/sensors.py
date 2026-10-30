"""Instrument noise for pressure sensors and temporary loggers (truth side).

The official BattLeDIM SCADA is reproduced by the Real model to 0.1 cm (only 0.01 m rounding), so the large gap
between the data and the utility's nominal model is model mismatch, which the Real truth produces physically. On top
we add realistic instrument noise:  observed = clean + offset + slow drift + white noise, rounded to 0.01 m.
A temporary logger has a fresh unknown offset (no history at that spot).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorNoise:
    offset_sigma: float = 0.05      # installation / calibration offset (m), constant per sensor
    drift_sigma: float = 0.02       # slow drift (m)
    drift_tau_days: float = 5.0
    white_sigma: float = 0.03       # reading noise (m)
    quant: float = 0.01             # data resolution (m)


def observe(clean: np.ndarray, time_s: np.ndarray, rng, noise: SensorNoise) -> np.ndarray:
    """clean: [T, S] true pressures -> noisy observations."""
    T, S = clean.shape
    n_days = int(np.ceil(time_s[-1] / 86400)) + 1
    rho = np.exp(-1.0 / noise.drift_tau_days)
    daily = np.empty((n_days + 1, S))
    daily[0] = rng.normal(0, noise.drift_sigma, S)
    for d in range(1, n_days + 1):
        daily[d] = rho * daily[d - 1] + rng.normal(0, noise.drift_sigma * np.sqrt(1 - rho ** 2), S)
    f = time_s / 86400.0
    d0 = np.floor(f).astype(int)
    w = (f - d0)[:, None]
    drift = (1 - w) * daily[d0] + w * daily[np.minimum(d0 + 1, n_days)]
    obs = clean + rng.normal(0, noise.offset_sigma, S)[None, :] + drift + rng.normal(0, noise.white_sigma, (T, S))
    return (np.round(obs / noise.quant) * noise.quant).astype(np.float32)


def logger_reading(clean_night_mean: float, rng, noise: SensorNoise, n_samples: int = 24) -> float:
    """Night (02-04 h) mean of a temporary logger: fresh offset + drift + averaged reading noise."""
    v = (clean_night_mean + rng.normal(0, noise.offset_sigma) + rng.normal(0, noise.drift_sigma)
         + rng.normal(0, noise.white_sigma / np.sqrt(n_samples)))
    return float(np.round(v / noise.quant) * noise.quant)
