"""The investigator's own beliefs about its field tools (inference side).

Deliberately rougher than the simulator's truth (configs/field.yaml), as a real utility only roughly knows how good
its tools are. Each function returns (likelihood over junctions, likelihood if there is no leak) for an outcome.
"""
from __future__ import annotations

import numpy as np

LISTEN_P1 = 0.45           # P(hear a 1 L/s leak inside the radius), grows with sqrt(size)
LISTEN_PMAX = 0.85
LISTEN_EDGE = 0.35         # harder near the edge of the radius
LISTEN_FP = 0.07
PINPOINT_M = 40.0
STEP_SD_LPS = 0.30         # step-test excess noise: meter + night-use wobble (plus a share of the expected use)
STEP_SD_REL = 0.18
LOGGER_SD_M = 0.10
CANV_A, CANV_MAX, CANV_LEN, CANV_FP = 0.5, 0.9, 45.0, 0.07     # the investigator's rough belief about neighbours' ears
CANV_DAMP_P, CANV_DAMP_LEN, CANV_DAMP_FP = 0.3, 40.0, 0.04
CANV_TAU = 0.8                                                  # tempered: the real town is not the model
CANV_RADIUS, CANV_CONTACT = 200.0, 0.6
_Q = np.geomspace(0.3, 15, 40)


def p_hear(q: float) -> float:
    return min(LISTEN_PMAX, LISTEN_P1 * np.sqrt(max(q, 0.1)))


class ToolModels:
    def __init__(self, net, sections):
        self.net, self.sections = net, sections
        self.D = net.pipe_distance

    def step_sd(self, area: str) -> float:
        """Meter noise + how much the area's night use varies from night to night (known from billing/AMR)."""
        return float(np.hypot(STEP_SD_LPS, self.sections.areas[area]["sd"]))

    def step(self, area: str, excess: float, q_hat: float, q_sd: float) -> tuple[float, float]:
        """excess = measured drop - expected customer use of the area. -> (lik if the leak is inside, lik if not)."""
        sd = self.step_sd(area)
        g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / s
        mu, s = np.log(max(q_hat, 1.0)), 0.9 if q_sd > 0.5 else 0.4    # broad size prior unless measured
        pq = np.exp(-0.5 * ((np.log(_Q) - mu) / s) ** 2)
        pq /= pq.sum()
        return float((g(excess, _Q, sd) * pq).sum()), float(g(excess, 0.0, sd))

    def step_power(self, area: str, q_hat: float, q_sd: float) -> tuple[float, float]:
        """-> (P(the area lights up | leak inside), P(it lights up | no leak inside)) under this model: 'lit' = likelihood ratio > 20, as on
        the board. A small leak in an area with noisy night use rarely lights it."""
        sd = self.step_sd(area)
        xs = np.linspace(-4 * sd, _Q[-1] + 4 * sd, 400)
        lit = np.array([(lambda a: a[0] > 20 * a[1])(self.step(area, x, q_hat, q_sd)) for x in xs])
        mu, s = np.log(max(q_hat, 1.0)), 0.9 if q_sd > 0.5 else 0.4
        pq = np.exp(-0.5 * ((np.log(_Q) - mu) / s) ** 2)
        pq /= pq.sum()
        dens = lambda m: np.exp(-0.5 * ((xs - m) / sd) ** 2)
        frac = lambda m: float((dens(m) * lit).sum() / dens(m).sum())
        return float(sum(w * frac(q) for w, q in zip(pq, _Q))), frac(0.0)

    def listen(self, c: int, R: float, heard: bool, at: int | None, q_hat: float) -> tuple[np.ndarray, float]:
        d = self.D[c]
        inside = d <= R
        p = np.where(inside, p_hear(q_hat) * (1 - LISTEN_EDGE * d / R), 0.0)
        if not heard:
            p_pen = 0.5 * p
            return (1 - p_pen) * (1 - LISTEN_FP), 1 - LISTEN_FP
        ring = max(int(inside.sum()), 1)
        if at is None:
            return p + LISTEN_FP / ring, LISTEN_FP / ring
        k = np.exp(-0.5 * (self.D[at] / PINPOINT_M) ** 2)
        k = k / max(k[inside].sum(), 1e-12)
        return p * k + LISTEN_FP / ring, LISTEN_FP / ring

    # ---- neighbour canvass: how many of n residents around `center` heard hissing at night / saw damp ground
    def canvass_ring(self, c: int, radius: float = CANV_RADIUS) -> tuple[np.ndarray, np.ndarray, int]:
        """-> (ring nodes, resident weights, expected number of answers). Residents ~ water demand (public data)."""
        ring = np.where(self.D[c] <= radius)[0]
        dem = self.net.base_demand / self.net.base_demand.max()
        w = 0.3 + dem[ring]
        return ring, w, int(np.clip(round(float(w.sum()) * CANV_CONTACT), 4, 40))

    def canvass_p(self, ring: np.ndarray, w: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
        """P(one answering resident reports hissing / damp ground | leak at each junction j of size q) -> two [J] arrays."""
        wn = w / w.sum()
        Dr = self.D[ring]
        h = (np.exp(-Dr / CANV_LEN) * wn[:, None]).sum(0)
        hd = (np.exp(-Dr / CANV_DAMP_LEN) * wn[:, None]).sum(0)
        p_hear = CANV_FP + (1 - CANV_FP) * min(CANV_MAX, CANV_A * np.sqrt(max(q, 0.1))) * h
        p_damp = CANV_DAMP_FP + (1 - CANV_DAMP_FP) * CANV_DAMP_P * hd
        return p_hear, p_damp

    def canvass(self, c: int, n: int, heard: int, damp: int, q_hat: float) -> tuple[np.ndarray, float]:
        from scipy.stats import binom
        ring, w, _ = self.canvass_ring(c)
        ph, pd = self.canvass_p(ring, w, q_hat)
        ll = binom.logpmf(heard, n, ph) + 0.5 * binom.logpmf(damp, n, pd)
        ll0 = binom.logpmf(heard, n, CANV_FP) + 0.5 * binom.logpmf(damp, n, CANV_DAMP_FP)
        return np.exp(CANV_TAU * (ll - ll0.max())), float(np.exp(CANV_TAU * (ll0 - ll0.max())))

    def logger(self, n: int, reading: float, offset: float, q_hat: float, fp) -> tuple[np.ndarray, float]:
        pred = fp.base_p_night[n] + offset - max(q_hat, 0.5) * fp.g_night[:, n]
        g = lambda x, m: np.exp(-0.5 * ((x - m) / LOGGER_SD_M) ** 2)
        return g(reading, pred), float(g(reading, fp.base_p_night[n] + offset))
