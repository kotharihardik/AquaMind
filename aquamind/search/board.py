"""The evidence board: every piece of evidence is one item; the leak map is recomputed from all items (code only).

    log P(leak at j | data) = log prior(j)                      pipe length at j (leaks happen per km of pipe)
                            + tau * pressure log-lik(j)          (aquamind/search/pressure.py)
                            + sum over messages  log L_r(j)      (aquamind/search/report_lik.py)
                            + sum over field checks log L_c(j)   step tests, listening, loggers, dry digs
    P(leak) compares the best leak explanation with "no leak" (all messages unrelated, no pressure drop, field checks
    as they would look without a leak). P(success) of a dig = P(leak) x P(leak within reach of the dig spot).
Items stay separate so a call-back can CHANGE a message's evidence (e.g. "it is from a house pipe") instead of
stacking a new guess on top, and so every number the agent sees can be traced to evidence ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .tool_models import ToolModels

CORR = 0.5          # how strongly overlapping reports discount each other (guide: model correlated reports conservatively)


@dataclass
class MessageItem:
    rid: str
    read: dict                       # the reader's fields (reader-2.0), never raw text
    w: np.ndarray | None             # where it points (None = no usable place)
    source: str                      # here / elsewhere / unknown
    eps: float | None                # chance it is unrelated to a network leak (None = not leak evidence)
    w_up: np.ndarray | None = None   # "it runs down from X" -> the source is near X
    notes: list = field(default_factory=list)
    calls: int = 0
    power: float = 1.0               # correlated reports count less than independent ones (set by the board)

    def usable(self) -> bool:
        return self.eps is not None and self.w is not None


class Board:
    def __init__(self, net, sections, rep_model, log_prior: np.ndarray, pressure_ll: np.ndarray, pressure_ll0: float,
                 prior_leak: float, q_hat: float, q_sd: float, dig_radius_m: float = 90.0):
        self.net, self.sections, self.rep = net, sections, rep_model
        self.base = log_prior + pressure_ll
        self.base_none = pressure_ll0
        self.prior_leak, self.q_hat, self.q_sd = prior_leak, q_hat, q_sd
        self.corr = CORR
        self.cover = (net.pipe_distance <= dig_radius_m).astype(np.float32)
        self.tools = ToolModels(net, sections)
        self.messages: dict[str, MessageItem] = {}
        self.checks: list[dict] = []           # {"label", "ll": [J], "ll0": float, "kind", "support": [J] bool}
        self.dry: set[int] = set()
        self._cache = None

    # ------------------------------------------------------------------ items
    def touch(self):
        self._cache = None
        self.version = getattr(self, "version", 0) + 1

    def _footprint(self):
        """[J leak sites, J places] exposure for the current size estimate (cached), or None when footprints are off."""
        if getattr(self, "foot", None) is None:
            return None
        key = round(float(np.log(max(self.q_hat, 0.5))), 1)
        if getattr(self, "_foot_key", None) != key:
            self._foot_E = self.foot["model"].exposure(self.q_hat)
            self._foot_key = key
        return self._foot_E

    def message_loglik(self, m: MessageItem) -> tuple[np.ndarray, float]:
        if self.rep.kind(m.read) == "discolour":
            E = self._footprint()
            if E is None:
                return np.zeros(len(self.base)), 0.0
            f = self.foot
            s = E @ m.w                                       # exposure of "somewhere at this place", per leak site
            hh = float(f["W"] @ m.w)                          # households there (relative)
            b = max(f["bg"] * hh / f["W"].sum(), 1e-9)       # usual rate of brown-water complaints from this place
            rho = 1.0 + f["kappa"] * hh * s / b
            return m.power * np.log((1 - m.eps) * rho + m.eps), 0.0
        k = self.rep.kernel(m.w, m.source, pooling="pooling" in (m.read.get("symptoms") or []),
                            kind=self.rep.kind(m.read))
        if m.w_up is not None:                      # the source is near the place the water runs down from
            up = self.rep.k_near @ m.w_up
            k = k * (0.15 + 0.85 * up / max(up.max(), 1e-12))
        bg = float(self.rep.background @ m.w)
        return m.power * np.log((1 - m.eps) * k + m.eps * bg + 1e-300), m.power * float(np.log(bg + 1e-300))

    def add_check(self, label: str, kind: str, lik: np.ndarray, lik_none: float, support: np.ndarray | None = None):
        self.checks.append({"label": label, "kind": kind, "ll": np.log(np.maximum(lik, 1e-300)),
                            "ll0": float(np.log(max(lik_none, 1e-300))), "support": support})
        self.touch()

    # ------------------------------------------------------------------ the map
    def _compute(self):
        lp, lnone = self.base.copy(), self.base_none
        prev = []
        for m in self.messages.values():
            if m.usable():
                # reports about the same area share causes (one hose, one leak): do not multiply them as independent
                m.power = 1.0 / (1.0 + self.corr * sum(float(m.w @ w2 / np.sqrt((m.w @ m.w) * (w2 @ w2) + 1e-300)) for w2 in prev))
                prev.append(m.w)
                l, l0 = self.message_loglik(m)
                lp += l
                lnone += l0
        for c in self.checks:
            lp += c["ll"]
            lnone += c["ll0"]
        E = self._footprint()
        if E is not None and self.foot["silence"] > 0:        # a leak that should have scoured many pipes, yet nobody complained
            lp -= self.foot["silence"] * self.foot["kappa"] * (E @ self.foot["W"])
        ev = logsumexp(lp)
        post = np.exp(lp - ev)
        lo = np.log(self.prior_leak / (1 - self.prior_leak)) + ev - lnone
        p_leak = float(1 / (1 + np.exp(-np.clip(lo, -40, 40))))
        cov = self.cover @ post
        cand = getattr(self, "cand", None)
        if cand is not None:                       # ADR-016: a crew digs only where a leak was hypothesised and simulated
            if not cand.any():
                self._cache = {"post": post, "p_leak": p_leak, "cov": cov, "dig": int(np.argmax(cov)), "p_success": 0.0}
                return self._cache
            cov = np.where(cand, cov, -1.0)
        j = int(np.argmax(cov))
        # Physical Pinpoint Anchor: if listening directly pinpointed a junction, dig at the pinpoint epicenter if its coverage is substantial
        pins = [c["at"] for c in self.checks if c.get("kind") == "listen_hit" and c.get("at") is not None]
        if pins:
            pin = pins[-1]
            if (cand is None or cand[pin]) and cov[pin] >= 0.70 * cov[j]:
                j = pin
        self._cache = {"post": post, "p_leak": p_leak, "cov": cov, "dig": j, "p_success": p_leak * float(max(cov[j], 0.0))}
        return self._cache

    @property
    def state(self) -> dict:
        return self._cache or self._compute()

    @property
    def post(self) -> np.ndarray:
        return self.state["post"]

    @property
    def p_leak(self) -> float:
        return self.state["p_leak"]

    def best_dig(self) -> tuple[int, float]:
        return self.state["dig"], self.state["p_success"]

    def mass_outside(self) -> float:
        """Share of the leak map on junctions nobody has hypothesised (ADR-016): the leak may be somewhere not yet considered."""
        cand = getattr(self, "cand", None)
        return 0.0 if cand is None else float(self.post[~cand].sum())

    def section_mass(self) -> np.ndarray:
        return self.sections.mass(self.post) * self.p_leak

    def section_mass_nomsg(self) -> np.ndarray:
        """Section mass of the leak map WITHOUT any citizen message: sensors, documents and field checks only. Complaints can mislead (a false
        complaint made more credible by a call-back); the hedged first check also covers where this map points (ADR-014)."""
        if getattr(self, "_nm_version", None) != self.version:
            lp = self.base.copy()
            for c in self.checks:
                lp += c["ll"]
            post = np.exp(lp - logsumexp(lp))
            self._nm_mass, self._nm_version = self.sections.mass(post) * self.p_leak, self.version
        return self._nm_mass

    def pinpoints(self, j: int, radius_m: float = 150.0) -> list[str]:
        """Listening surveys that pinpointed leak noise within radius of junction j."""
        return [c["label"] for c in self.checks if c["kind"] == "listen_hit" and c.get("at") is not None
                and self.net.pipe_distance[c["at"], j] <= radius_m]

    def area_support(self, j: int) -> list[str]:
        """Step tests that lit up the section holding junction j."""
        return [c["label"] for c in self.checks if c.get("support") is not None and c["support"][j]]

    def confirmed_sightings(self, j: int, radius_m: float = 150.0) -> list[str]:
        """Messages whose writer confirmed on a call-back that water comes out of the road right there."""
        out = []
        for rid, m in self.messages.items():
            if m.usable() and m.calls and m.source == "here" and m.eps < 0.5:
                if self.net.pipe_distance[int(np.argmax(m.w)), j] <= radius_m:
                    out.append(rid)
        return out

    def part_support(self, j: int) -> list[str]:
        """Step tests that lit a PART of a section (second night) holding junction j: the extra night flow is measured
        inside a few hundred metres of main. It is physical evidence that no message can fake (ADR-008)."""
        return [lab for lab in self.area_support(j) if self.sections.areas[lab.split(":")[-1]]["parent"] is not None]

    def field_support(self, j: int, final: bool = False) -> list[str]:
        """Evidence a crew would accept before opening the road (SOP-10): leak noise pinpointed near the spot, or a
        lit part of a section (the step test narrowed the extra night flow to a few hundred metres), or a lit section
        plus a citizen who confirmed water coming out of the road there. Messages alone can be wrong or faked, so they
        never suffice. When no check can add anything (`final`), any lit step-test area counts (ADR-008)."""
        pins = self.pinpoints(j)
        if pins:
            return pins
        parts = self.part_support(j)
        if parts:
            return parts
        area, seen = self.area_support(j), self.confirmed_sightings(j)
        if area and seen:
            return area + seen
        return area if (final and area) else []

    # ------------------------------------------------------------------ field results
    def add_step_result(self, label: str, steps: list[dict]):
        """One step-test night: several sections, each with its inflow drop and the expected customer use."""
        for s in steps:
            if s.get("failed"):
                continue
            mask = self.sections.areas[s["section"]]["mask"]
            excess = s["inflow_drop_lps"] - s["expected_customer_use_lps"]
            lik_in, lik_out = self.tools.step(s["section"], excess, self.q_hat, self.q_sd)
            lik = np.where(mask, lik_in, lik_out)
            lit = lik_in > 20 * lik_out
            self.add_check(f"{label}:{s['section']}", "step", lik, lik_out, support=mask if lit else None)
            if lit:                                      # the size is now measured much better than from the inflow
                self.q_hat, self.q_sd = max(excess, 0.5), 0.4 + 0.15 * max(excess, 0.5)

    def add_listen_result(self, label: str, center: int, radius_m: float, heard: bool, at: int | None):
        lik, lik0 = self.tools.listen(center, radius_m, heard, at, self.q_hat)
        self.add_check(label, "listen_hit" if heard else "listen_quiet", lik, lik0)
        self.checks[-1]["center"] = center
        if heard:
            self.checks[-1]["at"] = at

    def add_canvass_result(self, label: str, center: int, n: int, heard: int, damp: int):
        lik, lik0 = self.tools.canvass(center, n, heard, damp, self.q_hat)
        self.add_check(label, "canvass", lik, lik0)
        self.checks[-1]["center"] = center

    def canvassed(self) -> list[int]:
        return [c["center"] for c in self.checks if c["kind"] == "canvass"]

    def add_logger_result(self, label: str, node: int, reading_m: float, offset_m: float, fp):
        lik, lik0 = self.tools.logger(node, reading_m, offset_m, self.q_hat, fp)
        self.add_check(label, "logger", lik, lik0)

    def add_dry_dig(self, label: str, node: int):
        self.dry.add(node)
        lik = np.where(self.cover[node] > 0, 0.03, 1.0)
        self.add_check(label, "dry_dig", lik, 1.0)

    # ------------------------------------------------------------------ copies for look-ahead
    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.__dict__.update(self.__dict__)
        b.messages = {k: MessageItem(**{**v.__dict__, "notes": list(v.notes)}) for k, v in self.messages.items()}
        b.checks = list(self.checks)
        b.dry = set(self.dry)
        if getattr(self, "cand", None) is not None:
            b.cand = self.cand.copy()
            b.base = self.base.copy()
        b._cache = self._cache
        return b
