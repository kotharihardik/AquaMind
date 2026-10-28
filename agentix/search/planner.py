"""Value of the next field check, with look-ahead (code only).

The old planner scored a check by the immediate rise in P(success) (myopic). A step test that narrows the search
to one section rarely makes a dig possible at once, so it was under-valued, and cases ended with budget unspent.
Here every candidate check is valued by what it leads to by the END of the budget:

    value(a) = P(the leak is found by a dig) - 0.5 * E[dry digs]      over sampled "true" situations
    sample: the leak at j ~ current map (or no leak, with 1 - P(leak)); the outcome of a ~ the investigator's own tool
            models; then the case continues with the base policy below until the checks run out.
Base policy (also the code-only planner): dig when P(success) >= the bar and a field check supports the spot;
otherwise listen where the map is concentrated, else step-test the six sections that hold most of the map.
Common random numbers across candidates make the comparison between them fair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .tool_models import CANV_DAMP_FP, CANV_FP, CANV_RADIUS, LISTEN_EDGE, LISTEN_FP, p_hear

LISTEN_R = 250.0
DIG_BAR = 0.8
FINAL_BAR = 0.35          # bar for the last-resort dig, when no check can add anything (ADR-008)
CANVASS_BLOCK = 200.0
CANVASS_COST = 0.02       # a canvass spends no field check but costs about a night of time: prefer a check when equal
HOT_MASS = 0.2            # a tested section holding this share of the map is split into parts even if not "lit"
CONCENTRATED = 0.45       # listen when one 250 m circle holds this share of the map


@dataclass
class Option:
    kind: str                       # step | listen
    args: dict                      # step: {"sections": [...]} ; listen: {"center": j}
    label: str
    p_inside: float                 # P(leak inside the checked area), from the current map
    value: float = 0.0              # expected P(found) - 0.5 E[dry digs] by the end of the budget
    found: float = 0.0
    dry: float = 0.0
    notes: list = field(default_factory=list)


@dataclass
class Lite:
    """A light copy of the board for look-ahead: log map, 'no leak' score, size estimate, field support."""
    lp: np.ndarray
    lnone: float
    prior_leak: float
    q_hat: float
    q_sd: float
    support: np.ndarray            # [J] bool: a field check points here
    tested: set = field(default_factory=set)
    quiet: list = field(default_factory=list)       # centres of listening surveys that heard nothing
    lit: list = field(default_factory=list)         # step-test areas that showed extra flow
    lit_mask: np.ndarray | None = None              # [J] bool: inside any lit area (support for the last-resort dig)
    canvassed: list = field(default_factory=list)   # centres of neighbour canvasses already done
    mass_nm: np.ndarray | None = None               # section mass without messages (hedge, top level only)

    def copy(self):
        return Lite(self.lp.copy(), self.lnone, self.prior_leak, self.q_hat, self.q_sd, self.support.copy(),
                    set(self.tested), list(self.quiet), list(self.lit),
                    None if self.lit_mask is None else self.lit_mask.copy(), list(self.canvassed))

    def post_pleak(self):
        m = self.lp.max()
        e = np.exp(self.lp - m)
        z = e.sum()
        lo = np.log(self.prior_leak / (1 - self.prior_leak)) + m + np.log(z) - self.lnone
        return (e / z).astype(np.float32), float(1 / (1 + np.exp(-np.clip(lo, -40, 40))))

    def add(self, lik, lik0):
        self.lp = self.lp + np.log(np.maximum(lik, 1e-300))
        self.lnone += float(np.log(max(lik0, 1e-300)))


class Planner:
    def __init__(self, net, sections, tools, dig_radius_m: float = 90.0, n_samples: int = 96, seed: int = 0,
                 dig_bar: float = DIG_BAR, final_bar: float = FINAL_BAR, hedge: float = 0.0,
                 allow_step: bool = True):
        self.allow_step = allow_step
        self.hedge = hedge
        self.net, self.sections, self.tools = net, sections, tools
        self.dig_bar, self.final_bar = dig_bar, final_bar
        self.D = net.pipe_distance
        self.cover = (self.D <= dig_radius_m).astype(np.float32)
        self.D = self.D.astype(np.float32)
        self.listen_cover = (self.D <= LISTEN_R).astype(np.float32)
        self.canvass_cover = (self.D <= CANV_RADIUS).astype(np.float32)
        # detection-weighted cover: hearing is harder near the edge of the circle
        self.listen_w = (self.listen_cover * (1 - LISTEN_EDGE * self.D / LISTEN_R)).astype(np.float32)
        self.n, self.seed = n_samples, seed

    # ------------------------------------------------------------------ state from the board
    def lite(self, board) -> Lite:
        st = board.state
        post, p = st["post"], st["p_leak"]
        lp = np.log(np.maximum(post, 1e-300))
        p = min(max(p, 1e-9), 1 - 1e-9)
        lnone = float(np.log(board.prior_leak / (1 - board.prior_leak)) + np.log((1 - p) / p))
        support = np.zeros(len(post), bool)
        for c in board.checks:
            if c["kind"] == "listen_hit" and c.get("at") is not None:
                support |= self.D[c["at"]] <= 150
        tested = {c["label"].split(":")[-1] for c in board.checks if c["kind"] == "step"}
        quiet = [c["center"] for c in board.checks if c["kind"] == "listen_quiet"]
        lit = [c["label"].split(":")[-1] for c in board.checks if c["kind"] == "step" and c.get("support") is not None]
        lit_mask = np.zeros(len(post), bool)
        for c in board.checks:
            if c["kind"] == "step" and c.get("support") is not None:
                lit_mask |= c["support"]
                if self.sections.areas[c["label"].split(":")[-1]]["parent"] is not None:
                    support |= c["support"]                   # a lit part of a section is physical support (ADR-008)
        return Lite(lp, lnone, board.prior_leak, board.q_hat, board.q_sd, support, tested, quiet, lit, lit_mask,
                    board.canvassed(), board.section_mass_nomsg() if self.hedge > 0 else None)

    # ------------------------------------------------------------------ candidate checks
    def candidates(self, s: Lite, max_sections: int = 6, n_listen: int = 3, labels: bool = True,
                   canvass_left: int = 0, hedge_on: bool = False) -> list[Option]:
        post, p = s.post_pleak()
        mass = self.sections.mass(post) * p
        opts = []
        if self.allow_step:
            lit = [sid for sid in s.lit if sid in self.sections.ids]
            hot = [sid for k, sid in enumerate(self.sections.ids) if sid in s.tested and mass[k] >= HOT_MASS and sid not in lit]
            parts = [a for sid in lit + hot for a in self.sections.parts(sid) if a not in s.tested]
            if parts:                     # second night: split the lit section into its parts
                pm = {a: self.sections.area_mass(a, post) * p for a in parts}
                ids = sorted(pm, key=lambda a: -pm[a])[:max_sections]
                opts.append(Option("step", {"sections": ids}, f"night step test of the parts {', '.join(ids)} of the lit "
                                   f"section" if labels else "", float(sum(pm[a] for a in ids))))
            order = [k for k in np.argsort(-mass) if self.sections.ids[k] not in s.tested][:max_sections]
            if hedge_on and s.mass_nm is not None and self.hedge > 0:         # reserve slots for the complaint-free map: complaints can mislead
                n_nm = int(round(self.hedge * max_sections))
                nm_order = [k for k in np.argsort(-s.mass_nm) if self.sections.ids[k] not in s.tested]
                keep = [k for k in np.argsort(-mass) if self.sections.ids[k] not in s.tested][: max_sections - n_nm]
                extra = [k for k in nm_order if k not in keep][:n_nm]
                order = keep + extra
                order += [k for k in np.argsort(-mass) if self.sections.ids[k] not in s.tested and k not in order][: max_sections - len(order)]
            if order and not (parts and mass[order].sum() < 0.15):
                ids = [self.sections.ids[k] for k in order]
                opts.append(Option("step", {"sections": ids}, f"night step test of sections {', '.join(ids)}" if labels else "",
                                   float(mass[order].sum())))
        q = post.astype(np.float32)
        blocked = np.zeros(len(post), bool)
        for c in s.quiet:                                    # crews do not re-survey the same streets
            blocked |= self.D[c] <= 120
        for _ in range(n_listen):
            score = self.listen_w @ (q * ~blocked)
            score[blocked] = -1
            c = int(np.argmax(score))
            opts.append(Option("listen", {"center": c}, f"listen within 250 m of junction {self.net.junctions[c]}" if labels
                               else "", float((self.listen_cover[c] @ post) * p)))
            q = q * (1 - self.listen_cover[c])
        if canvass_left > 0:                  # ask the residents around the most likely spots (no field check used)
            q = post.astype(np.float32)
            blocked = np.zeros(len(post), bool)
            for c in s.canvassed:
                blocked |= self.D[c] <= CANVASS_BLOCK
            for _ in range(2):
                score = self.canvass_cover @ (q * ~blocked)
                score[blocked] = -1
                c = int(np.argmax(score))
                if score[c] <= 0:
                    break
                opts.append(Option("canvass", {"center": c}, f"canvass the residents within {CANV_RADIUS:.0f} m of junction "
                                   f"{self.net.junctions[c]}" if labels else "", float((self.canvass_cover[c] @ post) * p)))
                q = q * (1 - self.canvass_cover[c])
        return opts

    def base_choice(self, s: Lite) -> Option | None:
        opts = self.candidates(s, n_listen=1, labels=False)
        listen = [o for o in opts if o.kind == "listen"]
        step = [o for o in opts if o.kind == "step"]
        post, p = s.post_pleak()
        if listen and (not step or listen[0].p_inside >= CONCENTRATED * p
                       or listen[0].p_inside >= 0.6 * step[0].p_inside):
            return listen[0]
        return step[0] if step else (listen[0] if listen else None)

    # ------------------------------------------------------------------ simulated outcomes
    def _simulate(self, o: Option, s: Lite, j: int | None, q: float, rng) -> None:
        if o.kind == "step":
            for sid in o.args["sections"]:
                mask = self.sections.areas[sid]["mask"]
                inside = j is not None and bool(mask[j])
                excess = rng.normal(q if inside else 0.0, self.tools.step_sd(sid))
                lin, lout = self.tools.step(sid, excess, s.q_hat, s.q_sd)
                s.add(np.where(mask, lin, lout), lout)
                if lin > 20 * lout:
                    s.q_hat, s.q_sd = max(excess, 0.5), 0.5
                    s.lit.append(sid)
                    s.lit_mask = mask.copy() if s.lit_mask is None else (s.lit_mask | mask)
                    if self.sections.areas[sid]["parent"] is not None:
                        s.support |= mask
                s.tested.add(sid)
            return
        if o.kind == "canvass":
            c = o.args["center"]
            ring, w, n = self.tools.canvass_ring(c)
            ph, pd = self.tools.canvass_p(ring, w, q)
            heard = int(rng.binomial(n, ph[j] if j is not None else CANV_FP))
            damp = int(rng.binomial(n, pd[j] if j is not None else CANV_DAMP_FP))
            lik, lik0 = self.tools.canvass(c, n, heard, damp, s.q_hat)
            s.add(lik, lik0)
            s.canvassed.append(c)
            return
        c = o.args["center"]
        d = self.D[c, j] if j is not None else np.inf
        ph = p_hear(q) * (1 - LISTEN_EDGE * d / LISTEN_R) if d <= LISTEN_R else 0.0
        if rng.random() < ph:
            at = int(np.argmin(np.abs(self.D[j] - abs(rng.normal(0, 25)))))
            heard = True
        elif rng.random() < LISTEN_FP:
            ring = np.where(self.D[c] <= LISTEN_R)[0]
            at, heard = int(rng.choice(ring)), True
        else:
            at, heard = None, False
        lik, lik0 = self.tools.listen(c, LISTEN_R, heard, at, s.q_hat)
        s.add(lik, lik0)
        if heard:
            s.support |= self.D[at] <= 150
        else:
            s.quiet.append(c)

    def _finish(self, s: Lite, j: int | None, q: float, checks: int, digs_left: int, rng) -> tuple[float, int]:
        """Continue with the base policy; -> (found 0/1, dry digs)."""
        dry = 0
        for _ in range(checks + digs_left + 1):
            post, p = s.post_pleak()
            cov = self.cover @ post
            n = int(np.argmax(cov))
            last = checks <= 0
            ok = s.support[n] or (last and s.lit_mask is not None and s.lit_mask[n])
            if p * cov[n] >= (self.final_bar if last else self.dig_bar) and ok and digs_left > 0:
                if j is not None and self.D[n, j] <= 100:
                    return 1.0, dry
                dry += 1
                digs_left -= 1
                s.add(np.where(self.cover[n] > 0, 0.03, 1.0), 1.0)
                continue
            if checks <= 0:
                break
            o = self.base_choice(s)
            if o is None:
                break
            self._simulate(o, s, j, q, rng)
            checks -= 1
        return 0.0, dry

    # ------------------------------------------------------------------ public
    def evaluate(self, board, checks_left: int, digs_left: int = 2, canvass_left: int = 0) -> list[Option]:
        s0 = self.lite(board)
        opts = self.candidates(s0, canvass_left=canvass_left if checks_left > 0 else 0, hedge_on=True)
        if checks_left <= 0:
            return opts
        post, p = s0.post_pleak()
        rng0 = np.random.default_rng(self.seed)
        scen = []
        for _ in range(self.n):
            j = int(rng0.choice(len(post), p=post)) if rng0.random() < p else None
            q = float(np.exp(rng0.normal(np.log(max(s0.q_hat, 0.8)), 0.5 if s0.q_sd > 0.6 else 0.2)))
            scen.append((j, q, int(rng0.integers(1 << 30))))
        for o in opts:
            f = d = 0.0
            for j, q, sd in scen:
                rng = np.random.default_rng(sd)
                s = s0.copy()
                self._simulate(o, s, j, q, rng)
                found, dry = self._finish(s, j, q, checks_left - (0 if o.kind == "canvass" else 1), digs_left, rng)
                f += found
                d += dry
            o.found, o.dry = f / self.n, d / self.n
            o.value = o.found - 0.5 * o.dry - (CANVASS_COST if o.kind == "canvass" else 0.0)
        # near-ties (within Monte-Carlo noise) go to the check that covers more of the map now
        return sorted(opts, key=lambda o: -(o.value + 0.1 * o.p_inside))
