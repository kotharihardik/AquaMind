"""Explanations other than a leak (ADR-017, open world): the model may be wrong in a way nobody wrote down.

The leak map assumes the utility's model is right except for one leak. When it is not, a closed valve, a drifted pressure-reducing valve, a
mis-calibrated sensor or meter, or a night draw is fitted as "a leak at the junction that best mimics it", with P(leak) close to 1. This
module lets anyone (an agent, a rule, a search algorithm) PROPOSE a structural change of the model, and code then:
  1. simulates it on the nominal model (charged to the case's simulation budget),
  2. fits its free number (a PRV's drift, a sensor's or a meter's bias) by integrating over a grid with a prior,
  3. scores it against the sensor evidence alone and together with every hypothesised leak (superposition: drop = change + leak),
  4. folds every explanation into the leak map as a mixture, so P(leak) and the dig gate see it,
  5. turns an inspection's result into confirmed / refuted.
Proposals are generic primitives of the network, not a list of known faults:
  {"what": "pipe_closed", "pipe": "p695"} or {"what": "pipe_closed", "near": ["n342", "Market Road"]}   an isolation valve shut
  {"what": "valve_setting", "valve": "PRV-2"}                                                           a valve's setting moved
  {"what": "instrument", "sensor": "n769"} / {"what": "instrument", "meter": "p227"}                    an instrument reads wrong
  {"what": "night_draw", "near": "n607"}                                                                 water drawn at night, not lost
No number here comes from an agent: the agent names WHAT might be different; code decides how well it fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from ..town.network import series_chain
from .physics_tool import BudgetExceeded

INLET_METERS = ("p227", "p235")
PRIOR = {"pipe_closed": 0.08, "valve_setting": 0.05, "instrument": 0.06, "night_draw": 0.05}   # P(some fault of this kind | an alarm)
SCALE = np.linspace(-2.5, 2.5, 41)          # PRV drift in units of the 3 m probe (about -7.5 .. +7.5 m)
CLOSE_SCALE = np.exp(np.linspace(np.log(0.5), np.log(2.0), 15))   # the nominal model gets a closure's effect right only to within a factor
CLOSE_SCALE_SD = 0.35                                              # (pipe roughness / diameters / demands differ from the real town)
SCALE_PRIOR_SD = 1.2
PROBE_M = -3.0
SENSOR_BIAS_SD = 0.6                        # prior sd of a sensor's calibration error (m)
METER_GRID = np.linspace(0.5, 10.0, 20)     # meter over-reading (L/s)
CONC_LEAK, CONC_DRAW, CONC_SD = 0.40, 0.90, 0.15     # share of the night's extra inflow inside its busiest 3 h: a leak runs all night, a draw does not
INSPECT = {"pipe_closed": "valve", "valve_setting": "prv", "instrument": None, "night_draw": "watch"}


@dataclass
class Explanation:
    eid: str
    what: str                       # pipe_closed | valve_setting | instrument | night_draw
    element: str                    # pipe, valve, sensor node, meter link, or junction
    source: str                     # who proposed it
    logprior: float
    status: str = "open"            # open | confirmed | refuted
    d_night: np.ndarray | None = None
    d_day: np.ndarray | None = None
    note: str = ""
    post: float = 0.0
    fit: dict = field(default_factory=dict)

    def inspect_args(self) -> tuple[str, str]:
        if self.what == "instrument":
            return ("meter" if self.element in INLET_METERS else "sensor"), self.element
        if self.what == "night_draw" and self.fit.get("section"):
            return "watch", self.fit["section"]                  # a night patrol of the section's hydrants
        return INSPECT[self.what], self.element

    def label(self, describe=None) -> str:
        where = describe(self.element) if describe and self.what == "night_draw" else self.element
        return {"pipe_closed": f"isolation valve on pipe {self.element} shut", "valve_setting": f"{self.element} setting moved"
                + (f" ({self.fit.get('drift_m', 0):+.1f} m best fit)" if self.fit else ""),
                "instrument": f"{'meter' if self.element in INLET_METERS else 'sensor'} {self.element} reads wrong"
                + (f" (best fit {self.fit.get('bias', 0):+.2f})" if self.fit else ""),
                "night_draw": f"water drawn at night near {where}"}[self.what]


class ExplanationEngine:
    def __init__(self, search):
        self.s = search
        self.net = search.net
        self._loop = None

    # ------------------------------------------------------------------ the map's structure (public facts)
    def loop_pipes(self) -> list[str]:
        if self._loop is None:
            import networkx as nx
            g = nx.Graph()
            for name, link in self.net.wn.links():
                g.add_edge(link.start_node_name, link.end_node_name, name=name)
            br = {g.edges[e]["name"] for e in nx.bridges(g)}
            self._loop = [p for p in self.net.wn.pipe_name_list if p not in br and self.net.wn.get_link(p).diameter >= 0.1]
        return self._loop

    def series_chain(self, pipe: str) -> list[str]:
        """The pipes in series with `pipe` (joined only through junctions with exactly two pipes): one stretch of main. Closing any of them
        has the same effect, and a crew checking 'the valve on p691' walks that stretch and opens every valve box on it."""
        return series_chain(self.net.wn, pipe)

    def pipes_near(self, nodes: list[int], radius_m: float = 500.0, k: int = 10) -> list[str]:
        """Loop pipes within `radius_m` (along the pipes) of any of `nodes`, the ones carrying most water first (nominal night flow)."""
        phys = self.s._phys()
        flow = phys.base.get("flow_night")
        D = self.net.pipe_distance
        out = []
        for p in self.loop_pipes():
            u, v = self.net.pipe_endpoints(p)
            iu, iv = self.net.index.get(u), self.net.index.get(v)
            dmin = min(min(D[i, n] for n in nodes) for i in (iu, iv) if i is not None) if (iu is not None or iv is not None) else np.inf
            if dmin <= radius_m:
                out.append((abs(float(flow[p])) if flow is not None and p in flow.index else 0.0, dmin, p))
        out.sort(key=lambda x: (-x[0], x[1]))
        seen, keep = set(), []
        for _, _, p in out:                        # pipes in series are one candidate: closing any of them does the same
            c = tuple(series_chain(self.net.wn, p))
            if c not in seen:
                seen.add(c)
                keep.append(p)
        return keep[:k]

    def resolve_nodes(self, tb, items) -> list[int]:
        """Junction ids, sensor ids, sections or place names -> junction indices (public map only)."""
        from .hypotheses import place_weights
        out = []
        for x in items if isinstance(items, list) else [items]:
            x = str(x).strip()
            if x in self.net.index:
                out.append(self.net.index[x])
            elif x in self.s.sections.areas:
                m = np.asarray(self.s.sections.areas[x]["mask"], bool)
                out.append(int(np.argmax(m * self.s.town.net.base_demand + m)))
            else:
                w = place_weights(tb.places, x) if tb is not None else None
                if w is not None:
                    out.append(int(np.argmax(w)))
        return out

    def prv_reach(self, sensor_names: list[str], min_m: float = 0.5) -> dict[str, list[str]]:
        """Map fact: which of these sensors move when each pressure-reducing valve's setting moves by 3 m (nominal model, computed once per
        process and not charged: it is part of knowing your own network, like the pipe map)."""
        phys = self.s._phys()
        if not hasattr(phys, "_prv_reach"):
            phys._prv_reach = {}
            for v in self.net.wn.valve_name_list:
                r = phys._sim(None, mod=("valve_setting", v, PROBE_M))
                phys._prv_reach[v] = phys.base["night"] - r["night"]
        return {v: [s for s in sensor_names if abs(float(d[self.net.index[s]])) >= min_m] for v, d in phys._prv_reach.items()}

    def picture(self, board, tb=None) -> list[str]:
        """The anomaly as code sees it: plain facts for any brain (agent or rule)."""
        ev, sidx, names = board._hyp["ev"], board._hyp["sidx"], board.sensor_names
        z = ev.drop_m / np.sqrt(ev.drop_var)
        sens = ", ".join((f"{n} fell {d:.2f} m (z={zz:.0f})" if d >= 0 else f"{n} ROSE {-d:.2f} m (z={-zz:.0f})")
                         for n, d, zz in sorted(zip(names, ev.drop_m, z), key=lambda x: -abs(x[2])))
        out = ["ANOMALY PICTURE (code; change in the last 6 h against the demand-corrected history):",
               f"  sensors: {sens}",
               f"  zone inflow meters (inlets {', '.join(INLET_METERS)}) minus the tank pump: {'up' if ev.q_hat >= 0 else 'down'} {abs(ev.q_hat):.1f} L/s"
               + (f" (sd {ev.q_sd:.1f})" if ev.q_sd else "")]
        sh = getattr(board, "inflow_shape", None)
        if sh:
            out.append(f"  the latest night's extra inflow: {sh['night_extra_lps']:+.1f} L/s on average"
                       + (f"; {sh['share_in_busiest_3h']:.0%} of it inside its busiest 3 hours (spread evenly = about 40 %)"
                          if sh["share_in_busiest_3h"] is not None else " (not clearly above normal)"))
        evb = board._hyp.get("ev_block")
        if evb is not None:
            zb = evb.drop_m / np.sqrt(evb.drop_var)
            t1 = board._hyp["block_end_s"]
            out.append(f"  during last night's busiest 3 h of extra inflow (until {int(t1 % 86400 // 3600):02d}:00): "
                       + ", ".join((f"{n} fell {d:.2f} m (z={zz:.0f})" if d >= 0 else f"{n} ROSE {-d:.2f} m (z={-zz:.0f})")
                                   for n, d, zz in sorted(zip(names, evb.drop_m, zb), key=lambda x: -abs(x[2])))
                       + f"; inflow up {evb.q_hat:.1f} L/s then")
        big = [n for n, zz in zip(names, np.abs(z)) if zz > 4]
        if len(big) == 1:
            out.append(f"  only {big[0]} changed clearly; the other sensors did not")
        cand = np.where(board.cand)[0].tolist()
        phys = self.s._phys()
        if cand:
            post = board.post
            j = int(max(cand, key=lambda x: post[x]))
            gn, gd = phys.fingerprint(j)
            g = ev.night_frac * gn[sidx] + (1 - ev.night_frac) * gd[sidx] - ev.beta
            q = float(np.clip((g * ev.drop_m / ev.drop_var).sum() / max((g * g / ev.drop_var).sum(), 1e-9), 0.3, 15.0))
            res = (ev.drop_m - q * g) / np.sqrt(ev.drop_var + (0.25 * q * g) ** 2 + 1e-4)
            bad = [f"{n} ({r:+.0f} sd)" for n, r in zip(names, res) if abs(r) > 3]
            out.append(f"  best leak hypothesised so far ({self.net.junctions[j]}, best size {q:.1f} L/s): "
                       + (f"leaves unexplained {', '.join(bad)}" if bad else "explains every sensor within 3 sd")
                       + f"; it needs zone inflow up about {q:.1f} L/s (measured {ev.q_hat:+.1f})")
        reach = self.prv_reach(names)
        out.append("  map: sensors that move when a pressure-reducing valve's setting moves: "
                   + "; ".join(f"{v} -> {', '.join(r) or 'none of the sensors'}" for v, r in reach.items()))
        return out

    # ------------------------------------------------------------------ proposing
    def propose(self, board, spec: dict, source: str, tb=None) -> dict:
        """-> {"ok", "added": [eid...], "msg"}. Expands a proposal into concrete elements, simulates and scores each."""
        what = str(spec.get("what", "")).strip()
        ex = board.expl
        have = {(e.what, e.element) for e in ex.values()}
        els: list[str] = []
        if what == "pipe_closed":
            if spec.get("pipe"):
                els = [str(spec["pipe"])] if str(spec["pipe"]) in self.net.wn.pipe_name_list else []
            else:
                nodes = self.resolve_nodes(tb, spec.get("near") or [])
                els = self.pipes_near(nodes, k=int(spec.get("k", 10))) if nodes else []
        elif what == "valve_setting":
            v = str(spec.get("valve", ""))
            els = [v] if v in self.net.wn.valve_name_list else []
        elif what == "instrument":
            x = str(spec.get("sensor") or spec.get("meter") or "")
            els = [x] if (x in board.sensor_names or x in INLET_METERS) else []
        elif what == "night_draw":
            near = spec.get("near") or []
            nodes = self.resolve_nodes(tb, [x for x in (near if isinstance(near, list) else [near]) if str(x) != "best_fit"])
            if "best_fit" in [str(x) for x in (near if isinstance(near, list) else [near])]:
                nodes += self.best_draw_sites(board)
            D = self.net.pipe_distance
            secs = [str(x) for x in (near if isinstance(near, list) else [near]) if str(x) in self.s.sections.areas]
            els = list(dict.fromkeys(self.net.junctions[int(j)] for n in nodes for j in np.argsort(D[n])[:6] if D[n, j] <= 300))
            for sid in secs:                                # a draw somewhere in a section: up to 4 hydrant sites spread over it
                m = np.where(np.asarray(self.s.sections.areas[sid]["mask"], bool))[0]
                pick = []
                for j in sorted(m, key=lambda x: -self.s.town.net.base_demand[x]):
                    if all(D[j, o] > 250 for o in pick):
                        pick.append(int(j))
                els += [self.net.junctions[j] for j in pick[:4]]
            els = list(dict.fromkeys(els))
        else:
            return {"ok": False, "added": [], "msg": "what must be pipe_closed, valve_setting, instrument or night_draw"}
        if not els:
            return {"ok": False, "added": [], "msg": f"nothing on the map matches {spec}"}
        added = []
        chains = {tuple(series_chain(self.net.wn, e.element)) for e in ex.values() if e.what == "pipe_closed"}
        for el in els:
            if (what, el) in have or (what == "pipe_closed" and tuple(series_chain(self.net.wn, el)) in chains):
                continue
            e = Explanation(f"X{len(ex) + 1}", what, el, source, self._logprior(what, board))
            try:
                self._simulate(board, e)
            except BudgetExceeded as err:
                if added:
                    break
                return {"ok": False, "added": [], "msg": str(err)}
            ex[e.eid] = e
            added.append(e.eid)
        if added:
            self.refresh(board)
        return {"ok": bool(added), "added": added, "msg": f"{len(added)} explanation(s) simulated and scored"}

    def _logprior(self, what: str, board) -> float:
        n = {"pipe_closed": len(self.loop_pipes()), "valve_setting": len(self.net.wn.valve_name_list),
             "instrument": len(board.sensor_names) + len(INLET_METERS), "night_draw": len(self.net.junctions)}[what]
        return float(np.log(PRIOR[what] / n))

    def _simulate(self, board, e: Explanation) -> None:
        phys = self.s._phys()
        if e.what == "pipe_closed":
            e.d_night, e.d_day = phys.scenario("pipe_closed", e.element)
        elif e.what == "valve_setting":
            e.d_night, e.d_day = phys.scenario("valve_setting", e.element, PROBE_M)
        elif e.what == "night_draw":
            j = self.net.index[e.element]
            phys.fingerprint(j)                                     # a draw moves pressure like a leak there (charged once)
            e.fit = {"section": self.s.sections.ids[int(self.s.sections.of[j])]}

    # ------------------------------------------------------------------ scoring
    def _ev_under(self, board, e: Explanation | None, ev=None):
        """-> list of (log weight, evidence with this explanation's effect removed, sensor idx, sensor var inflation)."""
        ev, sidx = (ev if ev is not None else board._hyp["ev"]), board._hyp["sidx"]
        if e is None:
            return [(0.0, ev, sidx, None)]
        nf = ev.night_frac
        if e.what in ("pipe_closed", "valve_setting"):
            d = nf * e.d_night[sidx] + (1 - nf) * e.d_day[sidx]
            if e.what == "pipe_closed":
                lp = -0.5 * (np.log(CLOSE_SCALE) / CLOSE_SCALE_SD) ** 2
                lp -= logsumexp(lp)
                return [(float(lp[i]), _shift(ev, a * d, 0.0), sidx, None) for i, a in enumerate(CLOSE_SCALE)]
            lp = -0.5 * (SCALE / SCALE_PRIOR_SD) ** 2
            lp -= logsumexp(lp)
            return [(float(lp[i]), _shift(ev, a * d, 0.0), sidx, None) for i, a in enumerate(SCALE)]
        if e.what == "instrument":
            if e.element in INLET_METERS:          # an over-reading inlet meter: the extra inflow is not real, and the demand regression
                lp = np.full(len(METER_GRID), -np.log(len(METER_GRID)))        # that used it shows a fake rise of beta * b at every sensor
                return [(float(lp[i]), _shift(ev, -ev.beta * b, b), sidx, None) for i, b in enumerate(METER_GRID)]
            k = board.sensor_names.index(e.element)
            infl = np.zeros(len(sidx))
            infl[k] = SENSOR_BIAS_SD ** 2           # its reading = truth + bias, bias ~ N(0, sd): integrated out exactly
            return [(0.0, ev, sidx, infl)]
        return [(0.0, ev, sidx, None)]             # night_draw: handled as the leak-like term at its junction (refresh)

    def refresh(self, board) -> None:
        """Rebuild the leak map's pressure part as a mixture over 'no structural change' and every live explanation."""
        h = board._hyp
        live = [e for e in board.expl.values() if e.status != "refuted"]
        conf = [e for e in live if e.status == "confirmed"]
        mix = [(None, 0.0)] + [(e, e.logprior) for e in live if e.what != "night_draw"]
        draws = [e for e in live if e.what == "night_draw"]
        if conf:                                  # an inspection found it: the explanation is certain (one fault at a time)
            mix = [(e, 0.0) for e in conf if e.what != "night_draw"]
            draws = [e for e in conf if e.what == "night_draw"]
        n_struct = sum(PRIOR.values()) if not conf else 0.0
        cfg, phys = self.s.cfg, self.s._phys()
        cand = np.where(board.cand)[0].tolist()
        tau, rho = cfg.tau_p, cfg.rho_p
        conc_leak, conc_draw = board.inflow_conc
        LL0, LLJ, labels = [], [], []
        evb = h.get("ev_block")                    # last night's busiest 3 h: a second window every hypothesis is scored on
        for e, lw in mix:
            w = (np.log(max(1 - n_struct, 1e-6)) if e is None else lw)
            parts0, partsJ = [], []
            under_b = self._ev_under(board, e, evb) if evb is not None else None
            for i, (lwi, ev_f, sidx, infl) in enumerate(self._ev_under(board, e)):
                llj, ll0 = _loglik(phys, ev_f, cand, sidx, infl)
                if under_b is not None:            # the same explanation (same fitted number) on the night window
                    _, evb_f, _, inflb = under_b[i]
                    lljb, ll0b = _loglik(phys, evb_f, cand, sidx, inflb)
                    llj, ll0 = llj + lljb, ll0 + ll0b
                parts0.append(lwi + tau * ll0)
                partsJ.append(lwi + tau * llj if len(cand) else np.zeros(0))
            l0 = float(logsumexp(parts0))
            lj = logsumexp(np.stack(partsJ), axis=0) if len(cand) else np.zeros(0)
            lj = np.logaddexp(np.log(1 - rho) + lj, np.log(rho) + l0) if len(cand) else lj
            LL0.append(w + l0 + conc_leak)
            LLJ.append(w + lj + conc_leak)
            labels.append(e.eid if e else None)
        for e in draws:                            # a draw at j: no leak; at night the pressure looks like a leak at j, by day like nothing
            j = self.net.index[e.element]
            llj, ll0 = _loglik(phys, h["ev"], [j], h["sidx"], None)
            main = float(np.logaddexp(float(llj[0]), ll0) - np.log(2))   # the draw may or may not have been running in the main window
            if evb is not None:
                lljb, _ = _loglik(phys, evb, [j], h["sidx"], None)
                main += float(lljb[0])
            l0 = (0.0 if conf else e.logprior) + tau * main + conc_draw
            # a night step test sees a draw running inside a section like a leak there (the board adds every check's 'no leak' term to
            # this mixture, so the draw gets the difference); listening and digging do not find a draw
            l0 += sum(float(c["ll"][j] - c["ll0"]) for c in board.checks if c["kind"] == "step")
            LL0.append(l0)
            LLJ.append(np.full(len(cand), -np.inf))
            labels.append(e.eid)
        LL0 = np.array(LL0)
        none = float(logsumexp(LL0))
        board.base_none = none + h["zone0"]
        nd = [k for k, lab in enumerate(labels) if lab is None or board.expl[lab].what != "night_draw"]
        unk = float(logsumexp(LL0[nd])) if nd else none     # a junction nobody hypothesised: as likely as 'no leak' under every explanation
        board.base = self.s.log_prior + unk + h["zone"]
        if len(cand):
            M = np.stack(LLJ)                      # [explanations, candidates]
            board.base[cand] = self.s.log_prior[cand] + logsumexp(M, axis=0) + h["zone"][cand]
        # posterior of each explanation (pressure and inflow evidence; every other item of the board is the same under all of them)
        pl, lpr = board.prior_leak, self.s.log_prior
        non = np.setdiff1d(np.arange(len(lpr)), cand)
        tot = []
        for k, lab in enumerate(labels):
            a = np.log(1 - pl) + LL0[k]
            is_draw = lab is not None and board.expl[lab].what == "night_draw"
            b = -np.inf if is_draw else np.log(pl) + float(logsumexp(np.concatenate([lpr[cand] + LLJ[k], lpr[non] + LL0[k]])))
            tot.append(np.logaddexp(a, b))
        tot = np.array(tot)
        post = np.exp(tot - logsumexp(tot))
        board.expl_none = float(post[labels.index(None)]) if None in labels else 0.0
        if getattr(board, "prior_alarm", None) is not None:      # an alarm opened the case: it is evidence of a leak only for the share of
            pe = self.s.cfg.prior_leak_explained                  # the anomaly that no other explanation accounts for
            board.prior_leak = pe + (board.prior_alarm - pe) * board.expl_none
        for lab, pk in zip(labels, post):
            if lab is not None:
                board.expl[lab].post = float(pk)
        for e in board.expl.values():
            if e.status == "refuted":
                e.post = 0.0
            if e.what in ("valve_setting", "instrument") and e.status != "refuted":
                e.fit = self._best_fit(board, e)
        board.touch()

    def _best_fit(self, board, e: Explanation) -> dict:
        ev, sidx = board._hyp["ev"], board._hyp["sidx"]
        if e.what == "valve_setting":
            nf = ev.night_frac
            d = nf * e.d_night[sidx] + (1 - nf) * e.d_day[sidx]
            w = 1 / (ev.drop_var + 0.01 ** 2)
            a = float(np.clip((d * w * ev.drop_m).sum() / max((d * w * d).sum(), 1e-9), SCALE[0], SCALE[-1]))
            return {"drift_m": round(a * PROBE_M, 2)}
        if e.element in INLET_METERS:
            return {"bias": round(max(ev.q_hat, 0.0), 1)}
        k = board.sensor_names.index(e.element)
        return {"bias": round(-float(ev.drop_m[k]), 2)}

    def best_draw_sites(self, board, k: int = 3) -> list[int]:
        """Hypothesised junctions where a draw would best explain last night's busiest 3 h (already simulated, so free), spread apart."""
        evb = board._hyp.get("ev_block")
        cand = np.where(board.cand)[0].tolist()
        if evb is None or not cand:
            return []
        llj, _ = _loglik(self.s._phys(), evb, cand, board._hyp["sidx"], None)
        out = []
        for i in np.argsort(-llj):
            j = cand[int(i)]
            if all(self.net.pipe_distance[j, o] > 600 for o in out):
                out.append(j)
            if len(out) >= k:
                break
        return out

    def groups(self, board) -> list[dict]:
        """Explanations one inspection settles together: a pipe's whole stretch of main, or the draws within reach of one night watch.
        -> [{"kind", "target", "post" (summed), "eids", "label"}], largest first."""
        out, used = [], set()
        ex = sorted([e for e in board.expl.values() if e.status == "open"], key=lambda e: -e.post)
        D = self.net.pipe_distance
        for e in ex:
            if e.eid in used:
                continue
            kind, tgt = e.inspect_args()
            if e.what == "night_draw":                             # one patrol covers the section's hydrants
                mem = [x for x in ex if x.what == "night_draw" and x.eid not in used and x.inspect_args() == (kind, tgt)]
            else:
                mem = [e]
            used |= {x.eid for x in mem}
            out.append({"kind": kind, "target": tgt, "post": float(sum(x.post for x in mem)), "eids": [x.eid for x in mem],
                        "label": e.label() + (f" (and {len(mem) - 1} more within reach)" if len(mem) > 1 else "")})
        return sorted(out, key=lambda g: -g["post"])

    # ------------------------------------------------------------------ inspections
    def apply_inspection(self, board, kind: str, target: str, found: bool, source: str = "inspection") -> None:
        """A crew checked one element. Found: that explanation is confirmed (created if nobody had proposed it). Not found: refuted."""
        chain = set(self.series_chain(target)) if kind == "valve" else set()
        match = [e for e in board.expl.values() if e.inspect_args() == (kind, target)
                 or (kind == "valve" and e.what == "pipe_closed" and e.element in chain)
                 or (kind == "watch" and e.what == "night_draw" and (
                     (target in self.s.sections.areas and self.s.sections.areas[target]["mask"][self.net.index[e.element]])
                     or (target in self.net.index and self.net.pipe_distance[self.net.index[e.element], self.net.index[target]] <= 300)))]
        if found and not match:
            what = {"valve": "pipe_closed", "prv": "valve_setting", "sensor": "instrument", "meter": "instrument", "watch": "night_draw"}[kind]
            el = target
            if kind == "watch" and target in self.s.sections.areas:      # the patrol names the hydrant; here its section stands for it
                m = np.asarray(self.s.sections.areas[target]["mask"], bool)
                el = self.net.junctions[int(np.argmax(m))]
            e = Explanation(f"X{len(board.expl) + 1}", what, el, source, self._logprior(what, board))
            try:
                self._simulate(board, e)
            except BudgetExceeded:
                pass
            board.expl[e.eid] = e
            match = [e]
        for e in match:
            e.status = "confirmed" if found else "refuted"
            e.note = "confirmed by inspection" if found else "inspected: not this"
        if found:
            for e in board.expl.values():
                if e not in match:
                    e.status = "refuted" if e.status == "open" else e.status
        self.refresh(board)


def _shift(ev, d: np.ndarray, dq: float):
    from dataclasses import replace
    return replace(ev, drop_m=ev.drop_m - d, q_hat=ev.q_hat - dq, z=(ev.drop_m - d) / np.sqrt(ev.drop_var))


def _loglik(phys, ev, cand, sidx, infl):
    if infl is not None:
        from dataclasses import replace
        ev = replace(ev, drop_var=ev.drop_var + infl)
    return phys.pressure_loglik(ev, cand, sidx)


def _night_excess(time_s, flow, now_s, history_days: float = 7.0):
    t = np.asarray(time_s, float)
    Qin = (flow[:, 0] + flow[:, 1]) / 3.6
    Qp = flow[:, 2] / 3.6 if flow.shape[1] > 2 else np.zeros_like(Qin)
    q = Qin - Qp
    slot = ((t % 86400) // 3600).astype(int)
    base = t < history_days * 86400
    prof = np.array([q[base & (slot == h)].mean() if (base & (slot == h)).any() else np.nan for h in range(24)])
    hours = [22, 23, 0, 1, 2, 3, 4, 5]
    last22 = np.floor((now_s - 22 * 3600) / 86400) * 86400 + 22 * 3600              # the latest 22:00 before now
    night_start = last22 if now_s - last22 >= 4 * 3600 else last22 - 86400          # the current night if 4 h of it are in, else the last one
    ex = []
    for k, hh in enumerate(hours):
        m = (t >= night_start + k * 3600) & (t < night_start + (k + 1) * 3600) & (t <= now_s)
        ex.append(q[m].mean() - prof[hh] if m.any() else np.nan)
    sd = float(np.nanstd([q[base & (slot == h)].std() for h in hours]) + 0.3)
    return np.array(ex), sd, night_start


def night_block(time_s, flow, now_s) -> float:
    """End time (s) of last night's busiest 3 hours of extra inflow."""
    ex, _, t0 = _night_excess(time_s, flow, now_s)
    pos = np.clip(np.nan_to_num(ex), 0, None)
    i = int(np.argmax([pos[k:k + 3].sum() for k in range(6)]))         # (hours after now are 0: the block ends no later than now)
    return float(t0 + (i + 3) * 3600)


def inflow_concentration(time_s, flow, now_s, history_days: float = 7.0) -> tuple[float, float, float]:
    """How the extra NIGHT inflow of the last night (22-06 h) is spread over time: (share inside its busiest 3 h, extra L/s, its sd).
    A leak adds water all night (share ~ 3/8); a draw adds it in a block (share ~ 1). Code only."""
    ex, sd, _ = _night_excess(time_s, flow, now_s, history_days)
    ex = ex[~np.isnan(ex)]
    if len(ex) < 4:
        return float("nan"), 0.0, sd
    if ex.mean() < 2 * sd / np.sqrt(len(ex)):
        return float("nan"), float(ex.mean()), sd
    pos = np.clip(ex, 0, None)
    best = max(pos[i:i + 3].sum() for i in range(len(pos) - 2))
    return float(best / max(pos.sum(), 1e-9)), float(ex.mean()), sd


def conc_terms(c: float) -> tuple[float, float]:
    """log-likelihood of the night-inflow shape under (a leak or no change, a night draw)."""
    if not np.isfinite(c):
        return 0.0, 0.0
    g = lambda m: float(-0.5 * ((c - m) / CONC_SD) ** 2)
    return g(CONC_LEAK), g(CONC_DRAW)
