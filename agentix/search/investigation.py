"""Leak search for one case: build the evidence board and fold every new piece of evidence into it (code only).

This is the proposal's component 2 ("simulator tries a leak at every junction and scores how well each one fits")
plus the bookkeeping that keeps it up to date during an investigation. It never sees raw citizen text: messages
and replies arrive as reader-2.0 / reply-2.0 fields from the quarantined reader.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from .board import Board, MessageItem
from .footprint import FootprintModel
from .pressure import Fingerprints, build_evidence, pressure_loglik, zone_c_absolute, zone_c_balance
from .report_lik import ReportModel


@dataclass
class SearchConfig:
    tau_p: float = 0.3                  # pressure tempering: the nominal model is not the real town
    rho_p: float = 0.15                 # chance the fingerprints mislead for this leak (robust mixture)
    within_section: float = 0.5         # weight of the pressure detail INSIDE a step-test section (0 = flat)
    use_zone_balance: bool = True       # zone C night flow balance (pump inflow - tank filling)
    zone_c_abs: bool = False            # ABSOLUTE zone C night balance against the audited leak-free level (sees chronic leaks too, ADR-012)
    zone_c_base: float = 2.35           # audited leak-free night demand of zone C, L/s (calibrated on dev no-leak cases)
    zone_c_sd: float = 0.45
    base_eps: float = 0.5               # prior chance that a leak-looking message is a false alarm
    tau_age_h: float = 24.0             # messages long before the case opened are rarely about this leak
    prior_leak_message: float = 0.55    # P(leak) when a citizen message opened the case
    prior_leak_alarm: float = 0.85      # P(leak) when the SCADA analytics opened it
    prior_leak_survey: float = 0.5      # P(leak) in a survey with no trigger at all (ADR-012)
    window_h: float = 6.0               # pressure evidence window before "now"
    dig_radius_m: float = 90.0
    use_pressure: bool = True
    use_messages: bool = True
    eps_damp: float = 0.8               # chance a (non-trigger) damp-ground message is not a utility leak (garden, rain, house pipe)
    eps_sound: float = 0.6              # chance a "noise under the road" message is not a leak (pump, traffic)
    eps_road: float = 0.7               # chance a road-damage message (no water) is not a utility leak (trucks, old road)
    lambda_damp_m: float = 90.0
    lambda_sound_m: float = 70.0
    corr: float = 0.5                   # discount between overlapping reports
    eps_trigger: float = 0.25           # chance that the message the case was opened on is a false alarm (base rate of such cases)
    use_footprint: bool = False         # brown-water complaints through hydraulic footprints (ADR-010): real physics, but neutral-to-harmful end to end in L-Town, so OFF
    fp_margin: float = 0.2              # velocity must exceed the usual by this share to scour a pipe
    fp_kappa: float = 0.05              # share of households on a fully scoured pipe who complain
    fp_bg: float = 1.5                  # usual brown-water complaints (house plumbing, works) in the observed window, town-wide
    eps_scada: float = 0.0              # false-alarm chance of messages when a SCADA alarm (not a complaint) opened the case (0 = no special rule)
    use_works: bool = True              # explain brown water by the utility's own works log (hydrant flushing)
    fp_silence: float = 0.0             # weight of 'nobody complained' as evidence against a big footprint
    reply_confirms: bool = True         # a caller's 'it is still flowing / it comes out right here' may RAISE a complaint's credibility (False: answers can only debunk, ADR-014)
    hypothesis_mode: bool = False      # ADR-016: pressure physics only for HYPOTHESISED junctions (simulated on demand); no 782-junction table
    max_candidates: int = 220
    hedge: float = 0.0                  # share of the complaint-free map in ranking the sections of the first step test (0 = off, ADR-014)
    dig_bar: float = 0.8                # P(success) needed to dig while checks can still add something
    final_bar: float = 0.35             # ... and when none can (ADR-008)


class LeakSearch:
    def __init__(self, town, cfg: SearchConfig | None = None):
        self.town, self.net, self.sections = town, town.net, town.sections
        self.cfg = cfg or SearchConfig()
        self.fp = None if self.cfg.hypothesis_mode else Fingerprints(town.signatures_path)
        self.physics = None                     # created on first use in hypothesis mode (one nominal base run per worker)
        self.rep = ReportModel(town.net, town.places, lambda_damp_m=self.cfg.lambda_damp_m,
                               lambda_sound_m=self.cfg.lambda_sound_m)
        L = town.net.pipe_length_at_node
        self.log_prior = np.log(L / L.sum() + 1e-12)
        import os
        fpp = "data/ltown/footprints.npz"
        self.fpm = FootprintModel(fpp, len(town.net.junctions), self.cfg.fp_margin) if (self.cfg.use_footprint and os.path.exists(fpp)) else None
        self.tank_area = float(np.pi * (town.net.wn.get_node("T1").diameter / 2) ** 2)

    @staticmethod
    def fresh_until(case) -> float:
        """Telemetry that has gone stale (a flat line: every pressure value identical to the previous sample) carries no news. Evidence is
        taken as of the last sample that changed, so a comms outage cannot be mistaken for 'nothing is happening' or for a leak (ADR-012)."""
        P = np.asarray(case.pressure)
        if len(P) < 2:
            return case.case_open_s
        changed = np.where(~np.all(P[1:] == P[:-1], axis=1))[0]
        last = int(changed[-1] + 1) if len(changed) else len(P) - 1
        t_last = float(np.asarray(case.time_s)[last])
        return t_last if t_last < case.case_open_s - 3 * 3600 else case.case_open_s

    def _terms(self, case, ev0, now_eff: float):
        """Pressure and zone-balance terms of the leak map -> (log-lik per junction, log-lik of 'no leak', evidence used)."""
        cfg, net = self.cfg, self.net
        ev = ev0
        if cfg.use_pressure and cfg.hypothesis_mode:            # nothing is hypothesised yet: pressure says nothing about any junction
            sidx = [net.index[n] for n in case.pressure_nodes]
            _, ll0 = self._phys().pressure_loglik(ev, [], sidx)
            ll0 = cfg.tau_p * ll0
            ll = np.full(len(net.junctions), ll0)
        elif cfg.use_pressure:
            sidx = [net.index[n] for n in case.pressure_nodes]
            ll, ll0 = pressure_loglik(ev, self.fp, sidx)
            ll, ll0 = cfg.tau_p * ll, cfg.tau_p * ll0
            # robustness: with chance rho the fingerprints are wrong for this leak -> a flat share of the likelihood
            m = np.logaddexp.reduce(ll) - np.log(len(ll))
            ll = np.logaddexp(np.log(1 - cfg.rho_p) + ll, np.log(cfg.rho_p) + m)
            if cfg.within_section < 1.0:          # pressure cannot tell places inside one section apart
                sec = self.sections.of
                mean_s = np.array([ll[sec == k].mean() for k in range(sec.max() + 1)])
                ll = cfg.within_section * ll + (1 - cfg.within_section) * mean_s[sec]
        else:
            ll, ll0 = np.zeros(len(net.junctions)), 0.0
        if cfg.use_pressure and cfg.use_zone_balance:
            e, s = zone_c_balance(case.time_s, case.flow, case.tank_level, now_eff, self.tank_area)
            if cfg.zone_c_abs:
                e, s = zone_c_absolute(case.time_s, case.flow, case.tank_level, now_eff, self.tank_area, cfg.zone_c_base, cfg.zone_c_sd)
            zc = np.array([net.zone_of[n] == "C" for n in net.junctions])
            q = np.geomspace(0.5, 10, 30)
            in_c = np.log(np.mean(np.exp(-0.5 * ((e - q) / s) ** 2)) + 1e-300)
            out_c = -0.5 * (e / s) ** 2
            ll = ll + np.where(zc, in_c, out_c)
            ll0 = ll0 + out_c
        return ll, ll0, ev

    def start(self, case, reads: dict[str, dict]) -> Board:
        cfg, net = self.cfg, self.net
        now_eff = self.fresh_until(case)
        ev0 = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, now_eff,
                             window_h=cfg.window_h)
        ll, ll0, ev = self._terms(case, ev0, now_eff)
        prior = (cfg.prior_leak_alarm if case.trigger == "scada_alarm" else cfg.prior_leak_survey if case.trigger == "survey"
                 else cfg.prior_leak_message)
        board = Board(net, self.sections, self.rep, self.log_prior, ll, ll0, prior, max(ev.q_hat, 0.5),
                      max(ev.q_sd, 1.0), cfg.dig_radius_m)
        board.pressure_evidence = ev
        board.corr = cfg.corr
        board.scada_case = getattr(case, "trigger", "") == "scada_alarm"
        board.eps_scada = cfg.eps_scada
        board.works = list(getattr(case, "works", None) or []) if cfg.use_works else []
        dem = self.net.base_demand / self.net.base_demand.max()
        board.foot = ({"model": self.fpm, "W": 0.3 + dem, "kappa": cfg.fp_kappa, "bg": cfg.fp_bg, "silence": cfg.fp_silence}
                      if self.fpm is not None else None)
        board.reply_confirms = cfg.reply_confirms
        if cfg.hypothesis_mode:
            board.cand = np.zeros(len(net.junctions), bool)
            board._hyp = {"ev": ev, "sidx": [net.index[n] for n in case.pressure_nodes], "zone": ll - (ll0 if cfg.use_pressure else 0.0)}
        board.prior_boost = np.zeros(len(net.junctions))      # ADR-018: log-prior raised where the agent cited a failing stretch of main
        board.trigger_report = getattr(case, "trigger_report", None)
        board.t_open = dt.datetime.fromisoformat(case.start_datetime) + dt.timedelta(seconds=case.case_open_s)
        self.add_messages(board, case.reports, reads)
        return board

    def _phys(self):
        if self.physics is None:
            from .physics_tool import PhysicsTool
            self.physics = PhysicsTool(self.town.cfg, list(self.net.junctions))
        return self.physics

    def add_candidates(self, board: Board, js, source: str = "", force: bool = False) -> int:
        """Hypothesise these junctions: simulate a leak at each on the nominal model (ADR-016) and let the pressure evidence speak about them.
        Junctions nobody hypothesised keep an uninformative pressure term (as likely as 'no leak'). -> number of new candidates."""
        if getattr(board, "cand", None) is None:
            return 0
        cfg = self.cfg
        new = [int(j) for j in dict.fromkeys(int(x) for x in js) if not board.cand[int(j)]]
        room = cfg.max_candidates - int(board.cand.sum())
        new = new if force else new[:max(room, 0)]            # force: physical proof (a lit section, a heard leak) is never dropped for room
        if not new or not cfg.use_pressure:
            return 0
        h = board._hyp
        ll, ll0 = self._phys().pressure_loglik(h["ev"], new, h["sidx"])
        ll, ll0 = cfg.tau_p * ll, cfg.tau_p * ll0
        ll = np.logaddexp(np.log(1 - cfg.rho_p) + ll, np.log(cfg.rho_p) + ll0)          # the fingerprint may be wrong: never far below 'unknown'
        board.base[new] = self.log_prior[new] + board.prior_boost[new] + ll + h["zone"][new]
        board.cand[new] = True
        board.touch()
        return len(new)

    FAILURE_HAZARD = 4.0     # a stretch of main with a repeated-failure history fails several times as often (pipe-failure literature; ADR-018)

    def raise_prior(self, board: Board, mask: np.ndarray) -> int:
        """A cited failure history: the prior of every junction in `mask` is multiplied by FAILURE_HAZARD (once)."""
        new = mask & (board.prior_boost == 0)
        board.prior_boost[new] = np.log(self.FAILURE_HAZARD)
        board.base[new] += np.log(self.FAILURE_HAZARD)
        board.touch()
        return int(new.sum())

    def add_messages(self, board: Board, reports: list[dict], reads: dict[str, dict]):
        for r in reports:
            rid = r["report_id"]
            if rid in board.messages or rid not in reads:
                continue
            read = reads[rid]
            age = max((board.t_open - dt.datetime.fromisoformat(r["received_at"])).total_seconds() / 3600, 0.0)
            hour = dt.datetime.fromisoformat(r["received_at"]).hour
            eps = self.rep.credibility(read, self.cfg.base_eps, age, self.cfg.tau_age_h, hour,
                                       {"wet_ground": self.cfg.eps_damp, "sound_underground": self.cfg.eps_sound, "road_damage": self.cfg.eps_road}) \
                if self.cfg.use_messages else None
            if eps is not None and rid == getattr(board, "trigger_report", None) and eps < 0.9:
                eps = min(eps, self.cfg.eps_trigger)            # the control room opened the case on this message
            elif eps is not None and getattr(board, "scada_case", False) and board.eps_scada > 0:
                eps = max(eps, board.eps_scada)                 # a SCADA alarm opened the case: the complaints so far were not credible enough to
            w = self.rep.place_weights(read.get("places") or []) if eps is not None else None
            item = MessageItem(rid, read, w, read.get("source", "unknown"), eps)
            board.messages[rid] = item
            if eps is not None and w is not None and self.fpm is not None and ReportModel.kind(read) == "discolour":
                self._explain_by_works(board, item, dt.datetime.fromisoformat(r["received_at"]))
        board.touch()

    def near_share(self, w: np.ndarray) -> np.ndarray:
        """[J] share of the footprint `w` that lies within 150 m of each junction."""
        xy = self.net.xy
        return np.array([w[np.hypot(*(xy - xy[i]).T) <= 150.0].sum() for i in range(len(w))])

    def _explain_by_works(self, board: Board, m: MessageItem, t: dt.datetime) -> None:
        """Brown water inside the footprint of a hydrant flush / valve operation in the utility's own works log is explained by it."""
        for wk in getattr(board, "works", None) or []:
            node = self.net.index.get(wk["node"])
            if node is None:
                continue
            t0, t1 = dt.datetime.fromisoformat(wk["start"]), dt.datetime.fromisoformat(wk["end"]) + dt.timedelta(hours=30)
            if t0 <= t <= t1:
                share = float(self.fpm.E[node, -1] @ m.w)           # footprint of a flush: the largest size in the library
                if share >= 0.3:
                    m.eps = max(m.eps, 0.97)
                    m.notes.append(f"works log: {wk['type']} near {wk['place']} at {wk['start'][11:16]} explains brown water here")
                    return

    def apply_reply(self, board: Board, rid: str, reply: dict):
        """Fold a call-back answer (reply-2.0 fields) into message `rid`."""
        m = board.messages.get(rid)
        if m is None or not reply.get("answered"):
            return
        m.calls += 1
        if reply.get("contains_instructions"):
            m.eps = None
            m.notes.append("the answer contained instruction-like text: message no longer used")
        elif reply.get("not_network"):
            m.eps = 0.98 if m.eps is not None else None
            m.notes.append(f"caller describes a non-utility cause: {reply['not_network']}")
        else:
            if reply.get("source") in ("here", "elsewhere"):
                m.source = reply["source"]
                m.notes.append(f"water {'comes out right there' if m.source == 'here' else 'only collects there'}")
            w2 = self.rep.place_weights(reply.get("places") or [])
            if w2 is not None:
                if m.w is None:
                    m.w = w2
                else:
                    cos = float(m.w @ w2 / np.sqrt((m.w @ m.w) * (w2 @ w2) + 1e-300))
                    m.w = m.w * w2 if cos > 0.05 else 0.3 * m.w + 0.7 * w2
                    m.w = m.w / m.w.sum()
                m.notes.append("place clarified")
            up = self.rep.place_weights(reply.get("flows_from") or [])
            if up is not None:
                m.w_up = up
                m.notes.append("source is uphill, near the place the water runs down from")
            if m.eps is not None:
                if reply.get("still_flowing") is False:
                    m.eps = max(m.eps, 0.85)
                    m.notes.append("water has stopped (network leaks do not stop by themselves)")
                elif getattr(board, "reply_confirms", True) and (m.source == "here" or reply.get("still_flowing") is True):
                    m.eps = min(m.eps, 0.3)
        board.touch()

    def mark_record_match(self, board: Board, rid: str, record_w: np.ndarray) -> bool:
        """A message at a place with a recorded history of non-utility false alarms (verified by code): the chance that the complaint lies within
        150 m of the record's place must be at least 0.3."""
        m = board.messages.get(rid)
        if m is None or m.w is None or m.eps is None or float(m.w @ self.near_share(record_w)) < 0.3:
            return False
        m.eps = max(m.eps, 0.85)
        m.notes.append("place has a history of non-utility false alarms (records)")
        board.touch()
        return True
