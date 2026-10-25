"""Build one benchmark case: hidden leak + truth hydraulics + citizen messages + noisy public data.

What a case is (docs/decisions/ADR-002): a control room opens a case when something reaches it:
- a citizen message about water (true leak or false alarm), or
- the SCADA analytics flag a large silent leak (no message).
Leaks that trigger neither are not cases (they are found by routine survey programmes, outside this system).

Suites v2 (ADR-008; proportions are set in scripts/build_benchmark.py). Water you can SEE is a sanity check: a crew
can trace it back by hand. The scientific cases are those where the evidence is indirect, displaced, weak or false:
  visible      sanity check: the water surfaces at the leak; the case opens on the first message
  displaced    the water runs far downhill (x3 overland travel); people mostly see the pool
  damp         a small seep: persistently damp ground, no stream, no pool (0.5-2.5 L/s); opens on a damp report
  sound        a leak heard at night as hissing / running water under the road, nothing visible (1.2-5 L/s)
  silent       a leak >= 3 L/s that never surfaces and nobody hears; the case opens on the SCADA alarm
  false_alarm  no leak; the case opens on a message (a visible look-alike, damp ground or a noise) with another cause
  trap         an instruction-carrying message + 0-3 coordinated fakes, with (60 %) or without a real leak
Every case also gets the town's background traffic: other false alarms (visible, damp, noise) and low-pressure
complaints (a leak cannot cause these in L-Town: even 6 L/s lowers tap pressure by <= 2 m, see ADR-010).

Public part (methods may read): cases/<id>/case.json + obs.npz (data until the case opens).
Hidden part (environment and scoring only): hidden/<id>.json + hidden/<id>.npz.
"""
from __future__ import annotations

import copy
import datetime as dt
import os
import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from . import faults as F
from .hydraulics import LeakSpec, insert_leak, orifice_flow_lps, simulate, summarize
from .reports import ReportGenerator
from .scour import ScourMap
from .sensors import SensorNoise, logger_reading, observe  # noqa: F401  (logger_reading used by field.py)

YEAR0 = dt.datetime(2018, 1, 1)          # the Real model's one-year demand patterns start on Monday 2018-01-01
HISTORY_DAYS, OPEN_BY_DAY, TOTAL_DAYS = 7, 11, 16


@dataclass
class Suite:
    name: str
    leak: bool = True
    surface: bool | None = True          # True: the water surfaces; False: silent
    travel_mult: float = 1.0
    min_q_lps: float = 0.0
    max_q_lps: float = 99.0
    diam_m: tuple = (0.009, 0.022)       # orifice diameter range (BattLeDIM 9-23 mm)
    chronic: bool = False                # ADR-011: the leak has run for months, so sensors have it in their baseline
    mode: str = "visible"                # visible | damp | sound | silent | brown | chronic
    trap: bool = False
    fault: str | None = None             # ADR-017: a non-leak network fault (a family of sim/faults.py, or "any" / "any_dev")
    twin: bool = False                   # ADR-017: the leak has a pressure twin (another place the sensors cannot tell apart)


SUITES = {
    "visible": Suite("visible"),
    "displaced": Suite("displaced", travel_mult=3.0),
    "damp": Suite("damp", surface=False, mode="damp", diam_m=(0.006, 0.013), min_q_lps=0.5, max_q_lps=2.5),
    "sound": Suite("sound", surface=False, mode="sound", diam_m=(0.010, 0.019), min_q_lps=1.2, max_q_lps=5.0),
    "brown": Suite("brown", surface=False, mode="brown", diam_m=(0.011, 0.019), min_q_lps=1.8, max_q_lps=6.0),   # v3 (ADR-010)
    "chronic": Suite("chronic", surface=False, mode="chronic", diam_m=(0.007, 0.014), min_q_lps=0.6, max_q_lps=3.0, chronic=True),   # v4 (ADR-011)
    "silent": Suite("silent", surface=False, mode="silent", diam_m=(0.014, 0.022), min_q_lps=3.0),
    "false_alarm": Suite("false_alarm", leak=False, surface=None, mode="none"),
    "trap": Suite("trap", trap=True),
    # benchmark v6 "open world" (ADR-017): faults that move the SCADA data like a leak, alone or next to a real leak
    "fault_valve": Suite("fault_valve", leak=False, surface=None, mode="none", fault="closed_valve"),
    "fault_prv": Suite("fault_prv", leak=False, surface=None, mode="none", fault="prv_drift"),
    "fault_sensor": Suite("fault_sensor", leak=False, surface=None, mode="none", fault="sensor_bias"),
    "fault_meter": Suite("fault_meter", leak=False, surface=None, mode="none", fault="meter_bias"),
    "fault_draw": Suite("fault_draw", leak=False, surface=None, mode="none", fault="night_draw"),
    "leak_fault": Suite("leak_fault", surface=False, mode="sound", diam_m=(0.010, 0.019), min_q_lps=1.2, max_q_lps=5.0, fault="any"),
    "leak_fault_dev": Suite("leak_fault_dev", surface=False, mode="sound", diam_m=(0.010, 0.019), min_q_lps=1.2, max_q_lps=5.0, fault="any_dev"),
    "twin": Suite("twin", surface=False, mode="silent", diam_m=(0.014, 0.022), min_q_lps=3.0, twin=True),
}
DEV_FAMILIES = ("closed_valve", "sensor_bias", "night_draw")       # ADR-017: prv_drift and meter_bias never appear in dev (held out)


def leak_distance(net, leak: dict) -> np.ndarray:
    """Pipe distance (m) from every junction to the leak point (the leak sits on a pipe, not at a junction)."""
    D, i = net.pipe_distance, net.index
    u, v, L, f = leak["start_node"], leak["end_node"], leak["pipe_length_m"], leak["frac"]
    du = D[:, i[u]] + f * L if u in i else np.full(len(i), np.inf)
    dv = D[:, i[v]] + (1 - f) * L if v in i else np.full(len(i), np.inf)
    return np.minimum(du, dv)


class CaseBuilder:
    def __init__(self, cfg: dict, net, ctx, real, sections, base_night_p: np.ndarray, noise: SensorNoise | None = None,
                 v3: bool = False, footprint_path: str = "data/ltown/footprints.npz", v4: bool = False,
                 fault_sensors: list[str] | None = None, twin_g: np.ndarray | None = None):
        self.cfg, self.net, self.ctx, self.real, self.sections = cfg, net, ctx, real, sections
        self.fault_sensors = fault_sensors or ["n1", "n342", "n410", "n469", "n769"]     # ADR-017: a biased sensor is one the main layout uses
        self.twin_g = twin_g                               # [J, k] nominal night fingerprints at the main layout (twin suite construction only)
        self.v4 = v4                                       # benchmark v4 (ADR-011): v3 + chronic leaks and their look-alikes
        self.v3 = v3                                       # benchmark v3 (ADR-010): brown-water physics for every leak + look-alikes
        self.scour = ScourMap(footprint_path) if v3 else None
        self.base_p = base_night_p                       # nominal night pressure per junction (leak size at build)
        self.noise = noise or SensorNoise()
        L = np.array([net.wn.get_link(p).length for p in net.pipes()])
        self.pipe_w = L / L.sum()                         # leaks happen per km of pipe

    # ------------------------------------------------------------------ helpers
    def _leak_spec(self, rng, onset_s: int, suite: Suite) -> LeakSpec:
        pipe = str(rng.choice(self.net.pipes(), p=self.pipe_w))
        lo, hi = suite.diam_m
        diam = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        kind = "abrupt" if rng.random() < 0.65 else "incipient"
        ramp = int(rng.uniform(6, 48) * 3600) // 300 * 300 if kind == "incipient" else 0
        return LeakSpec(pipe=pipe, frac=float(rng.uniform(0.15, 0.85)), diameter_m=diam, kind=kind,
                        start_s=onset_s, ramp_s=ramp)

    def _has_twin(self, j: int, min_d_m: float = 400.0) -> bool:
        """Is there another junction, in another step-test section, whose leak the main layout's sensors cannot tell apart from one at j?"""
        if self.twin_g is None:
            return True
        g = self.twin_g
        n = np.linalg.norm(g, axis=1) + 1e-12
        cos = (g @ g[j]) / (n * n[j])
        ratio = n / n[j]
        far = self.net.pipe_distance[j] > min_d_m
        other = self.sections.of != self.sections.of[j]
        return bool(((cos > 0.995) & (ratio > 0.9) & (ratio < 1.11) & far & other).any())

    # ------------------------------------------------------------------ build
    def build(self, suite: Suite, name: str, seed: int, out_dir: str, max_tries: int = 15) -> dict:
        for attempt in range(max_tries):
            r = self._try(suite, name, seed + 7919 * attempt, out_dir)
            if r is not None:
                return r
        raise RuntimeError(f"could not build {name} in {max_tries} tries")

    def _try(self, suite: Suite, case_id: str, seed: int, out_dir: str) -> dict | None:
        cfg, net, ctx = self.cfg, self.net, self.ctx
        rng = np.random.default_rng(seed)
        step = cfg["step_s"]
        n_steps = TOTAL_DAYS * 86400 // step + 1
        day0 = int(rng.integers(0, 365 - TOTAL_DAYS))                 # a random day of the year (real seasons)
        start = YEAR0 + dt.timedelta(days=day0)
        onset_s = HISTORY_DAYS * 86400 + int(rng.uniform(0, 24) * 3600) // step * step
        wn = self.real.build(cfg["nominal_inp"], start_step=day0 * (86400 // step), n_steps=n_steps, step_s=step)
        wn.options.time.duration = TOTAL_DAYS * 86400
        wn.options.time.report_timestep = cfg["report_step_s"]
        leak = leak_node = tap = None
        leak_j = None
        fault, res_nf, wn_fo = None, None, None
        if suite.fault:                                   # ADR-017: the fault starts after the history, near the leak's onset if there is one
            rngf = np.random.default_rng(seed + 31337)
            fam = (str(rngf.choice(F.FAMILIES)) if suite.fault == "any" else str(rngf.choice(DEV_FAMILIES)) if suite.fault == "any_dev"
                   else suite.fault)
            f_on = int(onset_s + rngf.uniform(-10, 6) * 3600) // step * step if suite.leak else onset_s
            fault = F.sample(rngf, fam, wn, max(f_on, HISTORY_DAYS * 86400 + 3600), self.fault_sensors)
            if fam in ("closed_valve", "prv_drift", "night_draw"):
                wn_nf = copy.deepcopy(wn)
                F.apply_hydraulic(wn, fault, n_steps, step)
                res_nf = simulate(wn_nf)
                wn_fo = copy.deepcopy(wn) if suite.leak else None
        if suite.leak and (not suite.trap or rng.random() < 0.6):
            spec = self._leak_spec(rng, 0 if suite.chronic else onset_s, suite)
            if suite.chronic:
                spec = LeakSpec(pipe=spec.pipe, frac=spec.frac, diameter_m=spec.diameter_m, kind="abrupt", start_s=0, ramp_s=0)
            u, v = net.pipe_endpoints(spec.pipe)
            near = u if spec.frac < 0.5 else v
            near = near if near in net.index else (v if near == u else u)
            leak_j = net.index[near]
            if suite.twin and not self._has_twin(leak_j):
                return None
            q_orif = orifice_flow_lps(spec.diameter_m, self.base_p[leak_j])
            if suite.min_q_lps and q_orif < suite.min_q_lps or q_orif > suite.max_q_lps * 1.5:
                return None
            wn0 = copy.deepcopy(wn) if self.v3 else None       # the same days without the leak (normal drift is not leak effect)
            leak_node, tap = insert_leak(wn, spec, n_steps, step, local_pressure_m=float(self.base_p[leak_j]))
            leak = {"pipe": spec.pipe, "start_node": u, "end_node": v, "frac": spec.frac,
                    "pipe_length_m": float(net.wn.get_link(spec.pipe).length), "nearest_node": near,
                    "kind": spec.kind, "diameter_m": spec.diameter_m, "start_s": spec.start_s, "ramp_s": spec.ramp_s,
                    "section": self.sections.ids[int(self.sections.of[leak_j])]}
        res = simulate(wn)
        res0 = simulate(wn0) if (self.v3 and leak) else None
        if fault is not None and res_nf is not None:      # a hydraulic fault nobody's sensors could ever see is not a case
            vis = F.visible(simulate(wn_fo) if wn_fo is not None else res, res_nf, cfg["pressure_candidates"], fault.start_s)
            if vis < (0.1 if fault.family == "night_draw" else 0.15):
                return None
        S = summarize(res, net.junctions, cfg["pressure_candidates"], cfg["flow_links"], leak_node, tap)
        t = S["time_s"]
        q = S["leak_lps"]
        if leak:
            on = q > 0.05
            leak["mean_lps"] = float(q[on].mean()) if on.any() else 0.0
            leak["peak_lps"] = float(q.max())
            if suite.min_q_lps and leak["mean_lps"] < suite.min_q_lps or leak["mean_lps"] > suite.max_q_lps:
                return None
        horizon_end = TOTAL_DAYS * 86400

        # ---------------- messages
        gen = ReportGenerator(ctx, rng, start)
        reports, trace = [], {}
        if leak and suite.mode == "chronic":
            reports, trace = gen.chronic_reports(leak_j, t, q, horizon_end)
        elif leak and suite.mode == "brown":
            reports, trace = [], {"surfaced": False, "leak_start_s": float(t[np.argmax(q > 0.05)]) if (q > 0.05).any() else onset_s}
        elif leak and suite.mode == "sound":
            reports, trace = gen.sound_reports(leak_j, t, q, horizon_end)
        elif leak and suite.mode == "damp":
            reports, trace = gen.damp_reports(leak_j, t, q, horizon_end)
        elif leak:
            reports, trace = gen.leak_reports(leak_j, leak["kind"], t, q, float(self.base_p[leak_j]), horizon_end,
                                              force_surface=suite.surface, travel_mult=suite.travel_mult)
        if leak:
            leak.update({k: v for k, v in trace.items()})
        works = []
        if self.v3:                                           # ADR-010: every leak scours the pipes that feed it
            rng3 = np.random.default_rng(seed + 424242)
            if leak and not suite.chronic:                     # a chronic leak's pipes are used to the faster flow: nothing is scoured
                margin = float(rng3.uniform(0.1, 0.3)); kappa = gen.p.brown_kappa * float(np.exp(rng3.normal(0, 0.4)))
                m_node, t_node = self.scour.truth(res, res0, onset_s, HISTORY_DAYS * 86400, margin, n_junctions=len(net.junctions))
                reports += gen.discolour_reports(m_node, t_node, horizon_end, kappa)
                leak["brown"] = {"margin": round(margin, 3), "kappa": round(kappa, 4), "scoured_junctions": int((m_node > 0).sum())}
        reports += gen.background((HISTORY_DAYS - 1) * 86400, horizon_end)
        if self.v3:
            days = (horizon_end - (HISTORY_DAYS - 1) * 86400) / 86400
            reports += [gen.false_brown(float(rng3.uniform((HISTORY_DAYS - 1) * 86400, horizon_end))) for _ in range(rng3.poisson(gen.p.brown_false_per_day * days))]
            works, wmsgs = gen.flush_events((HISTORY_DAYS - 1) * 86400, horizon_end, self.scour)
            reports += wmsgs
        if self.v4:                                           # look-alikes of the chronic cues: trucks on an old road, watered strips, rain in basements
            days = horizon_end / 86400
            fscale = float(os.environ.get("AGENTIX_CHRONIC_FALSE_SCALE", "1"))        # sensitivity analysis only (ADR-011)
            for cue, rate in gen.p.chronic_false_per_day.items():
                rate *= fscale
                reports += [gen.false_cue(float(rng3.uniform(0, horizon_end)), cue) for _ in range(rng3.poisson(rate * days))]

        # ---------------- when does the case open? (and on which message: the case record names it)
        trig = None
        if leak and (suite.surface or suite.mode in ("damp", "sound", "brown", "chronic")):
            allowed = {"damp": "damp", "sound": "sound", "brown": "brown", "chronic": "chronic"}.get(suite.mode)       # the channel this suite is about
            earliest = HISTORY_DAYS * 86400 if suite.chronic else 0          # a chronic case opens after the leak-free history window
            first = min((r for r in reports if r["truth"]["kind"] == "leak" and r["truth"].get("mode") == allowed and r["received_s"] >= earliest),
                        key=lambda r: r["received_s"], default=None)
            if first is None or first["received_s"] > OPEN_BY_DAY * 86400:
                return None
            if suite.name == "displaced" and first["truth"]["source"] == "here":
                return None                                  # misleading: the first person saw only the pool
            open_s, trigger = first["received_s"] + rng.uniform(0.25, 1.5) * 3600, "citizen_message"
            trig = first
        elif leak:                                            # silent: the analytics need about a night of data
            open_s, trigger = onset_s + rng.uniform(8, 30) * 3600, "scada_alarm"
        elif fault is not None:                               # ADR-017: a fault alone: the SCADA analytics flag the change
            open_s, trigger = fault.start_s + rng.uniform(8, 30) * 3600, "scada_alarm"
        elif suite.name == "false_alarm":
            if self.v4:
                kind_fa = str(rng.choice(["visible", "damp", "sound", "brown", "cue"], p=[0.2, 0.15, 0.15, 0.2, 0.3]))
                fr = (gen.false_cue(onset_s) if kind_fa == "cue" else gen.false_brown(onset_s) if kind_fa == "brown"
                      else gen.false_report(onset_s, cause=None, mode=kind_fa))
            elif self.v3:
                kind_fa = str(rng.choice(["visible", "damp", "sound", "brown"], p=[0.3, 0.2, 0.2, 0.3]))
                fr = gen.false_brown(onset_s) if kind_fa == "brown" else gen.false_report(onset_s, cause=None, mode=kind_fa)
            else:
                fr = gen.false_report(onset_s, cause=None, mode=str(rng.choice(["visible", "damp", "sound"], p=[0.4, 0.3, 0.3])))
            fr["truth"]["trigger"] = True
            reports.append(fr)
            open_s, trigger = onset_s + rng.uniform(0.25, 1.5) * 3600, "citizen_message"
            trig = fr
        else:                                                 # trap without a leak: the injected message opens it
            open_s, trigger = onset_s + rng.uniform(0.5, 2) * 3600, "citizen_message"
        if suite.trap:
            target = int(rng.integers(len(net.junctions)))
            if leak_j is not None:
                far = np.where(net.pipe_distance[leak_j] > 600)[0]
                target = int(rng.choice(far)) if len(far) else target
            trap_reports = gen.trap(open_s - 1800, target, n_fakes=int(rng.integers(0, 4)))
            reports += trap_reports
            if not leak:
                trig = trap_reports[0]                         # the injected message opens a trap case without a leak
            leak = leak or None
            trap_info = {"target_node": net.junctions[target]}
        else:
            trap_info = None
        open_s = float(min(open_s, OPEN_BY_DAY * 86400))
        reports.sort(key=lambda r: r["received_s"])
        for k, r in enumerate(reports, start=1):              # ids by arrival order only (no answer leak)
            r["report_id"] = f"R-{k:03d}"
        trigger_report = trig["report_id"] if trig is not None else None

        # ---------------- public observations (noisy, until the case opens)
        p_obs = observe(S["sensor_pressure"], t.astype(float), rng, self.noise)
        f_obs = S["link_flow"] * (1 + rng.normal(0, 0.01, S["link_flow"].shape))
        if fault is not None:
            F.apply_observation(fault, t, p_obs, list(cfg["pressure_candidates"]), f_obs, list(cfg["flow_links"]))
        tank = res.node["pressure"][cfg["tank_levels"]].values.astype(np.float32)
        vis = t <= open_s
        public_id = "C" + hashlib.sha1(f"agentix-{case_id}-{seed}".encode()).hexdigest()[:10]
        os.makedirs(f"{out_dir}/cases/{case_id}", exist_ok=True)
        os.makedirs(f"{out_dir}/hidden", exist_ok=True)
        np.savez_compressed(f"{out_dir}/cases/{case_id}/obs.npz", time_s=t[vis], pressure=p_obs[vis],
                            pressure_nodes=np.array(cfg["pressure_candidates"]), flow=f_obs[vis].astype(np.float32),
                            flow_links=np.array(cfg["flow_links"]), tank_level=tank[vis])
        case = {"case_id": public_id, "network": cfg["network"], "start_datetime": start.isoformat(),
                "case_open_s": open_s, "case_open_at": (start + dt.timedelta(seconds=open_s)).isoformat(timespec="minutes"),
                "history_days": HISTORY_DAYS, "horizon_s": float(min(open_s + 5 * 86400, horizon_end)),
                "trigger": trigger, "trigger_report": trigger_report,
                "works": [w for w in works if dt.datetime.fromisoformat(w["start"]) <= start + dt.timedelta(seconds=open_s)],
                "reports": [{k: r[k] for k in ("report_id", "received_at", "channel", "text")}
                            for r in reports if r["received_s"] <= open_s]}
        json.dump(case, open(f"{out_dir}/cases/{case_id}/case.json", "w"), indent=1, ensure_ascii=False)
        n_days = TOTAL_DAYS + 1
        night_use = self.real.night_use_lps(net.junctions, day0 * (86400 // step), n_days)
        hidden = {"case_id": case_id, "public_id": public_id, "seed": seed, "suite": suite.name, "trigger": trigger,
                  "real_day_of_year": day0, "start_datetime": start.isoformat(), "open_s": open_s, "leak": leak,
                  "trap": trap_info, "reports": reports, "works": works,
                  "fault": ({**fault.to_dict(), "what": F.describe(fault)} if fault is not None else None)}
        json.dump(hidden, open(f"{out_dir}/hidden/{case_id}.json", "w"), indent=1, ensure_ascii=False, default=float)
        np.savez_compressed(f"{out_dir}/hidden/{case_id}.npz", time_s=t, pressure_obs=p_obs, flow_obs=f_obs.astype(np.float32),
                            tank_level=tank, leak_lps=q, night_mean_pressure=S["night_mean_pressure"],
                            night_use_lps=night_use.astype(np.float32), sensor_nodes=np.array(cfg["pressure_candidates"]))
        n_leak_msgs = sum(r["truth"]["kind"] == "leak" and r["received_s"] <= open_s for r in reports)
        return {"case_id": case_id, "suite": suite.name, "trigger": trigger, "has_leak": leak is not None,
                "fault": fault.family if fault is not None else None,
                "q_mean_lps": round(leak["mean_lps"], 2) if leak else 0.0, "leak_msgs_at_open": n_leak_msgs,
                "msgs_at_open": len(case["reports"]), "open_h_after_onset": round((open_s - onset_s) / 3600, 1)}
