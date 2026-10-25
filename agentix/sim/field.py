"""The investigation environment for one case: a method acts, answers come from the hidden truth.

Every call is logged with time, cost and result. Errors are returned as {"ok": False, "error": ...} and never
raised, so tool-call accuracy and recovery can be measured. With common random numbers (always on), the luck of an
action depends only on WHICH action it is (case, kind, target, repeat count), so two methods doing the same thing
get the same answer.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import zlib

import numpy as np
import yaml

from . import faults as F
from .cases import leak_distance
from .citizen import Citizen
from .operator import Operator
from .sensors import SensorNoise, logger_reading


class CaseEnv:
    def __init__(self, bench_dir: str, case_name: str, net, sections, places, ctx, layout: list[str],
                 field_yaml: str = "configs/field.yaml", text_variant: str = "realistic", max_checks: int | None = None,
                 noise: SensorNoise | None = None):
        self.net, self.sections, self.places, self.layout = net, sections, places, list(layout)
        self.ctx = ctx
        self.cfg = yaml.safe_load(open(field_yaml))
        self.case = json.load(open(f"{bench_dir}/cases/{case_name}/case.json"))
        self.hid = json.load(open(f"{bench_dir}/hidden/{case_name}.json"))
        self.arr = np.load(f"{bench_dir}/hidden/{case_name}.npz")
        if text_variant == "realistic":
            alt = _realistic(f"{bench_dir}/hidden/{case_name}.realistic.json")
            for r in self.hid["reports"]:
                r["text"] = alt.get(r["report_id"], r["text"])
        self.name = case_name
        self.leak = self.hid["leak"]
        self.start = dt.datetime.fromisoformat(self.case["start_datetime"])
        self.now = float(self.case["case_open_s"])
        self.horizon = float(self.case["horizon_s"])
        self.max_checks = max_checks if max_checks is not None else self.cfg["budget"]["field_checks"]
        self.checks_used, self.loggers, self.canvasses = 0, {}, 0
        self.noise = noise or SensorNoise()
        self.operator = Operator()
        self.citizen = Citizen(net, places, ctx.street_of, ctx.nearest_landmark, self.leak)
        self.log: list[dict] = []
        self.done, self.outcome = False, None
        self._n: dict[str, int] = {}
        self._sensor_cols = [list(self.arr["sensor_nodes"]).index(n) for n in self.layout]
        self._dist = leak_distance(net, self.leak) if self.leak else np.full(len(net.junctions), np.inf)
        fd = self.hid.get("fault")                         # ADR-017: a non-leak fault (truth); found and fixed only by an inspection
        self.fault = F.FaultSpec(**{k: (tuple(v) if k == "draw_h" else v) for k, v in fd.items() if k != "what"}) if fd else None
        self.fault_found, self.fault_fixed_s = False, None

    # ------------------------------------------------------------------ bookkeeping
    def _rng(self, kind: str, key: str) -> np.random.Generator:
        k = f"{kind}|{key}"
        n = self._n[k] = self._n.get(k, 0) + 1
        return np.random.default_rng(zlib.crc32(f"{self.name}|{k}|{n}".encode()))

    def _record(self, action: str, args: dict, result: dict, field_check: bool = False) -> dict:
        self.log.append({"t_s": self.now, "action": action, "args": args, "result": result, "field_check": field_check})
        return result

    def _err(self, action: str, args: dict, msg: str) -> dict:
        return self._record(action, args, {"ok": False, "error": msg})

    def clock(self, t_s: float | None = None) -> dt.datetime:
        return self.start + dt.timedelta(seconds=float(self.now if t_s is None else t_s))

    def checks_left(self) -> int:
        return max(self.max_checks - self.checks_used, 0)

    def _use_check(self, action: str, args: dict) -> dict | None:
        if self.done:
            return self._err(action, args, "the case is closed")
        if self.checks_used >= self.max_checks:
            return self._err(action, args, f"field-check budget used up ({self.max_checks} per case)")
        if self.now >= self.horizon:
            return self._err(action, args, "the case deadline has passed")
        self.checks_used += 1
        return None

    def _next_night_end(self) -> float:
        d = math.floor((self.now - 1 * 3600) / 86400) + 1          # the next 01:00-05:00 window
        return d * 86400 + 5 * 3600

    def leak_q_at(self, t_s: float) -> float:
        return float(np.interp(t_s, self.arr["time_s"], self.arr["leak_lps"])) if self.leak else 0.0

    def distance_to_leak(self, node: str) -> float:
        return float(self._dist[self.net.index[node]])

    # ------------------------------------------------------------------ what a method can see
    def observe(self) -> dict:
        t = self.arr["time_s"]
        vis = t <= self.now
        reps = [{k: r[k] for k in ("report_id", "received_at", "channel", "text")}
                for r in self.hid["reports"] if r["received_s"] <= self.now]
        return {"now_s": self.now, "time_s": t[vis], "pressure": self.arr["pressure_obs"][vis][:, self._sensor_cols],
                "pressure_nodes": self.layout, "flow": self.arr["flow_obs"][vis], "tank_level": self.arr["tank_level"][vis],
                "reports": reps, "loggers": dict(self.loggers), "checks_left": self.checks_left()}

    # ------------------------------------------------------------------ free actions
    def call(self, report_id: str, question: str) -> dict:
        """Call back the person who sent `report_id`. Free; at most 2 calls per message."""
        args = {"report_id": report_id, "question": str(question)[:400]}
        rep = next((r for r in self.hid["reports"] if r["report_id"] == report_id), None)
        if rep is None or rep["received_s"] > self.now:
            return self._err("call", args, f"unknown message '{report_id}'")
        if sum(e["action"] == "call" and e["args"]["report_id"] == report_id and e["result"].get("ok")
               for e in self.log) >= self.cfg["call"]["max_per_message"]:
            return self._err("call", args, "this person was already called twice")
        if not str(question).strip():
            return self._err("call", args, "empty question")
        rng = self._rng("call", f"{report_id}|{question.strip().lower()}")
        delay = rng.uniform(*self.cfg["call"]["delay_h"]) * 3600
        reply = self.citizen.answer(rep, question, rng)
        return self._record("call", args, {"ok": True, "reply_text": reply, "delay_s": delay})

    def advance(self, seconds: float) -> None:
        """Time passes (e.g. waiting for call-backs, which run in parallel)."""
        self.now = min(self.now + max(seconds, 0.0), self.horizon)

    def wait(self, hours: float) -> dict:
        if not 0 < hours <= 24:
            return self._err("wait", {"hours": hours}, "hours must be in (0, 24]")
        self.advance(hours * 3600)
        return self._record("wait", {"hours": hours}, {"ok": True, "now_s": self.now})

    def request_permit(self, street: str) -> dict:
        """Municipal permit for digging on a recently resurfaced road (SOP-6). Takes about 12 hours."""
        if not any(p["type"] == "street" and p["name"].lower() == str(street).lower() for p in self.places.places.values()):
            return self._err("request_permit", {"street": street}, f"unknown street '{street}'")
        self.advance(12 * 3600)
        return self._record("request_permit", {"street": street}, {"ok": True, "granted": True})

    def notify(self, target: str, message: str) -> dict:
        """'hospital:<name>', 'reporters', 'supervisor'."""
        return self._record("notify", {"target": str(target)[:80], "message": str(message)[:300]}, {"ok": True})

    def request_approval(self, action: str, target, certificate: dict) -> dict:
        d = self.operator.review(action, target, certificate)
        return self._record("request_approval", {"action": action, "target": target, "certificate": certificate},
                            {"ok": True, **d})

    # ------------------------------------------------------------------ field checks (budgeted)
    def step_test(self, section_ids: list[str], token: str | None) -> dict:
        args = {"sections": list(section_ids)}
        c = self.cfg["step_test"]
        bad = [s for s in section_ids if s not in self.sections.areas]
        if bad or not section_ids or len(section_ids) > c["max_sections"] or len(set(section_ids)) != len(section_ids):
            return self._err("step_test", args, f"give 1-{c['max_sections']} distinct section ids (unknown: {bad[:3]})")
        if not self.operator.use(token, "step_test", section_ids):
            return self._err("step_test", args, "a valid operator approval token for exactly these sections is needed")
        e = self._use_check("step_test", args)
        if e:
            return e
        self.now = self._next_night_end()
        day = int(self.now // 86400)
        q = self.leak_q_at(self.now - 2 * 3600)
        use = self.arr["night_use_lps"]
        out = []
        for sid in section_ids:
            area = self.sections.areas[sid]
            rng = self._rng("step", f"{sid}|{day}")
            if rng.random() < c.get("valve_fail", 0.0):
                out.append({"section": sid, "failed": "a boundary valve would not close; no reading"})
                continue
            true_use = float(use[min(day, len(use) - 1)][area["mask"]].sum())
            leak_in = self.leak is not None and bool(area["mask"][self.net.index[self.leak["nearest_node"]]])
            drop = true_use * (1 + rng.normal(0, c["use_rel_noise"])) + (q if leak_in else 0.0) + rng.normal(0, c["meter_noise_lps"])
            drop += self._draw_in(area["mask"], day)
            out.append({"section": sid, "inflow_drop_lps": round(float(drop), 2),
                        "expected_customer_use_lps": float(area["expected"])})
        return self._record("step_test", args, {"ok": True, "night_of": self.clock().date().isoformat(), "steps": out},
                            field_check=True)

    def _draw_in(self, mask, day: int) -> float:
        """A night draw inside the section during the step-test window (01-05 h) shows up as extra flow (its overlap with the window)."""
        f = self.fault
        if f is None or f.family != "night_draw" or self.fault_fixed_s is not None or not mask[self.net.index[f.element]]:
            return 0.0
        t = day * 86400 + np.arange(1 * 3600, 5 * 3600, 300)
        return float(f.value * F.draw_on(t, f).mean())

    # ------------------------------------------------------------------ inspections (ADR-017): a crew checks one element of the network
    INSPECT_H = {"valve": 4.0, "prv": 2.0, "sensor": 2.0, "meter": 3.0}

    def inspect(self, kind: str, target: str) -> dict:
        """A field check of one element: 'valve' (is the isolation valve on this pipe open?), 'prv' (outlet pressure of a pressure-reducing
        valve against its setting), 'sensor' (portable reference gauge next to a pressure sensor), 'meter' (clamp-on meter next to an inlet
        flow meter), 'watch' (a crew watches the hydrants within 300 m of a junction through the night). A fault that is found is fixed."""
        args = {"kind": kind, "target": target}
        wn = self.net.wn
        ok = {"valve": lambda x: x in wn.pipe_name_list, "prv": lambda x: x in wn.valve_name_list,
              "sensor": lambda x: x in self.layout, "meter": lambda x: x in F.INLET_METERS, "watch": lambda x: x in self.net.index}
        if kind not in ok:
            return self._err("inspect", args, "kind must be valve, prv, sensor, meter or watch")
        if not ok[kind](str(target)):
            return self._err("inspect", args, f"unknown {kind} '{target}'")
        e = self._use_check("inspect", args)
        if e:
            return e
        f, rng = self.fault, self._rng("inspect", f"{kind}|{target}")
        live = f is not None and self.fault_fixed_s is None
        hit = False
        if kind == "watch":
            self.now = min(math.floor((self.now - 6 * 3600) / 86400) * 86400 + 86400 + 6 * 3600, self.horizon)
            hit = (live and f.family == "night_draw" and self.net.pipe_distance[self.net.index[target], self.net.index[f.element]] <= 300
                   and rng.random() < 0.9)
            res = ({"ok": True, "found": True, "what": f"a water tanker was filling from the hydrant at {f.element} during the night "
                    f"(about {f.value:.0f} L/s); the crew stopped it"} if hit else {"ok": True, "found": False, "what": "nothing unusual at the hydrants"})
        else:
            self.advance(self.INSPECT_H[kind] * 3600)
            if kind == "valve":
                from ..town.network import series_chain                # the crew walks the stretch of main and opens every valve box on it
                hit = live and f.family == "closed_valve" and f.element in series_chain(self.net.wn, target)
                res = {"ok": True, "found": hit, "what": f"the isolation valve on {target} was CLOSED; it has been reopened" if hit
                       else f"the valve on {target} is open"}
            elif kind == "prv":
                setting = float(wn.get_link(target).initial_setting)
                dev = (f.value if (live and f.family == "prv_drift" and f.element == target) else 0.0) + float(rng.normal(0, 0.15))
                hit = abs(dev) > 1.0
                res = {"ok": True, "found": hit, "what": f"{target} outlet {setting + dev:.1f} m against a setting of {setting:.1f} m"
                       + ("; it has been reset" if hit else "")}
            elif kind == "sensor":
                b = (f.value if (live and f.family == "sensor_bias" and f.element == target) else 0.0) + float(rng.normal(0, 0.03))
                hit = abs(b) > 0.15
                res = {"ok": True, "found": hit, "what": f"sensor {target} reads {b:+.2f} m against the reference gauge"
                       + ("; it has been recalibrated" if hit else "")}
            else:
                b = (f.value if (live and f.family == "meter_bias" and f.element == target) else 0.0) + float(rng.normal(0, 0.3))
                hit = abs(b) > 1.0
                res = {"ok": True, "found": hit, "what": f"meter {target} reads {b:+.1f} L/s against the clamp-on meter"
                       + ("; it has been recalibrated" if hit else "")}
        if hit and live:
            self.fault_found, self.fault_fixed_s = True, self.now
        return self._record("inspect", args, res, field_check=True)

    def listen(self, center: str, radius_m: float) -> dict:
        args = {"center": center, "radius_m": radius_m}
        c = self.cfg["listen"]
        if center not in self.net.index:
            return self._err("listen", args, f"unknown junction '{center}'")
        if not 0 < radius_m <= c["max_radius_m"]:
            return self._err("listen", args, f"radius must be in (0, {c['max_radius_m']}] m")
        e = self._use_check("listen", args)
        if e:
            return e
        self.now = min(self.now + c["duration_h"] * 3600, self.horizon)
        rng = self._rng("listen", center)
        d = self.distance_to_leak(center)
        q = self.leak_q_at(self.now)
        p = min(c["p_detect_max"], c["p_detect_1lps"] * math.sqrt(max(q, 0.0))) * (1 - c["far_penalty"] * d / radius_m) \
            if d <= radius_m else 0.0
        u1, u2 = rng.random(), rng.random()
        res = {"ok": True, "heard": False}
        if u1 < p:                                        # heard the real leak: the crew pinpoints it
            target = abs(rng.normal(0, c["pinpoint_sigma_m"]))
            res = {"ok": True, "heard": True, "pinpointed_at": self.net.junctions[int(np.argmin(np.abs(self._dist - target)))]}
        elif u2 < c["false_positive"]:                    # a noise that sounds like a leak
            ring = np.where(self.net.pipe_distance[self.net.index[center]] <= radius_m)[0]
            res = {"ok": True, "heard": True, "pinpointed_at": self.net.junctions[int(rng.choice(ring))]}
        return self._record("listen", args, res, field_check=True)

    def canvass(self, center: str, radius_m: float | None = None) -> dict:
        """Survey the residents around `center`: how many heard hissing at night / saw damp ground. A tick-box form
        (counts only, no free text). Not a field check (no budget), but it takes until the next morning."""
        c = self.cfg["canvass"]
        radius_m = float(radius_m or c["radius_m"])
        args = {"center": center, "radius_m": radius_m}
        if self.done:
            return self._err("canvass", args, "the case is closed")
        if center not in self.net.index:
            return self._err("canvass", args, f"unknown junction '{center}'")
        if not 0 < radius_m <= 300:
            return self._err("canvass", args, "radius must be in (0, 300] m")
        if self.canvasses >= c["max_per_case"]:
            return self._err("canvass", args, f"canvass limit reached ({c['max_per_case']} per case)")
        if self.now >= self.horizon:
            return self._err("canvass", args, "the case deadline has passed")
        self.canvasses += 1
        ci = self.net.index[center]
        ring = np.where(self.net.pipe_distance[ci] <= radius_m)[0]
        w = self.ctx.residents[ring]
        rng = self._rng("canvass", center)
        n = int(np.clip(round(float(w.sum()) * c["contact_rate"]), 4, 40))
        who = rng.choice(ring, size=n, p=w / w.sum())
        q = self.leak_q_at(self.now)
        d = self._dist[who]
        if self.leak is not None and q > 0:
            h = np.minimum(c["p_hear_max"], c["p_hear_a"] * math.sqrt(q)) * np.exp(-d / c["len_m"])
            damp_leak = bool(self.leak.get("patch_nodes")) or bool(self.leak.get("surfaced"))
            sd = (c["p_damp"] * np.exp(-d / c["damp_len_m"])) if damp_leak else 0.0
        else:
            h, sd = 0.0, 0.0
        p_hear = c["false_pos"] + (1 - c["false_pos"]) * h
        p_damp = c["damp_false_pos"] + (1 - c["damp_false_pos"]) * sd
        heard = int((rng.random(n) < p_hear).sum())
        damp = int((rng.random(n) < p_damp).sum())
        d_end = math.floor((self.now - 7 * 3600) / 86400) + 1                      # answers by 07:00 next morning
        self.now = min(max(self.now + 3 * 3600, d_end * 86400 + 7 * 3600), self.horizon)
        return self._record("canvass", args, {"ok": True, "contacted": n, "heard_hissing": heard, "saw_damp": damp,
                                              "ready_at": self.clock().isoformat(timespec="minutes")})

    def place_logger(self, node: str) -> dict:
        args = {"node": node}
        if node not in self.net.index:
            return self._err("place_logger", args, f"unknown junction '{node}'")
        if len(self.loggers) >= self.cfg["logger"]["max"]:
            return self._err("place_logger", args, "logger limit reached")
        e = self._use_check("place_logger", args)
        if e:
            return e
        self.now = self._next_night_end()
        day = min(int(self.now // 86400), self.arr["night_mean_pressure"].shape[0] - 1)
        val = logger_reading(float(self.arr["night_mean_pressure"][day, self.net.index[node]]),
                             self._rng("logger", node), self.noise)
        self.loggers[node] = val
        return self._record("place_logger", args, {"ok": True, "night_mean_pressure_m": val}, field_check=True)

    # ------------------------------------------------------------------ decisions
    def dig(self, node: str, token: str | None, after_school_hours: bool = False) -> dict:
        args = {"node": node, "after_school_hours": after_school_hours}
        if self.done:
            return self._err("dig", args, "the case is closed")
        if node not in self.net.index:
            return self._err("dig", args, f"unknown junction '{node}'")
        if not self.operator.use(token, "dig", node):
            return self._err("dig", args, "a valid operator approval token for this spot is needed")
        if after_school_hours:
            ck = self.clock()
            h = ck.hour + ck.minute / 60
            if ck.weekday() < 5 and 7.5 <= h < 16.5:
                self.advance((16.5 - h) * 3600)
        self.advance(self.cfg["dig"]["duration_h"] * 3600)
        found = self.distance_to_leak(node) <= self.cfg["dig"]["success_radius_m"]
        if found:
            self.done, self.outcome = True, "found"
        return self._record("dig", args, {"ok": True, "leak_found": bool(found)})

    def close(self, verdict: str, summary: str = "") -> dict:
        """End the case without (another) dig: verdict 'no_leak' (false alarm) or 'hand_over' (not sure)."""
        if verdict not in ("no_leak", "hand_over"):
            return self._err("close", {"verdict": verdict}, "verdict must be 'no_leak' or 'hand_over'")
        self.done, self.outcome = True, self.outcome or verdict
        return self._record("close", {"verdict": verdict, "summary": summary[:300]}, {"ok": True})

    # ------------------------------------------------------------------ scoring (evaluation harness only)
    def score(self) -> dict:
        digs = [e for e in self.log if e["action"] == "dig" and e["result"].get("ok")]
        found = any(e["result"]["leak_found"] for e in digs)
        lost = 0.0
        if self.leak:
            t, q = self.arr["time_s"], self.arr["leak_lps"]
            m = t <= self.now
            lost = float(np.trapezoid(q[m], t[m]) / 1000)
        return {"has_leak": self.leak is not None, "found": found, "n_digs": len(digs),
                "wrong_digs": sum(not e["result"]["leak_found"] for e in digs),
                "outcome": "found" if found else (self.outcome or "unfinished"),
                "checks_used": self.checks_used, "canvasses": self.canvasses, "hours_open": round((self.now - self.case["case_open_s"]) / 3600, 1),
                "water_lost_m3": round(lost, 1),
                "fault": self.fault.family if self.fault else None, "fault_found": self.fault_found,
                # ADR-017: resolved = the leak found (if there is one) AND the fault found (if there is one), and no crew sent to a dry hole
                # when there was no leak
                "resolved": bool((found if self.leak else not digs and self.outcome in ("no_leak", "hand_over"))
                                 and (self.fault_found if self.fault else True)
                                 and (self.outcome != "hand_over" or self.leak is not None and found)),
                "leak_decision_ok": bool(found if self.leak else (not digs and self.outcome == "no_leak"))}


def _realistic(path: str) -> dict:
    import os
    return json.load(open(path)) if os.path.exists(path) else {}
