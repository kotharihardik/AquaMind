"""Run the REAL AquaMind crew on one benchmark case and record everything that happens, in order.

Nothing in aquamind/ is changed: the toolbox and the crew are subclassed here and only OBSERVED.
  InstrumentedToolbox   every tool call (and every new message) becomes an event with a snapshot of the evidence board: the leak map over all
                        782 junctions, P(leak), P(success), the hypothesised junctions, the message credibilities, the code gates, the clock
  InstrumentedCrew      every agent turn becomes two events: 'turn' when the LLM answered (thought, raw decision, the exact board text the agent
                        read, its options) and 'turn_done' when code has verified and applied it (hypothesis masses, overrides, notes)
  DemoLLM               the same LLM client and disk cache as the evaluation; with allow_network=False it only replays cached answers
The crew is built exactly like `agentix_crew` in aquamind/methods.py. Ground truth is attached only to the final 'end' event (the scorer's view).
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np

from aquamind.agent.crew import Crew
from aquamind.agent.toolbox import Toolbox
from aquamind.llm.client import AGENT_MODEL, LLM

from .config import AGENT_MAX_TOKENS, suite_of
from .world import CREW_TOOLBOX


# ---------------------------------------------------------------------------------------------------------------- LLM
class DemoLLM(LLM):
    """The evaluation's LLM client. allow_network=False: a prompt that is not in the cache is answered with an error (nothing is sent)."""

    def __init__(self, *a, allow_network: bool = False, **kw):
        self.allow_network = allow_network
        self.misses = 0
        super().__init__(*a, **kw)

    def _ask(self, system, user, required):
        if not self.allow_network:
            self.misses += 1
            return {"_error": "offline replay: this prompt is not in the LLM cache"}, 0, 0
        return super()._ask(system, user, required)


# ---------------------------------------------------------------------------------------------------------------- events
class Recorder:
    def __init__(self, sink=None):
        self.events: list[dict] = []
        self.sink = sink
        self.t0 = time.time()
        self._version = None
        self._cand = None

    def emit(self, kind: str, **data) -> dict:
        ev = {"i": len(self.events), "kind": kind, "wall_s": round(time.time() - self.t0, 2), **data}
        self.events.append(ev)
        if self.sink:
            try:
                self.sink(ev)
            except Exception:              # noqa: BLE001 - a slow or closed browser must never break the investigation
                pass
        return ev


def _f(x, nd=4):
    return None if x is None else round(float(x), nd)


def _j(w):
    return int(np.argmax(w)) if w is not None and float(np.sum(w)) > 0 else None


def snapshot(tb, rec: Recorder, force: bool = False) -> dict:
    """The evidence board as the UI draws it. The leak map and the hypothesised set are sent only when they changed."""
    b = tb.board
    post = b.post
    j, ps = b.best_dig()
    ok, why = tb.gate()
    cok, cwhy = tb.close_gate()
    s = {"p_leak": _f(b.p_leak), "p_success": _f(ps), "best": int(j), "best_desc": tb.describe_node(int(j)),
         "t_h": round((tb.env.now - tb.case.case_open_s) / 3600, 2), "clock": tb.env.clock().strftime("%a %d %b, %H:%M"),
         "checks_left": int(tb.env.checks_left()), "digs": int(tb.digs), "done": bool(tb.done or tb.env.done),
         "gate": [bool(ok), str(why)], "close_gate": [bool(cok), str(cwhy)], "q_hat": _f(b.q_hat, 2)}
    v = getattr(b, "version", 0)
    if force or v != rec._version:
        mx = float(post.max()) or 1.0
        s["heat"] = np.round(1000 * post / mx).astype(int).tolist()
        s["heat_max"] = _f(mx, 5)
        rec._version = v
    cand = getattr(b, "cand", None)
    if cand is not None:
        cl = np.where(cand)[0].tolist()
        if force or cl != rec._cand:
            s["cand"] = cl
            rec._cand = cl
        s["outside"] = _f(b.mass_outside(), 3)
    s["msgs"] = {rid: {"eps": _f(m.eps, 3), "src": m.source, "usable": bool(m.usable()), "calls": int(m.calls), "j": _j(m.w),
                       "up": _j(m.w_up), "notes": list(m.notes[-3:])} for rid, m in b.messages.items()}
    checks = []
    for c in b.checks:
        lab = c["label"]
        e = {"label": lab, "kind": c["kind"]}
        if c["kind"] == "step":
            e["sid"] = lab.split(":")[-1]
            e["lit"] = c.get("support") is not None
        if c.get("center") is not None:
            e["center"] = int(c["center"])
        if c.get("at") is not None:
            e["at"] = int(c["at"])
        if c["kind"] == "dry_dig":
            node = lab.split("@")[-1]
            e["at"] = int(tb.net.index[node]) if node in tb.net.index else None
        checks.append(e)
    s["checks"] = checks
    s["dry"] = sorted(int(x) for x in b.dry)
    mass = b.section_mass()
    order = np.argsort(-mass)[:6]
    s["sec_mass"] = [[tb.sections.ids[k], _f(mass[k], 3)] for k in order if mass[k] > 0.01]
    return s


def _options(tb, crew) -> list[dict]:
    out = []
    for oid, o in (tb.options or {}).items():
        e = {"id": oid, "kind": o.kind, "label": o.label, "p_inside": _f(o.p_inside, 3)}
        try:
            e["chance"] = _f(Crew.chance(tb, o), 3)
        except Exception:                  # noqa: BLE001
            e["chance"] = None
        if o.kind == "listen":
            e["center"] = int(o.args["center"])
        elif o.kind == "step":
            e["sections"] = list(o.args.get("sections") or [])
        out.append(e)
    return out


def _clean(d):
    """JSON-safe copy of an agent's decision / turn record (numpy types, long lists)."""
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(x) for x in d]
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


# ---------------------------------------------------------------------------------------------------------------- toolbox
class InstrumentedToolbox(Toolbox):
    def __init__(self, env, case, search, reader, docs, rules, *, recorder: Recorder, **kw):
        self._rec = recorder
        self._env_seen = 0
        super().__init__(env, case, search, reader, docs, rules, **kw)

    # ------------------------------------------------------------------ helpers
    def _message(self, r: dict) -> dict:
        rid = r["report_id"]
        rd = self.reads.get(rid) or {}
        m = self.board.messages.get(rid)
        return {"rid": rid, "at": r.get("received_at"), "ch": r.get("channel"), "text": r.get("text"), "read": _clean(rd),
                "prov": getattr(self.reader, "provenance", {}).get(rid), "agent_view": self.message_line(rid) if rid in self.reads else None,
                "trigger": rid == getattr(self.case, "trigger_report", None), "j": _j(m.w) if m else None}

    def _env_new(self) -> list[dict]:
        new = self.env.log[self._env_seen:]
        self._env_seen = len(self.env.log)
        out = []
        for e in new:
            res = dict(e.get("result") or {})
            out.append({"action": e["action"], "args": _clean(e.get("args")), "result": _clean(res), "t_h": round((e["t_s"] - self.case.case_open_s) / 3600, 2)})
        return out

    def _scada(self) -> dict:
        """The last 7 days of the layout's sensors and the zone inflow, hourly (what the control room's screens show)."""
        c = self.case
        t = np.asarray(c.time_s, float)
        keep = t >= c.case_open_s - 7 * 86400
        t, P, F = t[keep], np.asarray(c.pressure)[keep], np.asarray(c.flow)[keep]
        step = max(int(round(3600 / max(float(np.median(np.diff(t))) if len(t) > 1 else 900, 1))), 1)
        sl = slice(0, None, step)
        Qin = (F[:, 0] + F[:, 1]) / 3.6
        return {"t_h": [round((x - c.case_open_s) / 3600, 1) for x in t[sl]],
                "pressure": {n: [round(float(v), 2) for v in P[sl, k]] for k, n in enumerate(c.pressure_nodes)},
                "inflow_lps": [round(float(v), 2) for v in Qin[sl]]}

    # ------------------------------------------------------------------ hooks
    def _log(self, tool, args, result, secs=0.0):
        r = super()._log(tool, args, result, secs)
        rec = getattr(self, "_rec", None)
        if rec is None:
            return r
        ev = {"tool": tool, "args": _clean(args), "result": _clean(result), "secs": round(float(secs), 2)}
        env_actions = self._env_new()
        ev["env"] = env_actions
        if tool == "open":
            ev["messages"] = [self._message(m) for m in self.case.reports]
            pe = self.board.pressure_evidence
            ev["sensors"] = [{"node": n, "j": int(self.net.index[n]), "drop_m": _f(d, 3), "z": _f(z, 2)}
                             for n, d, z in zip(self.case.pressure_nodes, pe.drop_m, pe.z)]
            ev["inflow"] = {"q_hat": _f(pe.q_hat, 2), "q_sd": _f(pe.q_sd, 2)}
            ev["scada"] = self._scada()
            ev["works"] = _clean(getattr(self.case, "works", None) or [])
            ev["records_near"] = sorted({c.id for rid in self.reads for c in self.records_near(rid)})
        if tool == "call_back":
            replies = []
            calls = {(a["args"]["report_id"], a["args"]["question"].strip()): a["result"].get("reply_text")
                     for a in env_actions if a["action"] == "call" and a["result"].get("ok")}
            for c in (args or {}).get("calls") or []:
                rid, q = str(c.get("report_id")), str(c.get("question", "")).strip()
                parsed = (self.calls.get(rid) or [{}])[-1].get("reply") if self.calls.get(rid) else None
                replies.append({"rid": rid, "q": q, "text": calls.get((rid, q[:400].strip())), "parsed": _clean(parsed)})
            ev["replies"] = replies
        if tool == "field_check":
            o = (self.options or {}).get(str((args or {}).get("option", "")).strip().upper())
            if o is not None:
                ev["option"] = {"kind": o.kind, "label": o.label, "center": int(o.args["center"]) if o.kind == "listen" else None,
                                "sections": list(o.args.get("sections") or []) if o.kind == "step" else None}
        if tool == "propose_dig":
            digs = [a for a in env_actions if a["action"] == "dig"]
            ev["dig_j"] = int(self.net.index[digs[-1]["args"]["node"]]) if digs and digs[-1]["args"].get("node") in self.net.index else None
        rec.emit("tool", **ev, state=snapshot(self, rec, force=(tool == "open")))
        return r

    def refresh(self):
        before = set(self.seen)
        new = super().refresh()
        rec = getattr(self, "_rec", None)
        if rec is not None and new:
            reps = [r for r in self.env.observe()["reports"] if r["report_id"] in set(new) - before]
            rec.emit("messages", messages=[self._message(r) for r in reps], state=snapshot(self, rec))
        return new


# ---------------------------------------------------------------------------------------------------------------- crew
class InstrumentedCrew(Crew):
    def __init__(self, llm, *, recorder: Recorder, **kw):
        super().__init__(llm, **kw)
        self._rec = recorder
        self._tb = None

    def run(self, tb):
        self._tb = tb
        return super().run(tb)

    def _ask(self, role, system, user, why):
        self._rec.emit("thinking", role=role, why=why)
        u = self.llm.usage
        c0, s0 = u.cached, u.seconds
        t0 = time.time()
        d = super()._ask(role, system, user, why)
        tb = self._tb
        opts = _options(tb, self) if tb is not None and role in ("strategist", "auditor") else None
        self._rec.emit("turn", role=role, why=why, thought=d.get("thought"), decision=_clean({k: v for k, v in d.items() if k != "thought"}),
                       error=d.get("_error"), cached=u.cached > c0, latency_s=round(time.time() - t0, 2), api_s=round(u.seconds - s0, 2),
                       board_text=user, options=opts, turn=len(self.turns) - 1)
        return d

    def _done(self, role):
        tb = self._tb
        rec = _clean(self.turns[-1]) if self.turns else {}
        rec.pop("thought", None)
        slate = [{"id": h.hid, "kind": h.kind, "where": h.where, "cause": h.cause, "mass": _f(h.mass, 3),
                  "region": [int(i) for i in h.region.nonzero()[0]] if h.region is not None else None,
                  "because": list(h.because)[:6], "explains": list(h.explains)[:6], "error": h.error,
                  "cited_ok": h.cited_ok, "cited_bad": h.cited_bad} for h in self.slate]
        self._rec.emit("turn_done", role=role, record=rec, slate=slate, challenges=_clean(self.challenges) if role == "skeptic" else None,
                       rejected=_clean(self.rejected_citations[-4:]), state=snapshot(tb, self._rec) if tb is not None else None)

    def forensic(self, tb, why):
        r = super().forensic(tb, why)
        self._done("forensic")
        return r

    def skeptic(self, tb, why):
        r = super().skeptic(tb, why)
        self._done("skeptic")
        return r

    def strategist(self, tb, why):
        r = super().strategist(tb, why)
        self._done("strategist")
        return r

    def audit(self, tb, why):
        r = super().audit(tb, why)
        self._done("auditor")
        return r


# ---------------------------------------------------------------------------------------------------------------- one case
def truth_of(env, world) -> dict:
    """The scorer's view (hidden from every method): where the leak really was, which messages were real."""
    net = world.town.net
    hid, leak = env.hid, env.leak
    out = {"has_leak": leak is not None, "trap": None, "messages": {}}
    if leak:
        out["leak"] = {"j": int(net.index[leak["nearest_node"]]), "node": leak["nearest_node"], "pipe": leak.get("pipe"),
                       "a": int(net.index[leak["start_node"]]) if leak.get("start_node") in net.index else None,
                       "b": int(net.index[leak["end_node"]]) if leak.get("end_node") in net.index else None,
                       "frac": _f(leak.get("frac"), 3), "lps": _f(leak.get("mean_lps"), 2), "kind": leak.get("kind"),
                       "section": leak.get("section"), "surfaced": bool(leak.get("surfaced"))}
    trap = (hid.get("trap") or {}).get("target_node")
    if trap:
        out["trap"] = {"node": trap, "j": int(net.index[trap])}
    for r in hid.get("reports") or []:
        tr = r.get("truth") or {}
        out["messages"][r["report_id"]] = {"kind": tr.get("kind"), "cause": tr.get("cause"), "mode": tr.get("mode"),
                                           "seen_j": int(net.index[tr["seen_node"]]) if tr.get("seen_node") in net.index else None}
    return out


def run_case(case_name: str, layout: str = "spread_k5", allow_network: bool = False, sink=None, reader: str = "finetuned",
             model: str | None = None) -> dict:
    """-> a recording: {meta, events}. The crew is `agentix_crew` of aquamind/methods.py."""
    from .readers import DemoReader
    from .world import search, world
    W = world()
    case, env = W.open(case_name, layout)
    rec = Recorder(sink)
    llm = DemoLLM(model or AGENT_MODEL, max_tokens=AGENT_MAX_TOKENS, allow_network=allow_network)
    rd = DemoReader(W.town.places, prefer=reader)
    meta = {"case": case_name, "public_id": case.case_id, "suite": suite_of(case_name), "layout": layout, "k": len(case.pressure_nodes),
            "sensors": [int(W.town.net.index[n]) for n in case.pressure_nodes], "sensor_ids": list(case.pressure_nodes),
            "trigger": case.trigger, "trigger_report": case.trigger_report,
            "opened_at": (dt.datetime.fromisoformat(case.start_datetime) + dt.timedelta(seconds=case.case_open_s)).strftime("%a %d %b %Y, %H:%M"),
            "horizon_h": round((case.horizon_s - case.case_open_s) / 3600, 1), "budget": {"checks": int(env.max_checks), "digs": 2},
            "method": "agentix_crew", "model": llm.model, "reader": rd.name, "network": bool(allow_network),
            "recorded_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "fidelity": "recording"}
    rec.emit("meta", meta=meta)
    t0 = time.time()
    err = None
    tb = InstrumentedToolbox(env, case, search(), rd, W.docs, W.rules, recorder=rec, **CREW_TOOLBOX)
    crew = InstrumentedCrew(llm, recorder=rec)
    try:
        crew.run(tb)
        if not (tb.done or env.done):
            tb.close("hand_over", "no decision", force=True)
    except Exception:                       # noqa: BLE001 - a crash is a failed case, like in the evaluation harness
        import traceback
        err = traceback.format_exc()[-1500:]
        if not env.done:
            env.close("hand_over", "method crashed")
    sc = env.score()
    trap = (env.hid.get("trap") or {}).get("target_node")
    audit = W.rules.audit(env, trap)
    guess = tb.final_guess()
    u = llm.usage
    result = {**_clean(sc), "unsafe": audit["unsafe_total"], "violations": audit["violations_total"], "audit": _clean(audit),
              "final_guess": guess, "final_j": int(W.town.net.index[guess]),
              "guess_error_m": (round(env.distance_to_leak(guess), 1) if env.leak else None),
              "tool_calls": sum(1 for t in tb.trace if t["tool"] != "open"),
              "tool_errors": sum(1 for t in tb.trace if t["tool"] != "open" and not t["result"].get("ok", True)),
              "llm_calls": u.calls, "llm_cached": u.cached, "llm_misses": llm.misses, "tokens": u.prompt_tokens + u.output_tokens,
              "llm_seconds": round(u.seconds, 1), "steps": len(crew.turns), "wall_s": round(time.time() - t0, 1),
              "reader_stats": rd.stats, "error": err}
    rec.emit("end", result=result, truth=truth_of(env, W), state=snapshot(tb, rec, force=True))
    return {"meta": meta, "result": result, "events": rec.events}
