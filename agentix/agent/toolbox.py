"""The agent's toolbox: every action a 'brain' (LLM or rules) can take, each one checked by code before it runs.

Safety by design (patterns from Beurer-Kellner et al., "Design Patterns for Securing LLM Agents against Prompt
Injections", 2025):
- ACTION SELECTOR: a brain picks tools and menu items; it can never name an arbitrary dig location. A dig always
  targets the spot the EVIDENCE supports (code's best dig spot), so text inside a report cannot steer a crew.
- CODE GATE: a dig request needs P(success) >= 0.5 and no strong alternative explanation BEFORE the operator is
  even asked; then the operator must approve. The mechanical SOP steps (permit, hospital notice, school-hours
  timing, informing reporters) are done by code, so they never depend on a brain remembering them.
- DUAL LLM / CONTEXT MINIMIZATION: raw citizen text is only read by the quarantined reader. Tools return only
  fixed words, canonical place names and numbers. Document hits lose any instruction-like sentence.
- MONITOR: every call is validated (known tool, valid ids, budget, step limits); errors are returned, not raised.

The same toolbox serves the LLM agent AND the rule-based brain, so they are compared on equal terms.
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..active.investigation import OPEN_QUESTION, Investigation, Option
from ..env.operator import SUSPICIOUS
from .timeparse import parse_when

DIG_THRESHOLD = 0.5          # P(success) needed before a dig may even be requested
ALT_LIMIT = 0.3              # a non-leak explanation above this blocks digging (check it first)
MAX_DIGS = 2                 # SOP-2
TOOLS = ["do", "ask", "search_docs", "test_event", "test_sensor_fault", "exclude_sensor", "wait", "recheck",
         "request_permit", "notify", "propose_dig", "abstain"]


class Toolbox:
    def __init__(self, inv: Investigation, env, case, retriever, rulebook, budget_cu: float = 1.6,
                 use_rag: bool = True, use_gate: bool = True, allow_alternatives: bool = True):
        self.inv, self.env, self.case = inv, env, case
        self.ret, self.rules = retriever, rulebook
        self.budget, self.use_rag, self.use_gate = budget_cu, use_rag, use_gate
        self.allow_alt = allow_alternatives          # ablation: no testing of other explanations
        self.board = inv.start(case, env)
        self.net = inv.net
        self.menu_cache: dict[str, Option] = {}
        self.doc_hits: dict[str, dict] = {}
        self.digs = 0
        self.done = False
        self.events: list[dict] = []          # what happened, for the brain's memory and the logs
        self.searches: dict[str, list] = {}
        self.failed: dict[str, str] = {}      # calls that failed (so an identical retry is refused at once)
        # this case's own work orders and maintenance notes are short, so every brain sees all of them from the
        # start (instruction-like sentences removed); search_docs is for the SOP and for finding things again
        self.notes = {c.id: {"text": _clean(c.text), "provenance": c.provenance}
                      for c in retriever.chunks if c.provenance == "internal_note"} if use_rag else {}
        self.doc_hits.update(self.notes)

    # ------------------------------------------------------------------ small helpers
    def p_success(self) -> float:
        return self.board.belief().best_dig()[1]

    def best_node(self) -> str:
        return self.net.junctions[self.board.belief().best_dig()[0]]

    def alt_max(self) -> tuple[str | None, float]:
        """The strongest non-leak explanation and its probability."""
        ex = self.board.explanations()
        alts = [(k, v) for k, v in ex.items() if k not in ("leak", "none")]
        return max(alts, key=lambda x: x[1]) if alts else (None, 0.0)

    def sensor_outlier(self, min_drop_m: float = 0.15, ratio: float = 3.0):
        """One sensor that dropped much more than all others (SOP-10: may be a fault, not a leak)."""
        d = np.abs(self.inv.ev0.drop_m)
        if len(d) < 3:
            return None
        order = np.argsort(-d)
        if d[order[0]] >= min_drop_m and d[order[0]] >= ratio * max(d[order[1]], 0.02):
            return self.case.pressure_nodes[int(order[0])], float(self.inv.ev0.drop_m[order[0]]), float(d[order[1]])
        return None

    def menu(self) -> list[Option]:
        opts = self.inv.menu(self.board, budget_left=self.budget - self.inv.state["spent"])
        if not self.env.night_left():                    # field checks need a night; only questions remain
            opts = [o for o in opts if o.kind == "ask"]
        self.menu_cache = {f"M{i + 1}": o for i, o in enumerate(opts[:6])}
        return opts[:6]

    def resolve_node(self, where: str) -> str | None:
        """'n123' or a place description -> one junction id.
        'Vegetable Market, Tank Bund Road' -> the Vegetable Market closest to Tank Bund Road (names repeat in a
        city, so the street helps pick the right one). A street alone -> its middle junction."""
        where = str(where).strip()
        if where in self.net.index:
            return where
        gaz = self.inv.gaz
        found = [pids for _, _, _, pids in gaz.find_mentions(where)] or [gaz.resolve(where)]
        pids = [p for group in found for p in group]
        if not pids:
            return None
        marks = [gaz.places[p]["geometry"]["node"] for p in pids if "node" in gaz.places[p].get("geometry", {})]
        lines = [n for p in pids for n in gaz.places[p].get("geometry", {}).get("nodes", []) if n in self.net.index]
        if marks and lines:
            xy = self.net.xy[[self.net.index[n] for n in lines]]
            return min(marks, key=lambda m: float(np.min(np.hypot(*(xy - self.net.xy[self.net.index[m]]).T))))
        if marks:
            return marks[0]
        return lines[len(lines) // 2] if lines else None

    def resolve_place(self, text: str) -> dict | None:
        """A place description -> one gazetteer place (a landmark if one is named; the one nearest a named street)."""
        node = self.resolve_node(text)
        gaz = self.inv.gaz
        pids = [p for _, _, _, g in gaz.find_mentions(text) for p in g] or gaz.resolve(text)
        if not pids:
            return None
        marks = [p for p in pids if gaz.places[p].get("geometry", {}).get("node") == node]
        return gaz.places[marks[0] if marks else pids[0]]

    def grounded(self, basis: str) -> str:
        """Is a hypothesis backed by something real? 'doc' (a work order / log line we have read), 'report'
        (a report we have), or 'guess'. Citing an ID that does not exist counts as a guess."""
        words = str(basis).strip().upper().replace(",", " ").split()
        b = words[0] if words else ""
        if b in self.doc_hits and self.doc_hits[b]["provenance"] == "internal_note":
            return "doc"
        if b in self.inv.state["reader_outs"]:
            return "report"
        return "guess"

    def missing_steps(self, node: str, schedule: str) -> list[str]:
        """SOP steps still to do before digging at `node` (SOP-5 school hours, SOP-6 permit, SOP-7 hospital)."""
        todo = []
        for need in self.rules.requirements("dig", [node], self.env.clock()):
            if need["rule"] == "school_hours" and schedule != "after_school_hours":
                todo.append(f"{need['sop']}: use schedule='after_school_hours'")
            elif need["rule"] == "resurfaced_permit" and \
                    self.rules.resurfaced_node.get(node) not in self.env.permits:
                todo.append(f"{need['sop']}: {need['do']}")
            elif need["rule"] == "hospital_notice" and not any(t.startswith("hospital") for t in self.env.notified):
                todo.append(f"{need['sop']}: {need['do']}")
        return todo

    # ------------------------------------------------------------------ single entry point
    def call(self, tool: str, args: dict | None) -> dict:
        """Validate, run, record. Never raises."""
        if (self.done or self.env.done) and tool != "notify":       # informing people is still allowed
            return {"ok": False, "error": "case already closed"}
        if tool not in TOOLS:
            return {"ok": False, "error": f"unknown tool '{tool}'. Allowed: {TOOLS}"}
        if not self.allow_alt and tool in ("test_event", "test_sensor_fault", "exclude_sensor", "recheck"):
            return {"ok": False, "error": f"'{tool}' is not available in this setup"}
        key = tool + json.dumps(args or {}, sort_keys=True, default=str)
        if key in self.failed:                               # same call failed before: don't spend time on it again
            return {"ok": False, "error": f"repeat of a call that already failed ({self.failed[key][:80]})"}
        try:
            res = getattr(self, "t_" + tool)(**(args or {}))
        except (TypeError, ValueError) as e:
            res = {"ok": False, "error": f"bad arguments for {tool}: {str(e)[:120]}"}
        if not res.get("ok", True) and tool not in ("propose_dig", "wait", "recheck"):
            self.failed[key] = str(res.get("error"))
        self.events.append({"tool": tool, "args": args, "result": {k: v for k, v in res.items() if k != "explanations"}})
        return res

    # ------------------------------------------------------------------ the tools
    def t_do(self, option: str):
        """Run a prepared check from the menu by its id (M1, M2, ...)."""
        o = self.menu_cache.get(str(option).strip().upper())
        if o is None:
            return {"ok": False, "error": f"unknown menu option '{option}'; choose one of {list(self.menu_cache)}"}
        if self.inv.state["spent"] + o.cost > self.budget + 1e-9:
            return {"ok": False, "error": "not enough budget left for this check"}
        if o.kind == "step":                                 # SOP-7: hospitals hear about a night shut-off first
            nodes = [self.net.junctions[i] for i in o.args["S"]]
            for need in self.rules.requirements("step_test", nodes, self.env.clock()):
                self.env.notify(need["do"].split("notify('")[1].rstrip("')"), "night step test planned nearby")
        r = self.inv.execute(o, self.board, self.env)
        return {"ok": bool(r.get("ok")), "check": o.label, "result": _short(r),
                "p_success_now": round(self.p_success(), 3)}

    def t_ask(self, report_id: str, question: str = "", options: list | None = None):
        """Ask a reporter a question, optionally 'near which of these places?'. Free, at most 2 per report.
        The answer comes back already read by the quarantined reader (never as raw text)."""
        if report_id not in self.board.reports:
            return {"ok": False, "error": f"no usable report '{report_id}'"}
        question = str(question or OPEN_QUESTION)[:300]
        if re.search(SUSPICIOUS, question.lower()):
            return {"ok": False, "error": "question contains instruction-like text"}
        opts = []
        for o in (options or [])[:4]:
            p = self.resolve_place(str(o))
            if p and p["place_id"] not in [x["place_id"] for x in opts]:
                opts.append({"place_id": p["place_id"], "label": p["name"]})
        o = Option(f"ask@{report_id}", "ask", {"rid": report_id, "question": question, "options": opts}, 0.0, 2.0)
        r = self.inv.execute(o, self.board, self.env, question=question)
        parsed = r.get("parsed") or {}
        k = parsed.get("option_index")
        chose = (opts[k]["label"] if k is not None and k < len(opts) else
                 "none of those" if opts and k == len(opts) else None)
        safe = {"answered": bool(r.get("reply_text")), "source_seen": parsed.get("source_seen"), "chose_option": chose,
                "places": [self._canon(m["text"]) for m in parsed.get("mentions", [])],
                "flows_from": [self._canon(m["text"]) for m in parsed.get("upstream", [])],
                "not_public_pipe": parsed.get("non_supply", False)}
        return {"ok": bool(r.get("ok")), "error": r.get("error"), "answer": safe,
                "p_success_now": round(self.p_success(), 3)}

    def t_search_docs(self, query: str):
        """Search the SOP rules and this case's work orders / maintenance log (BM25)."""
        if not self.use_rag:
            return {"ok": False, "error": "document search is not available"}
        q = " ".join(str(query).lower().split())
        if q in self.searches:
            return {"ok": False, "error": "you already ran this search; its hits are on the board"}
        if len(self.searches) >= 4:
            return {"ok": False, "error": "search limit (4) reached; use what is on the board"}
        hits = []
        for c, _ in self.ret.search(q, k=4):
            text = _clean(c.text)
            self.doc_hits[c.id] = {"text": text, "provenance": c.provenance}
            hits.append({"id": c.id, "type": c.provenance, "text": text[:260]})
        self.searches[q] = [h["id"] for h in hits]
        return {"ok": True, "hits": hits}

    def t_test_event(self, where: str, start: str, end: str, basis: str = "guess"):
        """Test the explanation 'temporary water use (flushing / fire brigade / valve works) at `where`'.
        start/end: '14:15', '1415 hrs', '2 PM', or with a date like '10/10 14:15'."""
        node = self.resolve_node(where)
        if node is None:
            return {"ok": False, "error": f"unknown place '{where}'"}
        sd = self.env.start_dt
        t0 = parse_when(start, sd, self.case.case_open_s)
        t1 = parse_when(end, sd, t0 if t0 is not None else self.env.now)
        if t0 is None or t1 is None:
            return {"ok": False, "error": "start/end must contain a clock time like 14:15"}
        if t1 <= t0:
            t1 += 86400                       # runs past midnight
        g = self.grounded(basis)
        prior = {"doc": 0.6, "report": 0.4, "guess": 0.1}[g]
        name = f"event@{node}"
        if name in self.inv.state["alts_tested"]:
            return {"ok": False, "error": f"'{name}' was already tested; see HYPOTHESES on the board"}
        ex = self.inv.test_event(self.board, node, t0, t1, prior, name, spread=(g == "guess"))
        return {"ok": True, "hypothesis": name, "support": g, "explanations": _round(ex),
                "event_ends_in_h": round((t1 - self.env.now) / 3600, 1),
                "hint": "an event stops, a leak does not: wait until ~2 h after it ends, then recheck"}

    def t_test_sensor_fault(self, sensor: str, basis: str = "guess"):
        """Test the explanation 'this sensor is faulty and there is no leak'."""
        if sensor not in self.case.pressure_nodes:
            return {"ok": False, "error": f"'{sensor}' is not one of our sensors {self.case.pressure_nodes}"}
        if f"sensor_fault:{sensor}" in self.inv.state["alts_tested"]:
            return {"ok": False, "error": f"sensor fault of {sensor} was already tested; see HYPOTHESES on the board"}
        prior = {"doc": 0.6, "report": 0.15, "guess": 0.15}[self.grounded(basis)]
        ex = self.inv.test_sensor_fault(self.board, sensor, prior, f"sensor_fault:{sensor}")
        return {"ok": True, "explanations": _round(ex)}

    def t_exclude_sensor(self, sensor: str):
        """Stop using one sensor (for example after a maintenance note says it is broken)."""
        if sensor not in self.case.pressure_nodes:
            return {"ok": False, "error": f"'{sensor}' is not one of our sensors"}
        if sensor in self.inv.state.get("excluded_sensors", []):
            return {"ok": False, "error": f"'{sensor}' is already excluded"}
        ex = self.inv.exclude_sensor(self.board, sensor)
        return {"ok": True, "explanations": _round(ex), "p_success_now": round(self.p_success(), 3)}

    def t_wait(self, hours: float):
        """Let time pass (up to 12 h). New reports that arrive are read and added."""
        hours = float(hours)
        if not 0 < hours <= 12:
            return {"ok": False, "error": "wait between 0 and 12 hours"}
        n0 = len(self.inv.state["seen_reports"])
        r = self.env.wait(hours)
        self.inv.add_reports(self.board, self.env.observe()["reports"])
        return {"ok": bool(r.get("ok")), "now": self.env.clock().strftime("%a %H:%M"),
                "new_reports": len(self.inv.state["seen_reports"]) - n0}

    def t_recheck(self):
        """Look at the newest data (after any suspected event ended): is the anomaly still there?
        If it is a little too early (an event ended less than 2 h ago), wait the few hours needed first."""
        r = self.inv.recheck(self.board, self.env)
        if not r.get("ok") and 0 < r.get("wait_h", 99) <= 6 and self.env.now + r["wait_h"] * 3600 <= self.env.horizon:
            self.t_wait(r["wait_h"])
            r = self.inv.recheck(self.board, self.env)
        if r.get("ok"):
            r["explanations"] = _round(r["explanations"])
            r["p_success_now"] = round(self.p_success(), 3)
        return r

    def t_request_permit(self, street: str):
        """SOP-6 permit for a recently resurfaced street (takes about 12 hours)."""
        return self.env.request_permit(str(street))

    def t_notify(self, target: str, message: str = ""):
        """'hospital:<name>', 'reporter:<report_id>', 'supervisor' or 'traffic_police'."""
        if re.search(SUSPICIOUS, str(message).lower()):
            return {"ok": False, "error": "message contains instruction-like text"}
        return self.env.notify(str(target), str(message))

    def t_propose_dig(self, schedule: str = "now", reason: str = ""):
        """Ask to dig at the best-supported spot. Code gate first, then the operator."""
        if schedule not in ("now", "after_school_hours"):
            return {"ok": False, "error": "schedule must be 'now' or 'after_school_hours'"}
        if self.digs >= MAX_DIGS:
            return {"ok": False, "gate": "blocked", "error": "two digs already done (SOP-2)"}
        p, node = self.p_success(), self.best_node()
        alt, pa = self.alt_max()
        if self.use_gate:
            if p < DIG_THRESHOLD:
                return {"ok": False, "gate": "blocked",
                        "error": f"P(success) {p:.2f} < {DIG_THRESHOLD}: collect more evidence first"}
            if pa > ALT_LIMIT:
                return {"ok": False, "gate": "blocked", "error": f"'{alt}' still has {pa:.2f}: rule it out first"}
        sop_done = []
        for step in self.missing_steps(node, schedule):     # mechanical SOP steps are done by code, not left to luck
            if "after_school_hours" in step:
                schedule = "after_school_hours"
            elif "request_permit" in step:
                self.env.request_permit(self.rules.resurfaced_node[node])
            elif "notify" in step:
                self.env.notify(step.split("notify('")[1].rstrip("')"), "planned excavation nearby")
            sop_done.append(step)
        token, why = self.inv.approve(self.env, "dig", node, self.board, str(reason))
        if token is None:
            return {"ok": False, "operator": why}
        b = self.board.belief()
        j = self.net.index[node]
        r = self.env.request_dig(node, approval_token=token, schedule=schedule)
        self.digs += 1
        found = bool(r.get("leak_found"))
        if found:
            self.done = True
        else:
            self.board.add_check(f"failed_dig@{node}", 1 - b.cover[j] * 0.9, 1.0)
        if r.get("ok"):                                      # SOP-8: tell the people whose reports we used
            for rid in self.board.reports:
                self.env.notify(f"reporter:{rid}", "a crew inspected the reported area")
        return {"ok": bool(r.get("ok")), "dug_at": node, "leak_found": found, "error": r.get("error"),
                "sop_done": sop_done + (["SOP-8: reporters informed"] if r.get("ok") else [])}

    def t_abstain(self, summary: str = ""):
        """Close the case without digging and hand it to a human (with a short summary)."""
        self.env.abstain(str(summary)[:300])
        self.done = True
        return {"ok": True}

    def _canon(self, text: str) -> str:
        ids = self.inv.gaz.resolve(text)
        return self.inv.gaz.places[ids[0]]["name"] if ids else "an unknown place"


def _clean(text: str) -> str:
    """Remove any sentence that reads like an instruction to us (it is data, never an order)."""
    return re.sub(r"[^.|]*(" + SUSPICIOUS + r")[^.|]*", "[instruction-like text removed]", text, flags=re.I)


def _round(d: dict) -> dict:
    return {k: round(float(v), 3) for k, v in d.items()}


def _short(r: dict) -> dict:
    keep = ("ok", "error", "detected", "approx_node", "night_mean_pressure_m", "inflow_drop_lps",
            "expected_customer_drop_lps", "note")
    return {k: v for k, v in r.items() if k in keep}
