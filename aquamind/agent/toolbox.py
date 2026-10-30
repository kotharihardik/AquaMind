"""The investigation toolbox: every action a 'brain' (the LLM agent or the code-only planner) can take.

Safety by design (proposal Section 7, step 4; ADR-005). The temporary logger (place_logger) is a baseline-only tool (FixedRules, ReAct); the crew's
new-measurement tool is the night step test:
- QUARANTINE: raw citizen text is read only by the reader. The brain sees code-built summaries: fixed words,
  canonical place names from the map, numbers. Questions to citizens are free text, but answers come back as fields.
- ACTION SELECTOR: a dig has no location argument; it always goes to the spot the evidence supports most.
- CODE GATE before the operator is even asked: P(success) >= 0.8 and at least one field check (not a message)
  supports the spot; at most two digs (SOP-2).
- OPERATOR TOKEN: the duty operator approves the form; the one-time token is bound to the exact action and target.
- SOP paperwork (permits, hospital notices, school hours, informing citizens) is done by code from the SOP (RAG).
- Every call is validated; errors are returned (never raised) and logged, so tool-call accuracy can be measured.
"""
from __future__ import annotations

import time
import zlib

import numpy as np

from ..search.planner import DIG_BAR, FINAL_BAR, Option, Planner

SYM_WORDS = {"water_coming_out": "water coming out", "burst": "burst", "pooling": "water collected",
             "wet_ground": "wet ground", "sound_underground": "sound under road", "road_damage": "road damage",
             "low_pressure": "low pressure", "no_water": "no water", "dirty_water": "dirty water"}


class Toolbox:
    def __init__(self, env, case, search, reader, docs, rulebook, *, use_gate: bool = True, use_rag: bool = True,
                 use_field: bool = True, plan_samples: int = 96, use_canvass: bool = False,
                 use_planner: bool = True, min_checks_close: int = 0, auto_sop: bool = False,
                 allow_step: bool = True):
        self.env, self.case, self.search, self.reader = env, case, search, reader
        self.docs, self.rules = docs, rulebook
        self.net, self.places, self.sections = search.net, search.town.places, search.sections
        self.use_gate, self.use_rag, self.use_field, self.use_canvass = use_gate, use_rag, use_field, use_canvass
        self.min_checks_close = min_checks_close            # sweep before 'no leak': at least this many field checks unless P(leak) < 0.03
        self.auto_sop = auto_sop                            # True: the rule book in code does the SOP paperwork; False: only what an agent asks for (read from sop.md)
        self.use_planner = use_planner                      # False: no look-ahead values at all (the rule-based dispatcher baseline)
        self.allow_step = allow_step                        # False: no night step test (the temporary flow/pressure measurement) is offered or accepted
        self.trace: list[dict] = []
        t0 = time.time()
        self.reads = reader.read_many(case.reports)
        self.board = search.start(case, self.reads)
        self.seen = {r["report_id"] for r in case.reports}
        self.planner = Planner(self.net, self.sections, self.board.tools, n_samples=plan_samples,
                               seed=zlib.crc32(case.case_id.encode()) % 10_000,
                               dig_bar=search.cfg.dig_bar, final_bar=search.cfg.final_bar, hedge=search.cfg.hedge,
                               allow_step=allow_step)
        self.options: dict[str, Option] = {}
        self._plan_key = None
        self.attempts: dict[str, int] = {}              # call-backs made per message (answered or not): max 2 each
        self.repairs_used: set[str] = set()             # ADR-018: repair tickets the agent cited as a failing stretch
        self.digs = 0
        self.calls: dict[str, list] = {}
        self.records_used: dict[str, str] = {}
        self.done = False
        self._log("open", {}, {"messages": len(case.reports)}, time.time() - t0)

    # ------------------------------------------------------------------ bookkeeping
    def _log(self, tool, args, result, secs=0.0):
        self.trace.append({"t_case_h": round((self.env.now - self.case.case_open_s) / 3600, 2), "tool": tool,
                           "args": args, "result": result, "p_leak": round(self.board.p_leak, 3),
                           "p_success": round(self.board.best_dig()[1], 3), "secs": round(secs, 2)})
        return result

    def _err(self, tool, args, msg):
        return self._log(tool, args, {"ok": False, "error": msg})

    def refresh(self):
        """Read messages that arrived since the last look (time passes during an investigation)."""
        new = [r for r in self.env.observe()["reports"] if r["report_id"] not in self.seen]
        if new:
            self.reads.update(self.reader.read_many(new))
            self.seen |= {r["report_id"] for r in new}
            self.search.add_messages(self.board, new, self.reads)
        return [r["report_id"] for r in new]

    def describe_node(self, j: int) -> str:
        return f"{self.places.describe(j)} ({self.net.junctions[j]})"

    def message_line(self, rid: str) -> str:
        """A code-built summary of a message: fixed words + canonical map names only (never the raw text)."""
        rd, m = self.reads[rid], self.board.messages.get(rid)
        names = []
        for p in rd.get("places") or []:
            ids = self.places.resolve(p["text"], p.get("type"))
            names.append(self.places.places[ids[0]]["name"] if ids else "an unrecognised place")
        flags = []
        if rd.get("contains_instructions"):
            flags.append("CONTAINS INSTRUCTION-LIKE TEXT (quarantined, not evidence)")
        if rd.get("on_private_property"):
            flags.append("private property")
        what = ", ".join(SYM_WORDS.get(s, s) for s in rd.get("symptoms") or []) or "-"
        line = (("[OPENED THE CASE] " if rid == getattr(self.case, "trigger_report", None) else "") +
                f"{rid}: {rd.get('category')} | {what} | place: {'; '.join(names) or 'none'} | "
                f"source: {(m.source if m else rd.get('source'))}")
        if m and m.usable() and self.search.rep.kind(rd) == "discolour":
            line += (" | brown/dirty tap water: weak AREA-level evidence (the leak is on a feeder pipe, often 300-900 m away; the place itself is not "
                     "the leak)" + (f" | explained by the utility's own works" if m.eps >= 0.9 else f" | calls {m.calls}/2"))
        elif m and m.usable():
            line += f" | chance it is a false alarm {m.eps:.0%} | calls {m.calls}/2"
        elif not flags:
            line += " | not leak evidence"
        if m and m.notes:
            line += " | " + "; ".join(m.notes[-3:])
        return line + (" | " + "; ".join(flags) if flags else "")

    def influence(self, rid: str) -> float:
        """How much the leak map would change without this message (total-variation distance)."""
        m = self.board.messages.get(rid)
        if not (m and m.usable()):
            return 0.0
        p0 = self.board.post
        b = self.board.copy()
        b.messages[rid].eps = 0.999
        b.touch()
        return 0.5 * float(np.abs(b.post - p0).sum())

    def records_near(self, rid: str, k: int = 2) -> list:
        """Complaint/repair records at the place of a message (RAG)."""
        if not self.use_rag:
            return []
        rd = self.reads[rid]
        names = [self.places.places[ids[0]]["name"] for p in rd.get("places") or []
                 if (ids := self.places.resolve(p["text"], p.get("type")))]
        if not names:
            return []
        hits = self.docs.search(" ".join(names), k=6, provenance="record")
        out = []
        for c, s in hits:
            if any(n.lower() in c.text.lower() for n in names):         # tickets that mention the place; an agent judges what they mean
                out.append(c)
        return out[:k + 1]

    def plan(self) -> list[Option]:
        """Valued options (cached until the evidence changes, so asking twice is free)."""
        if not self.use_planner:
            return []
        left = self.env.checks_left() if self.use_field else 0
        cl = max(self.env.cfg["canvass"]["max_per_case"] - self.env.canvasses, 0) if (self.use_field and self.use_canvass) else 0
        key = (self.board.version, left, self.digs, cl)
        if key != self._plan_key:
            opts = self.planner.evaluate(self.board, left, 2 - self.digs, cl) if left else []
            self.options = {f"F{i + 1}": o for i, o in enumerate(opts[:4])}
            self._plan_key = key
        return list(self.options.values())

    def rule_option(self):
        """The simple dispatcher rule (no look-ahead): listen where the map is concentrated, else night step test of the sections holding most of it."""
        lite = self.planner.lite(self.board)
        cand = getattr(self.board, "cand", None)
        if cand is not None and cand.any():                  # ADR-016: only where somebody hypothesised a leak
            lite.lp = np.where(cand, lite.lp, -np.inf)
        o = self.planner.base_choice(lite)
        if o is None:
            return None
        o.label = (f"night step test of sections {', '.join(o.args['sections'])}" if o.kind == "step"
                   else f"listen within 250 m of junction {self.net.junctions[o.args['center']]}")
        self.options = {"F1": o}
        return o

    def add_option(self, oid: str, kind: str, args: dict, label: str) -> str | None:
        """Register a field check someone proposed (no look-ahead value): a night step test of up to 6 valid sections, or listening at a junction."""
        from ..search.planner import Option
        if kind == "step":
            if not self.allow_step:
                return None
            sids = [s for s in dict.fromkeys(args.get("sections") or []) if s in self.sections.areas][:6]
            if not sids:
                return None
            args = {"sections": sids}
        elif kind == "listen":
            if not isinstance(args.get("center"), (int, np.integer)):
                return None
        else:
            return None
        post = self.board.post * self.board.p_leak
        p_in = (sum(self.sections.area_mass(s, post) for s in args["sections"]) if kind == "step"
                else float(self.planner.listen_cover[args["center"]] @ post))
        self.options[oid] = Option(kind, args, label, float(p_in))
        return oid

    def can_call(self, rid: str) -> bool:
        m = self.board.messages.get(rid)
        return m is not None and m.usable() and self.attempts.get(rid, 0) < self.env.cfg["call"]["max_per_message"]

    def exhausted(self) -> bool:
        """No field check can add anything any more (budget or time spent, or none is worth its cost)."""
        if not self.use_field or self.env.checks_left() <= 0 or self.env.now >= self.env.horizon:
            return True
        if not self.use_planner:
            return False
        opts = self.plan()
        return bool(opts) and max(o.value for o in opts) <= 0.01

    def gate(self) -> tuple[bool, str]:
        j, p = self.board.best_dig()
        final = self.exhausted()
        sup = self.board.field_support(j, final)
        bar = self.planner.final_bar if final else self.planner.dig_bar
        if self.digs >= 2:
            return False, "two digs already done (SOP-2): hand the case over"
        if not self.use_gate:
            return True, "no code gate in this setup"
        if p < bar:
            return False, f"P(success) {p:.2f} is below the bar {bar:.2f}" + (" (last-resort bar: no check can add anything)" if final else "")
        if not sup:
            return False, "no field check (step test or listening) supports the spot yet; messages alone can be wrong or faked"
        return True, f"P(success) {p:.2f} >= {bar:.2f}; supported by {', '.join(sup[:3])}" + (" (last resort)" if final and p < self.planner.dig_bar else "")

    # ------------------------------------------------------------------ tools
    def call_back(self, calls: list[dict]) -> dict:
        """Call several people back (free, in parallel). calls: [{"report_id", "question"}]."""
        t0 = time.time()
        args = {"calls": calls}
        if not isinstance(calls, list) or not calls:
            return self._err("call_back", args, "give a list of {report_id, question}")
        asked, results, longest = [], [], 0.0
        for c in calls[:4]:
            rid, q = str(c.get("report_id", "")), str(c.get("question", "")).strip()
            if rid not in self.board.messages:
                results.append({"report_id": rid, "error": "unknown message id"})
                continue
            r = self.env.call(rid, q)
            if not r.get("ok"):
                results.append({"report_id": rid, "error": r["error"]})
                continue
            longest = max(longest, r["delay_s"])
            self.attempts[rid] = self.attempts.get(rid, 0) + 1
            asked.append((rid, q, r["reply_text"]))
        replies = self.reader.read_replies([(q, a) for _, q, a in asked])
        for (rid, q, a), rep in zip(asked, replies):
            self.search.apply_reply(self.board, rid, rep)
            self.calls.setdefault(rid, []).append({"question": q, "reply": rep})
            facts = [k for k in ("source", "not_network", "still_flowing") if rep.get(k) not in (None, "unknown")]
            results.append({"report_id": rid, "answered": rep["answered"],
                            "learned": {k: rep[k] for k in facts},
                            "places_named": len(rep.get("places") or []) + len(rep.get("flows_from") or [])})
        self.env.advance(longest)
        new = self.refresh()
        return self._log("call_back", args, {"ok": True, "results": results, "new_messages": new}, time.time() - t0)

    def _record_check(self, rid: str, record_id: str) -> tuple[bool, str]:
        """Does this record describe a non-utility false alarm at the place this message points to? (pure, no logging)"""
        c = self.docs.by_id.get(record_id) if self.use_rag else None
        if c is None or c.provenance != "record":
            return False, "unknown record id"
        m = self.board.messages.get(rid)
        if m is None or m.w is None or m.eps is None:
            return False, "the record's place is not where this message points"
        w = self._record_footprint(c.text)
        if w is None or float(m.w @ self.search.near_share(w)) < 0.3:
            return False, "the record's place is not where this message points"
        return True, ""

    def _record_footprint(self, text: str):
        """Junction weights of the places a ticket mentions (landmark disc / street), or None."""
        ws = []
        for _, _, _, ids in self.places.find_mentions(text):
            ws += [self.places.place_nodes(i) for i in ids[:1]]
        if not ws:
            return None
        w = sum(ws[:3])
        return w / w.sum() if w.sum() > 0 else None

    def record_applies(self, rid: str, record_id: str) -> bool:
        return self._record_check(rid, record_id)[0]

    def use_record(self, rid: str, record_id: str) -> dict:
        """Count a record of repeated non-utility false alarms at this message's place (verified by code)."""
        args = {"report_id": rid, "record_id": record_id}
        ok, why = self._record_check(rid, record_id)
        if not ok:
            return self._err("use_record", args, why)
        if not self.search.mark_record_match(self.board, rid, self._record_footprint(self.docs.by_id[record_id].text)):
            return self._err("use_record", args, "the record's place is not where this message points")
        self.records_used[rid] = record_id
        return self._log("use_record", args, {"ok": True, "effect": "message counted as a likely repeat false alarm"})

    def repair_tickets_near(self, k: int = 5) -> list:
        """RAG over the town archive: repair tickets (not complaint visits) whose place lies within 300 m of a usable complaint or a hypothesised
        junction. The agent judges what a ticket means; code only finds and places them."""
        if not self.use_rag:
            return []
        b, D = self.board, self.net.pipe_distance
        focus = [int(np.argmax(m.w)) for m in b.messages.values() if m.usable()]
        cand = getattr(b, "cand", None)
        if cand is not None and cand.any():
            focus += list(np.where(cand)[0][:: max(int(cand.sum()) // 40, 1)])
        if not focus:
            return []
        out = []
        for cid, c in self.docs.by_id.items():
            if c.provenance != "record" or "No utility pipe leak found" in c.text or cid in self.repairs_used:
                continue
            w = self._record_footprint(c.text)
            if w is None:
                continue
            j = int(np.argmax(w))
            d = min(float(D[j, f]) for f in focus)
            if d <= 300:
                out.append((d, c))
        return [c for _, c in sorted(out, key=lambda x: x[0])[:k]]

    def use_repair_record(self, record_id: str) -> dict:
        """The agent cites a ticket as a failing stretch of main. Code checks it is a repair ticket with a place on the map, raises the prior
        within 150 m of it by the failure hazard, and hypothesises that stretch (it is simulated like any other hypothesis)."""
        args = {"record_id": record_id}
        c = self.docs.by_id.get(str(record_id)) if self.use_rag else None
        if c is None or c.provenance != "record" or "No utility pipe leak found" in c.text:
            return self._err("use_repair_record", args, "not a repair ticket of the archive")
        w = self._record_footprint(c.text)
        if w is None:
            return self._err("use_repair_record", args, "the ticket's place is not on the map")
        mask = self.net.pipe_distance[int(np.argmax(w))] <= 150
        n = self.search.raise_prior(self.board, mask)
        added = self.search.add_candidates(self.board, np.where(mask)[0], f"history:{record_id}")
        self.repairs_used.add(str(record_id))
        return self._log("use_repair_record", args, {"ok": True, "junctions_raised": n, "hypothesised": added,
                                                     "where": self.describe_node(int(np.argmax(w)))})

    def search_docs(self, query: str) -> dict:
        if not self.use_rag:
            return self._err("search_docs", {"query": query}, "document search is not available in this setup")
        hits = self.docs.search(str(query), k=4)
        return self._log("search_docs", {"query": query}, {"ok": True, "hits": [{"id": c.id, "text": c.text[:300]}
                                                                                for c, _ in hits]})

    def _promote_physical(self) -> None:
        """A lit step-test area or a heard leak is physical proof: its junctions become hypotheses (simulated) whether or not any agent named them.
        Mechanical, no language: every toolbox method gets the same rule (the no-AI fallback already did)."""
        b = self.board
        if getattr(b, "cand", None) is None:
            return
        js = []
        for c in b.checks:
            if c.get("support") is not None:
                js += list(np.where(c["support"])[0])
            elif c["kind"] == "listen_hit" and c.get("at") is not None:
                js += list(np.where(self.net.pipe_distance[c["at"]] <= 150)[0])
        if js:
            self.search.add_candidates(b, js, "physical", force=True)

    def field_check(self, option_id: str) -> dict:
        t0 = time.time()
        args = {"option": option_id}
        if not self.use_field:
            return self._err("field_check", args, "field checks are not available in this setup")
        o = self.options.get(str(option_id).strip().upper())
        if o is None:
            return self._err("field_check", args, f"unknown option; choose one of {sorted(self.options)}")
        if o.kind == "step":
            if not self.allow_step:
                return self._err("field_check", args, "step tests / valve closures are disabled for practical real-world operations")
            sids = o.args["sections"]
            nodes = [n for s in sids for n in self.sections.by_id[s]["junctions"]]
            if self.use_rag:
                for need in self.rules.requirements("step_test", nodes, self.env.clock()):
                    self.env.notify(f"hospital:{need['hospital']}", "Night step test planned in your area (SOP-7).")
            post = self.board.post * self.board.p_leak
            cert = {"why": "locate a suspected leak", "evidence": [f"{s}: {self.sections.area_mass(s, post):.0%} of the leak map"
                                                               for s in sids]}
            ap = self.env.request_approval("step_test", sids, cert)
            r = self.env.step_test(sids, ap.get("token"))
            if r.get("ok"):
                self.board.add_step_result(f"step@{r['night_of']}", r["steps"])
                ok_steps = [s for s in r["steps"] if not s.get("failed")]
                lit_labels = {c["label"] for c in self.board.checks if c["kind"] == "step" and c.get("support") is not None}
                lit = [s["section"] for s in ok_steps if f"step@{r['night_of']}:{s['section']}" in lit_labels]
                r = {"ok": True, "kind": "step test", "excess_flow_lps": {s["section"]: round(
                    s["inflow_drop_lps"] - s["expected_customer_use_lps"], 2) for s in ok_steps}, "clearly_lit": lit,
                     "suggestive": [s["section"] for s in ok_steps if s["section"] not in lit and (lambda a: a[0] > 3 * a[1])(self.board.tools.step(
                         s["section"], s["inflow_drop_lps"] - s["expected_customer_use_lps"], self.board.q_hat, self.board.q_sd))],
                     "valve_failed": [s["section"] for s in r["steps"] if s.get("failed")]}
        elif o.kind == "canvass":
            c = o.args["center"]
            r = self.env.canvass(self.net.junctions[c])
            if r.get("ok"):
                self.board.add_canvass_result(f"canvass@{self.net.junctions[c]}", c, r["contacted"], r["heard_hissing"],
                                              r["saw_damp"])
                r = {"ok": True, "kind": "canvass", "where": self.describe_node(c), "residents_answered": r["contacted"],
                     "heard_hissing_at_night": r["heard_hissing"], "saw_damp_ground": r["saw_damp"]}
        else:
            c = o.args["center"]
            r = self.env.listen(self.net.junctions[c], 250.0)
            if r.get("ok"):
                at = self.net.index[r["pinpointed_at"]] if r.get("heard") else None
                self.board.add_listen_result(f"listen@{self.net.junctions[c]}", c, 250.0, r["heard"], at)
                r = {"ok": True, "kind": "listening", "where": self.describe_node(c), "heard": r["heard"],
                     **({"pinpointed": self.describe_node(at)} if at is not None else {})}
        self._promote_physical()
        self.refresh()
        r["p_success_now"] = round(self.board.best_dig()[1], 2)
        return self._log("field_check", {**args, "label": o.label}, r, time.time() - t0)

    def sop_context(self, j: int) -> list[str]:
        """What an auditor needs to apply the SOP to a dig at junction j: where and when it would happen, which schools and hospitals are near
        (facts of the map), and the SOP clauses retrieved from sop.md (RAG). Code states facts and text; deciding what applies is the agent's job."""
        when = self.env.clock()
        st = self.places.node_street(j)
        lms = []
        for p in self.places.places.values():
            if p["type"] == "landmark" and p.get("subtype") in ("school", "hospital"):
                d = float(np.hypot(*(self.net.xy[j] - self.net.xy[self.net.index[p["geometry"]["node"]]])))
                if d <= 250:
                    lms.append(f"{p['name']} ({p['subtype']}, {d:.0f} m away)")
        lines = [f"DIG SITE: {self.describe_node(j)}; street: {st['name'] if st else 'unknown'}; the dig would start {when.strftime('%A %H:%M')} (case clock)",
                 "SCHOOLS / HOSPITALS within 250 m: " + ("; ".join(lms) if lms else "none")]
        if self.use_rag:
            q = f"excavation dig {st['name'] if st else ''} school hospital resurfaced road permit weekday schools notify citizens approval"
            lines += ["SOP CLAUSES (retrieved from sop.md):"] + [f"  {c.id}: {c.text[:420]}" for c, _ in self.docs.search(q, k=6, provenance="policy")]
        return lines

    def _sop_execute(self, actions: list) -> tuple[bool, list[str], list[str]]:
        """Run the paperwork an agent asked for; -> (after_school_hours, reporters_to_inform, errors). Names must exist on the map."""
        after, inform, errs = False, [], []
        hospitals = {h.lower(): h for h in self.rules.hospitals}
        streets = {p["name"].lower(): p["name"] for p in self.places.places.values() if p["type"] == "street"}
        for a in actions or []:
            a = str(a).strip()
            kind, _, arg = a.partition(":")
            kind, arg = kind.strip().lower(), arg.strip()
            if kind == "after_school_hours":
                after = True
            elif kind == "request_permit" and arg.lower() in streets:
                self.env.request_permit(streets[arg.lower()])
            elif kind == "notify_hospital" and arg.lower() in hospitals:
                self.env.notify(f"hospital:{hospitals[arg.lower()]}", "Excavation planned nearby (SOP-7).")
            elif kind == "inform_reporters":
                inform.append("reporters")
            else:
                errs.append(a)
        return after, inform, errs

    def propose_dig(self, reason: str = "", sop_actions: list | None = None) -> dict:
        args = {"reason": str(reason)[:200], **({"sop_actions": [str(x) for x in sop_actions][:6]} if sop_actions else {})}
        ok, why = self.gate()
        if not ok:
            return self._err("propose_dig", args, f"code gate refused: {why}")
        j, p = self.board.best_dig()
        node = self.net.junctions[j]
        after_school, inform, bad = False, [], []
        if self.auto_sop:                                    # SOP paperwork done by the rule book in code (a hard-wired baseline)
            for need in self.rules.requirements("dig", [node], self.env.clock()):
                if need["rule"] == "school_hours":
                    after_school = True
                elif need["rule"] == "resurfaced_permit":
                    self.env.request_permit(need["street"])
                elif need["rule"] == "hospital_notice":
                    self.env.notify(f"hospital:{need['hospital']}", "Excavation planned nearby (SOP-7).")
            inform = ["reporters"]
        elif sop_actions:                                    # paperwork an agent asked for after reading the SOP
            after_school, inform, bad = self._sop_execute(sop_actions)
        ev = self.board.pressure_evidence
        evidence = (self.board.field_support(j) + self.board.area_support(j)
                    + [f"{rid} (false-alarm chance {m.eps:.0%})" for rid, m in self.board.messages.items()
                       if m.usable() and m.eps < 0.6]
                    + [f"pressure sensors: net inflow rise {ev.q_hat:+.1f} L/s, leak map P(leak) {self.board.p_leak:.2f}"])
        evidence = list(dict.fromkeys(evidence))
        cert = {"spot": self.describe_node(j), "p_success": round(p, 2), "evidence": evidence[:6] or ["leak map"],
                "why": "leak located by the investigation"}
        ap = self.env.request_approval("dig", node, cert)
        if not ap.get("approved"):
            return self._err("propose_dig", args, f"operator refused: {ap.get('reason')}")
        r = self.env.dig(node, ap["token"], after_school_hours=after_school)
        self.digs += 1
        if r.get("ok") and r["leak_found"]:
            if inform:
                self.env.notify("reporters", "The leak you reported was found and is being repaired (SOP-8).")
            self.done = True
        elif r.get("ok"):
            self.board.add_dry_dig(f"dry dig@{node}", j)
            if inform:
                self.env.notify("reporters", "We checked the spot; investigation continues (SOP-8).")
        return self._log("propose_dig", args, {"ok": r.get("ok", False), "spot": self.describe_node(j),
                                               "leak_found": r.get("leak_found"), "p_success": round(p, 2),
                                               **({"sop_unparsed": bad} if bad else {})})

    def close_gate(self) -> tuple[bool, str]:
        """May the case be closed without a (further) dig? Only when the leak is unlikely or no check can help."""
        if self.env.checks_left() <= 0 or not self.use_field or self.env.now >= self.env.horizon:
            return True, "no field checks or time left"
        used = sum(1 for t in self.trace if t["tool"] == "field_check" and t["result"].get("kind") != "canvass" and t["result"].get("ok", True))
        all_debunked = bool(self.board.messages) and all(
            (not m.usable()) or m.eps >= 0.85 or m.source == "not_network"
            for m in self.board.messages.values()
        )
        has_physical = any((c["kind"] == "step" and c.get("support") is not None) or c["kind"] == "listen_hit" for c in self.board.checks)
        if all_debunked and not has_physical and self.board.p_leak < 0.25:
            return True, f"all complaints debunked as non-network/false alarms and P(leak) {self.board.p_leak:.2f} < 0.25"
        if used < self.min_checks_close and self.board.p_leak >= 0.03:
            return False, f"sweep first: only {used} of at least {self.min_checks_close} field checks done (a leak can hide from a map that is wrong)"
        if self.board.p_leak < 0.15:
            return True, f"P(leak) {self.board.p_leak:.2f} < 0.15"
        if not self.use_planner:
            return False, f"P(leak) is still {self.board.p_leak:.2f} and checks are left"
        best = max((o.found for o in self.plan()), default=0.0)
        if best < 0.10:
            return True, f"no remaining check has a 10 % chance of finding the leak (best {best:.0%})"
        return False, (f"P(leak) is still {self.board.p_leak:.2f} and the best check has a {best:.0%} chance of "
                       f"finding the leak. Explaining a message does not explain the pressure/flow anomaly.")

    def close(self, verdict: str, summary: str = "", force: bool = False) -> dict:
        if verdict not in ("no_leak", "hand_over"):
            return self._err("close", {"verdict": verdict}, "verdict must be no_leak or hand_over")
        ok, why = self.close_gate()
        if not ok and not force:
            return self._err("close", {"verdict": verdict}, f"close gate refused: {why}")
        self.env.close(verdict, summary)
        self.done = True
        return self._log("close", {"verdict": verdict, "summary": summary[:300]}, {"ok": True})

    def final_guess(self) -> str:
        return self.net.junctions[self.board.best_dig()[0]]
