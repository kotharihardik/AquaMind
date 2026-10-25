"""RQ2 baselines (proposal Section 10) and the crews' current practice. Same environment, cases, budget and luck.

  react_agent    a standard tool-using LLM agent (ReAct): same model, same field tools and the leak search as a tool,
                 same step budget; reads raw messages, picks the dig location itself, no hypothesis board, no code gate
  plain_llm      the case as text (sensor summary + raw messages), one answer: where to dig, or no dig
  fixed_rules    candidates by the fixed no-AI rule (sensors only: without an LLM no message is understood, ADR-017), one temporary
                 pressure sensor where they disagree most, then dig the best one
  (ADR-016: no method, baseline or ablation uses the precomputed 782-junction fingerprint table; physics is simulated per named candidate)
  crew_practice  what crews do today without decision support: listen where people report (and near the sensor
                 that dropped most), dig where leak noise is heard (people read the messages: the rule reader stands in for them)
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..reader.rules import RuleReader
from ..search.hypotheses import place_weights
from ..search.tool_models import LOGGER_SD_M

STEP_BUDGET = 15


def _cert(evidence: list[str], why: str) -> dict:
    return {"why": why, "evidence": evidence}


def _resolve_place(world, text: str) -> int | None:
    """A free-text place or a junction id -> junction index (the LLM baselines name places themselves)."""
    t = str(text or "").strip()
    net, places = world.town.net, world.town.places
    if t in net.index:
        return net.index[t]
    m = re.search(r"\bn\d+\b", t)
    if m and m.group(0) in net.index:
        return net.index[m.group(0)]
    found = [p for _, _, _, ids in places.find_mentions(t) for p in ids] or places.resolve(t)
    if not found:
        return None
    w = sum(places.place_nodes(p) for p in found)
    return int(np.argmax(w))


def _sensor_summary(case) -> str:
    from ..search.pressure import build_evidence
    ev = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s)
    from ..agent.agentix import _dp, _dq
    s = ", ".join(_dp(n, d, z) for n, d, z in zip(case.pressure_nodes, ev.drop_m, ev.z))
    return f"pressure in the last 6 h vs normal (a leak makes pressure FALL and zone inflow RISE): {s}; zone inflow {_dq(ev.q_hat)}"


# ====================================================================== fixed rules (no LLM)
class FixedRules:
    """One temporary pressure logger where the candidate leaks disagree most, then dig the best candidate. Candidates come from the fixed
    no-AI rule (near the complaints and the sensors that dropped); physics only for them (ADR-016)."""
    name = "fixed_rules"

    def run_case(self, world, env, case) -> dict:
        from ..agent.toolbox import Toolbox
        from ..search.candidates import heuristic
        from ..methods import NullReader
        reads = NullReader()                                   # no LLM, so no language: the messages are not understood (ADR-017)
        ls = world.search(hypothesis_mode=True, use_messages=False)
        tb = Toolbox(env, case, ls, reads, world.docs, world.rules, use_planner=False)
        b = tb.board
        ls.add_candidates(b, heuristic(tb), "heuristic")
        cands = list(np.where(b.cand)[0]) if b.cand.any() else []
        steps, spot = 0, None
        if cands:
            top = sorted(cands, key=lambda j: -b.post[j])[:8]
            g = np.array([ls.physics.fingerprint(j)[0] for j in top])           # [top, J] night drop per L/s, simulated on demand
            n = int(np.argmax(g.std(axis=0)))
            node = world.town.net.junctions[n]
            r = env.place_logger(node)
            steps = 1
            if r.get("ok"):
                pred = ls.physics.base["night"][n] - max(b.q_hat, 0.5) * np.array([ls.physics.fingerprint(j)[0][n] for j in cands])
                lik0 = float(np.exp(-0.5 * ((r["night_mean_pressure_m"] - ls.physics.base["night"][n]) / LOGGER_SD_M) ** 2))
                lik = np.full(len(b.base), lik0)                                     # untested junctions: no information
                lik[cands] = np.exp(-0.5 * ((r["night_mean_pressure_m"] - pred) / LOGGER_SD_M) ** 2)
                b.add_check(f"logger@{node}", "logger", lik, lik0)
            j, p = b.best_dig()
            spot = world.town.net.junctions[j]
            ap = env.request_approval("dig", spot, _cert([f"temporary sensor at {node}", "candidate with the best fit"], "fixed rule"))
            if ap.get("approved"):
                env.dig(spot, ap["token"])
            steps += 2
        if not env.done:
            env.close("hand_over", "fixed rules done")
        return {"final_guess": spot, "steps": steps, "tool_calls": steps}


# ====================================================================== crews' current practice (no software)
class CrewPractice:
    name = "crew_practice"

    def run_case(self, world, env, case) -> dict:
        net, places = world.town.net, world.town.places
        reader = RuleReader(places)
        reads = reader.read_many(case.reports)
        rep = world.search(hypothesis_mode=True).rep                          # place names -> streets only (no leak fingerprints)
        leads = []
        for r in sorted(case.reports, key=lambda r: r["received_at"], reverse=True):     # newest first
            rd = reads[r["report_id"]]
            if rd["category"] == "leak" and not rd["contains_instructions"]:
                w = rep.place_weights(rd["places"])
                if w is not None:
                    leads.append(int(np.argmax(w)))
        from ..search.pressure import build_evidence
        ev = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s)
        if case.trigger == "scada_alarm" or not leads:
            s = case.pressure_nodes[int(np.argmax(ev.z))]
            leads.append(net.index[s])
        D = net.pipe_distance
        ring = []                                 # then walk the streets around each lead (300 m out, spread apart)
        for c in leads:
            around = [int(j) for j in np.where((D[c] > 200) & (D[c] < 450))[0]]
            pick = []
            for j in sorted(around, key=lambda j: -net.elevation[j]):     # uphill first: water runs downhill
                if all(D[j, k] > 300 for k in pick):
                    pick.append(j)
            ring += pick[:4]
        steps, guess, tried = 0, None, []
        for c in leads + ring:
            if env.done or env.checks_left() <= 0:
                break
            if any(net.pipe_distance[c, t] < 150 for t in tried):
                continue
            tried.append(c)
            r = env.listen(net.junctions[c], 250.0)
            steps += 1
            guess = guess if guess is not None else net.junctions[c]
            if r.get("ok") and r.get("heard"):
                spot = r["pinpointed_at"]
                guess = spot
                ap = env.request_approval("dig", spot, _cert(["citizen report", "leak noise heard"], "crew heard a leak"))
                if ap.get("approved"):
                    env.dig(spot, ap["token"])
                    steps += 1
        if not env.done:
            env.close("hand_over", "crews found nothing")
        return {"final_guess": guess, "steps": steps, "tool_calls": steps}


# ====================================================================== plain LLM (no tools, one answer)
PLAIN_SYSTEM = """You are a control-room engineer at a water utility. From the case below decide ONE thing: where to
send a crew to dig for a leak, or that no dig is justified. Name the dig place as a street and/or landmark from
the case, or a junction id. Answer JSON: {"decision": "dig" | "no_dig", "place": "...", "reason": "..."}"""


class PlainLLM:
    name = "plain_llm"

    def __init__(self, llm_factory):
        self.llm_factory = llm_factory

    def run_case(self, world, env, case) -> dict:
        llm = self.llm_factory()
        text = (f"Case opened by {case.trigger.replace('_', ' ')}"
                + (f" (message {case.trigger_report})" if getattr(case, "trigger_report", None) else "")
                + f". Sensors: {_sensor_summary(case)}.\n"
                f"Step-test sections: " + "; ".join(s["label"] for s in world.town.sections.items) + "\n"
                "Citizen messages:\n" + "\n".join(f"- {r['report_id']} ({r['received_at']}): {r['text']}"
                                                  for r in case.reports))
        d = llm.json(PLAIN_SYSTEM, text, ["decision"], tag="plain_llm")
        guess = None
        if d.get("decision") == "dig":
            j = _resolve_place(world, d.get("place"))
            if j is not None:
                spot = world.town.net.junctions[j]
                guess = spot
                ap = env.request_approval("dig", spot, _cert([str(d.get("reason", ""))[:200], "LLM judgement"], "plain LLM"))
                if ap.get("approved"):
                    env.dig(spot, ap["token"])
        if not env.done:
            env.close("hand_over", "plain LLM done")
        u = llm.usage
        return {"final_guess": guess, "steps": 1, "tool_calls": 1, "llm_calls": u.calls, "llm_cached": u.cached,
                "llm_seconds": u.seconds, "tokens": u.prompt_tokens + u.output_tokens}


# ====================================================================== standard ReAct agent
REACT_SYSTEM = """You are a leak investigation agent at a water utility. Use the tools, one per step, to find the leak
and dig at it, or close the case if there is no leak. You have at most {budget} steps and {checks} field checks
(step tests, listening surveys, temporary loggers; calls and searches are free).
Tools:
- test_places {{"places": ["S04", "Market Road", "n123"]}}: simulate a leak at every junction of these sections / places on the utility's model and
  return the best-fitting spots among everything you have tested so far (nothing is simulated that you do not name)
- call {{"report_id": "R-003", "question": "..."}}: call back the person who sent a message
- step_test {{"sections": ["S03", ...]}}: night step test of up to 6 sections (needs operator approval first)
- listen {{"place": "junction id or street/landmark"}}: acoustic survey within 250 m
- place_logger {{"place": "..."}}: temporary pressure logger for one night
- search_docs {{"query": "..."}}: search the SOP and the complaint/repair records
- request_approval {{"action": "dig" | "step_test", "target": "<place or list of sections>", "evidence": ["..."]}}
- dig {{"place": "...", "token": "..."}}: send a crew to dig (needs an approval token)
- close {{"verdict": "no_leak" | "hand_over"}}
Answer each step with JSON: {{"thought": "...", "tool": "...", "args": {{...}}}}"""


class ReActAgent:
    name = "react_agent"

    def __init__(self, llm_factory, budget: int = STEP_BUDGET):
        self.llm_factory, self.budget = llm_factory, budget

    def run_case(self, world, env, case) -> dict:
        llm = self.llm_factory()
        net = world.town.net
        ls = world.search(hypothesis_mode=True, use_messages=False)       # the agent reads the raw messages itself; the tool is physics only
        reads = {}
        board = ls.start(case, reads)
        sys = REACT_SYSTEM.format(budget=self.budget, checks=env.checks_left())
        hist = [f"Case opened by {case.trigger.replace('_', ' ')}"
                + (f" (message {case.trigger_report})" if getattr(case, "trigger_report", None) else "")
                + f". Sensors: {_sensor_summary(case)}.",
                "Sections: " + "; ".join(s["label"] for s in world.town.sections.items),
                "Messages:\n" + "\n".join(f"- {r['report_id']} ({r['received_at']}): {r['text']}" for r in case.reports)]
        n_head = len(hist)
        tokens, steps, calls, errors, guess = {}, 0, 0, 0, None
        trace = []
        for steps in range(1, self.budget + 1):
            if env.done:
                break
            d = llm.json(sys, "\n".join(hist[:n_head] + hist[n_head:][-11:]) + "\nNext step?", ["tool"], tag="react")
            tool, args = str(d.get("tool", "")), d.get("args") if isinstance(d.get("args"), dict) else {}
            res = self._act(world, env, board, ls, tool, args, tokens)
            calls += 1
            errors += not res.get("ok", True)
            if tool == "dig" and args.get("place"):
                j = _resolve_place(world, args["place"])
                guess = net.junctions[j] if j is not None else guess
            trace.append({"step": steps, "thought": d.get("thought"), "tool": tool, "args": args, "result": res})
            hist.append(f"Step {steps}: {tool} {json.dumps(args)[:200]} -> {json.dumps(res)[:400]}")
        if not env.done:
            env.close("hand_over", "ReAct agent stopped")
        u = llm.usage
        return {"final_guess": guess or net.junctions[board.best_dig()[0]], "steps": steps, "tool_calls": calls,
                "tool_errors": errors, "trace": trace, "llm_calls": u.calls, "llm_cached": u.cached,
                "llm_seconds": u.seconds, "tokens": u.prompt_tokens + u.output_tokens}

    def _act(self, world, env, board, ls, tool, args, tokens) -> dict:
        net, sec = world.town.net, world.town.sections
        if tool == "test_places":
            js = []
            for p in (args.get("places") or [])[:4]:
                p = str(p)
                if p in sec.areas:
                    js += list(np.where(sec.areas[p]["mask"])[0])
                elif p in net.index:
                    js += list(np.where(net.pipe_distance[net.index[p]] <= 250)[0])
                else:
                    w = place_weights(world.town.places, p)
                    if w is not None:
                        js += list(np.where(net.pipe_distance[int(np.argmax(w))] <= 250)[0])
            if not js:
                return {"ok": False, "error": "no known section, junction or place"}
            ls.add_candidates(board, js, "react")
            post = board.post
            top = [int(j) for j in np.argsort(-np.where(board.cand, post, -1))[:5]]
            return {"ok": True, "best_spots": [f"{world.town.places.describe(j)} ({net.junctions[j]}) {post[j]:.1%}" for j in top],
                    "share_outside_everything_tested": round(board.mass_outside(), 2)}
        if tool == "call":
            r = env.call(str(args.get("report_id")), str(args.get("question", "")))
            if r.get("ok"):
                env.advance(r["delay_s"])
                return {"ok": True, "reply": r["reply_text"] or "(no answer)"}
            return r
        if tool == "step_test":
            sids = [str(s) for s in args.get("sections") or []]
            r = env.step_test(sids, tokens.pop(("step_test", tuple(sorted(sids))), None))
            if r.get("ok"):
                board.add_step_result("react-step", r["steps"])
            return r
        if tool == "canvass":
            j = _resolve_place(world, args.get("place"))
            if j is None:
                return {"ok": False, "error": "unknown place"}
            r = env.canvass(net.junctions[j])
            if r.get("ok"):
                board.add_canvass_result("react-canvass", j, r["contacted"], r["heard_hissing"], r["saw_damp"])
            return r
        if tool in ("listen", "place_logger"):
            j = _resolve_place(world, args.get("place"))
            if j is None:
                return {"ok": False, "error": "unknown place"}
            if tool == "listen":
                r = env.listen(net.junctions[j], 250.0)
                if r.get("ok"):
                    at = net.index[r["pinpointed_at"]] if r.get("heard") else None
                    board.add_listen_result("react-listen", j, 250.0, r["heard"], at)
                    if at is not None:
                        r = {**r, "pinpointed_at": f"{world.town.places.describe(at)} ({r['pinpointed_at']})"}
                return r
            return env.place_logger(net.junctions[j])
        if tool == "search_docs":
            return {"ok": True, "hits": [{"id": c.id, "text": c.text[:250]} for c, _ in world.docs.search(str(args.get("query", "")))]}
        if tool == "request_approval":
            action = args.get("action")
            if action == "dig":
                j = _resolve_place(world, args.get("target"))
                if j is None:
                    return {"ok": False, "error": "unknown place"}
                target = net.junctions[j]
            elif action == "step_test":
                target = [str(s) for s in (args.get("target") if isinstance(args.get("target"), list) else [args.get("target")])]
            else:
                return {"ok": False, "error": "action must be dig or step_test"}
            r = env.request_approval(action, target, {"evidence": args.get("evidence") or [], "why": "react agent"})
            if r.get("approved"):
                tokens[("dig", target) if action == "dig" else ("step_test", tuple(sorted(target)))] = r["token"]
                return {"ok": True, "approved": True, "token": "<kept>"}
            return {"ok": True, "approved": False, "reason": r.get("reason")}
        if tool == "dig":
            j = _resolve_place(world, args.get("place"))
            if j is None:
                return {"ok": False, "error": "unknown place"}
            node = net.junctions[j]
            r = env.dig(node, tokens.pop(("dig", node), None))
            if r.get("ok") and not r["leak_found"]:
                board.add_dry_dig("react-dry", j)
            return r
        if tool == "close":
            return env.close(str(args.get("verdict", "hand_over")))
        return {"ok": False, "error": f"unknown tool '{tool}'"}


def registry(llm_factory) -> dict:
    return {"fixed_rules": FixedRules(), "crew_practice": CrewPractice(), "plain_llm": PlainLLM(llm_factory),
            "react_agent": ReActAgent(llm_factory)}
