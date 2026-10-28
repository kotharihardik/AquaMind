"""RQ2 Baseline: Standard ReAct Tool-Using LLM Agent (Proposal Section 10).

Same model, same field tools, and the leak search as a tool, same step budget (15).
Reads raw messages, picks dig location itself, no hypothesis board, and no code gate.
"""
from __future__ import annotations

import json
import re
import numpy as np

from ..search.hypotheses import place_weights

STEP_BUDGET = 15

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


def resolve_place(world, text: str) -> int | None:
    """Resolve free-text place or junction id to network junction index."""
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


def format_sensor_summary(case) -> str:
    from ..search.pressure import build_evidence
    ev = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s)
    from ..agent.agentix import _dp, _dq
    s = ", ".join(_dp(n, d, z) for n, d, z in zip(case.pressure_nodes, ev.drop_m, ev.z))
    return f"pressure in the last 6 h vs normal: {s}; zone inflow {_dq(ev.q_hat)}"


class ReActAgent:
    name = "react_agent"

    def __init__(self, llm_factory, budget: int = STEP_BUDGET, allow_logger: bool = True):
        self.llm_factory, self.budget, self.allow_logger = llm_factory, budget, allow_logger

    def run_case(self, world, env, case) -> dict:
        llm = self.llm_factory()
        net = world.town.net
        ls = world.search(hypothesis_mode=True, use_messages=False)
        reads = {}
        board = ls.start(case, reads)
        sys = REACT_SYSTEM.format(budget=self.budget, checks=env.checks_left())
        hist = [f"Case opened by {case.trigger.replace('_', ' ')}"
                + (f" (message {case.trigger_report})" if getattr(case, "trigger_report", None) else "")
                + f". Sensors: {format_sensor_summary(case)}.",
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
                j = resolve_place(world, args["place"])
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
            j = resolve_place(world, args.get("place"))
            if j is None:
                return {"ok": False, "error": "unknown place"}
            r = env.canvass(net.junctions[j])
            if r.get("ok"):
                board.add_canvass_result("react-canvass", j, r["contacted"], r["heard_hissing"], r["saw_damp"])
            return r
        if tool in ("listen", "place_logger"):
            j = resolve_place(world, args.get("place"))
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
            if not self.allow_logger:
                return {"ok": False, "error": "temporary pressure logger tool is disabled in this ablation"}
            return env.place_logger(net.junctions[j])
        if tool == "search_docs":
            return {"ok": True, "hits": [{"id": c.id, "text": c.text[:250]} for c, _ in world.docs.search(str(args.get("query", "")))]}
        if tool == "request_approval":
            action = args.get("action")
            if action == "dig":
                j = resolve_place(world, args.get("target"))
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
            j = resolve_place(world, args.get("place"))
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
