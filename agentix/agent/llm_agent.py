"""AgentiX investigator: an LLM planner on top of the safe toolbox ("LLM proposes, code verifies").

Each round ("plan, then execute"):
  1. code draws the case board (numbers, safe summaries, the menu with expected gains),
  2. the LLM answers with a thought, optional alternative explanations ("hypotheses") and a SHORT PLAN of 1-4
     actions,
  3. code tests new hypotheses for free (physics + Bayes), then runs the actions one by one through the toolbox,
     and stops early to ask the LLM again when something important changes (a check fails, the gate can pass,
     a new report arrives, an explanation's probability moves by more than 0.2),
  4. before a dig, a second LLM call (the critic) may hold it - but only with a testable, cited alternative,
  5. everything is logged (board, belief, menu/VOI table, thought, actions, tool results, tokens).
Planning a few steps at once uses 2-3x fewer LLM calls than one call per action (free-tier quotas are small).
If the LLM fails twice in a row or the rounds run out, a simple code fallback closes the case safely.
"""
from __future__ import annotations

from .board_view import render
from .prompts import CRITIC_SCHEMA, CRITIC_SYSTEM, PLANNER_SCHEMA, PLANNER_SYSTEM
from .toolbox import ALT_LIMIT, DIG_THRESHOLD, TOOLS, Toolbox

SHIFT = 0.2           # an explanation moving more than this is "important news": ask the LLM again


class LLMInvestigator:
    name = "agentix_llm"

    def __init__(self, llm, max_rounds: int = 6, use_critic: bool = True, use_hypotheses: bool = True,
                 use_keyword_scan: bool = True, name: str | None = None):
        self.llm = llm
        self.max_rounds = max_rounds
        self.use_critic, self.use_hyp, self.use_scan = use_critic, use_hypotheses, use_keyword_scan
        if name:
            self.name = name

    # ------------------------------------------------------------------ main loop
    def run(self, tb: Toolbox) -> list[dict]:
        trace, fails = [], 0
        if self.use_scan and self.use_hyp:              # code does the routine keyword reading first (shared with
            from .rule_agent import keyword_scan       # the rule brain); the LLM reads for what it missed
            trace += [{"round": "scan", **t} for t in keyword_scan(tb)]
        for rnd in range(self.max_rounds):
            if tb.done or tb.env.done:
                break
            board = render(tb)
            u0 = self.llm.usage.as_dict()
            d = self.llm.json(PLANNER_SYSTEM, board + "\nWhat do you do next?", PLANNER_SCHEMA, tag="planner")
            head = {"round": rnd, "t_h": _hours(tb), "p_success": round(tb.p_success(), 3),
                    "explanations": _round(tb.board.explanations()),
                    "menu": [(k, o.oid, round(o.cost, 2), round(o.gain, 3)) for k, o in tb.menu_cache.items()],
                    "board": board}
            actions = d.get("actions")
            if isinstance(d.get("action"), dict) and not actions:          # tolerate the one-action shape
                actions = [d["action"]]
            if "_error" in d or not isinstance(actions, list) or not actions:
                fails += 1
                trace.append({**head, "llm_error": d.get("_error", "no actions"), "tokens": _delta(u0, self.llm)})
                if fails >= 2:
                    break
                continue
            fails = 0
            tests = self.test_hypotheses(tb, d.get("hypotheses") or []) if self.use_hyp else []
            trace.append({**head, "thought": d.get("thought"), "hypotheses": d.get("hypotheses"), "tests": tests,
                          "plan": actions, "tokens": _delta(u0, self.llm)})
            self.execute_plan(tb, actions[:4], trace, rnd)
        if not (tb.done or tb.env.done):
            trace.append({"step": "fallback", **close_safely(tb)})
        return trace

    def execute_plan(self, tb: Toolbox, actions: list, trace: list, rnd: int):
        """Run the planned actions in order; stop early on important news (then the LLM plans again)."""
        p0 = tb.p_success()
        ex0 = tb.board.explanations()
        n0 = len(tb.inv.state["seen_reports"])
        for k, act in enumerate(actions):
            if tb.done or tb.env.done:
                return
            tool = act.get("tool") if isinstance(act, dict) else None
            args = (act.get("args") or {}) if isinstance(act, dict) else {}
            rec = {"round": rnd, "step": k, "action": {"tool": tool, "args": args}, "critic": None}
            if tool == "propose_dig" and self.use_critic and self.gate_open(tb):
                u0 = self.llm.usage.as_dict()
                rec["critic"] = critic = self.critic(tb)
                rec["tokens"] = _delta(u0, self.llm)
                if critic.get("held"):
                    rec["result"] = {"ok": False, "error": "critic held the dig to test an alternative"}
                    trace.append(rec)
                    tb.events.append({"tool": "critic", "args": {}, "result": {"ok": False, "error": critic["concern"]}})
                    return
            res = tb.call(tool, args) if tool in TOOLS else {"ok": False, "error": f"unknown tool '{tool}'"}
            rec.update({"result": _jsonable(res), "p_after": round(tb.p_success(), 3)})
            trace.append(rec)
            if self.news(tb, res, p0, ex0, n0):
                return

    @staticmethod
    def news(tb: Toolbox, res: dict, p0: float, ex0: dict, n0: int) -> bool:
        """Did something happen that the LLM should look at before doing more?"""
        if not res.get("ok", True):
            return True
        p = tb.p_success()
        if (p >= DIG_THRESHOLD) != (p0 >= DIG_THRESHOLD) or abs(p - p0) > SHIFT:
            return True
        if len(tb.inv.state["seen_reports"]) > n0:
            return True
        ex = tb.board.explanations()
        return any(abs(ex.get(k, 0) - ex0.get(k, 0)) > SHIFT for k in set(ex) | set(ex0))

    # ------------------------------------------------------------------ abduction: test what the LLM proposes
    def test_hypotheses(self, tb: Toolbox, hyps: list) -> list[dict]:
        out = []
        for h in hyps[:3]:
            if not isinstance(h, dict):
                continue
            if h.get("kind") == "temporary_water_use" and h.get("where") and h.get("start") and h.get("end"):
                node = tb.resolve_node(h["where"])
                if node is None or f"event@{node}" in tb.board.alts:
                    continue
                r = tb.call("test_event", {"where": h["where"], "start": h["start"], "end": h["end"],
                                           "basis": h.get("basis", "guess")})
            elif h.get("kind") == "sensor_fault" and h.get("sensor"):
                if f"sensor_fault:{h['sensor']}" in tb.board.alts:
                    continue
                r = tb.call("test_sensor_fault", {"sensor": h["sensor"], "basis": h.get("basis", "guess")})
            else:
                continue
            out.append({"hypothesis": h, "result": _jsonable(r)})
        return out

    # ------------------------------------------------------------------ the critic (second look before a dig)
    @staticmethod
    def gate_open(tb: Toolbox) -> bool:
        return tb.p_success() >= DIG_THRESHOLD and tb.alt_max()[1] <= ALT_LIMIT

    def critic(self, tb: Toolbox) -> dict:
        """Grounded veto: a 'hold' only counts if it names a cited, untested alternative that code can test.
        Code then tests it; the dig is held only if the alternative now blocks the gate."""
        d = self.llm.json(CRITIC_SYSTEM, render(tb) + "\nThe investigator asks to dig now. Your verdict?",
                          CRITIC_SCHEMA, tag="critic")
        out = {"verdict": d.get("verdict", "go"), "concern": str(d.get("concern", ""))[:200], "held": False}
        h = d.get("hypothesis")
        if out["verdict"] == "hold" and isinstance(h, dict):
            basis_ok = tb.grounded(h.get("basis", "")) != "guess"
            out["grounded"] = basis_ok
            if basis_ok:
                out["tests"] = self.test_hypotheses(tb, [h])
                out["held"] = not self.gate_open(tb)
        return out


def dig_with_sop(tb: Toolbox, reason: str) -> dict:
    """Propose the dig (the toolbox does the SOP steps and informs the reporters)."""
    return _jsonable(tb.call("propose_dig", {"schedule": "now", "reason": reason}))


def close_safely(tb: Toolbox) -> dict:
    """Code fallback when the brain stops: dig only if the gate passes (doing the SOP steps first), else abstain."""
    if LLMInvestigator.gate_open(tb) and tb.digs < 2:
        r = dig_with_sop(tb, "fallback: gate passed")
        if r.get("leak_found"):
            return {"action": "fallback_dig", "result": r}
    tb.call("abstain", {"summary": "handing over: evidence not strong enough for a dig"})
    return {"action": "fallback_abstain"}


def _hours(tb) -> float:
    return round((tb.env.now - tb.case.case_open_s) / 3600, 2)


def _round(d: dict) -> dict:
    return {k: round(float(v), 3) for k, v in d.items()}


def _delta(u0: dict, llm) -> dict:
    u1 = llm.usage.as_dict()
    return {k: round(u1[k] - u0[k], 2) for k in ("calls", "cached", "prompt_tokens", "output_tokens", "seconds")}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x
