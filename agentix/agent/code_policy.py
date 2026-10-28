"""The code autopilot, and AgentiX with every LLM switched off (ablation 'agentix_code').

Since ADR-014 the autopilot contains NO language at all: no question templates, no phrase tests on records, no pattern readers for documents.
Asking a person a good question, reading a work order or a permit, and judging whether a document explains a complaint are language jobs; in a real
control room they are done by a person, so without an LLM agent nothing is asked and no document is read. What is left is the routine that needs no
language:
1. Follow the planner's best field check until the code gate opens; then dig.
2. Close as 'no leak' when the leak is unlikely, else hand over.
`agentix_code` is this autopilot with a RULE-based dispatcher (no look-ahead planner); `agentix_code_planner` keeps the planner (AI removed, tool kept): it uses the messages the reader extracted as evidence (the citizen reader is held constant across all methods)
but never calls anyone back and never reads a document. The crew (crew.py) runs the same autopilot's field routine under LLM agents that do the language work.
(The earlier version with question templates is kept as an inert text file in research/archive/ only so that the v2-v4 numbers in the ADRs can be traced.)
"""
from __future__ import annotations


class Autopilot:
    """Routine steps of an investigation, one action at a time. Holds no LLM and no language."""

    def __init__(self):
        self.refused = False

    def step(self, tb, prefer: str | None = None) -> str:
        """-> 'done' | 'dig' | 'check' | 'stop' (nothing left that is worth doing)."""
        if tb.done or tb.env.done:
            return "done"
        if tb.gate()[0] and not self.refused:
            self.refused = not tb.propose_dig("gate open").get("ok")
            return "dig"
        self.refused = False
        if getattr(tb.board, "cand", None) is not None:      # ADR-016: no investigator -> the fixed fallback decides where to look
            from ..search.candidates import heuristic
            tb.search.add_candidates(tb.board, heuristic(tb), "heuristic")
            if tb.gate()[0] and not self.refused:
                self.refused = not tb.propose_dig("gate open").get("ok")
                return "dig"
        if not tb.use_planner:                               # rule-based dispatcher: no look-ahead values
            if tb.exhausted() or tb.rule_option() is None:
                return "stop"
            tb.field_check("F1")
            return "check"
        opts = tb.plan()
        if not opts or opts[0].value <= 0.01:
            return "stop"
        tb.field_check(prefer if prefer in tb.options else "F1")
        return "check"

    def finish(self, tb) -> None:
        if not (tb.done or tb.env.done):
            b = tb.board
            j, ps = b.best_dig()
            has_hit = any(c["kind"] == "listen_hit" for c in b.checks)
            has_part = bool(b.part_support(j))
            no_leak = b.p_leak < 0.35 or (ps < 0.10 and not has_hit and not has_part)
            tb.close("no_leak" if no_leak else "hand_over", "investigation finished", force=True)


class CodePolicy:
    name = "agentix_code"

    def __init__(self, max_rounds: int = 12):
        self.max_rounds = max_rounds

    def run(self, tb) -> list[dict]:
        ap = Autopilot()
        for _ in range(self.max_rounds):
            if ap.step(tb) in ("done", "stop"):
                break
        ap.finish(tb)
        return tb.trace
