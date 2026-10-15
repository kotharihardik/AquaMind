"""The matched NON-LLM brain: fixed rules on the very same toolbox as the LLM agent.

It is a strong, honest opponent. It reads the case notes with keyword rules and a time-range pattern, tests the
explanations it finds, waits and rechecks, lets the toolbox do the SOP steps, and otherwise takes the menu item
with the best expected gain per crew-unit (the Phase 2 greedy VOI policy).
Where it loses to the LLM agent, that is the value of the LLM. Where it wins, the LLM was not needed.
The keyword rules were written while looking at the DEV documents only.

`keyword_scan` (the routine reading) is shared: the LLM agent runs it too, so the LLM starts from everything the
rules can find and adds its own reading on top.
"""
from __future__ import annotations

import datetime as dt
import re

from ..reader.cues import find_cues, safe_time
from .llm_agent import close_safely, dig_with_sop
from .timeparse import find_range
from .toolbox import ALT_LIMIT, DIG_THRESHOLD, Toolbox

EVENT_WORDS = r"flush|hydrant|valve|shutdown|isolate|fire brigade"
FAULT_WORDS = r"fault|unreliable|offset|drift|do not use|flooded|damaged|too low"


class RuleInvestigator:
    name = "rule_agent"

    def __init__(self, max_steps: int = 12, min_gain_per_cu: float = 0.05, read_docs: bool = True):
        self.max_steps, self.min_rate, self.read_docs = max_steps, min_gain_per_cu, read_docs

    def run(self, tb: Toolbox) -> list[dict]:
        trace = keyword_scan(tb, self.read_docs)
        rechecked = set()
        for _ in range(self.max_steps):
            if tb.done or tb.env.done:
                break
            alt, pa = tb.alt_max()
            a = tb.board.alts.get(alt, {}) if alt else {}
            if pa > ALT_LIMIT and a.get("kind") == "sensor_fault" and pa > 0.5:
                trace.append(_do(tb, "exclude_sensor", {"sensor": a["sensor"]}))
                continue
            ev = event_to_check(tb, rechecked)
            if ev:                                   # an event stops, a leak does not: wait, then look again
                wait_h = (tb.board.alts[ev]["end_s"] + 2.1 * 3600 - tb.env.now) / 3600
                if wait_h > 0:
                    trace.append(_do(tb, "wait", {"hours": round(min(12.0, wait_h), 2)}))
                trace.append(_do(tb, "recheck", {}))
                rechecked.add(ev)
                continue
            if tb.p_success() >= DIG_THRESHOLD and pa <= ALT_LIMIT and tb.digs < 2:
                r = dig_with_sop(tb, "gate passed")
                trace.append({"action": {"tool": "propose_dig"}, "result": r})
                if not r.get("dug_at"):          # operator or gate said no: stop and hand over
                    break
                continue
            opts = tb.menu()
            if not opts or opts[0].rate() < self.min_rate:
                break
            trace.append(_do(tb, "do", {"option": "M1"}))
            if not trace[-1]["result"].get("ok"):
                break
        if not (tb.done or tb.env.done):
            trace.append({"step": "close", **close_safely(tb)})
        return trace


def event_to_check(tb: Toolbox, done: set, min_p: float = 0.05) -> str | None:
    """A tested event that still matters and can be rechecked before the case deadline."""
    ex = tb.board.explanations()
    for k, a in tb.board.alts.items():
        if a.get("kind") == "event" and k not in done and ex.get(k, 0) >= min_p \
                and a["end_s"] + 2.1 * 3600 <= tb.env.horizon:
            return k
    return None


# ---------------------------------------------------------------------- the keyword scan (shared with the LLM agent)
def keyword_scan(tb: Toolbox, read_docs: bool = True) -> list[dict]:
    """Routine reading done by code: keyword rules over the case notes and report cues, plus the one-sensor
    alert. Every explanation found is TESTED (physics + Bayes) and appears in the board's hypothesis ledger."""
    trace = read_documents(tb) if read_docs else []
    trace += read_report_cues(tb)
    out = tb.sensor_outlier()                      # SOP-10: one sensor standing out may be a fault
    if out and f"sensor_fault:{out[0]}" not in tb.inv.state["alts_tested"]:
        trace.append(_do(tb, "test_sensor_fault", {"sensor": out[0], "basis": "E-P"}))
    return trace


def read_documents(tb: Toolbox) -> list[dict]:
    """Work order / maintenance lines with event or fault keywords -> test that explanation."""
    out = []
    for did, hit in list(tb.notes.items()):
        text = hit["text"]
        low = text.lower()
        if re.search(EVENT_WORDS, low) and not re.search(r"left open|no shutdown|not hydraulic", low):
            rng = find_range(text)
            names = [s for s, _, _, _ in tb.inv.gaz.find_mentions(text)]
            place = ", ".join(names[:2]) if names else None
            if rng and place:
                out.append(_do(tb, "test_event", {"where": place, "start": rng[0], "end": rng[1], "basis": did}))
        if re.search(FAULT_WORDS, low):
            for s in tb.case.pressure_nodes:
                if re.search(rf"\b{s}\b", low):
                    out.append(_do(tb, "test_sensor_fault", {"sensor": s, "basis": did}))
    return out


def read_report_cues(tb: Toolbox) -> list[dict]:
    """Fire brigade / tanker / hydrant in a report -> test a temporary use there. The end is unknown, so we
    assume up to 6 h after it started (SOP-9)."""
    out = []
    for rid, o in tb.inv.state["reader_outs"].items():
        cues = set(find_cues(o.get("summary_en", "")))
        if not cues & {"fire brigade", "tanker filling", "hydrant", "flushing"}:
            continue
        place = next((m["text"] for m in o["location"]["mentions"] if tb.resolve_node(m["text"])), None)
        if place is None:
            continue
        got = dt.datetime.fromisoformat(tb.inv.state["received"][rid])
        since = safe_time(o.get("time", {}).get("onset_value"))
        t0 = got - dt.timedelta(hours=1)
        if since and ":" in since:
            h, m = map(int, since.split(":"))
            t0 = got.replace(hour=h, minute=m)
            t0 = t0 - dt.timedelta(days=1) if t0 > got else t0
        start, end = t0.strftime("%d/%m %H:%M"), (t0 + dt.timedelta(hours=6)).strftime("%d/%m %H:%M")
        out.append(_do(tb, "test_event", {"where": place, "start": start, "end": end, "basis": rid}))
    return out


def _do(tb: Toolbox, tool: str, args: dict) -> dict:
    r = tb.call(tool, args)
    return {"action": {"tool": tool, "args": args}, "result": {k: v for k, v in r.items() if k != "hits"},
            "p_after": round(tb.p_success(), 3)}
