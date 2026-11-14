"""Rebuild an investigation timeline from the evaluation logs (any of the 210 cases, any logged method).

The logs keep, per case, the tool calls in order (with P(leak) and P(success) after each) and the agent turns in order, but not the leak map
itself and not the order BETWEEN turns and tool calls. This module puts the two lists back together from how the crew loop works
(aquamind/agent/crew.py) and produces the same event format as a recording (demo_ui/instrument.py) with fidelity 'log': no heat map, but every
thought, question, hypothesis region, field result and dig. Messages are the benchmark's own texts, read again by the same fine-tuned reader
(from its cache), so pins land where the agents' code placed them.
"""
from __future__ import annotations

import json
import re
import threading

import numpy as np

from .config import BENCH, LOGS, suite_of

# method -> log file prefix relative to logs/evaluation; {v} = the NEWEST version that exists (the final runs; v1 = the mid-report runs,
# which stay replayable with version=1). Runs logged since the final evaluation also carry the step order ("seq"), leak-map snapshots ("heat")
# and the citizens' own words, so their replays are as complete as a recording.
SOURCES = {
    "agentix_crew": "agentix/crew/{k}/v{v}_{k}_agentix_crew",
    "react_agent": "baselines/react_agent/k5/v{v}_k5_react_agent",
    "crew_practice": "baselines/crew_practice/{k}/v{v}_{k}_crew_practice",
    "no_gate": "ablations/no_gate/k5/v{v}_k5_ablation_no_gate",
    "one_shot": "ablations/one_shot/k5/v{v}_k5_ablation_one_shot",
}
METHOD_NAMES = {"agentix_crew": "AquaMind crew", "react_agent": "ReAct agent", "crew_practice": "Crews' current practice",
                "no_gate": "Crew without the safety gate", "one_shot": "Crew, one shot"}
FORENSIC_TOOLS = ("use_record", "use_repair_record", "triangulate", "call_back")
POST_CHECK = ("a physical check", "leak noise", "boundary valve", "the last check", "listening check was quiet", "step test showed",
              "a field check was quiet", "the Skeptic challenged the case after")

_lock = threading.Lock()
_index: dict[str, dict] = {}


def _versions(method: str, layout: str) -> tuple[str | None, list[int]]:
    """Args: method = method name; layout = e.g. 'spread_k5'.
    Returns: (log path pattern with '{v}', sorted version numbers that exist), or (None, []) if the method is unknown.
    """
    k = layout.replace("spread_", "")
    pat = SOURCES.get(method)
    if not pat:
        return None, []
    pat = pat.replace("{k}", k)
    folder, name = pat.rsplit("/", 1)
    rx = re.compile("^" + re.escape(name).replace(re.escape("{v}"), r"(\d+)") + r"\.jsonl$")
    return pat, sorted(int(m.group(1)) for f in (LOGS / folder).glob("v*.jsonl") if (m := rx.match(f.name)))


def _base(method: str, layout: str, version: int | None = None) -> str | None:
    """Args: method = method name; layout = e.g. 'spread_k5'; version = log version (None = the newest; 1 = the mid-report run).
    Returns: log path prefix of that method's run, or None if unknown (or that version does not exist).
    """
    pat, vs = _versions(method, layout)
    if not pat:
        return None
    if version is not None:
        return pat.replace("{v}", str(int(version))) if int(version) in vs else None
    return pat.replace("{v}", str(vs[-1] if vs else 1))



def _load_index(base: str) -> dict:
    """{case: {"row": result row, "trace": [entries]}} for one logged run (loaded once)."""
    with _lock:
        if base in _index:
            return _index[base]
        out: dict[str, dict] = {}
        rp, tp = LOGS / f"{base}.jsonl", LOGS / f"{base}_trace.jsonl"
        if rp.exists():
            for line in open(rp):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                r.pop("trace", None)
                out.setdefault(r["case"], {"row": r, "trace": []})["row"] = r
        if tp.exists():
            for line in open(tp):
                try:
                    t = json.loads(line)
                except ValueError:
                    continue
                out.setdefault(t["case"], {"row": None, "trace": []})["trace"].append(t)
        _index[base] = out
        return out


def available(method: str, layout: str = "spread_k5") -> list[str]:
    """Args: method = method name; layout = sensor layout.
    Returns: sorted case names that have a log entry.
    """
    base = _base(method, layout)
    return sorted(_load_index(base)) if base else []


def outcomes(method: str, layout: str = "spread_k5") -> dict[str, dict]:
    """Args: method = method name; layout = sensor layout.
    Returns: {case: {'outcome', 'found', 'err', ...}} from the logs.
    """
    base = _base(method, layout)
    if not base:
        return {}
    idx = _load_index(base)
    return {c: {"outcome": (d["row"] or {}).get("outcome"), "found": (d["row"] or {}).get("found"), "err": (d["row"] or {}).get("guess_error_m"),
                "digs": (d["row"] or {}).get("n_digs"), "unsafe": (d["row"] or {}).get("unsafe")} for c, d in idx.items()}


# ---------------------------------------------------------------- helpers
_NODE = re.compile(r"\((n\d+)\)")
_SEC = re.compile(r"\bS\d{2}(?:\.\d+)?\b")


def _node_in(text, net) -> int | None:
    """Find a junction id mentioned in a text, e.g. '(n123)'.

    Args: text = any text; net = Network.
    Returns: junction index or None.
    """
    m = _NODE.findall(str(text or ""))
    for n in reversed(m):
        if n in net.index:
            return int(net.index[n])
    m = re.findall(r"\bn\d+\b", str(text or ""))
    for n in m:
        if n in net.index:
            return int(net.index[n])
    return None


def _messages(case_name: str, open_s: float):
    """All messages of the case (realistic texts), with their arrival time in hours after the opening."""
    hid = json.load(open(f"{BENCH}/hidden/{case_name}.json"))
    try:
        alt = json.load(open(f"{BENCH}/hidden/{case_name}.realistic.json"))
    except (OSError, ValueError):
        alt = {}
    out = []
    for r in hid["reports"]:
        out.append({"report_id": r["report_id"], "received_at": r["received_at"], "channel": r["channel"],
                    "text": alt.get(r["report_id"], r["text"]), "t_h": (r["received_s"] - open_s) / 3600})
    return hid, out


def _merge(tools: list[dict], turns: list[dict]) -> list[tuple[str, dict]]:
    """Interleave turns and tool calls the way the crew loop produced them (exact when the log has step numbers)."""
    if tools and turns and all("seq" in t for t in tools + turns):
        both = [("tool", t) for t in tools] + [("turn", t) for t in turns]
        return sorted(both, key=lambda x: x[1]["seq"])
    seq: list[tuple[str, dict]] = []
    p = 0

    def flush(q):
        """Append tool events up to position q to the merged sequence.

        Args: q = position in the tool list.
        """
        nonlocal p
        while p < min(q, len(tools)):
            seq.append(("tool", tools[p]))
            p += 1

    def next_of(names, start=None):
        """Args: names = tool names; start = search start (None = current position).
        Returns: position of the next tool call with one of these names, or None.
        """
        for q in range(p if start is None else start, len(tools)):
            if tools[q].get("tool") in names:
                return q
        return None

    for t in turns:
        role, why = t.get("role"), str(t.get("why") or "")
        if role == "forensic":
            block = []
            n_fm = len(t.get("failing_mains") or [])
            want = (["use_repair_record"] * n_fm) + (["triangulate"] if t.get("triangulate") else []) + (["call_back"] if t.get("questions") else [])
            if any(why.startswith(x) or x in why for x in POST_CHECK):
                q = next_of(("field_check",))
                f = next_of(FORENSIC_TOOLS)
                if q is not None and (f is None or q < f):
                    flush(q + 1)
            if want:
                # the block starts at the first forensic tool from here (use_record calls of the same turn come first)
                q = next_of(FORENSIC_TOOLS)
                if q is not None:
                    flush(q)
                    while p < len(tools) and tools[p].get("tool") == "use_record":
                        block.append(tools[p]); p += 1
                    for name in want:
                        if p < len(tools) and tools[p].get("tool") == name:
                            block.append(tools[p]); p += 1
            seq.append(("turn", t))
            seq += [("tool", x) for x in block]
        elif role == "skeptic":
            if any(x in why for x in POST_CHECK):
                q = next_of(("field_check",))
                f = next_of(FORENSIC_TOOLS)
                if q is not None and (f is None or q < f):
                    flush(q + 1)
            seq.append(("turn", t))
        elif role == "strategist":
            q = next_of(("field_check",))
            if q is not None:
                flush(q)
            seq.append(("turn", t))
            if q is not None:
                flush(q + 1)
        elif role == "auditor":
            q = next_of(("propose_dig", "field_check", "close"))
            if q is not None:
                flush(q)
            seq.append(("turn", t))
            if q is not None:
                flush(q + 1)
        else:
            seq.append(("turn", t))
    flush(len(tools))
    return seq


def build(method: str, case_name: str, layout: str = "spread_k5", version: int | None = None) -> dict | None:
    """Rebuild a replay (event stream) of a logged run from its trace, for the UI.

    Args: method = 'agentix_crew' | 'react_agent' | ...; case_name = case; layout = sensor layout;
          version = log version (None = the newest = the final run; 1 = the mid-report run).
    Returns: recording-like dict {'meta', 'events', ...}, or None if not in the logs.
    """
    from .instrument import truth_of
    from .readers import DemoReader
    from .world import search, world
    base = _base(method, layout, version)
    if not base:
        return None
    all_versions = _versions(method, layout)[1]
    ver = int(version) if version is not None else (all_versions[-1] if all_versions else 1)
    entry = _load_index(base).get(case_name)
    if not entry:
        return None
    W = world()
    net = W.town.net
    case, env = W.open(case_name, layout)
    hid, msgs = _messages(case_name, case.case_open_s)
    reader = DemoReader(W.town.places, prefer="finetuned", allow_model=False)
    rep = search().rep
    reads = {}
    for m in msgs:
        reads[m["report_id"]] = reader.read(m)
    anchors = {}
    for rid, rd in reads.items():
        w = rep.place_weights(rd.get("places") or [])
        anchors[rid] = int(np.argmax(w)) if w is not None and float(w.sum()) > 0 else None

    from aquamind.search.pressure import build_evidence
    ev0 = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s)
    row = entry["row"] or {}
    tools = [t for t in entry["trace"] if t.get("kind") == "tool_call"]
    turns = [t for t in entry["trace"] if t.get("kind") == "llm_turn"]
    is_react = method == "react_agent"
    seq = _merge(tools, turns) if (turns and not is_react) else [("tool", t) for t in tools]

    meta = {"case": case_name, "public_id": case.case_id, "suite": suite_of(case_name), "layout": layout, "k": len(case.pressure_nodes),
            "sensors": [int(net.index[n]) for n in case.pressure_nodes], "sensor_ids": list(case.pressure_nodes), "trigger": case.trigger,
            "trigger_report": case.trigger_report, "opened_at": "", "horizon_h": round((case.horizon_s - case.case_open_s) / 3600, 1),
            "budget": {"checks": 5, "digs": 2}, "method": method, "method_name": METHOD_NAMES.get(method, method),
            "reader": row.get("reader"), "fidelity": "log", "source": f"logs/evaluation/{base}_trace.jsonl",
            "map": any(t.get("heat") for t in tools), "ordered": bool(tools) and all("seq" in t for t in tools),
            "version": ver, "stage": "final" if (all_versions and ver == all_versions[-1] and len(all_versions) > 1) else "mid" if ver == 1 else "earlier"}
    import datetime as dt
    meta["opened_at"] = (dt.datetime.fromisoformat(case.start_datetime) + dt.timedelta(seconds=case.case_open_s)).strftime("%a %d %b %Y, %H:%M")

    # running state
    st = {"p_leak": None, "p_success": 0.0, "t_h": None if method == "react_agent" else 0.0, "checks_left": 5, "digs": 0, "done": False, "checks": [], "dry": [], "msgs": {},
          "slate": [], "cand": [], "best": None}
    shown: set[str] = set()
    events: list[dict] = [{"i": 0, "kind": "meta", "meta": meta}]
    lit: dict[str, bool] = {}

    def msg_payload(m):
        """Args: m = message dict.
        Returns: the message as the UI shows it.
        """
        rid = m["report_id"]
        return {"rid": rid, "at": m["received_at"], "ch": m["channel"], "text": m["text"], "read": reads.get(rid), "prov": reader.provenance.get(rid),
                "agent_view": None, "trigger": rid == case.trigger_report, "j": anchors.get(rid)}

    def state():
        """Returns: a copy of the current replay state (one snapshot per event)."""
        s = {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)) for k, v in st.items() if k != "slate"}
        s["msgs"] = {rid: dict(v) for rid, v in st["msgs"].items()}
        return s

    def emit(kind, **d):
        """Add one event to the replay.

        Args: kind = event type; **d = event fields.
        """
        events.append({"i": len(events), "kind": kind, **d})

    def reveal(t_h):
        """Show the messages that arrived by this case time.

        Args: t_h = hours since case opening.
        """
        new = [m for m in msgs if m["report_id"] not in shown and m["t_h"] <= t_h + 1e-6]
        for m in new:
            shown.add(m["report_id"])
            rd = reads.get(m["report_id"]) or {}
            st["msgs"][m["report_id"]] = {"eps": None, "src": rd.get("source"), "usable": rd.get("category") in ("leak", "quality") and not rd.get("contains_instructions"),
                                          "calls": 0, "j": anchors.get(m["report_id"]), "up": None, "notes": []}
        return new

    first = reveal(0.0)
    sensors = [{"node": n, "j": int(net.index[n]), "drop_m": round(float(d), 3), "z": round(float(z), 2)} for n, d, z in zip(case.pressure_nodes, ev0.drop_m, ev0.z)]
    checks_used = 0
    pending: list = []                     # the Forensic's 'turn_done' waits until its own tool calls (records, calls) are shown

    def turn_done(t, rec):
        """Emit the end of a turn: its hypotheses (slate) and challenges, with the state after it.

        Args: t = logged turn; rec = its decision fields."""
        slate = rec.get("slate")
        if slate:
            st["slate"] = slate
            cand = set(st["cand"])
            for h in slate:
                cand |= set(h.get("region") or [])
            st["cand"] = sorted(cand)
        emit("turn_done", role=t.get("role"), record=rec, slate=slate, challenges=rec.get("challenges"), rejected=None, state=state())

    for kind, t in seq:
        if pending and (kind == "turn" or t.get("tool") not in ("use_record", "use_repair_record", "triangulate", "call_back")):
            turn_done(*pending.pop())
        if kind == "turn":
            rec = {k: v for k, v in t.items() if k not in ("case_no", "case", "layout", "step", "kind", "thought", "seq")}
            if is_react:
                continue
            emit("turn", role=t.get("role"), why=t.get("why"), thought=t.get("thought"), decision=rec, error=t.get("error"), cached=None,
                 board_text=None, options=None, turn=t.get("turn"))
            if t.get("role") == "forensic":
                pending.append((t, rec))
            else:
                turn_done(t, rec)
            continue
        tool = t.get("tool")
        res, args = t.get("result") or {}, t.get("args") or {}
        heat = None
        if t.get("heat"):                                   # leak-map snapshot: the 40 most likely junctions -> map scaled 0..1000
            dense = np.zeros(len(net.junctions))
            for j, w in t["heat"]:
                dense[int(j)] = float(w)
            mx = float(dense.max())
            if mx > 0:
                heat = np.round(1000 * dense / mx).astype(int).tolist()
                st["best"] = int(t["heat"][0][0])
        if t.get("p_leak") is not None:
            st["p_leak"], st["p_success"] = t["p_leak"], t.get("p_success", st["p_success"])
        if t.get("t_case_h") is not None:
            st["t_h"] = t["t_case_h"]
        newm = reveal(st["t_h"]) if (tool != "open" and st["t_h"] is not None) else []
        if newm:
            emit("messages", messages=[msg_payload(m) for m in newm], state=state())
        if is_react and t.get("thought"):
            emit("turn", role="react", why=f"step {t.get('step')}", thought=t.get("thought"), decision={"tool": tool, "args": args}, error=None,
                 cached=None, board_text=None, options=None, turn=t.get("step"))
        ev = {"tool": tool, "args": args, "result": res, "secs": t.get("secs"), "env": []}
        if tool == "open":
            ev.update(messages=[msg_payload(m) for m in first], sensors=sensors, inflow={"q_hat": round(float(ev0.q_hat), 2), "q_sd": round(float(ev0.q_sd), 2)},
                      scada=None, works=hid.get("works") or [])
            if st["p_leak"] is None:
                st["p_leak"] = t.get("p_leak")
        elif tool == "call_back":
            replies = []
            for c, r in zip(args.get("calls") or [], res.get("results") or []):
                rid = str(c.get("report_id"))
                learned = r.get("learned") or {}
                replies.append({"rid": rid, "q": r.get("question") or c.get("question"), "text": r.get("reply_text"),
                                "parsed": {"answered": r.get("answered"), **learned}})
                m = st["msgs"].get(rid)
                if m is not None and r.get("answered"):
                    m["calls"] += 1
                    if learned.get("not_network"):
                        m["eps"], m["usable"] = 0.98, True
                        m["notes"] = (m["notes"] + [f"caller describes a non-utility cause: {learned['not_network']}"])[-3:]
                    elif learned.get("still_flowing") is False:
                        m["notes"] = (m["notes"] + ["water has stopped (network leaks do not stop by themselves)"])[-3:]
                    elif learned.get("source") == "here":
                        m["notes"] = (m["notes"] + ["water comes out right there"])[-3:]
            ev["replies"] = replies
        elif tool == "call" and is_react:
            rid = str(args.get("report_id"))
            ev["replies"] = [{"rid": rid, "q": args.get("question"), "text": res.get("reply"), "parsed": None}]
        elif tool == "use_record" and res.get("ok"):
            m = st["msgs"].get(str(args.get("report_id")))
            if m is not None:
                m["eps"] = max(m["eps"] or 0, 0.85)
                m["notes"] = (m["notes"] + [f"place has a history of non-utility false alarms ({args.get('record_id')})"])[-3:]
        elif tool == "triangulate" and res.get("ok"):
            for rid in re.findall(r"R-\d+", res.get("msg", "").split(" placed")[0]):
                m = st["msgs"].get(rid)
                if m is not None:
                    m["src"], m["up"] = "elsewhere", anchors.get(str(args.get("origin")))
                    m["notes"] = (m["notes"] + [f"triangulation: water runs down from {args.get('origin')}'s place"])[-3:]
        elif tool in ("field_check", "step_test", "listen", "place_logger"):
            if res.get("ok"):
                checks_used += 1
            label = args.get("label") or ""
            kindr = res.get("kind") or ("step test" if tool == "step_test" else "listening" if tool == "listen" else tool)
            if kindr == "step test" and res.get("ok"):
                secs = list((res.get("excess_flow_lps") or {}).keys()) or [s["section"] for s in res.get("steps") or []]
                lit_now = set(res.get("clearly_lit") or [])
                if tool == "step_test":
                    for s in res.get("steps") or []:
                        ex = s.get("inflow_drop_lps", 0) - s.get("expected_customer_use_lps", 0)
                        if not s.get("failed") and ex > 1.0:
                            lit_now.add(s["section"])
                for sid in secs:
                    st["checks"].append({"label": f"step:{sid}", "kind": "step", "sid": sid, "lit": sid in lit_now})
                    lit[sid] = sid in lit_now
                ev["option"] = {"kind": "step", "label": label, "sections": secs, "center": None}
            elif kindr == "listening" and res.get("ok"):
                c = _node_in(label, net) if label else (_node_in(args.get("place"), net) if args.get("place") else None)
                if c is None and args.get("place"):
                    from aquamind.baselines.react_agent import resolve_place
                    c = resolve_place(W, args.get("place"))
                at = _node_in(res.get("pinpointed"), net) if res.get("heard") else None
                e = {"label": label or f"listen@{args.get('place')}", "kind": "listen_hit" if res.get("heard") else "listen_quiet"}
                if c is not None:
                    e["center"] = c
                if at is not None:
                    e["at"] = at
                st["checks"].append(e)
                ev["option"] = {"kind": "listen", "label": label or f"listen around {args.get('place')}", "center": c, "sections": None}
            st["checks_left"] = max(0, 5 - checks_used)
        elif tool in ("propose_dig", "dig"):
            if res.get("ok"):
                j = _node_in(res.get("spot"), net) if tool == "propose_dig" else None
                if j is None and tool == "dig":
                    from aquamind.baselines.react_agent import resolve_place
                    j = resolve_place(W, args.get("place"))
                ev["dig_j"] = j
                st["digs"] += 1
                if not res.get("leak_found") and j is not None:
                    st["dry"].append(j)
                    st["checks"].append({"label": f"dry dig@{j}", "kind": "dry_dig", "at": j})
                if res.get("leak_found"):
                    st["done"] = True
                    st["best"] = j
        elif tool == "close" and res.get("ok"):
            st["done"] = True
        if tool == "test_places" and res.get("best_spots"):
            js = [_node_in(s, net) for s in res["best_spots"]]
            ev["spots"] = [j for j in js if j is not None]
        s = state()
        if heat is not None:
            s["heat"] = heat
        emit("tool", **ev, state=s)

    if pending:
        turn_done(*pending.pop())
    result = {k: row.get(k) for k in ("has_leak", "found", "n_digs", "wrong_digs", "outcome", "checks_used", "hours_open", "water_lost_m3",
                                       "unsafe", "violations", "audit", "guess_error_m", "tool_calls", "tool_errors", "llm_calls", "llm_cached",
                                       "tokens", "steps", "wall_s", "key_action", "pred_spot")}
    guess = row.get("pred_spot")
    result["final_j"] = _node_in(guess, net) if guess else None
    emit("end", result=result, truth=truth_of(env, W), state=state())
    return {"meta": meta, "result": result, "events": events}
