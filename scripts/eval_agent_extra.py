"""Extra agent metrics for the AquaMind crew, computed from the evaluation logs (no LLM, no API calls).

The main metrics (aquamind/eval/metrics.py) judge the OUTCOME of a case. These judge HOW the agents worked, role by role:
  1. Hypothesis quality (Forensic Advocate): was the true leak inside a network-leak hypothesis - at the first look, at any time, at the end?
  2. Grounding: share of the evidence cited by hypotheses and by Skeptic challenges that the code checks accept.
  3. Decision quality (Strategist): how often code had to replace a pick that was far worse than the best option, or the pick was invalid.
  4. Safety review (Auditor): verdicts, and how often it objected in cases that ended with a dry dig versus cases with no dry dig.
  5. Interrogation: call-backs per case, share answered, share that taught the agents a fact (source, not-network, still flowing).
  6. Efficiency: field checks per found leak, LLM calls (incl. cached answers) per solved case, first-dig success, hours a found leak\n     ran, share of tool calls refused by code.
  7. Effort per role: agent turns per case.
Mid (v1) and final (v2) crew runs are compared on the same 210 cases for each sensor layout.

usage: .venv/bin/python -I scripts/eval_agent_extra.py [--bench data/benchmark/agentix_master] [--out logs/evaluation/extra]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())


def load_truth(bench: str) -> dict:
    """Args: bench = benchmark folder.
    Returns: {case name: hidden ground-truth dict}.
    """
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{bench}/hidden/*-*.json")}


def mean(x) -> float | None:
    """Args: x = list of numbers.
    Returns: mean rounded to 3 decimals, or None if empty.
    """
    return round(float(np.mean(x)), 3) if len(x) else None


def run_metrics(rows: list[dict], trace_path: str, truth: dict, node_index: dict) -> dict:
    """All process metrics of one crew run.

    Args: rows = result rows (one per case); trace_path = its trace jsonl; truth = load_truth(); node_index = {junction id: index}.
    Returns: metrics dict.
    """
    res = {r["case"]: r for r in rows}
    per = defaultdict(lambda: defaultdict(list))
    for line in open(trace_path):
        s = json.loads(line)
        if s.get("case") in res:
            per[s["case"]][s.get("kind") or "?"].append(s)
    hyp_first, hyp_any, hyp_last = [], [], []
    g_ok = g_bad = c_ok = c_bad = 0
    strat = Counter()
    aud = Counter()
    obj_dry, obj_clean = [], []
    cb = Counter()
    turns = Counter()
    for case, k in per.items():
        r, t = res[case], truth.get(case, {})
        tl = [s for s in k.get("llm_turn", [])]
        for s in tl:
            turns[s["role"]] += 1
        # 1. hypotheses vs the true leak
        if r["has_leak"] and t.get("leak"):
            lj = node_index.get(t["leak"]["nearest_node"])
            sec = t["leak"].get("section", "")
            slates = [s["slate"] for s in tl if s["role"] == "forensic" and s.get("slate")]

            def covers(sl):
                """Args: sl = one slate (list of hypothesis dicts). Returns: True if a network-leak hypothesis holds the leak junction."""
                for h in sl:
                    if not isinstance(h, dict) or h.get("kind") != "network_leak":
                        continue
                    if h.get("region") is not None:
                        if lj in set(h["region"]):
                            return True
                    elif sec and str(h.get("where", "")).upper().startswith(sec):
                        return True
                return False
            if slates:
                hyp_first.append(covers(slates[0]))
                hyp_any.append(any(covers(sl) for sl in slates))
                hyp_last.append(covers(slates[-1]))
        # 2. grounding (unique hypotheses per turn; Skeptic challenges)
        for s in tl:
            for h in s.get("slate") or []:
                if isinstance(h, dict):
                    g_ok += int(h.get("cited_ok") or 0)
                    g_bad += int(h.get("cited_bad") or 0)
            for ch in s.get("challenges") or []:
                if isinstance(ch, dict):
                    c_ok += int(ch.get("cited_ok") or 0)
                    c_bad += int(ch.get("cited_bad") or 0)
        # 3. Strategist
        for s in tl:
            if s["role"] == "strategist":
                strat["turns"] += 1
                strat["overridden"] += bool(s.get("override"))
                strat["invalid"] += s.get("note") == "no valid choice"
                strat["error"] += bool(s.get("error"))
        # 4. Auditor
        verdicts = [str(s.get("verdict")).lower() for s in tl if s["role"] == "auditor"]
        aud.update(verdicts)
        if verdicts:
            objected = any(v in ("object", "hold") for v in verdicts)
            (obj_dry if r.get("wrong_digs", 0) > 0 else obj_clean).append(objected)
        # 5. call-backs
        for s in k.get("tool_call", []):
            if s.get("tool") == "call_back":
                for x in (s.get("result") or {}).get("results") or []:
                    cb["asked"] += 1
                    cb["error"] += bool(x.get("error"))
                    cb["answered"] += bool(x.get("answered"))
                    cb["learned"] += bool(x.get("learned"))
    solved = [r for r in rows if (r["found"] if r["has_leak"] else r["n_digs"] == 0)]
    found = [r for r in rows if r["has_leak"] and r["found"]]
    n = max(len(per), 1)
    return {
        "cases": len(rows),
        "hypothesis_recall_first_look": mean(hyp_first), "hypothesis_recall_any_time": mean(hyp_any), "hypothesis_recall_final": mean(hyp_last),
        "hypothesis_grounding": round(g_ok / max(g_ok + g_bad, 1), 3), "skeptic_grounding": round(c_ok / max(c_ok + c_bad, 1), 3),
        "strategist_override_rate": round(strat["overridden"] / max(strat["turns"], 1), 3),
        "strategist_invalid_rate": round(strat["invalid"] / max(strat["turns"], 1), 3),
        "auditor_verdicts": dict(aud),
        "auditor_objection_rate_in_dry_dig_cases": mean(obj_dry), "auditor_objection_rate_in_other_cases": mean(obj_clean),
        "callbacks_per_case": round(cb["asked"] / n, 2), "callback_answered_rate": round(cb["answered"] / max(cb["asked"], 1), 3),
        "callback_learned_fact_rate": round(cb["learned"] / max(cb["asked"], 1), 3),
        "checks_per_found_leak": mean([r["checks_used"] for r in found]),
        "first_dig_success": round(sum(r["n_digs"] == 1 for r in found) / max(len(found), 1), 3),
        "hours_found_leak_ran": mean([r["hours_open"] for r in found if r.get("hours_open") is not None]),
        "llm_calls_per_case": mean([r["llm_calls"] + r.get("llm_cached", 0) for r in rows]),           # cached answers are calls too
        "llm_calls_per_solved_case": round(sum(r["llm_calls"] + r.get("llm_cached", 0) for r in rows) / max(len(solved), 1), 2),
        "tool_error_rate": round(sum(r["tool_errors"] for r in rows) / max(sum(r["tool_calls"] for r in rows), 1), 3),
        "turns_per_case": {k: round(v / n, 2) for k, v in sorted(turns.items())},
    }


def to_md(res: dict) -> str:
    """Args: res = {layout: {'mid': metrics, 'final': metrics}}.
    Returns: Markdown tables (one row per metric, mid vs final per layout).
    """
    keys = [("hypothesis_recall_first_look", "True leak inside a hypothesis: first look"), ("hypothesis_recall_any_time", "... at any time"),
            ("hypothesis_recall_final", "... at the end"), ("hypothesis_grounding", "Hypothesis citations accepted by code checks"),
            ("skeptic_grounding", "Skeptic citations accepted by code checks"), ("strategist_override_rate", "Strategist picks replaced by code"),
            ("strategist_invalid_rate", "Strategist invalid picks"),
            ("auditor_objection_rate_in_dry_dig_cases", "Auditor objected: cases with a dry dig"),
            ("auditor_objection_rate_in_other_cases", "Auditor objected: other cases"), ("callbacks_per_case", "Call-backs per case"),
            ("callback_answered_rate", "Call-backs answered"), ("callback_learned_fact_rate", "Call-backs that taught a fact"),
            ("checks_per_found_leak", "Field checks per found leak"), ("first_dig_success", "Found leaks dug right the first time"),
            ("hours_found_leak_ran", "Hours a found leak ran"), ("llm_calls_per_case", "LLM calls per case"),
            ("llm_calls_per_solved_case", "LLM calls per solved case"),
            ("tool_error_rate", "Tool calls refused by code (incl. M4 citation checks)")]
    lays = list(res)
    L = ["# Agent process metrics (mid vs final crew)", "", "| metric | " + " | ".join(f"{l} mid | {l} final" for l in lays) + " |",
         "|---|" + "---|---|" * len(lays)]
    for k, name in keys:
        L.append(f"| {name} | " + " | ".join(f"{res[l]['mid'].get(k)} | {res[l]['final'].get(k)}" for l in lays) + " |")
    L += ["", "Agent turns per case: " + "; ".join(f"{l} mid {res[l]['mid']['turns_per_case']}, final {res[l]['final']['turns_per_case']}" for l in lays), ""]
    return "\n".join(L)


def main():
    """Compute the metrics for every layout that has both a mid (v1) and a final (v2) crew log; write agent_extra.json/.md."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark/agentix_master")
    ap.add_argument("--out", default="logs/evaluation/extra")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from aquamind.eval.runner import World
    net = World(a.bench).town.net
    truth = load_truth(a.bench)
    T = "logs/evaluation/agentix/crew"
    res = {}
    for k in ("k5", "k3", "k10"):
        ok = all(os.path.exists(f"{T}/{k}/{v}_{k}_agentix_crew_trace.jsonl") for v in ("v1", "v2"))
        if not ok:
            continue
        res[k] = {}
        for v, lab in (("v1", "mid"), ("v2", "final")):
            rows = [json.loads(x) for x in open(f"{T}/{k}/{v}_{k}_agentix_crew.jsonl")]
            res[k][lab] = run_metrics(rows, f"{T}/{k}/{v}_{k}_agentix_crew_trace.jsonl", truth, net.index)
    json.dump(res, open(f"{a.out}/agent_extra.json", "w"), indent=2)
    open(f"{a.out}/agent_extra.md", "w").write(to_md(res))
    print(to_md(res))


if __name__ == "__main__":
    main()
