"""Result tables for the Results view, computed from the evaluation's own result rows (logs/evaluation/**.jsonl).

Every number the UI shows comes from these rows with the metric definitions of the mid report (Section 6), so the tables match Tables 1-4 of the
report and can be traced to a file: each table row carries the path of the rows it was computed from. The one-look comparison (Table 5) was
produced by scripts/run_one_look.py into a file that is not in the logs, so it is quoted from the report and labelled as such.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .config import CACHE, LOGS, SUITES

# ---------------------------------------------------------------- which result rows make each table (the mid-report runs)
MAIN = [
    ("agentix_crew", "AquaMind crew", "agentix/crew/k5/v1_k5_agentix_crew.jsonl", "Four LLM agents on a code search; gate + one-time token", True),
    ("react_agent", "ReAct agent", "baselines/react_agent/k5/v1_k5_react_agent.jsonl", "Same LLM and tools, 15 steps, no gate", False),
    ("crew_practice", "Crews' current practice", "baselines/crew_practice/k5/v1_k5_crew_practice.jsonl", "Listen where people complain, walk outward", False),
    ("plain_llm", "Plain LLM", "baselines/plain_llm/k5/v1_k5_plain_llm.jsonl", "Reads the case as text, answers once", False),
    ("classical", "Classical inversion (EPANET)", "baselines/bruteforce/k5/v3_k5_bruteforce.jsonl", "EPANET brute force with model errors, then listen", False),
    ("fixed_rules", "Fixed rules", "baselines/fixed_rules/k5/v2_k5_fixed_rules.jsonl", "No LLM: one logger, then dig the top spot", False),
]
ABLATIONS = [
    ("no_reports", "Citizen reports", "ablations/no_reports/k5/v1_k5_ablation_no_reports.jsonl", "Messages are never read", "RQ1"),
    ("no_temp_sensor", "Temporary sensor (step test)", "ablations/no_temp_sensor copy/k5/v1_k5_ablation_no_temp_sensor.jsonl",
     "No new flow measurement; listening stays", "Tool calling"),
    ("no_rag", "RAG", "ablations/no_rag/k5/v1_k5_ablation_no_rag.jsonl", "No SOP and no town archive", "RAG"),
    ("no_gate", "Safety gate", "ablations/no_gate/k5/v1_k5_ablation_no_gate.jsonl", "No evidence bar; token and 2-dig limit stay", "Safety"),
    ("one_shot", "Step-by-step loop", "ablations/one_shot/k5/v1_k5_ablation_one_shot.jsonl", "One shot: one look, then dig or close", "Reasoning"),
    ("no_ai", "All AI", "ablations/no_ai/k5/v2_k5_ablation_no_ai.jsonl", "Agents replaced by code; no text is read", "Agentic"),
]
ROLES = [
    ("no_forensic", "Forensic Advocate", "ablations/no_forensic/k5/v1_k5_ablation_no_forensic.jsonl", "A fixed rule names where to look; nobody is called back"),
    ("no_skeptic", "Critical Skeptic", "ablations/no_skeptic/k5/v1_k5_ablation_no_skeptic.jsonl", "The leading case is never attacked"),
    ("no_strategist", "Hydraulic Strategist", "ablations/no_strategist/k5/v1_k5_ablation_no_strategist.jsonl", "Code picks the check with the best chance"),
    ("no_auditor", "Safety & SOP Auditor", "ablations/no_auditor/k5/v1_k5_ablation_no_auditor.jsonl", "No last review and no SOP paperwork"),
    ("no_sop", "SOP clauses only", "ablations/no_sop/k5/v1_k5_ablation_no_sop.jsonl", "The Auditor works without the SOP"),
    ("no_triangulation", "Triangulation", "ablations/no_triangulation/k5/v1_k5_ablation_no_triangulation.jsonl", "Callers are not read as one story"),
]
SENSORS = {   # method -> {k: rows}
    "agentix_crew": ("AquaMind crew", {3: "agentix/crew/k3/v1_k3_agentix_crew.jsonl", 5: "agentix/crew/k5/v1_k5_agentix_crew.jsonl",
                                        10: "agentix/crew/k10/v1_k10_agentix_crew.jsonl"}),
    "no_ai": ("Same search, agents replaced by code", {3: "ablations/no_ai/k3/v1_k3_ablation_no_ai.jsonl", 5: "ablations/no_ai/k5/v2_k5_ablation_no_ai.jsonl",
                                                       10: "ablations/no_ai/k10/v1_k10_ablation_no_ai.jsonl"}),
    "crew_practice": ("Crews' current practice", {3: "baselines/crew_practice/k3/v1_k3_crew_practice.jsonl", 5: "baselines/crew_practice/k5/v1_k5_crew_practice.jsonl",
                                                  10: "baselines/crew_practice/k10/v1_k10_crew_practice.jsonl"}),
    "classical": ("Classical inversion (sensors only)", {3: "baselines/bruteforce/k3/v2_k3_bruteforce.jsonl", 5: "baselines/bruteforce/k5/v3_k5_bruteforce.jsonl",
                                                         10: "baselines/bruteforce/k10/v2_k10_bruteforce.jsonl"}),
    "fixed_rules": ("Fixed rules", {3: "baselines/fixed_rules/k3/v1_k3_fixed_rules.jsonl", 5: "baselines/fixed_rules/k5/v2_k5_fixed_rules.jsonl",
                                    10: "baselines/fixed_rules/k10/v1_k10_fixed_rules.jsonl"}),
}
MITIGATION = [("agentix_crew_mid", "Mid system (mitigations off)", "baselines/agentix_crew_mid/k5/v1_k5_agentix_crew_mid.jsonl"),
              ("agentix_crew", "Final system (mitigations M1-M5 on)", "agentix/crew/k5/v2_k5_agentix_crew.jsonl")]
# Table 5 of the mid report (scripts/run_one_look.py; its output file is not part of logs/evaluation): median error (m) of ONE look, no field checks
ONE_LOOK = {"source": "AgentiX_Mid_Report.pdf, Table 5 (scripts/run_one_look.py)", "k": [3, 5, 10], "rows": [
    ["Our search with messages", [460, 385, 362], True],
    ["Our search, sensors only", [720, 533, 530], False],
    ["LILA-style re-implementation", [1266, 886, 736], False],
    ["Sensitivity matching", [1168, 924, 877], False],
    ["Report hotspot (messages only)", [1017, 1017, 1017], False]]}

_rows_cache: dict[str, list] = {}


def rows(rel: str) -> list[dict]:
    if rel not in _rows_cache:
        p = LOGS / rel
        out = []
        if p.exists():
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not r.get("error"):
                    r.pop("trace", None)
                    out.append(r)
        _rows_cache[rel] = out
    return _rows_cache[rel]


def _boot(x, n=2000, seed=0):
    x = np.asarray(list(x), float)
    if len(x) == 0:
        return [None, None, None]
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x))).mean(1)
    return [round(100 * float(x.mean()), 1), round(100 * float(np.quantile(b, 0.025)), 1), round(100 * float(np.quantile(b, 0.975)), 1)]


def summary(rs: list[dict]) -> dict:
    """The metrics of the mid report (Section 6) for one method on one layout."""
    if not rs:
        return {"n": 0}
    leak = [r for r in rs if r.get("has_leak")]
    nol = [r for r in rs if not r.get("has_leak")]
    digs = sum(r.get("n_digs", 0) for r in rs)
    wrong = sum(r.get("wrong_digs", 0) for r in rs)
    errs = [r["guess_error_m"] for r in leak if r.get("guess_error_m") is not None]
    tc = sum(r.get("tool_calls", 0) for r in rs)
    te = sum(r.get("tool_errors", 0) for r in rs)
    traps = [r for r in rs if r.get("suite") == "trap"]
    pct = lambda xs: round(100 * float(np.mean(xs)), 1) if len(xs) else None
    return {
        "n": len(rs), "n_leak": len(leak), "n_noleak": len(nol),
        "correct": _boot([bool(r["found"]) for r in leak]),
        "within100": pct([r.get("guess_error_m") is not None and r["guess_error_m"] <= 100 for r in leak]),
        "median_m": round(float(np.median(errs))) if errs else None,
        "false_digs": round(100 * wrong / digs, 1) if digs else None,
        "unsafe": pct([r.get("unsafe", 0) > 0 for r in rs]),
        "unsafe_n": int(sum(r.get("unsafe", 0) > 0 for r in rs)),
        "tool_acc": round(100 * (tc - te) / tc, 1) if tc else None,
        "violations": pct([r.get("violations", 0) > 0 for r in rs]),
        "steps": round(float(np.mean([r.get("steps", 0) for r in rs])), 1),
        "llm_calls": round(float(np.mean([r.get("llm_calls", 0) + r.get("llm_cached", 0) for r in rs])), 1),
        "tokens": round(float(np.mean([r.get("tokens", 0) for r in rs]))),
        "checks": round(float(np.mean([r.get("checks_used", 0) for r in rs])), 2),
        "noleak_clean": [int(sum(r.get("n_digs", 0) == 0 for r in nol)), len(nol)],
        "traps": [int(sum(bool(r.get("found")) or (not r.get("has_leak") and r.get("n_digs", 0) == 0) for r in traps)), len(traps)],
        "followed_trap": int(sum(((r.get("audit") or {}).get("hard") or {}).get("followed_trap", 0) > 0 for r in traps)),
        "unapproved": int(sum(((r.get("audit") or {}).get("hard") or {}).get("unapproved_attempt", 0) for r in rs)),
        "first_dig": pct([bool(r["found"]) and r.get("wrong_digs", 0) == 0 for r in leak]),
        "hours_to_find": round(float(np.mean([r.get("hours_open", 0) for r in leak if r.get("found")])), 1) if any(r.get("found") for r in leak) else None,
    }


def _mcnemar(a: list[dict], b: list[dict]) -> dict | None:
    """Exact McNemar on correct digs, paired by case (leak cases both methods ran)."""
    from scipy.stats import binomtest
    A = {r["case"]: bool(r["found"]) for r in a if r.get("has_leak")}
    B = {r["case"]: bool(r["found"]) for r in b if r.get("has_leak")}
    common = sorted(set(A) & set(B))
    if not common:
        return None
    oa = sum(A[c] and not B[c] for c in common)
    ob = sum(B[c] and not A[c] for c in common)
    p = binomtest(oa, oa + ob).pvalue if oa + ob else 1.0
    return {"only_ours": oa, "only_theirs": ob, "p": float(p), "n": len(common)}


def _by_suite(rs: list[dict]) -> dict:
    out = {}
    for r in rs:
        s = r.get("suite") or "?"
        ok = bool(r.get("found")) if r.get("has_leak") else r.get("n_digs", 0) == 0
        e = out.setdefault(s, [0, 0])
        e[0] += int(ok)
        e[1] += 1
    return out


def _reader() -> dict:
    out = {"source": "finetune/metrics/reader_results.json, finetune/notebooks/results_rules_test_real.json, finetune/dataset/llm_reference.json"}
    try:
        R = json.load(open("finetune/metrics/reader_results.json"))
    except (OSError, ValueError):
        R = {}
    def pick(d, nd=3):
        o = d.get("overall", d) if isinstance(d, dict) else {}
        return {"category": o.get("category"), "symptoms_f1": o.get("symptoms_f1"), "places_f1": o.get("places_relaxed_f1"),
                "time_f1": o.get("since_relaxed_f1", o.get("since")), "json_valid": o.get("json_valid", o.get("valid")), "n": o.get("n")}
    rows_ = []
    if "finetuned/test_real" in R:
        rows_.append({"name": "Ours: Qwen2.5-3B + QLoRA", "ours": True, **pick(R["finetuned/test_real"])})
    if "base_zeroshot/test_real" in R:
        rows_.append({"name": "Same model, no tuning", **pick(R["base_zeroshot/test_real"])})
    try:
        rr = json.load(open("finetune/notebooks/results_rules_test_real.json"))
        rows_.append({"name": "Rule reader (regular expressions)", **pick(rr), "time_f1": None, "json_valid": None})
    except (OSError, ValueError):
        pass
    try:
        lr = json.load(open("finetune/dataset/llm_reference.json"))["test_real"]
        rows_.append({"name": "Nemotron 550B, prompted (100 reports)", **pick(lr)})
    except (OSError, ValueError, KeyError):
        pass
    out["test_real"] = rows_
    hand = []
    if "finetuned/test_hand" in R:
        hand.append({"name": "Ours (fine-tuned)", "ours": True, **pick(R["finetuned/test_hand"])})
    if "base_zeroshot/test_hand" in R:
        hand.append({"name": "Same model, no tuning", **pick(R["base_zeroshot/test_hand"])})
    try:
        hand.append({"name": "Nemotron 550B, prompted", **pick(json.load(open("finetune/dataset/llm_reference.json"))["test_hand"])})
    except (OSError, ValueError, KeyError):
        pass
    out["test_hand"] = hand
    try:
        log = json.load(open("finetune/metrics/train_log.json"))
        out["train"] = [[e["step"], e["loss"]] for e in log if "loss" in e]
        out["eval"] = [[e["step"], e["eval_loss"]] for e in log if "eval_loss" in e]
    except (OSError, ValueError):
        out["train"], out["eval"] = [], []
    try:
        out["data"] = json.load(open("finetune/dataset/data_card.json"))["splits"]
    except (OSError, ValueError, KeyError):
        out["data"] = {}
    return out


def build() -> dict:
    t0 = time.time()
    crew = rows(MAIN[0][2])
    main = []
    for key, name, rel, what, ours in MAIN:
        rs = rows(rel)
        e = {"key": key, "name": name, "what": what, "ours": ours, "file": f"logs/evaluation/{rel}", **summary(rs)}
        if not ours and rs:
            e["mcnemar"] = _mcnemar(crew, rs)
        e["suites"] = _by_suite(rs)
        main.append(e)
    full = {"key": "full", "name": "Nothing removed (full crew)", "what": "-", "file": f"logs/evaluation/{MAIN[0][2]}", **summary(crew)}
    abl = [full] + [{"key": k, "name": n, "what": w, "capability": c, "file": f"logs/evaluation/{rel}", **summary(rows(rel))} for k, n, rel, w, c in ABLATIONS]
    roles = [full] + [{"key": k, "name": n, "what": w, "file": f"logs/evaluation/{rel}", **summary(rows(rel))} for k, n, rel, w in ROLES]
    sensors = []
    for key, (name, files) in SENSORS.items():
        sensors.append({"key": key, "name": name, "ours": key == "agentix_crew",
                        "by_k": {str(k): {"file": f"logs/evaluation/{rel}", **summary(rows(rel))} for k, rel in files.items()}})
    mit = [{"key": k, "name": n, "file": f"logs/evaluation/{rel}", **summary(rows(rel)), "suites": _by_suite(rows(rel))} for k, n, rel in MITIGATION]
    suites = [{"id": s, "name": SUITES.get(s, (s, ""))[0], "blurb": SUITES.get(s, ("", ""))[1],
               "n": sum(1 for r in crew if r.get("suite") == s), "has_leak": any(r.get("has_leak") for r in crew if r.get("suite") == s)}
              for s in ["brown", "chronic", "cd_damp", "cd_damp_decoy", "cd_sound", "cd_sound_decoy", "displaced", "trap", "false_alarm", "visible"]]
    react = rows(MAIN[1][2])
    traps = []
    for r in crew:
        if r.get("suite") != "trap":
            continue
        o = next((x for x in react if x["case"] == r["case"]), None)
        traps.append({"case": r["case"], "crew": r.get("outcome"), "crew_err": r.get("guess_error_m"), "crew_found": r.get("found"),
                      "react": o.get("outcome") if o else None, "react_err": o.get("guess_error_m") if o else None,
                      "react_found": o.get("found") if o else None,
                      "react_unsafe": o.get("unsafe") if o else None,
                      "react_followed": ((o or {}).get("audit") or {}).get("hard", {}).get("followed_trap", 0)})
    return {"generated_s": round(time.time() - t0, 2), "main": main, "ablations": abl, "roles": roles, "sensors": sensors,
            "one_look": ONE_LOOK, "mitigation": mit, "suites": suites, "traps": traps, "reader": _reader(),
            "note": "Computed live from the result rows in logs/evaluation (one row per case). Definitions: mid report Section 6."}


def payload(refresh: bool = False) -> dict:
    p = CACHE / "results.json"
    newest = max((os.path.getmtime(LOGS / rel) for rel in [m[2] for m in MAIN] if (LOGS / rel).exists()), default=0)
    if not refresh and p.exists() and os.path.getmtime(p) > newest:
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass
    d = build()
    p.write_text(json.dumps(d))
    return d
