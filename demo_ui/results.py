"""Result tables for the Results view, computed from the evaluation's own result rows (logs/evaluation/**.jsonl).

Every number the UI shows comes from these rows with the metric definitions of the report (Section 6), and can be traced to a file: each table
row carries the path of the rows it was computed from. Each method has two runs: the FINAL run (fine-tuned reader, mitigations M1-M5, all 210
cases) and the MID-report run. The final run is used as soon as it holds all 210 cases; until then the mid run is shown and labelled 'mid run',
and it is compared with the crew of the same run. The mid numbers stay next to the final ones so the effect of the mitigations is visible.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .config import CACHE, LOGS, ROOT, SUITES

N_CASES = 210          # 60 dev + 150 frozen test
A, B = "ablations", "baselines"

# ---------------------------------------------------------------- which result rows make each table: (final run, mid-report run)
CREW = {3: ("agentix/crew/k3/v2_k3_agentix_crew.jsonl", "agentix/crew/k3/v1_k3_agentix_crew.jsonl"),
        5: ("agentix/crew/k5/v2_k5_agentix_crew.jsonl", "agentix/crew/k5/v1_k5_agentix_crew.jsonl"),
        10: ("agentix/crew/k10/v2_k10_agentix_crew.jsonl", "agentix/crew/k10/v1_k10_agentix_crew.jsonl")}
MAIN = [   # key, name, final, mid, what, ours
    ("agentix_crew", "AquaMind crew", *CREW[5], "Four LLM agents on a code search; gate + one-time token", True),
    ("react_agent", "ReAct agent", f"{B}/react_agent/k5/v2_k5_react_agent.jsonl", f"{B}/react_agent/k5/v1_k5_react_agent.jsonl",
     "Same LLM and tools, 15 steps, no gate", False),
    ("crew_practice", "Crews' current practice", f"{B}/crew_practice/k5/v2_k5_crew_practice.jsonl", f"{B}/crew_practice/k5/v1_k5_crew_practice.jsonl",
     "Listen where people complain, walk outward", False),
    ("plain_llm", "Plain LLM", f"{B}/plain_llm/k5/v2_k5_plain_llm.jsonl", f"{B}/plain_llm/k5/v1_k5_plain_llm.jsonl",
     "Reads the case as text, answers once", False),
    ("classical", "Classical inversion (EPANET)", f"{B}/bruteforce/k5/v4_k5_bruteforce.jsonl", f"{B}/bruteforce/k5/v3_k5_bruteforce.jsonl",
     "EPANET brute force with model errors, then listen", False),
    ("fixed_rules", "Fixed rules", f"{B}/fixed_rules/k5/v3_k5_fixed_rules.jsonl", f"{B}/fixed_rules/k5/v2_k5_fixed_rules.jsonl",
     "No LLM: one logger, then dig the top spot", False),
]
ABLATIONS = [   # key, name, final, mid, what, capability
    ("no_reports", "Citizen reports", f"{A}/no_reports/k5/v2_k5_ablation_no_reports.jsonl", f"{A}/no_reports/k5/v1_k5_ablation_no_reports.jsonl",
     "Messages are never read", "RQ1"),
    ("no_temp_sensor", "Temporary sensor (step test)", f"{A}/no_temp_sensor/k5/v2_k5_ablation_no_temp_sensor.jsonl",
     f"{A}/no_temp_sensor copy/k5/v1_k5_ablation_no_temp_sensor.jsonl", "No new flow measurement; listening stays", "Tool calling"),
    ("no_rag", "RAG", f"{A}/no_rag/k5/v2_k5_ablation_no_rag.jsonl", f"{A}/no_rag/k5/v1_k5_ablation_no_rag.jsonl", "No SOP and no town archive", "RAG"),
    ("no_gate", "Safety gate", f"{A}/no_gate/k5/v2_k5_ablation_no_gate.jsonl", f"{A}/no_gate/k5/v1_k5_ablation_no_gate.jsonl",
     "No evidence bar; token and 2-dig limit stay", "Safety"),
    ("one_shot", "Step-by-step loop", f"{A}/one_shot/k5/v2_k5_ablation_one_shot.jsonl", f"{A}/one_shot/k5/v1_k5_ablation_one_shot.jsonl",
     "One shot: one look, then dig or close", "Reasoning"),
    ("no_ai", "All AI", f"{A}/no_ai/k5/v3_k5_ablation_no_ai.jsonl", f"{A}/no_ai/k5/v2_k5_ablation_no_ai.jsonl", "Agents replaced by code; no text is read", "Agentic"),
]
ROLES = [   # key, name, final, mid, what   (no_strategist has an older v2 from October: its final run is v3; no_forensic's final run is
            # v4, the fair re-run of ADR-013 (v3 let code do the Forensic's job))
    ("no_forensic", "Forensic Advocate", f"{A}/no_forensic/k5/v4_k5_ablation_no_forensic.jsonl", f"{A}/no_forensic/k5/v1_k5_ablation_no_forensic.jsonl",
     "A fixed rule names where to look; nobody is called back; no ready-made map sweep (fair re-run, ADR-013)"),
    ("no_skeptic", "Critical Skeptic", f"{A}/no_skeptic/k5/v2_k5_ablation_no_skeptic.jsonl", f"{A}/no_skeptic/k5/v1_k5_ablation_no_skeptic.jsonl",
     "The leading case is never attacked"),
    ("no_strategist", "Hydraulic Strategist", f"{A}/no_strategist/k5/v3_k5_ablation_no_strategist.jsonl",
     f"{A}/no_strategist/k5/v1_k5_ablation_no_strategist.jsonl", "A simple rule picks the next check (no menu, no Strategist)"),
    ("no_auditor", "Safety & SOP Auditor", f"{A}/no_auditor/k5/v2_k5_ablation_no_auditor.jsonl", f"{A}/no_auditor/k5/v1_k5_ablation_no_auditor.jsonl",
     "No last review and no SOP paperwork"),
    ("no_sop", "SOP clauses only", f"{A}/no_sop/k5/v2_k5_ablation_no_sop.jsonl", f"{A}/no_sop/k5/v1_k5_ablation_no_sop.jsonl", "The Auditor works without the SOP"),
    ("no_triangulation", "Triangulation", f"{A}/no_triangulation/k5/v2_k5_ablation_no_triangulation.jsonl",
     f"{A}/no_triangulation/k5/v1_k5_ablation_no_triangulation.jsonl", "Callers are not read as one story"),
]
SENSORS = {   # method -> (name, {k: (final, mid)})
    "agentix_crew": ("AquaMind crew", CREW),
    "no_ai": ("Same search, agents replaced by code", {
        3: (f"{A}/no_ai/k3/v2_k3_ablation_no_ai.jsonl", f"{A}/no_ai/k3/v1_k3_ablation_no_ai.jsonl"),
        5: (f"{A}/no_ai/k5/v3_k5_ablation_no_ai.jsonl", f"{A}/no_ai/k5/v2_k5_ablation_no_ai.jsonl"),
        10: (f"{A}/no_ai/k10/v2_k10_ablation_no_ai.jsonl", f"{A}/no_ai/k10/v1_k10_ablation_no_ai.jsonl")}),
    "crew_practice": ("Crews' current practice", {k: (f"{B}/crew_practice/k{k}/v2_k{k}_crew_practice.jsonl", f"{B}/crew_practice/k{k}/v1_k{k}_crew_practice.jsonl")
                                                  for k in (3, 5, 10)}),
    "classical": ("Classical inversion (sensors only)", {
        3: (f"{B}/bruteforce/k3/v3_k3_bruteforce.jsonl", f"{B}/bruteforce/k3/v2_k3_bruteforce.jsonl"),
        5: (f"{B}/bruteforce/k5/v4_k5_bruteforce.jsonl", f"{B}/bruteforce/k5/v3_k5_bruteforce.jsonl"),
        10: (f"{B}/bruteforce/k10/v3_k10_bruteforce.jsonl", f"{B}/bruteforce/k10/v2_k10_bruteforce.jsonl")}),
    "fixed_rules": ("Fixed rules", {
        3: (f"{B}/fixed_rules/k3/v2_k3_fixed_rules.jsonl", f"{B}/fixed_rules/k3/v1_k3_fixed_rules.jsonl"),
        5: (f"{B}/fixed_rules/k5/v3_k5_fixed_rules.jsonl", f"{B}/fixed_rules/k5/v2_k5_fixed_rules.jsonl"),
        10: (f"{B}/fixed_rules/k10/v2_k10_fixed_rules.jsonl", f"{B}/fixed_rules/k10/v1_k10_fixed_rules.jsonl")}),
}
# the final code with every mitigation switched off, on the same cases (shows the gain comes from the mitigations, not from other changes)
SWITCHED_OFF = f"{B}/agentix_crew_mid/k5/v2_k5_agentix_crew_mid.jsonl"
MITIGATIONS = [
    ("M1", "Weak leads", "Agents see how strong each check was; a quiet spot can be checked again; suggestive flow areas become dig candidates."),
    ("M2", "Turn budget", "More Forensic turns across the case and a Skeptic turn after a quiet check (challenge WHERE, not WHETHER)."),
    ("M3", "Confirm the dig", "A listening check at a fresh spot confirms the dig; objections hold only digs without physical support."),
    ("M4", "Ticket type", "Only non-utility tickets count as false alarms; citing a repair ticket as one is refused by code."),
    ("M5", "Corner permits", "Every street at the dig junction is listed for the SOP-6 permit."),
]
# One look, no field checks (scripts/run_one_look.py). Final outputs; the mid-report table (Table 5) is the fallback.
ONE_LOOK_FILES = ["runs/final_queue/one_look_dev.jsonl", "runs/final_queue/one_look_test.jsonl"]
ONE_LOOK_METHODS = [("agentix_search", "Our search with messages", True), ("agentix_no_reports", "Our search, sensors only", False),
                    ("lila_style", "LILA-style re-implementation", False), ("sensitivity_match", "Sensitivity matching", False),
                    ("report_hotspot", "Report hotspot (messages only)", False)]
ONE_LOOK_MID = {"source": "AgentiX_Mid_Report.pdf, Table 5 (scripts/run_one_look.py)", "stage": "mid", "k": [3, 5, 10], "rows": [
    ["Our search with messages", [460, 385, 362], True],
    ["Our search, sensors only", [720, 533, 530], False],
    ["LILA-style re-implementation", [1266, 886, 736], False],
    ["Sensitivity matching", [1168, 924, 877], False],
    ["Report hotspot (messages only)", [1017, 1017, 1017], False]]}
EXTRA = {"agent": "extra/agent_extra.json", "rag": "extra/rag_eval.json"}     # scripts/eval_agent_extra.py, scripts/eval_rag.py

_rows_cache: dict[str, tuple] = {}


def rows(rel: str) -> list[dict]:
    """Args: rel = path of a result .jsonl under logs/evaluation.
    Returns: its rows, one per case (the last row of a case wins; cached until the file changes; [] if missing).
    """
    p = LOGS / rel
    if not p.exists():
        return []
    mt = os.path.getmtime(p)
    hit = _rows_cache.get(rel)
    if hit and hit[0] == mt:
        return hit[1]
    by_case = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("error"):
            r.pop("trace", None)
            by_case[r.get("case")] = r
    out = list(by_case.values())
    _rows_cache[rel] = (mt, out)
    return out


def _pick(final: str, mid: str | None) -> tuple[str, str, str | None]:
    """Choose the run a table row is computed from.

    Args: final = result file of the final run; mid = result file of the mid-report run (or None).
    Returns: (file, 'final' | 'mid', note or None). The final run counts only when it holds all N_CASES cases.
    """
    nf = len(rows(final))
    if nf >= N_CASES:
        return final, "final", None
    if mid and rows(mid):
        return mid, "mid", f"final run: {nf}/{N_CASES} cases logged" if nf else "final run not logged yet"
    return final, "final", f"{nf}/{N_CASES} cases logged"


def solved(r: dict) -> bool:
    """Args: r = one result row.
    Returns: True if the case was solved (leak dug within 100 m, or a no-leak case closed without a dig).
    """
    return bool(r.get("found")) if r.get("has_leak") else r.get("n_digs", 0) == 0


def _boot(x, n=2000, seed=0):
    """Bootstrap mean and 95% interval.

    Args: x = values (0/1 or numbers); n = resamples; seed = random seed.
    Returns: [mean, low, high] (None if empty).
    """
    x = np.asarray(list(x), float)
    if len(x) == 0:
        return [None, None, None]
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x))).mean(1)
    return [round(100 * float(x.mean()), 1), round(100 * float(np.quantile(b, 0.025)), 1), round(100 * float(np.quantile(b, 0.975)), 1)]


def summary(rs: list[dict]) -> dict:
    """The metrics of the report (Section 6) for one method on one layout.

    Args: rs = result rows.
    Returns: metrics dict.
    """
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
        "solved": [int(sum(solved(r) for r in rs)), len(rs)],
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
    """Args: rs = result rows.
    Returns: {category: [correct, total]}.
    """
    out = {}
    for r in rs:
        s = r.get("suite") or "?"
        ok = bool(r.get("found")) if r.get("has_leak") else r.get("n_digs", 0) == 0
        e = out.setdefault(s, [0, 0])
        e[0] += int(ok)
        e[1] += 1
    return out


def _reader() -> dict:
    """Returns: reader quality numbers (fine-tuned vs rules vs big LLM) and the training curve, for the Results view."""
    out = {"source": "finetune/metrics/reader_results.json, finetune/notebooks/results_rules_test_real.json, finetune/dataset/llm_reference.json"}
    try:
        R = json.load(open("finetune/metrics/reader_results.json"))
    except (OSError, ValueError):
        R = {}
    def pick(d, nd=3):
        """Args: d = metrics dict; nd = decimals.
        Returns: the main reader scores (category, symptoms F1, places F1, time F1, JSON valid, n).
        """
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


def _entry(final: str, mid: str | None, **fields) -> tuple[dict, list[dict], str]:
    """One table row from its final run (or its mid run while the final one is incomplete).

    Args: final, mid = result files; fields = extra fields of the row (key, name, what, ...).
    Returns: (row dict with the metrics, the result rows used, 'final' | 'mid').
    """
    rel, stage, note = _pick(final, mid)
    rs = rows(rel)
    e = {**fields, "file": f"logs/evaluation/{rel}", "stage": stage, "note": note, **summary(rs)}
    if stage == "final" and mid and rows(mid):
        m = summary(rows(mid))
        e["mid"] = {"file": f"logs/evaluation/{mid}", **{k: m.get(k) for k in ("correct", "solved", "median_m", "false_digs", "unsafe", "violations", "n")}}
    return e, rs, stage


def _paired(mid: list[dict], fin: list[dict]) -> dict:
    """Case-by-case comparison of two runs on solved cases (exact McNemar).

    Args: mid, fin = result rows of the mid and the final run.
    Returns: {'gained': cases only the final run solved, 'lost': cases only the mid run solved, 'p': McNemar p, 'n': paired cases}.
    """
    from scipy.stats import binomtest
    M, F = {r["case"]: r for r in mid}, {r["case"]: r for r in fin}
    common = sorted(set(M) & set(F))
    def brief(r):
        """Args: r = result row. Returns: the few fields the UI shows."""
        return {"outcome": r.get("outcome"), "found": bool(r.get("found")), "err": r.get("guess_error_m"), "digs": r.get("n_digs", 0)}
    pick = lambda want: [{"case": c, "suite": F[c].get("suite"), "has_leak": bool(F[c].get("has_leak")), "final": brief(F[c]), "mid": brief(M[c])}
                         for c in common if solved(F[c]) == want and solved(M[c]) != want]
    g, lost = pick(True), pick(False)
    return {"gained": g, "lost": lost, "n": len(common), "p": float(binomtest(len(g), len(g) + len(lost)).pvalue) if g or lost else 1.0}


def _mitigation() -> dict:
    """Mid system against the final system (mitigations M1-M5), on the same 210 cases, for 3, 5 and 10 sensors.

    Returns: {'layouts': [{k, mid, final, gained, lost, p}], 'suites': per case type at 5 sensors, 'switched_off': final code with the
              mitigations off (None until that run is complete), 'items': the five mitigations}.
    """
    out = {"layouts": [], "suites": [], "switched_off": None,
           "items": [{"id": i, "title": t, "what": w} for i, t, w in MITIGATIONS]}
    for k, (fin, mid) in CREW.items():
        F, M = rows(fin), rows(mid)
        if len(F) < N_CASES or not M:
            continue
        out["layouts"].append({"k": k, "mid": {"file": f"logs/evaluation/{mid}", **summary(M)}, "final": {"file": f"logs/evaluation/{fin}", **summary(F)},
                               **_paired(M, F)})
        if k == 5:
            sm, sf = _by_suite(M), _by_suite(F)
            out["suites"] = [{"id": s, "name": SUITES.get(s, (s, ""))[0], "mid": sm.get(s, [0, 0]), "final": sf.get(s, [0, 0])} for s in SUITES if s in sf]
    off = rows(SWITCHED_OFF)
    if len(off) >= N_CASES:
        out["switched_off"] = {"file": f"logs/evaluation/{SWITCHED_OFF}", **summary(off)}
    return out


def _one_look() -> dict:
    """Median error (m) of ONE look, no field checks, per method and sensor count (all layouts of scripts/run_one_look.py).

    Returns: {'source', 'stage', 'k', 'rows': [[name, [median per k], ours]], 'n'}; the mid-report table if the final outputs are missing.
    """
    by: dict[tuple, list] = {}
    cases = set()
    for f in ONE_LOOK_FILES:
        p = ROOT / f
        if not p.exists():
            return ONE_LOOK_MID
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            cases.add(r["case"])
            if r.get("error_m") is not None:
                by.setdefault((r["method"], r["k"]), []).append(r["error_m"])
    ks = [3, 5, 10]
    if not all((m, k) in by for m, _, _ in ONE_LOOK_METHODS for k in ks):
        return ONE_LOOK_MID
    return {"source": " + ".join(ONE_LOOK_FILES) + " (scripts/run_one_look.py, fine-tuned reader)", "stage": "final", "k": ks, "n": len(cases),
            "rows": [[name, [round(float(np.median(by[(m, k)]))) for k in ks], ours] for m, name, ours in ONE_LOOK_METHODS],
            "mid": ONE_LOOK_MID["rows"]}


def _extra(key: str) -> dict | None:
    """Args: key = 'agent' | 'rag'.
    Returns: the saved process metrics (scripts/eval_agent_extra.py, scripts/eval_rag.py) with their source path, or None if not computed.
    """
    p = LOGS / EXTRA[key]
    try:
        return {"source": f"logs/evaluation/{EXTRA[key]}", "data": json.loads(p.read_text())}
    except (OSError, ValueError):
        return None


SKEPTIC_RUNS = {   # stage -> (crew rows, crew trace, no-Skeptic rows, no-Skeptic trace)
    "final": ("agentix/crew/k5/v2_k5_agentix_crew.jsonl", "agentix/crew/k5/v2_k5_agentix_crew_trace.jsonl",
              f"{A}/no_skeptic/k5/v2_k5_ablation_no_skeptic.jsonl", f"{A}/no_skeptic/k5/v2_k5_ablation_no_skeptic_trace.jsonl"),
    "mid": ("agentix/crew/k5/v1_k5_agentix_crew.jsonl", "agentix/crew/k5/v1_k5_agentix_crew_trace.jsonl",
            f"{A}/no_skeptic/k5/v1_k5_ablation_no_skeptic.jsonl", f"{A}/no_skeptic/k5/v1_k5_ablation_no_skeptic_trace.jsonl"),
}
HIDDEN = ROOT / "data/benchmark/agentix_master/hidden"      # ground truth, read here only to score (as the evaluation scorer does)


def _skeptic_value() -> dict | None:
    """Where the Critical Skeptic helps: leak cases split by whether the Forensic's FIRST leading hypothesis (its first turn, before the
    Skeptic speaks, so the same in both runs) contains the true leak. The Skeptic's job is to attack a wrong first story.

    Returns: {stage: {'right': [cases, crew solved, no-Skeptic solved], 'wrong': [...], 'files': [...]}}, or None if a file is missing.
    """
    truth = {}
    for f in HIDDEN.glob("*.json"):
        if f.name.startswith("_"):
            continue
        d = json.loads(f.read_text())
        if d.get("leak"):
            truth[d["case_id"]] = int(d["leak"]["nearest_node"][1:]) - 1        # junction index (n1 -> 0)
    out = {}
    for stage, (crew_rows, crew_tr, ns_rows, ns_tr) in SKEPTIC_RUNS.items():
        C, S = {r["case"]: r for r in rows(crew_rows)}, {r["case"]: r for r in rows(ns_rows)}
        p = LOGS / crew_tr
        if not C or not S or not p.exists() or not (LOGS / ns_tr).exists():
            return None
        first = {}
        for line in open(p):
            if '"role": "forensic"' not in line:
                continue
            t = json.loads(line)
            if t.get("layout") == "spread_k5" and t["case"] not in first:
                first[t["case"]] = t.get("slate") or []
        g = {"right": [0, 0, 0], "wrong": [0, 0, 0]}
        for case, j in truth.items():
            if case not in C or case not in S or case not in first:
                continue
            net = [h for h in first[case] if h.get("kind") == "network_leak" and h.get("region")]
            lead = max(net, key=lambda h: h.get("mass", 0)) if net else None
            v = g["right" if lead and j in set(lead["region"]) else "wrong"]
            v[0] += 1
            v[1] += solved(C[case])
            v[2] += solved(S[case])
        out[stage] = {**g, "files": [f"logs/evaluation/{crew_rows}", f"logs/evaluation/{ns_rows}", f"logs/evaluation/{crew_tr}"]}
    return out


def build() -> dict:
    """Recompute every Results table from the evaluation logs (with CIs and McNemar tests).

    Returns: JSON-ready dict.
    """
    t0 = time.time()
    crew_e, crew, crew_stage = _entry(MAIN[0][2], MAIN[0][3], key=MAIN[0][0], name=MAIN[0][1], what=MAIN[0][4], ours=True)
    crew_mid = rows(MAIN[0][3])
    base = lambda stage: crew if stage == crew_stage else crew_mid       # the crew of the same run, for fair differences
    crew_e["suites"] = _by_suite(crew)
    main = [crew_e]
    for key, name, fin, mid, what, ours in MAIN[1:]:
        e, rs, stage = _entry(fin, mid, key=key, name=name, what=what, ours=ours)
        e["mcnemar"] = _mcnemar(base(stage), rs) if rs else None
        e["suites"] = _by_suite(rs)
        main.append(e)
    full = {**crew_e, "key": "full", "name": "Nothing removed (full crew)", "what": "-"}
    full.pop("suites", None)

    def variant(fin, mid, **fields):
        """Args: fin, mid = result files; fields = row fields. Returns: an ablation row with its change against the crew of the same run."""
        e, rs, stage = _entry(fin, mid, **fields)
        b = summary(base(stage))
        e["base_correct"] = b["correct"][0] if b.get("n") else None
        e["base_violations"] = b.get("violations")
        e["delta"] = round(e["correct"][0] - e["base_correct"], 1) if e.get("n") and e["base_correct"] is not None else None
        return e
    abl = [full] + [variant(fin, mid, key=k, name=n, what=w, capability=c) for k, n, fin, mid, w, c in ABLATIONS]
    roles = [full] + [variant(fin, mid, key=k, name=n, what=w) for k, n, fin, mid, w in ROLES]
    sensors = []
    for key, (name, files) in SENSORS.items():
        by_k = {}
        for k, (fin, mid) in files.items():
            by_k[str(k)] = _entry(fin, mid)[0]
        sensors.append({"key": key, "name": name, "ours": key == "agentix_crew", "by_k": by_k})
    suites = [{"id": s, "name": SUITES.get(s, (s, ""))[0], "blurb": SUITES.get(s, ("", ""))[1],
               "n": sum(1 for r in crew if r.get("suite") == s), "has_leak": any(r.get("has_leak") for r in crew if r.get("suite") == s)}
              for s in ["brown", "chronic", "cd_damp", "cd_damp_decoy", "cd_sound", "cd_sound_decoy", "displaced", "trap", "false_alarm", "visible"]]
    react = {r["case"]: r for r in rows(_pick(MAIN[1][2], MAIN[1][3])[0])}
    traps = []
    for r in sorted(crew, key=lambda r: r["case"]):
        if r.get("suite") != "trap":
            continue
        o = react.get(r["case"])
        traps.append({"case": r["case"], "has_leak": bool(r.get("has_leak")), "crew": r.get("outcome"), "crew_err": r.get("guess_error_m"),
                      "crew_found": r.get("found"), "crew_digs": r.get("n_digs", 0),
                      "react": o.get("outcome") if o else None, "react_err": o.get("guess_error_m") if o else None,
                      "react_found": o.get("found") if o else None, "react_digs": o.get("n_digs", 0) if o else None,
                      "react_unsafe": o.get("unsafe") if o else None,
                      "react_followed": ((o or {}).get("audit") or {}).get("hard", {}).get("followed_trap", 0)})
    everything = main + abl[1:] + roles[1:] + [x for s in sensors for x in s["by_k"].values()]
    pending = sorted({e["file"].replace("logs/evaluation/", "") for e in everything if e.get("stage") == "mid"})
    return {"generated_s": round(time.time() - t0, 2), "main": main, "ablations": abl, "roles": roles, "sensors": sensors,
            "one_look": _one_look(), "mitigation": _mitigation(), "suites": suites, "traps": traps, "reader": _reader(),
            "agent_extra": _extra("agent"), "rag": _extra("rag"), "skeptic": _skeptic_value(), "stage": crew_stage, "pending": pending,
            "n_cases": N_CASES,
            "note": "Computed live from the result rows in logs/evaluation (one row per case). Final runs: fine-tuned reader, mitigations M1-M5, "
                    "60 dev + 150 frozen test cases." + (" Rows marked 'mid run' wait for their final run and are compared with the mid crew."
                                                         if pending else "")}


def _sources() -> list:
    """Returns: every file the Results data is computed from (to know when the cache is stale)."""
    fs = [LOGS / f for _, _, fin, mid, *_ in MAIN + ABLATIONS + ROLES for f in (fin, mid)]
    fs += [LOGS / f for _, files in SENSORS.values() for pair in files.values() for f in pair]
    fs += [LOGS / SWITCHED_OFF] + [LOGS / f for f in EXTRA.values()] + [ROOT / f for f in ONE_LOOK_FILES]
    fs += [LOGS / f for run in SKEPTIC_RUNS.values() for f in run]
    return fs


def payload(refresh: bool = False) -> dict:
    """Args: refresh = rebuild even if the cache is newer than the logs.
    Returns: the Results data.
    """
    p = CACHE / "results.json"
    newest = max([os.path.getmtime(f) for f in _sources() if f.exists()] + [os.path.getmtime(__file__)])
    if not refresh and p.exists() and os.path.getmtime(p) > newest:
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass
    d = build()
    p.write_text(json.dumps(d))
    return d
