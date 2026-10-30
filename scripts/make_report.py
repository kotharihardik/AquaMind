"""Result tables (proposal Section 11 metrics) from rows.jsonl files -> a Markdown file.

usage: python -I scripts/make_report.py <out.md> <rows.jsonl> [more rows.jsonl ...] [--one-look one_look.jsonl]
Every rate has a 95 % bootstrap interval over cases; paired comparisons with AquaMind use the exact McNemar test.
"""
import collections
import json
import sys

import numpy as np
from scipy.stats import binomtest

# ADR-008: visible water is a sanity check; the headline is the pooled non-obvious cases
GROUPS = {"visible water (sanity check)": ["visible"],
          "NON-OBVIOUS (displaced + damp + sound + brown water + chronic + SCADA-only)": ["displaced", "damp", "sound", "brown", "chronic", "silent"]}
ORDER = ["agentix", "agentix_code", "agentix_code_canvass", "agentix_no_reports_code", "agentix_no_reports_canvass", "agentix_canvass", "react_agent", "plain_llm", "fixed_rules", "crew_practice", "agentix_no_reports",
         "agentix_no_fieldchecks", "agentix_no_rag", "agentix_no_gate", "agentix_one_shot"]
NAMES = {"agentix": "**AquaMind (ours)**", "agentix_code_canvass": "exploratory: code-only planner + neighbour canvass",
         "agentix_no_reports_code": "code-only planner, no citizen messages", "agentix_canvass": "exploratory: AquaMind + neighbour canvass", "agentix_no_reports_canvass": "exploratory: no messages, with neighbour canvass", "agentix_code": "AquaMind without the LLM (code-only planner)",
         "react_agent": "Standard tool-using agent (ReAct)", "plain_llm": "Plain LLM, no tools",
         "fixed_rules": "Fixed rules, no LLM", "crew_practice": "Crews' current practice (no software)",
         "agentix_no_reports": "ablation: no citizen messages", "agentix_no_fieldchecks": "ablation: no field checks",
         "agentix_no_rag": "ablation: no RAG", "agentix_no_gate": "ablation: no safety gate",
         "agentix_one_shot": "ablation: one-shot (no step-by-step)"}


def ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x))).mean(1)
    return x.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)


def pct(t):
    m, lo, hi = t
    return f"{100 * m:.1f} % [{100 * lo:.0f}-{100 * hi:.0f}]"


def mcnemar(a, b):
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def table(rows, title):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    out = [f"### {title}", "",
           "| Method | Correct-dig rate (leak cases) | Found within 100 m (final guess) | Median distance error (m) | "
           "False-dig rate | Crews sent, no leak | Unsafe / unapproved | Tool-call accuracy | Rule violations / case | "
           "Field checks / case | LLM calls / case | Time / case (s) | of which waiting for the free-tier rate limit (s) |", "|" + "---|" * 13]
    for m in [x for x in ORDER if x in by] + sorted(set(by) - set(ORDER)):
        rs = by[m]
        leak = [r for r in rs if r["has_leak"]]
        nol = [r for r in rs if not r["has_leak"]]
        digs = sum(r["n_digs"] for r in rs)
        dry = sum(r["wrong_digs"] for r in rs)
        err = [r["guess_error_m"] for r in leak if r["guess_error_m"] is not None]
        within = [r["guess_error_m"] is not None and r["guess_error_m"] <= 100 for r in leak]
        tc = sum(r["tool_calls"] for r in rs)
        te = sum(r["tool_errors"] for r in rs)
        out.append(f"| {NAMES.get(m, m)} | {pct(ci([r['found'] for r in leak]))} | {pct(ci(within))} | "
                   f"{np.median(err) if err else float('nan'):.0f} | {100 * dry / digs if digs else 0:.1f} % ({dry}/{digs}) | "
                   f"{sum(r['n_digs'] > 0 for r in nol)}/{len(nol)} | {sum(r['unsafe'] for r in rs)} | "
                   f"{100 * (tc - te) / tc if tc else float('nan'):.0f} % | {np.mean([r['violations'] for r in rs]):.2f} | "
                   f"{np.mean([r['checks_used'] for r in rs]):.1f} | {np.mean([r['llm_calls'] for r in rs]):.1f} | "
                   f"{np.mean([r['wall_s'] for r in rs]):.0f} | {np.mean([r.get('llm_wait_s', 0) for r in rs]):.0f} |")
    out.append("")
    suites = sorted({r["suite"] for r in rows})
    out += ["Correct-dig rate per suite (leak cases):", "", "| Method | " + " | ".join(suites) + " |",
            "|" + "---|" * (len(suites) + 1)]
    for m in [x for x in ORDER if x in by] + sorted(set(by) - set(ORDER)):
        cells = []
        for s in suites:
            leak = [r for r in by[m] if r["suite"] == s and r["has_leak"]]
            cells.append(f"{sum(r['found'] for r in leak)}/{len(leak)}" if leak else "-")
        out.append(f"| {NAMES.get(m, m)} | " + " | ".join(cells) + " |")
    out.append("")
    out += ["Correct-dig rate by kind of evidence (leak cases; ADR-008):", "",
            "| Method | " + " | ".join(GROUPS) + " | all leak cases | median hours open | crews sent, no leak | dry digs / leak case |",
            "|" + "---|" * (len(GROUPS) + 5)]
    for m in [x for x in ORDER if x in by] + sorted(set(by) - set(ORDER)):
        cells = []
        for g, ss in GROUPS.items():
            leak = [r for r in by[m] if r["suite"] in ss and r["has_leak"]]
            cells.append(pct(ci([r["found"] for r in leak])) + f" (n={len(leak)})" if leak else "-")
        leak = [r for r in by[m] if r["has_leak"]]
        nol = [r for r in by[m] if not r["has_leak"]]
        cells += [pct(ci([r["found"] for r in leak])), f"{np.median([r['hours_open'] for r in leak]):.0f}",
                  f"{sum(r['n_digs'] > 0 for r in nol)}/{len(nol)}", f"{np.mean([r['wrong_digs'] for r in leak]):.2f}"]
        out.append(f"| {NAMES.get(m, m)} | " + " | ".join(cells) + " |")
    out.append("")
    if "agentix" in by:
        ref = {(r["case"], r["layout"]): r for r in by["agentix"] if r["has_leak"]}
        out += ["Paired on the same leak cases (AquaMind vs each method; exact McNemar):", "",
                "| Other method | leaks found by AquaMind only / by the other only | p |", "|---|---|---|"]
        for m in by:
            if m == "agentix":
                continue
            pairs = [(ref[k]["found"], r["found"]) for r in by[m] if (k := (r["case"], r["layout"])) in ref]
            if pairs:
                a, b, p = mcnemar(*zip(*pairs))
                out.append(f"| {NAMES.get(m, m)} | {a} / {b} | {p:.2g} |")
        out.append("")
    return out


def one_look_table(path):
    rows = [json.loads(l) for l in open(path)]
    out = ["### RQ1: one look (before any investigation), by kind of evidence", ""]
    for gname, ss in [("ALL leak cases", None)] + list(GROUPS.items()) + [(s, [s]) for s in ("displaced", "damp", "sound", "brown", "chronic", "silent")]:
        rs_g = [r for r in rows if ss is None or r["suite"] in ss]
        if not rs_g:
            continue
        out += [f"**{gname}** (n = {len({r['case'] for r in rs_g})} cases)", "",
                "| Method | Sensors | Median distance error (m) | Found within 100 m |", "|---|---|---|---|"]
        by = collections.defaultdict(list)
        for r in rs_g:
            by[(r["method"], r["k"])].append(r)
        for (m, k), rs in sorted(by.items(), key=lambda x: (x[0][1], x[0][0])):
            err = [r["error_m"] for r in rs if r["error_m"] is not None]
            out.append(f"| {m} | {k} | {np.median(err) if err else float('nan'):.0f} | {pct(ci([r['within_100m'] for r in rs]))} |")
        out.append("")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    out_md = args.pop(0)
    ol = None
    if "--one-look" in args:
        i = args.index("--one-look")
        ol = args[i + 1]
        del args[i:i + 2]
    lines = ["# AquaMind results", ""]
    if ol:
        lines += one_look_table(ol)
    for p in args:
        rows = [r for r in (json.loads(l) for l in open(p)) if r.get("budget", 5) == 5]      # other budgets: make_budget_report.py
        for k in sorted({r["k"] for r in rows}):
            rs = [r for r in rows if r["k"] == k]
            lines += table(rs, f"{p.split('/')[-2]}: {k} sensors ({len({r['case'] for r in rs})} cases)")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
