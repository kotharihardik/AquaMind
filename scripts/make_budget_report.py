"""Value of citizen complaints vs the field-check budget, per kind of evidence (ADR-008/010) -> Markdown.
usage: python -I scripts/make_budget_report.py out.md rows.jsonl [more rows.jsonl ...]   (rows from scripts/run_eval.py, --max-checks sets the budget)
"""
import collections, json, sys
import numpy as np

GROUPS = {"visible (sanity)": ["visible"], "displaced pool": ["displaced"], "damp ground": ["damp"], "night sound": ["sound"],
          "brown water": ["brown"], "CHRONIC (sensors blind)": ["chronic"], "SCADA only": ["silent"], "trap": ["trap"],
          "NON-OBVIOUS pooled": ["displaced", "damp", "sound", "brown", "chronic", "silent"]}
NAMES = {"agentix_code": "autopilot, complaints used", "agentix_no_reports_code": "autopilot, NO complaints (sensors + field checks)",
         "agentix_code_brown": "autopilot + brown-water footprints", "agentix_code_brown_nowork": "  ... without the works log",
         "agentix": "AgentiX (LLM)", "fixed_rules": "fixed rules", "crew_practice": "crews' current practice"}

def ci(x, n=1000):
    x = np.asarray(x, float); rng = np.random.default_rng(0)
    b = rng.choice(x, size=(n, len(x))).mean(1)
    return x.mean(), np.quantile(b, .025), np.quantile(b, .975)

def main(out, files):
    rows = [json.loads(l) for f in files for l in open(f)]
    for r in rows:
        r.setdefault("budget", 5)
    lines = ["# Value of citizen complaints, by field-check budget and kind of evidence (benchmark v3)", ""]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["method"], r["budget"], r["k"])].append(r)
    methods = [m for m in NAMES if any(k[0] == m for k in by)]
    lines += ["## Correct-dig rate (leak cases), 5 sensors, by number of field checks allowed", "",
              "| Method | checks | " + " | ".join(GROUPS) + " | dry digs / leak case | crews on no-leak |", "|" + "---|" * (len(GROUPS) + 4)]
    for budget in sorted({k[1] for k in by}):
        for m in methods:
            rs = by.get((m, budget, 5), [])
            if not rs:
                continue
            leak = [r for r in rs if r["has_leak"]]; nol = [r for r in rs if not r["has_leak"]]
            cells = []
            for g, ss in GROUPS.items():
                x = [r["found"] for r in leak if r["suite"] in ss]
                cells.append(f"{100*np.mean(x):.0f} % ({sum(x)}/{len(x)})" if x else "-")
            lines.append(f"| {NAMES[m]} | {budget} | " + " | ".join(cells) + f" | {np.mean([r['wrong_digs'] for r in leak]):.2f} | {sum(r['n_digs']>0 for r in nol)}/{len(nol)} |")
    lines += ["", "## Gain from complaints (autopilot with minus without), percentage points, with the 95 % bootstrap interval of the paired difference", "",
              "| checks | " + " | ".join(GROUPS) + " |", "|" + "---|" * (len(GROUPS) + 1)]
    for budget in sorted({k[1] for k in by}):
        a = {(r["case"]): r for r in by.get(("agentix_code", budget, 5), []) if r["has_leak"]}
        b = {(r["case"]): r for r in by.get(("agentix_no_reports_code", budget, 5), []) if r["has_leak"]}
        cells = []
        for g, ss in GROUPS.items():
            d = [a[c]["found"] - b[c]["found"] for c in a if c in b and a[c]["suite"] in ss]
            if d:
                m, lo, hi = ci(d); cells.append(f"{100*m:+.0f} [{100*lo:+.0f}, {100*hi:+.0f}]")
            else:
                cells.append("-")
        lines.append(f"| {budget} | " + " | ".join(cells) + " |")
    lines += ["", "## Correct-dig rate at 5 field checks by number of sensors", "", "| Method | sensors | all leak cases | NON-OBVIOUS pooled | crews on no-leak |", "|---|---|---|---|---|"]
    for m in methods:
        for k in (3, 5, 10):
            rs = by.get((m, 5, k), [])
            if rs:
                leak = [r for r in rs if r["has_leak"]]; nol = [r for r in rs if not r["has_leak"]]
                no = [r["found"] for r in leak if r["suite"] in GROUPS["NON-OBVIOUS pooled"]]
                lines.append(f"| {NAMES[m]} | {k} | {100*np.mean([r['found'] for r in leak]):.1f} % | {100*np.mean(no):.1f} % | {sum(r['n_digs']>0 for r in nol)}/{len(nol)} |")
    open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
