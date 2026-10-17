"""Summarise RQ1 static results into markdown tables (test split; thresholds chosen on dev).

usage: python -I scripts/experiments/report_static.py runs/phase2/static_dev.jsonl runs/phase2/static_test.jsonl docs/phase2/results_static.md
"""
import sys

import numpy as np
import pandas as pd

from agentix.eval.metrics import IsotonicCalibrator, aurc, bootstrap_ci, mcnemar, precision_at_coverage

dev_p, test_p, out_md = sys.argv[1:4]
dev, test = pd.read_json(dev_p, lines=True), pd.read_json(test_p, lines=True)
FAM = ["random", "engineering", "info"]          # 3 layouts per k for the main table
LEAK_SUITES = ["S1_standard", "S2_silent", "S3_misleading", "S7_traps"]
METHODS = ["naive_first_report", "naive_latest_report", "report_hotspot", "sensitivity_match", "lila_style",
           "bicik_ds", "mu_gate", "ours_pressure_only", "ours_reports_only", "ours_no_downhill", "ours_no_recency",
           "ours_full", "ours_full_oracle_reader"]
lines = []
P = lines.append


def fmt(m, lo, hi):
    return f"{100*m:.1f} [{100*lo:.1f}–{100*hi:.1f}]"


def leak_rows(df, k=None):
    d = df[df.suite.isin(LEAK_SUITES) & df.family.isin(FAM)]
    if k is not None:
        d = d[d.k == k]
    return d


P("# Phase 2 — RQ1 static results (test split)\n")
P(f"Test cases: {test.case.nunique()} ({test[test.has_leak].case.nunique()} with a leak). Layouts per k: random, "
  "engineering, info (3 × cases per cell). Numbers = % with 95 % bootstrap CI. Hit = dig within 100 m of the leak "
  "(pipe distance). Methods that may abstain (mu_gate) are scored on the node they WOULD dig (forced).\n")

# ---- table 1: hit@100 by k, all leak suites
P("## Table 1 — correct dig (hit@100) by number of sensors, all leak cases\n")
P("| Method | k=3 | k=5 | k=10 | all 33 |")
P("|---|---|---|---|---|")
for m in METHODS:
    cells = []
    for k in (3, 5, 10):
        d = leak_rows(test, k)
        x = d[d.method == m].hit100.fillna(False).astype(float)
        cells.append(fmt(*bootstrap_ci(x)))
    d33 = test[(test.layout == "all33") & test.suite.isin(LEAK_SUITES) & (test.method == m)].hit100.fillna(False)
    cells.append(fmt(*bootstrap_ci(d33.astype(float))))
    P(f"| {m} | " + " | ".join(cells) + " |")

# ---- table 2: visible vs hidden at k=5
P("\n## Table 2 — k=5: visible vs hidden leaks, and within-300 m / top-3 search areas\n")
P("| Method | hit@100 visible | hit@100 hidden | hit@300 all | found in top-3 areas | median miss (m) |")
P("|---|---|---|---|---|---|")
d5 = leak_rows(test, 5)
for m in METHODS:
    d = d5[d5.method == m]
    vis = d[d.visible].hit100.fillna(False).astype(float)
    hid = d[~d.visible].hit100.fillna(False).astype(float)
    top3 = (d.search_rank <= 3).astype(float)
    P(f"| {m} | {100*vis.mean():.1f} | {100*hid.mean():.1f} | {100*d.hit300.fillna(False).mean():.1f} | "
      f"{100*top3.mean():.1f} | {d.dist_m.median():.0f} |")

# ---- table 3: by suite at k=5 for key methods
P("\n## Table 3 — k=5 hit@100 by suite\n")
key = ["naive_latest_report", "report_hotspot", "bicik_ds", "ours_pressure_only", "ours_reports_only", "ours_full",
       "ours_full_oracle_reader"]
P("| Method | " + " | ".join(LEAK_SUITES) + " |")
P("|---|" + "---|" * len(LEAK_SUITES))
for m in key:
    d = d5[d5.method == m]
    P(f"| {m} | " + " | ".join(f"{100*d[d.suite == s].hit100.fillna(False).mean():.1f}" for s in LEAK_SUITES) + " |")

# ---- table 4: paired significance (ours_full vs each baseline), k=5, all leak suites
P("\n## Table 4 — paired McNemar test, ours_full vs baseline (k=5, leak cases)\n")
P("| Baseline | baseline-only wins | ours-only wins | p-value |")
P("|---|---|---|---|")
ours = d5[d5.method == "ours_full"].set_index(["case", "layout"]).hit100.fillna(False)
for m in METHODS:
    if m == "ours_full":
        continue
    b = d5[d5.method == m].set_index(["case", "layout"]).hit100.fillna(False).reindex(ours.index).fillna(False)
    n_b, n_o, p = mcnemar(ours.values, b.values)
    P(f"| {m} | {n_b} | {n_o} | {p:.2g} |")

# ---- table 5: selective decisions with calibrated confidence (isotonic fit on DEV, applied to TEST)
def ece(conf, hit, bins=(0, .05, .1, .2, .3, .5, 1.01)):
    conf, hit = np.asarray(conf, float), np.asarray(hit, float)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            e += m.sum() * abs(conf[m].mean() - hit[m].mean())
    return e / len(conf)


P("\n## Table 5 — honest confidence and selective digging (k=5 and 10; calibration fit on DEV, scored on TEST)\n")
P("Precision@X% = share of correct digs if we dig only the X % most confident leak cases. False dispatch = share of "
  "no-leak cases whose calibrated confidence exceeds the threshold that gave >= 40 % precision on dev.\n")
P("| Method | k | ECE raw | ECE calibrated | precision@10% | precision@20% | AURC | false dispatch @dev-40%-thr |")
P("|---|---|---|---|---|---|---|---|")
for m in ["ours_full", "ours_full_oracle_reader", "ours_reports_only", "ours_pressure_only", "bicik_ds", "report_hotspot"]:
    for k in (5, 10):
        dv = dev[(dev.method == m) & (dev.k == k) & dev.family.isin(FAM)]
        dl = dv[dv.suite.isin(LEAK_SUITES)]
        cal = IsotonicCalibrator().fit(dl.confidence.values, dl.hit100.fillna(False).values)
        cdev = cal(dl.confidence.values)
        okd = dl.hit100.fillna(False).values
        order = np.argsort(-cdev)
        prec = np.cumsum(okd[order]) / np.arange(1, len(okd) + 1)
        good = np.where(prec >= 0.4)[0]
        thr = cdev[order][good.max()] if len(good) else np.inf
        tv = test[(test.method == m) & (test.k == k) & test.family.isin(FAM)]
        tl = tv[tv.suite.isin(LEAK_SUITES)]
        hit = tl.hit100.fillna(False).values
        ct = cal(tl.confidence.values)
        nl = tv[tv.suite == "S4_noleak"]
        fd = (cal(nl.confidence.values) >= thr).mean() if len(nl) else float("nan")
        P(f"| {m} | {k} | {ece(tl.confidence, hit):.3f} | {ece(ct, hit):.3f} | "
          f"{100*precision_at_coverage(ct, hit, 0.10):.1f} % | {100*precision_at_coverage(ct, hit, 0.20):.1f} % | "
          f"{aurc(ct, hit):.3f} | {100*fd:.1f} % |")
d = test[(test.method == "mu_gate") & test.family.isin(FAM)]
for k in (5, 10):
    dl = d[(d.k == k) & d.suite.isin(LEAK_SUITES)]
    a = dl.acted
    P(f"| mu_gate (own contract: digs {100*a.mean():.1f} % of leak cases) | {k} | – | – | precision of its digs: "
      f"{100*dl[a].hit100.fillna(False).mean() if a.any() else float('nan'):.1f} % | – | – | "
      f"{100*d[(d.k == k) & (d.suite == 'S4_noleak')].acted.mean():.1f} % |")
P("\n## Runtime (ms per case, mean over all rows)\n")
P(test.groupby("method").ms.mean().round(1).to_frame().to_markdown())
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
