"""Summarise Phase 2b active-investigation results (test split) into markdown.

usage: python -I scripts/experiments/report_active.py runs/phase2/active_test.jsonl docs/phase2/results_active.md
"""
import sys

import numpy as np
import pandas as pd

from agentix.eval.metrics import bootstrap_ci, mcnemar

src, out_md = sys.argv[1:3]
d = pd.read_json(src, lines=True)
d["correct"] = d.outcome == "correct_dig"
ORDER = ["dig_now", "dig_if_confident", "one_more_sensor", "random_actions", "acoustic_sweep", "greedy_voi", "oracle"]
L, N = d[d.has_leak], d[~d.has_leak]
lines = []
P = lines.append
P("# Phase 2b — active investigation results (test split)\n")
P(f"Test cases: {d.case.nunique()} ({L.case.nunique()} with a leak, {N.case.nunique()} no-leak controls). "
  f"Errors: {int(d.error.notna().sum())}. Costs in crew-units (CU = 1 crew-day; a dig = 1.5 CU). "
  "Correct = a dig within 100 m of the leak before the 3-day horizon. Dig precision = correct digs / all digs.\n")
for lay in sorted(d.layout.unique()):
    P(f"## Layout {lay}\n")
    P("| Policy | correct digs (95 % CI) | wrong digs / case | dig precision | gives up | no-leak false dispatch | "
      "action cost (CU) | hours used | water lost (m³) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for p in ORDER:
        l = L[(L.layout == lay) & (L.policy == p)]
        n = N[(N.layout == lay) & (N.policy == p)]
        if l.empty:
            continue
        m, lo, hi = bootstrap_ci(l.correct.astype(float))
        prec = l.correct.sum() / max(l.n_digs.sum(), 1)
        P(f"| {p} | {100*m:.1f} [{100*lo:.1f}–{100*hi:.1f}] | {l.wrong_digs.mean():.2f} | {100*prec:.0f} % | "
          f"{100*(l.outcome == 'abstain').mean():.0f} % | {100*(n.n_digs > 0).mean():.0f} % | "
          f"{l.action_cost_cu.mean():.2f} | {l.hours_from_open.mean():.0f} | {l.water_lost_m3.median():.0f} |")
    P("")
lay = "info_k5"
P(f"## Visible vs hidden leaks ({lay})\n")
P("| Policy | correct, visible | correct, hidden | wrong digs/case, visible | wrong digs/case, hidden |")
P("|---|---|---|---|---|")
for p in ORDER:
    l = L[(L.layout == lay) & (L.policy == p)]
    v, h = l[l.visible], l[~l.visible]
    P(f"| {p} | {100*v.correct.mean():.1f} | {100*h.correct.mean():.1f} | {v.wrong_digs.mean():.2f} | {h.wrong_digs.mean():.2f} |")
P(f"\n## Paired tests on correct digs ({lay}): greedy_voi vs each policy\n")
P("| Other policy | other-only correct | greedy-only correct | p (McNemar) | wrong digs: greedy vs other |")
P("|---|---|---|---|---|")
g = L[(L.layout == lay) & (L.policy == "greedy_voi")].set_index("case")
for p in ORDER:
    if p == "greedy_voi":
        continue
    o = L[(L.layout == lay) & (L.policy == p)].set_index("case").reindex(g.index)
    n_o, n_g, pv = mcnemar(g.correct.values, o.correct.fillna(False).values)
    P(f"| {p} | {n_o} | {n_g} | {pv:.2g} | {g.wrong_digs.mean():.2f} vs {o.wrong_digs.mean():.2f} |")
P(f"\n## Which checks the greedy policy used ({lay}, all test cases)\n")
acts = pd.Series([a for x in d[(d.layout == lay) & (d.policy == "greedy_voi")].actions for a in x]).value_counts()
P(acts.to_frame("count").to_markdown())
B = 3.0
P(f"\n## Net value per case ({lay}): {B:.0f} CU per leak found − action cost (includes digs)\n")
P("| Policy | net value (CU/case) |")
P("|---|---|")
for p in ORDER:
    x = d[(d.layout == lay) & (d.policy == p)]
    P(f"| {p} | {(B * x.correct - x.action_cost_cu).mean():.2f} |")
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
