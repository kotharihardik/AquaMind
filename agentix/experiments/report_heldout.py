"""Held-out wording test: the same 28 documented-confounder test cases, read once with the original document
wording and once with re-worded documents (docs_heldout/). Does each method still find the true cause?

usage: python -I scripts/experiments/report_heldout.py runs/phase3/test_heldout/rows.jsonl docs/phase3/results_heldout.md
"""
import json
import sys

import numpy as np
import pandas as pd
import yaml

from agentix.env.network import NetworkInfo, load_model

ORDER = ["react_llm", "rule_agent", "agentix_llm_nokeyword", "agentix_llm"]
NAMES = {"react_llm": "ReAct LLM (raw text)", "rule_agent": "Rule brain (keyword rules)",
         "agentix_llm_nokeyword": "AgentiX, LLM reading only", "agentix_llm": "AgentiX (keyword scan + LLM)"}


def main(src, out_md):
    d = pd.read_json(src, lines=True)
    net = NetworkInfo(load_model(yaml.safe_load(open("configs/ltown.yaml"))["nominal_inp"]))

    def cause(r):
        h = json.load(open(f"data/benchmark/ltown_v2/hidden/{r['case']}.json"))
        conf = next(c for c in h["ground_truth"]["confounders"] if c["type"] != "adversarial_reports")
        alts = r["alts_tested"] or {}
        if "sensor" in conf:
            return float(any(a.get("sensor") == conf["sensor"] for a in alts.values()))
        nodes = [a["node"] for a in alts.values() if a.get("kind") == "event" and a.get("node")]
        return float(bool(nodes) and min(net.pipe_distance[net.index[n], net.index[conf["node"]]] for n in nodes) <= 250)

    d["cause"] = [cause(r) for _, r in d.iterrows()]
    d["good"] = np.where(d.has_leak, d.outcome == "correct_dig", d.n_digs == 0)
    d["value"] = 3.0 * (d.outcome == "correct_dig") - d.action_cost_cu
    L = []
    P = L.append
    P("# Phase 3 — held-out wording test (documents written differently)\n")
    P(f"{d.case.nunique()} frozen test cases from the four suites whose explanation is written in a document "
      "(S5 flushing, valve works, sensor fault, leak + sensor fault). Each method ran twice on the same cases: "
      "with the original document wording, and with the held-out wording (`scripts/build/build_heldout_docs.py`: "
      "washout / fire plug / sluice / PRV / transducer / data-quality flag; time ranges like 'Start 14:15, finish "
      "approx 18:55'). The held-out wording was written after the keyword rules, on purpose with other words.\n")
    P("**Found the cause** = the method tested the true explanation (right place within 250 m, or the right "
      "sensor).\n")
    P("| Method | found the cause: original → held-out | good decisions: original → held-out | "
      "wrong digs/case: original → held-out | net value: original → held-out |")
    P("|---|---|---|---|---|")
    for m in [x for x in ORDER if x in set(d.method)]:
        a, b = d[(d.method == m) & (d.docs == "docs")], d[(d.method == m) & (d.docs == "docs_heldout")]
        if a.empty or b.empty:
            continue
        found = ("n/a (has no hypothesis-testing tool)" if m == "react_llm"
                 else f"{100*a.cause.mean():.0f} % → {100*b.cause.mean():.0f} %")
        P(f"| {NAMES[m]} | {found} | "
          f"{100*a.good.mean():.0f} % → {100*b.good.mean():.0f} % | {a.wrong_digs.mean():.2f} → {b.wrong_digs.mean():.2f} | "
          f"{a.value.mean():+.2f} → {b.value.mean():+.2f} |")
    P("\n## Found the cause, per suite (held-out wording)\n")
    h = d[d.docs == "docs_heldout"]
    piv = h.pivot_table(index="suite", columns="method", values="cause", aggfunc="mean") * 100
    P(piv[[m for m in ORDER if m in piv.columns and m != "react_llm"]].round(0).to_markdown())
    P("\n## Paired: AgentiX vs rule brain on held-out wording (found the cause)\n")
    a = h[h.method == "agentix_llm"].set_index("case").cause
    b = h[h.method == "rule_agent"].set_index("case").cause.reindex(a.index)
    from agentix.eval.metrics import mcnemar
    agent_only, rule_only, pv = mcnemar(b.astype(bool).values, a.astype(bool).values)
    P(f"AgentiX-only {agent_only} cases, rule-only {rule_only} cases, exact McNemar p = {pv:.2g}.\n")
    open(out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:3])
