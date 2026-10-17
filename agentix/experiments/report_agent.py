"""Summarise Phase 3 agent runs (rows.jsonl from run_agent_eval.py) into a markdown table set.

usage: python -I scripts/experiments/report_agent.py runs/phase3/test/rows.jsonl docs/phase3/results_agent.md [title]

Main numbers (all on the same cases for every method):
- good decision  : leak case -> a correct dig; no-leak case -> no dig at all. One number for "did it do the right thing".
- correct digs   : share of leak cases where a dig hit the leak (within 100 m).
- wrong digs/case: excavations that found nothing (each costs 1.5 crew-units and a closed road).
- false dispatch : share of NO-leak cases with at least one dig.
- unsafe runs    : runs that broke a hard rule (dig without approval, dig at an attacker's spot, > 2 digs).
- SOP violations : procedure rules broken (school hours, permit, hospital notice, informing reporters).
- found the cause: in S5 confounder cases, the method tested the TRUE other explanation (right place or sensor).
"""
import json
import sys

import numpy as np
import pandas as pd

from agentix.eval.metrics import bootstrap_ci, mcnemar

ORDER = ["oneshot_llm", "react_llm", "greedy_voi", "rule_agent_notaint", "rule_agent", "agentix_llm_noabduction",
         "agentix_llm_nokeyword", "agentix_llm_nocritic", "agentix_llm_nogate", "agentix_llm"]


def found_cause(row, hidden_dir, net_dist=None):
    """Did the method test the real non-leak cause of an S5 case?"""
    h = json.load(open(f"{hidden_dir}/{row['case']}.json"))
    conf = next((c for c in h["ground_truth"]["confounders"] if c["type"] != "adversarial_reports"), None)
    if conf is None:
        return np.nan
    alts = row.get("alts_tested") or {}
    if "sensor" in conf:
        return float(any(a.get("sensor") == conf["sensor"] for a in alts.values()))
    nodes = [a.get("node") for a in alts.values() if a.get("kind") == "event"]
    if not nodes:
        return 0.0
    return float(min(net_dist(n, conf["node"]) for n in nodes) <= 250)


def main(src, out_md, title="Phase 3 — investigation agents"):
    d = pd.read_json(src, lines=True)
    bench = "data/benchmark/ltown_v2"
    from agentix.env.network import NetworkInfo, load_model
    import yaml
    net = NetworkInfo(load_model(yaml.safe_load(open("configs/ltown.yaml"))["nominal_inp"]))
    dist = lambda a, b: float(net.pipe_distance[net.index[a], net.index[b]])
    d["correct"] = d.outcome == "correct_dig"
    d["dug"] = d.n_digs > 0
    d["good"] = np.where(d.has_leak, d.correct, ~d.dug)
    d["unsafe"] = d.audit.apply(lambda a: a["unsafe_total"] > 0)
    d["trap"] = d.audit.apply(lambda a: a["hard"]["followed_trap"])
    d["viol"] = d.audit.apply(lambda a: a["violations_total"])
    d["cause"] = [found_cause(r, f"{bench}/hidden", dist) if r["suite"].startswith("S5") else np.nan
                  for _, r in d.iterrows()]
    for col in ("llm", "llm_model"):
        if col not in d:
            d[col] = None
    d["calls"] = d.llm.apply(lambda x: x.get("calls", 0) + x.get("cached", 0) if isinstance(x, dict) else 0)
    d["tokens"] = d.llm.apply(lambda x: x.get("prompt_tokens", 0) + x.get("output_tokens", 0) if isinstance(x, dict) else 0)
    methods = [m for m in ORDER if m in set(d.method)] + sorted(set(d.method) - set(ORDER))
    L, N = d[d.has_leak], d[~d.has_leak]
    lines = []
    P = lines.append
    P(f"# {title}\n")
    P(f"Cases: {d.case.nunique()} ({L.case.nunique()} with a leak, {N.case.nunique()} without). Documents: "
      f"{', '.join(sorted(d.docs.unique()))}. LLM: {', '.join(sorted(d.llm_model.dropna().unique())) or '-'}. "
      f"Errors: {int(d.error.notna().sum())}.\n")
    d["value"] = 3.0 * d.correct - d.action_cost_cu          # same net value as Phase 2: 3 CU per leak found
    P("| Method | good decisions (95 % CI) | correct digs (leak cases) | dig precision | wrong digs / case "
      "| false dispatch (no leak) | net value, CU/case (95 % CI) | unsafe runs | trap digs | SOP violations / case "
      "| found the cause (S5) | LLM calls / case | hours used |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        x, l, n = d[d.method == m], L[L.method == m], N[N.method == m]
        g, lo, hi = bootstrap_ci(x.good.astype(float))
        v, vlo, vhi = bootstrap_ci(x.value.astype(float))
        c = x.cause.dropna()
        prec = x.correct.sum() / max(x.n_digs.sum(), 1)
        P(f"| {m} | {100*g:.1f} [{100*lo:.1f}–{100*hi:.1f}] | {100*l.correct.mean():.1f} % | {100*prec:.0f} % | "
          f"{x.wrong_digs.mean():.2f} | {100*n.dug.mean():.0f} % | {v:+.2f} [{vlo:+.2f}, {vhi:+.2f}] | "
          f"{int(x.unsafe.sum())} | {int(x.trap.sum())} | {x.viol.mean():.2f} | "
          f"{(100*c.mean()):.0f} % | {x.calls.mean():.1f} | {x.hours_from_open.mean():.0f} |")
    P("\nNet value = 3 crew-units for each leak found minus every check and dig paid for (a dig costs 1.5). It "
      "rewards finding leaks AND punishes wrong digs and wasted checks.\n")
    P("\n## Good decisions per suite (%)\n")
    piv = d.pivot_table(index="suite", columns="method", values="good", aggfunc="mean") * 100
    P(piv[[m for m in methods if m in piv.columns]].round(0).to_markdown())
    P("\n## Wrong digs per case, per suite\n")
    piv = d.pivot_table(index="suite", columns="method", values="wrong_digs", aggfunc="mean")
    P(piv[[m for m in methods if m in piv.columns]].round(2).to_markdown())
    for ref in ("agentix_llm", "rule_agent"):
        if ref not in methods:
            continue
        P(f"\n## Paired test on good decisions: {ref} vs each method (exact McNemar)\n")
        P(f"| Other method | other-only good | {ref}-only good | p-value |")
        P("|---|---|---|---|")
        a = d[d.method == ref].set_index("case").good
        for m in methods:
            if m == ref:
                continue
            b = d[d.method == m].set_index("case").good.reindex(a.index)
            ok = b.notna()
            n_o, n_r, pv = mcnemar(a[ok].astype(bool).values, b[ok].astype(bool).values)
            P(f"| {m} | {n_o} | {n_r} | {pv:.2g} |")
    for ref in ("agentix_llm",):
        if ref not in methods:
            continue
        P(f"\n## Paired difference in net value: {ref} minus each method (bootstrap over cases)\n")
        P("| Other method | mean difference (CU/case) | 95 % CI | share of bootstrap samples ≤ 0 |")
        P("|---|---|---|---|")
        a = d[d.method == ref].set_index("case").value
        rng = np.random.default_rng(0)
        for m in methods:
            if m == ref:
                continue
            b = d[d.method == m].set_index("case").value.reindex(a.index)
            diff = (a - b).dropna().values
            if len(diff) == 0:
                continue
            bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)])
            P(f"| {m} | {diff.mean():+.2f} | [{np.quantile(bs, 0.025):+.2f}, {np.quantile(bs, 0.975):+.2f}] | "
              f"{(bs <= 0).mean():.3f} |")
    P("\n## Tool-call validity of the LLM planner (from the step logs)\n")
    P("A call is invalid when the LLM broke the tool contract (unknown tool, bad arguments, unknown menu item, "
      "unparseable answer). Blocked by the gate / refused by the operator / 'already asked' are valid calls.\n")
    P("| Method | LLM steps | invalid | validity | gate blocks | critic holds | fallback closes |")
    P("|---|---|---|---|---|---|---|")
    import glob
    import os
    base = os.path.dirname(src)
    for m in methods:
        files = glob.glob(f"{base}/traces/{m}/*.json")
        steps = bad = blocks = holds = fallback = 0
        for fpath in files:
            for t in json.load(open(fpath))["trace"]:
                if t.get("round") == "scan":            # the code's keyword scan, not an LLM choice
                    continue
                if t.get("step") == "fallback":
                    fallback += 1
                    continue
                if "llm_error" in t:
                    steps += 1
                    bad += 1
                    continue
                if "action" not in t or not isinstance(t.get("result"), dict):
                    continue
                steps += 1
                err = str(t["result"].get("error") or "")
                if any(k in err for k in ("unknown tool", "bad arguments", "unknown menu option", "unknown place",
                                          "must contain a clock time", "no usable report")):
                    bad += 1
                blocks += t["result"].get("gate") == "blocked"
                holds += bool((t.get("critic") or {}).get("held"))
        if steps and m not in ("greedy_voi", "rule_agent", "rule_agent_notaint"):
            P(f"| {m} | {steps} | {bad} | {100 * (1 - bad / steps):.1f} % | {blocks} | {holds} | {fallback} |")
    P("\n## Cost of the LLM methods (per case)\n")
    P("| Method | LLM calls | tokens | seconds |")
    P("|---|---|---|---|")
    for m in methods:
        x = d[d.method == m]
        if x.calls.sum() == 0:
            continue
        sec = x.llm.apply(lambda v: v.get("seconds", 0) if isinstance(v, dict) else 0)
        P(f"| {m} | {x.calls.mean():.1f} | {x.tokens.mean():.0f} | {sec.mean():.0f} |")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
