"""Phase 3 figures (static PNGs for the docs). Same validated palette as Phase 2: ours = blue slot, other methods
neutral grey (identity also given by the y-axis label, so colour is never the only cue). Exact numbers are in
docs/phase3/results_*.md (the table view).

usage: python -I scripts/experiments/make_phase3_figures.py runs/phase3/test_v2/rows.jsonl docs/phase3/figures \
           [runs/phase3/test_heldout/rows.jsonl]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentix.eval.metrics import bootstrap_ci

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
SURF, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
OURS, WARN = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"oneshot_llm": "One-shot LLM", "react_llm": "ReAct LLM (raw text, no gate)", "greedy_voi": "Greedy VOI (Phase 2)",
         "rule_agent_notaint": "Rule brain, no taint", "rule_agent": "Rule brain (same toolbox)",
         "agentix_llm_noabduction": "AgentiX − abduction", "agentix_llm_nokeyword": "AgentiX − keyword scan",
         "agentix_llm_nocritic": "AgentiX − critic",
         "agentix_llm_nogate": "AgentiX − gate", "agentix_llm": "AgentiX LLM agent (ours)"}
d = pd.read_json(src, lines=True)
d = d[d.docs == "docs"]
d["good"] = np.where(d.has_leak, d.outcome == "correct_dig", d.n_digs == 0)
order = [m for m in NAMES if m in set(d.method)]

# ---------------- Figure 1: good decisions with 95 % CI, wrong digs as a second panel (separate axes, no dual axis)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 0.5 * len(order) + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 1]})
for i, m in enumerate(order):
    x = d[d.method == m]
    g, lo, hi = bootstrap_ci(x.good.astype(float))
    col = OURS if m == "agentix_llm" else NEUTRAL
    a1.plot([100 * lo, 100 * hi], [i, i], color=col, lw=2, solid_capstyle="round")
    a1.plot(100 * g, i, "o", ms=9, color=col, mec=SURF, mew=2)
    a1.annotate(f"{100 * g:.0f}%", (100 * hi, i), xytext=(6, 0), textcoords="offset points", va="center",
                color=INK2, fontsize=9)
    w = x.wrong_digs.mean()
    a2.barh(i, w, height=0.55, color=col)
    a2.annotate(f"{w:.2f}", (w, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
a1.set_yticks(range(len(order)), [NAMES[m] for m in order])
a1.set_xlabel("good decisions, % of cases (95 % CI)\nleak → correct dig; no leak → no dig")
a1.set_xlim(0, 100)
a2.set_xlabel("wrong digs per case (lower is better)")
for a in (a1, a2):
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
fig.suptitle("Same cases, budget and operator: plain LLM agents fail; the safe LLM agent is the most careful",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(f"{out}/fig1_good_decisions.png", dpi=150)
plt.close(fig)

# ---------------- Figure 2: per-suite good decisions, three key methods (small multiples by suite)
keys = [m for m in ("greedy_voi", "rule_agent", "agentix_llm", "react_llm") if m in order]
suites = sorted(d.suite.unique())
fig, ax = plt.subplots(figsize=(10.8, 4.2))
xs = np.arange(len(suites))
wdt = 0.8 / len(keys)
shades = {"agentix_llm": OURS, "rule_agent": "#1baf7a", "greedy_voi": "#eda100", "react_llm": WARN}  # validated slots
SHORT = {"S1_standard": "S1\nstandard", "S2_silent": "S2\nsilent", "S3_misleading": "S3\nmisleading",
         "S4_noleak": "S4\nno leak", "S5_fault_doc": "S5 sensor\nfault (note)", "S5_fault_undoc": "S5 sensor\nfault (none)",
         "S5_flush_doc": "S5\nflushing", "S5_flush_fire": "S5 fire\nbrigade", "S5_leak_fault_doc": "S5 leak +\nfault",
         "S5_valve_doc": "S5\nvalve works", "S7_traps": "S7\ntraps"}
for k, m in enumerate(keys):
    ys = [100 * d[(d.method == m) & (d.suite == s)].good.mean() for s in suites]
    ax.bar(xs + (k - (len(keys) - 1) / 2) * wdt, ys, width=wdt - 0.03, color=shades[m], label=NAMES[m])
ax.set_xticks(xs, [SHORT.get(s, s) for s in suites], fontsize=8)
ax.set_ylabel("good decisions, %")
ax.set_ylim(0, 105)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.12))
fig.tight_layout()
fig.savefig(f"{out}/fig2_per_suite.png", dpi=150)
plt.close(fig)
# ---------------- Figure 3 (optional): held-out wording, "found the true cause" before vs after re-wording
if len(sys.argv) > 3:
    import json as _json
    import yaml
    from agentix.env.network import NetworkInfo, load_model
    h = pd.read_json(sys.argv[3], lines=True)
    net = NetworkInfo(load_model(yaml.safe_load(open("configs/ltown.yaml"))["nominal_inp"]))

    def _cause(r):
        hid = _json.load(open(f"data/benchmark/ltown_v2/hidden/{r['case']}.json"))
        conf = next(c for c in hid["ground_truth"]["confounders"] if c["type"] != "adversarial_reports")
        alts = r["alts_tested"] or {}
        if "sensor" in conf:
            return float(any(a.get("sensor") == conf["sensor"] for a in alts.values()))
        nodes = [a["node"] for a in alts.values() if a.get("kind") == "event" and a.get("node")]
        return float(bool(nodes) and min(net.pipe_distance[net.index[n], net.index[conf["node"]]] for n in nodes) <= 250)

    h["cause"] = [_cause(r) for _, r in h.iterrows()]
    show = [("rule_agent", "Rule brain (keyword rules)", "#1baf7a"),
            ("agentix_llm_nokeyword", "AgentiX, LLM reading only", "#7fb2ea"),
            ("agentix_llm", "AgentiX (keyword scan + LLM)", OURS)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m, label, col in show:
        ys = [100 * h[(h.method == m) & (h.docs == dv)].cause.mean() for dv in ("docs", "docs_heldout")]
        ax.plot([0, 1], ys, color=col, lw=2, marker="o", ms=9, mec=SURF, mew=2)
        ax.annotate(f"{label}  {ys[1]:.0f}%", (1, ys[1]), xytext=(10, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
        ax.annotate(f"{ys[0]:.0f}%", (0, ys[0]), xytext=(-10, 0), textcoords="offset points", va="center",
                    ha="right", color=INK2, fontsize=9)
    ax.set_xticks([0, 1], ["original wording", "held-out wording\n(new words, new time formats)"])
    ax.set_xlim(-0.35, 2.1)
    ax.set_ylim(0, 75)
    ax.set_ylabel("found the true cause, % of 28 test cases")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("When notes are written in unfamiliar words, keyword rules break; the LLM keeps reading",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_heldout_wording.png", dpi=150)
    plt.close(fig)
print("figures written to", out)
