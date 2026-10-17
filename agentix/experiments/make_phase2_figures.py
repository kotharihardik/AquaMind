"""Phase 2 figures (static PNGs for docs). Palette = dataviz reference categorical slots (validated, light mode);
low-contrast slots carry direct labels; tables with exact numbers live in docs/phase2/*.md (table view).

usage: python -I scripts/experiments/make_phase2_figures.py runs/phase2/static_test.jsonl runs/phase2/active_test.jsonl docs/phase2/figures
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

static_p, active_p, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
SURF, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------- Figure 1: one-look localization vs number of sensors
s = pd.read_json(static_p, lines=True)
LEAK = ["S1_standard", "S2_silent", "S3_misleading", "S7_traps"]
fam = ["random", "engineering", "info"]
series = [("ours_full", "Our engine (pressure + reports)"), ("ours_pressure_only", "Our engine, pressure only"),
          ("lila_style", "LILA-style"), ("mu_gate", "Mu-style gate (forced)"),
          ("naive_latest_report", "Naive: dig at newest report")]
xs = [3, 5, 10, 33]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ends = []
for (m, label), col in zip(series, SLOTS):
    ys = []
    for k in xs:
        d = s[(s.method == m) & s.suite.isin(LEAK)]
        d = d[d.layout == "all33"] if k == 33 else d[(d.k == k) & d.family.isin(fam)]
        ys.append(100 * d.hit100.fillna(False).mean())
    ax.plot(range(len(xs)), ys, color=col, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=label)
    ends.append(ys[-1])
placed = []                                    # nudge end labels apart so close values never collide
for v in sorted(ends, reverse=True):
    y = v if not placed or placed[-1] - v >= 0.6 else placed[-1] - 0.6
    placed.append(y)
    ax.annotate(f"{v:.1f}%", (len(xs) - 1, v), xytext=(len(xs) - 1 + 0.08, y), textcoords="data",
                va="center", color=INK2, fontsize=9)
ax.set_xticks(range(len(xs)), ["3", "5", "10", "33 (all)"])
ax.set_xlabel("pressure sensors in the town")
ax.set_ylabel("correct digs (within 100 m), % of leak cases")
ax.set_ylim(0, 14)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("One look is not enough: best static method digs correctly in < 11 % of cases", loc="left",
             fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{out}/fig1_static_hit_vs_sensors.png", dpi=150)
plt.close(fig)

# ---------------- Figure 2: active investigation - correct digs vs wrong digs (two panels, one axis each)
a = pd.read_json(active_p, lines=True)
a = a[(a.layout == "info_k5") & a.has_leak]
order = ["dig_now", "one_more_sensor", "random_actions", "acoustic_sweep", "greedy_voi"]
names = {"dig_now": "Dig now (one look)", "one_more_sensor": "One-more-sensor (2024)", "random_actions": "Random checks",
         "acoustic_sweep": "Always listen, then dig", "greedy_voi": "Ours: greedy value of information"}
corr = [100 * (a[a.policy == p].outcome == "correct_dig").mean() for p in order]
wrong = [a[a.policy == p].wrong_digs.mean() for p in order]
cols = [SLOTS[0] if p == "greedy_voi" else NEUTRAL for p in order]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
y = np.arange(len(order))
for axx, vals, title, fmt in ((axs[0], corr, "Correct digs (% of leak cases)", "{:.0f}%"),
                              (axs[1], wrong, "Wrong digs per leak case", "{:.2f}")):
    axx.barh(y, vals, color=cols, height=0.6, edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, vals):
        axx.text(v, yi, "  " + fmt.format(v), va="center", color=INK2, fontsize=9)
    axx.set_title(title, loc="left", fontsize=10, color=INK)
    axx.grid(axis="x", color=GRID, lw=0.8)
    axx.set_axisbelow(True)
    axx.set_xlim(0, max(vals) * 1.3)
axs[0].set_yticks(y, [names[p] for p in order])
fig.suptitle("Investigating before digging (5 sensors, test cases)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(f"{out}/fig2_active_correct_vs_wrong.png", dpi=150)
plt.close(fig)
print("saved figures to", out)
