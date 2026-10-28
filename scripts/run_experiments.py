"""Unified Experiment Runner for AgentiX Final Evaluation Phase.

Orchestrates baseline and ablation experiments with modular support for:
- Fast Non-LLM execution across 3, 5, 10 sensors (completes in minutes, zero API cost).
- LLM Pilot on 5-sensor configuration (spread_k5) for baselines (ReAct, Plain LLM) and ablations.
- Full multi-sensor evaluation (spread_k3, spread_k5, spread_k10).
- Automatic versioned logging in `logs/evaluation/run_vXXX/` (report.md, summary.json, cases.jsonl).

Usage Examples:
  # 1. Fast non-LLM baselines & ablations across 3, 5, 10 sensors:
  .venv/bin/python -I scripts/run_experiments.py --mode fast-non-llm --split dev

  # 2. Pilot LLM evaluation on 5 sensors (5-case sanity check or full):
  .venv/bin/python -I scripts/run_experiments.py --mode llm-pilot-5sensor --split dev --limit 5

  # 3. Dedicated Hidden Leaks evaluation (chronic leaks where sensors are blind):
  .venv/bin/python -I scripts/run_experiments.py --mode fast-non-llm --suites chronic

  # 4. RQ1 One-Look First-Look Comparison:
  .venv/bin/python -I scripts/run_experiments.py --mode rq1-first-look
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="AgentiX Final Evaluation Orchestrator")
    ap.add_argument("--mode", choices=["fast-non-llm", "llm-pilot-5sensor", "rq1-first-look", "custom"],
                    default="fast-non-llm", help="experiment execution mode")
    ap.add_argument("--bench", default="data/benchmark/agentix_master" if os.path.exists("data/benchmark/agentix_master") else "data/benchmark/agentix_ltown")
    ap.add_argument("--split", default="dev", choices=["dev", "test"], help="dataset split")
    ap.add_argument("--layouts", default=None, help="comma-separated layouts (default depends on mode)")
    ap.add_argument("--methods", default=None, help="comma-separated methods (default depends on mode)")
    ap.add_argument("--suites", default="", help="filter by suites, e.g. 'chronic' for hidden leaks")
    ap.add_argument("--limit", type=int, default=0, help="limit number of cases (0 = all)")
    ap.add_argument("--workers", type=int, default=3, help="parallel worker processes (default 3 to match 3 API keys)")
    ap.add_argument("--reader", default="rules", choices=["rules", "llm", "local"], help="reader implementation")
    ap.add_argument("--log-dir", default="logs/evaluation", help="directory for versioned evaluation run logs")
    ap.add_argument("--title", default="", help="custom run title for log report")

    a = ap.parse_args()

    py_bin = sys.executable

    if a.mode == "rq1-first-look":
        out_file = "runs/rq1/one_look_latest.jsonl"
        cmd = [
            py_bin, "-I", "scripts/run_one_look.py",
            "--bench", a.bench,
            "--split", a.split,
            "--reader", a.reader,
            "--out", out_file,
        ]
        print(f"Running RQ1 One-Look First Look Evaluation on {a.bench} ({a.split})...")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"RQ1 One-look evaluation complete. Saved to {out_file}")
        return

    # Determine default layouts and methods by mode
    if a.mode == "fast-non-llm":
        layouts = a.layouts or "spread_k3,spread_k5,spread_k10"
        methods = a.methods or "classical_direct_inversion,fixed_rules,crew_practice,agentix_code"
        title = a.title or f"Fast Non-LLM Baselines & Code Ablation across 3, 5, 10 Sensors ({a.split})"
    elif a.mode == "llm-pilot-5sensor":
        layouts = a.layouts or "spread_k5"
        methods = a.methods or "react_agent,plain_llm,agentix_crew"
        title = a.title or f"LLM Pilot Evaluation on 5 Sensors ({a.split})"
    else: # custom
        layouts = a.layouts or "spread_k5"
        methods = a.methods or "agentix_crew"
        title = a.title or f"Custom Evaluation ({a.split})"

    cmd = [
        py_bin, "-I", "scripts/run_eval.py",
        "--bench", a.bench,
        "--split", a.split,
        "--methods", methods,
        "--layouts", layouts,
        "--workers", str(a.workers),
        "--reader", a.reader,
        "--log-dir", a.log_dir,
        "--title", title,
    ]
    if a.limit > 0:
        cmd.extend(["--limit", str(a.limit)])
    if a.stride > 1:
        cmd.extend(["--stride", str(a.stride)])
    if a.suites:
        cmd.extend(["--suites", a.suites])

    print("=" * 70)
    print(f" AgentiX Final Evaluation Phase — Mode: {a.mode.upper()}")
    print(f" Layouts: {layouts} (3, 5, 10 sensor configs)")
    print(f" Methods: {methods}")
    print(f" Split: {a.split} | Suites: {a.suites or 'all'} | Limit: {a.limit or 'all'}")
    print(f" Log Destination: {a.log_dir}")
    print("=" * 70)
    print("Executing command:", " ".join(cmd), "\n")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
