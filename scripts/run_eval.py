"""Run methods on a split and write one JSON row per (method, case, layout). Resumable: finished rows are skipped.

usage: python -I scripts/run_eval.py --bench data/benchmark/agentix_ltown --split dev|test --methods a,b
                                     --layouts spread_k5 [--workers 2] [--limit N] --out runs/x/rows.jsonl
The test split is read from test_frozen.json and its hash is checked (the list was frozen before any final run).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from agentix.eval.splits import cases_for

_W = {}


def _init(bench, model, reader):
    from agentix.eval.runner import World
    from agentix.methods import registry
    _W["world"], _W["methods"] = World(bench), registry(model, reader)


def _job(job):
    from agentix.eval.runner import run_case
    method, name, layout, text, max_checks = job
    r = run_case(_W["world"], _W["methods"][method], name, layout, text, max_checks)
    r["budget"] = max_checks or 5
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark/agentix_ltown")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--layouts", default="spread_k5")
    ap.add_argument("--text", default="realistic")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suites", default="")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth case (a stratified subset for the expensive LLM baselines)")
    ap.add_argument("--regime", default="", help="stress regimes (agentix/regime.py): survey, plastic, plastic_blind, drift, outage")
    ap.add_argument("--max-checks", type=int, default=None, help="field-check budget (default 5); the value-of-evidence curve uses 2 and 3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reader", default="rules", help="rules | llm | local: the same reader for every method")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["AGENTIX_REGIME"] = a.regime
    names = cases_for(a.bench, a.split)
    if a.suites:
        names = [n for n in names if n.split("-")[1] in a.suites.split(",")]
    if a.stride > 1:
        names = names[:: a.stride]
    if a.limit:
        names = names[: a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["method"], r["case"], r["layout"], r.get("budget", 5), r.get("regime", "")))
    jobs = [(m, n, l, a.text, a.max_checks) for l in a.layouts.split(",") for m in a.methods.split(",") for n in names
            if (m, n, l, a.max_checks or 5, ",".join(sorted(x for x in a.regime.split(",") if x))) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.bench, a.model, a.reader)) as ex, open(a.out, "a") as f:
        for k, fut in enumerate(as_completed([ex.submit(_job, j) for j in jobs]), 1):
            r = fut.result()
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            print(f"{k}/{len(jobs)} {r['method']:24s} {r['case']:24s} {r['layout']:12s} {r['outcome']:10s} "
                  f"digs={r['n_digs']} checks={r['checks_used']} {r['wall_s']:.0f}s"
                  + (" ERROR" if r["error"] else ""), flush=True)
    print(f"done in {time.time() - t0:.0f} s")
