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
    ap.add_argument("--workers", type=int, default=3, help="number of parallel worker processes (default 3 to match 3 API keys)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suites", default="")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth case (a stratified subset for the expensive LLM baselines)")
    ap.add_argument("--regime", default="", help="stress regimes (agentix/regime.py): survey, plastic, plastic_blind, drift, outage")
    ap.add_argument("--max-checks", type=int, default=None, help="field-check budget (default 5); the value-of-evidence curve uses 2 and 3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reader", default="rules", help="rules | llm | local: the same reader for every method")
    ap.add_argument("--out", default="", help="optional explicit jsonl output file (if omitted, logs to evaluation_logs)")
    ap.add_argument("--log-dir", default="logs/evaluation", help="directory for versioned evaluation run logs")
    ap.add_argument("--title", default="", help="optional title for this evaluation run")
    ap.add_argument("--fresh-log", action="store_true", help="start a NEW log version (v2, v3, ...) instead of appending to the newest log of the same evaluation")
    ap.add_argument("--log-layouts", default="", help="(rarely needed) log several layouts in one folder; by default each layout has its own log folder")
    ap.add_argument("--log-rows", default="", help="glob(s), comma separated, of results files of this evaluation: finished cases found there but missing from the log are added to it")
    a = ap.parse_args()
    os.environ["AGENTIX_REGIME"] = a.regime

    from agentix.eval.logger import LoggerSet
    run_title = a.title or f"{a.methods} on {a.split} ({a.layouts})"
    # every case of the benchmark in one fixed order: a case's number in the log is its position here (stable over resumes and splits)
    order = []
    for sp in ("dev", "val", "test"):
        try:
            order += cases_for(a.bench, sp)
        except FileNotFoundError:
            pass
    methods, layouts = a.methods.split(","), a.layouts.split(",")
    # one log (five files) per method and layout; the same evaluation (bench, method, reader, text, regime, budget) resumes it, whatever the split
    logs = LoggerSet(methods, layouts, a.log_dir, a.bench, f"reader={a.reader} | text={a.text} | regime={a.regime or '-'} | checks={a.max_checks or 5}",
                     run_title=run_title, case_order=order, resume=not a.fresh_log)
    out_path = a.out or f"runs/eval/{'+'.join(methods)[:80]}_{'_'.join(layouts)}_{a.split}.jsonl"     # full result rows incl. trace: the source of scripts/make_metrics.py

    names = cases_for(a.bench, a.split)
    if a.suites:
        names = [n for n in names if n.split("-")[1] in a.suites.split(",")]
    if a.stride > 1:
        names = names[:: a.stride]
    if a.limit:
        names = names[: a.limit]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done, prior = set(), []
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            prior.append(r)
            done.add((r["method"], r["case"], r["layout"], r.get("budget", 5), r.get("regime", "")))
    import glob
    wanted, lay_set = set(methods), set(layouts)
    for pat in [g for g in a.log_rows.split(",") if g]:
        for fpath in sorted(glob.glob(pat)):
            if os.path.abspath(fpath) == os.path.abspath(out_path):
                continue
            for line in open(fpath):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("method") in wanted and r.get("layout") in lay_set and not r.get("error"):
                    prior.append(r)
    added = logs.backfill([r for r in prior if not r.get("error")])     # cases finished earlier, in any results file of this evaluation, join the same logs
    jobs = [(m, n, l, a.text, a.max_checks) for l in layouts for m in methods for n in names
            if (m, n, l, a.max_checks or 5, ",".join(sorted(x for x in a.regime.split(",") if x))) not in done]
    print("\n".join(logs.describe()) + f"\n{added} cases backfilled from the results files; benchmark has {len(order)} cases", flush=True)
    print(f"Starting run: {run_title}", flush=True)
    print(f"{len(jobs)} runs to do ({len(done)} already done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.bench, a.model, a.reader)) as ex, open(out_path, "a") as f:
        for k, fut in enumerate(as_completed([ex.submit(_job, j) for j in jobs]), 1):
            r = fut.result()
            logs.log_case(r)                                    # cases, table, metrics, trace and rows of THIS method and layout, rewritten in order
            f.write(json.dumps(r, default=str) + "\n")          # the full row incl. trace
            f.flush()
            print(f"{k}/{len(jobs)} {r['method']:24s} {r['case']:24s} {r['layout']:12s} {r['outcome']:10s} "
                  f"digs={r['n_digs']} checks={r['checks_used']} {r['wall_s']:.0f}s"
                  + (" ERROR" if r["error"] else ""), flush=True)

    logs.finalize(metadata={"bench": a.bench, "split": a.split, "methods": a.methods, "layouts": a.layouts,
                            "suites": a.suites, "regime": a.regime, "reader": a.reader})
    print(f"Done in {time.time() - t0:.0f} s", flush=True)
    for (m, lay), lg in logs.loggers.items():
        print(f"[{lg.version}] {m} / {lay}: cases {lg.txt_file} | table {lg.overview_file.name} | metrics {lg.metrics_file.name} | trace {lg.trace_file.name}")
    print(f"Master ledger: {a.log_dir}/master_index.json")
