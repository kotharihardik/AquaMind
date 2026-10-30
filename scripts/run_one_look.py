"""RQ1: one look (no investigation) for every method x layout x leak case -> distance error and found within 100 m.

usage: python -I scripts/run_one_look.py --split dev|test --out runs/x/one_look.jsonl [--reader rules|llm]
Layouts: the 3 random layouts + the spread layout for 3, 5 and 10 sensors (the same for every method).
"""
import argparse
import json
import os

import numpy as np

from aquamind.baselines.rq1 import METHODS, one_look
from aquamind.eval.runner import World
from aquamind.eval.splits import cases_for
from aquamind.methods import make_reader

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark/agentix_ltown")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--reader", default="rules")
    ap.add_argument("--text", default="realistic")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = World(a.bench)
    llm = None
    if a.reader == "llm":
        from aquamind.llm.client import LLM
        llm = LLM()
    reader = make_reader(a.reader, w, llm)
    names = cases_for(a.bench, a.split)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for name in names:
            reads = None
            for layout in sorted(w.layouts):
                case, env = w.open(name, layout, a.text)
                if not env.leak:
                    break
                reads = reads or reader.read_many(case.reports)
                dist = env._dist
                for m in METHODS:
                    j, _ = one_look(w, case, reads, m)
                    d = float(dist[j]) if j is not None else None
                    f.write(json.dumps({"case": name, "suite": env.hid["suite"], "layout": layout, "k": len(env.layout),
                                        "method": m, "error_m": d, "within_100m": d is not None and d <= 100,
                                        "surfaced": bool(env.leak.get("surfaced"))}) + "\n")
            print(name, flush=True)
