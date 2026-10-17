"""RQ1 static evaluation: every method x every layout x every case of a split -> per-case JSONL.

usage: python -I scripts/experiments/run_static_eval.py <bench_dir> <split: dev|test> <out.jsonl> [layout,layout,...]
Split rule (fixed, ADR-006): dev = case index % 10 in {0,1,2}; test = the rest.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

DEFAULT_LAYOUTS = ["random_k3_s0", "engineering_k3", "info_k3", "random_k5_s0", "engineering_k5", "info_k5",
                   "random_k10_s0", "engineering_k10", "info_k10", "all33"]
_W = {}


def _init(bench):
    from agentix.baselines.static import (BicikDS, LilaStyle, MuGate, NaiveReport, ReportHotspot,
                                          SensitivityMatch)
    from agentix.engine.fusion import EngineConfig, EvidenceEngine
    from agentix.engine.pressure_lik import SignatureLibrary
    from agentix.engine.report_lik import ReportModel
    from agentix.env.network import NetworkInfo, load_model
    from agentix.reader.gazetteer_index import GazetteerIndex
    from agentix.reader.rule_reader import RuleReader

    cfg = yaml.safe_load(open("configs/ltown.yaml"))
    net = NetworkInfo(load_model(cfg["nominal_inp"]))
    lib = SignatureLibrary(cfg["signature_cache"])
    gaz = GazetteerIndex(json.load(open(f"{bench}/gazetteer/gazetteer_ltown.json")), net)
    rep = ReportModel(net, gaz)
    rep_flat = ReportModel(net, gaz, downhill=False)
    E = lambda **kw: EvidenceEngine(net, lib, rep, EngineConfig(**kw))
    methods = {
        "ours_full": (E(), "rule"),
        "ours_full_oracle_reader": (E(), "oracle"),
        "ours_pressure_only": (E(use_reports=False), "rule"),
        "ours_reports_only": (E(use_pressure=False), "rule"),
        "ours_no_downhill": (EvidenceEngine(net, lib, rep_flat, EngineConfig()), "rule"),
        "ours_no_recency": (E(tau_age_h=1e6), "rule"),
        "naive_first_report": (NaiveReport(net, rep, "first"), "rule"),
        "naive_latest_report": (NaiveReport(net, rep, "latest"), "rule"),
        "report_hotspot": (ReportHotspot(net, rep), "rule"),
        "sensitivity_match": (SensitivityMatch(net, lib), "rule"),
        "lila_style": (LilaStyle(net), "rule"),
        "bicik_ds": (BicikDS(net, lib, rep), "rule"),
        "mu_gate": (MuGate(net, lib), "rule"),
    }
    _W.update(net=net, lib=lib, gaz=gaz, reader=RuleReader(gaz), methods=methods, bench=bench,
              cover=(net.pipe_distance <= 90).astype(float))


def _run_layout(args):
    from agentix.baselines.static import Decision
    from agentix.engine.fusion import EngineResult
    from agentix.engine.preprocess import build_evidence
    from agentix.env.loader import load_layouts, load_public_case
    from agentix.eval.metrics import search_rank
    from agentix.eval.truth import CaseTruth

    lname, cases = args
    bench, net = _W["bench"], _W["net"]
    layout = load_layouts(bench)[lname]
    rows = []
    for cid in cases:
        case = load_public_case(bench, cid, layout)
        tr = CaseTruth(bench, cid, net)
        ev = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s)
        outs = {"rule": [_W["reader"].read(r) for r in case.reports]}
        gold = tr.public_gold()
        outs["oracle"] = [gold[r["report_id"]] for r in case.reports]
        dist = tr.dist_to_leak_all() if tr.has_leak else None
        for mname, (m, rname) in _W["methods"].items():
            t0 = time.time()
            out = m.infer(case, outs[rname], evidence=ev) if hasattr(m, "infer") else m.decide(case, outs[rname], ev)
            if isinstance(out, EngineResult):
                dec = Decision(out.dig_node, out.confidence, out.posterior,
                               {"p_leak": out.p_leak, "coverage": out.dig_coverage, **{k: v for k, v in
                                out.diagnostics.items() if k in ("q_hat_lps", "eff_n_nodes", "pressure_misfit")}})
            else:
                dec = out
            node = dec.dig_node or dec.info.get("dig_if_forced")
            row = {"case": cid, "layout": lname, "k": len(layout), "family": lname.split("_k")[0],
                   "method": mname, **tr.strata(), "acted": dec.dig_node is not None,
                   "confidence": float(dec.confidence), "ms": round((time.time() - t0) * 1000, 1)}
            if tr.has_leak and node is not None:
                d = float(dist[net.index[node]])
                row.update({"dist_m": d, "hit100": d <= 100, "hit300": d <= 300,
                            "search_rank": (search_rank(dec.scores, dist, _W["cover"]) if dec.scores is not None
                                            else (1 if d <= 100 else 11))})
            row.update({k: v for k, v in dec.info.items() if k in ("p_leak", "coverage", "q_hat_lps", "eff_n_nodes",
                                                                    "pressure_misfit")})
            rows.append(row)
    return rows


if __name__ == "__main__":
    from agentix.env.loader import list_cases

    bench, split, out = sys.argv[1:4]
    layouts = sys.argv[4].split(",") if len(sys.argv) > 4 else DEFAULT_LAYOUTS
    keep = (0, 1, 2) if split == "dev" else (3, 4, 5, 6, 7, 8, 9)
    cases = [c for c in list_cases(bench) if int(c[-4:]) % 10 in keep]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=_init, initargs=(bench,)) as ex, open(out, "w") as f:
        for rows in ex.map(_run_layout, [(l, cases) for l in layouts]):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            print(f"layout {rows[0]['layout']} done ({len(rows)} rows) {time.time() - t0:.0f}s", flush=True)
    print(f"{len(cases)} cases x {len(layouts)} layouts in {time.time() - t0:.0f}s -> {out}")
