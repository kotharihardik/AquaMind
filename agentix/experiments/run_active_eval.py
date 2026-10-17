"""Phase 2b: run investigation policies inside CaseEnv on a split -> per-case JSONL (outcome, cost, time, digs).

usage: python -I scripts/experiments/run_active_eval.py <bench_dir> <dev|test> <out.jsonl> [layout,...] [policy,...]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import yaml

_W = {}
POLICIES = ["dig_now", "dig_if_confident", "one_more_sensor", "acoustic_sweep", "random_actions", "greedy_voi",
            "oracle"]


def _init(bench):
    from agentix.engine.fusion import EngineConfig, EvidenceEngine
    from agentix.engine.pressure_lik import SignatureLibrary
    from agentix.engine.report_lik import ReportModel
    from agentix.env.network import NetworkInfo, load_model
    from agentix.reader.gazetteer_index import GazetteerIndex
    from agentix.reader.rule_reader import RuleReader

    cfg = yaml.safe_load(open("configs/ltown.yaml"))
    net = NetworkInfo(load_model(cfg["nominal_inp"]))
    gaz_json = json.load(open(f"{bench}/gazetteer/gazetteer_ltown.json"))
    gaz = GazetteerIndex(gaz_json, net)
    eng = EvidenceEngine(net, SignatureLibrary(cfg["signature_cache"]), ReportModel(net, gaz), EngineConfig())
    _W.update(net=net, gaz=gaz, gaz_json=gaz_json, eng=eng, reader=RuleReader(gaz), bench=bench)


def _run(job):
    from agentix.active import policies as P
    from agentix.active.investigation import Investigation
    from agentix.env.actions import CaseEnv, case_seed
    from agentix.env.loader import load_layouts, load_public_case
    from agentix.eval.truth import CaseTruth

    cid, lname, pname = job
    bench, net = _W["bench"], _W["net"]
    layout = load_layouts(bench)[lname]
    case = load_public_case(bench, cid, layout)
    env = CaseEnv(bench, cid, net, layout, "configs/costs.yaml", _W["gaz_json"], seed=case_seed(cid, lname))
    inv = Investigation(_W["eng"], _W["reader"], _W["gaz"], seed=7, operator=False)     # Phase-2 setting
    base, _, kw = pname.partition(":")          # e.g. "greedy_voi:budget_cu=1.6;dig_threshold=0.4" (policies split by |)
    params = {k: float(v) for k, v in (x.split("=") for x in kw.split(";") if x)}
    pol = {"dig_now": P.DigNow, "dig_if_confident": P.DigIfConfident, "one_more_sensor": P.OneMoreSensor,
           "acoustic_sweep": P.AcousticSweep, "random_actions": P.RandomActions, "greedy_voi": P.GreedyVOI,
           "oracle": P.Oracle}[base](**params)
    t0 = time.time()
    try:
        pol.run(inv, case, env)
        err = None
    except Exception as e:  # keep the run going; record the failure
        err = repr(e)
    tr = CaseTruth(bench, cid, net)
    s = env.score()
    acts = [e["action"] for e in env.log]
    return {"case": cid, "layout": lname, "k": len(layout), "policy": pname, **tr.strata(), **s,
            "n_digs": acts.count("request_dig"), "actions": acts, "error": err,
            "sec": round(time.time() - t0, 2)}


if __name__ == "__main__":
    from agentix.env.loader import list_cases

    bench, split, out = sys.argv[1:4]
    layouts = sys.argv[4].split(",") if len(sys.argv) > 4 else ["info_k5", "info_k10"]
    pols = sys.argv[5].split("|") if len(sys.argv) > 5 else POLICIES
    keep = (0, 1, 2) if split == "dev" else (3, 4, 5, 6, 7, 8, 9)
    cases = [c for c in list_cases(bench) if int(c[-4:]) % 10 in keep]
    jobs = [(c, l, p) for l in layouts for p in pols for c in cases]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(4, initializer=_init, initargs=(bench,)) as ex, open(out, "w") as f:
        for k, r in enumerate(ex.map(_run, jobs, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            if k % 200 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")
