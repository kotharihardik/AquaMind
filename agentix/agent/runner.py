"""Run one 'brain' on one case and score it (outcome, digs, safety audit, LLM cost, full step log).

Everything a brain needs is built here the same way for every method, so comparisons stay fair:
same case, same sensor layout, same toolbox, same budget, same operator, same rules.
"""
from __future__ import annotations

import json
import time

import yaml

from ..active.investigation import Investigation
from ..engine.fusion import EngineConfig, EvidenceEngine
from ..engine.pressure_lik import SignatureLibrary
from ..engine.report_lik import ReportModel
from ..env.actions import CaseEnv, case_seed
from ..env.loader import load_layouts, load_public_case
from ..env.network import NetworkInfo, load_model
from ..eval.truth import CaseTruth
from ..rag.retriever import load_case_corpus
from ..reader.gazetteer_index import GazetteerIndex
from ..reader.rule_reader import RuleReader
from ..safety.rules import RuleBook
from .toolbox import Toolbox

SOP = "data/corpus/ltown/sop_leak_response.md"


class World:
    """The shared, case-independent parts (network, engine, reader, rules). Build once per process."""

    def __init__(self, bench: str, config: str = "configs/ltown.yaml"):
        cfg = yaml.safe_load(open(config))
        self.bench = bench
        self.net = NetworkInfo(load_model(cfg["nominal_inp"]))
        self.gaz_json = json.load(open(f"{bench}/gazetteer/gazetteer_ltown.json"))
        self.gaz = GazetteerIndex(self.gaz_json, self.net)
        self.eng = EvidenceEngine(self.net, SignatureLibrary(cfg["signature_cache"]), ReportModel(self.net, self.gaz),
                                  EngineConfig())
        self.reader = RuleReader(self.gaz)
        self.rules = RuleBook("configs/rules.yaml", self.gaz_json, self.net)
        self.layouts = load_layouts(bench)

    def open_case(self, cid: str, layout: str, budget_cu: float = 1.6, use_rag: bool = True, use_gate: bool = True,
                  docs_dir: str = "docs", taint: bool = True, allow_alternatives: bool = True):
        lay = self.layouts[layout]
        case = load_public_case(self.bench, cid, lay)
        env = CaseEnv(self.bench, cid, self.net, lay, "configs/costs.yaml", self.gaz_json, seed=case_seed(cid, layout))
        inv = Investigation(self.eng, self.reader, self.gaz, seed=7, taint=taint)
        ret = load_case_corpus(SOP, f"{self.bench}/cases/{cid}", docs_dir=docs_dir)
        tb = Toolbox(inv, env, case, ret, self.rules, budget_cu=budget_cu, use_rag=use_rag, use_gate=use_gate,
                     allow_alternatives=allow_alternatives)
        return tb


def score(world: World, tb: Toolbox, cid: str, layout: str) -> dict:
    """Outcome + safety audit for a finished run (uses hidden truth: evaluation only)."""
    env = tb.env
    tr = CaseTruth(world.bench, cid, world.net)
    trap = next((c.get("target_node") for c in tr.h["ground_truth"].get("confounders", [])
                 if c["type"] == "adversarial_reports"), None)
    audit = world.rules.audit(env, trap_node=trap)
    acts = [e["action"] for e in env.log]
    alts = {k: {"kind": a.get("kind"), "node": a.get("node"), "sensor": a.get("sensor")}
            for k, a in tb.inv.state["alts_tested"].items()}         # every explanation ever tested
    alts_final = sorted(tb.board.alts)
    return {"case": cid, "layout": layout, **tr.strata(), **env.score(), "n_digs": acts.count("request_dig"),
            "actions": acts, "audit": audit, "alts_tested": alts, "alts_on_board": alts_final,
            "excluded_sensors": tb.inv.state.get("excluded_sensors", []), "tainted": sorted(tb.inv.state["tainted"]),
            "final_explanations": {k: round(float(v), 3) for k, v in tb.board.explanations().items()},
            "final_p_success": round(tb.p_success(), 3)}


def run_brain(world: World, brain, cid: str, layout: str, **open_kw) -> dict:
    t0 = time.time()
    tb = world.open_case(cid, layout, **open_kw)
    llm = getattr(brain, "llm", None)
    u0 = llm.usage.as_dict() if llm else None
    try:
        trace, err = brain.run(tb), None
    except Exception as e:                       # keep a long evaluation going; record the failure
        trace, err = [], repr(e)[:300]
    row = {"brain": brain.name, **score(world, tb, cid, layout), "error": err, "sec": round(time.time() - t0, 1),
           "n_steps": len(trace)}
    if llm:
        u1 = llm.usage.as_dict()
        row["llm"] = {k: round(u1[k] - u0[k], 2) for k in u1}
        row["llm_model"] = llm.name
    return row, trace
