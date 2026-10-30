"""Run one method on one case and score it (evaluation harness: the only place truth meets methods' outputs)."""
from __future__ import annotations

import os
import time
import traceback

from .. import regime
from ..case import load_layouts, load_public_case
from ..rag.retriever import load_corpus
from ..safety.rulebook import RuleBook
from ..search.investigation import LeakSearch, SearchConfig
from ..sim.field import CaseEnv
from ..sim.world import SimTown
from ..town.assets import Town


class World:
    """Everything that is loaded once per process."""

    def __init__(self, bench: str, config: str = "configs/ltown.yaml"):
        regime.tools_for_agent()
        self.bench = bench
        self.town = Town(config)
        self.sim = SimTown(self.town)
        self.layouts = load_layouts(bench)
        rec = f"{bench}/records.md" if os.path.exists(f"{bench}/records.md") else self.town.records_path    # a benchmark may bring its own town archive
        self.docs = load_corpus(self.town.sop_path, rec)
        self.rules = RuleBook("configs/rules.yaml", self.town.places, self.town.net, self.town.sections)
        self._search = {}

    def search(self, **kw) -> LeakSearch:
        key = tuple(sorted(kw.items()))
        if key not in self._search:
            self._search[key] = LeakSearch(self.town, SearchConfig(**kw))
        return self._search[key]

    def open(self, name: str, layout: str, text: str = "realistic", max_checks: int | None = None):
        nodes = self.layouts[layout]
        case = load_public_case(self.bench, name, nodes, text)
        env = CaseEnv(self.bench, name, self.town.net, self.town.sections, self.town.places, self.sim.ctx, nodes,
                      text_variant=text, max_checks=max_checks)
        regime.apply(case, env, name)
        return case, env


def run_case(world: World, method, name: str, layout: str, text: str = "realistic", max_checks: int | None = None) -> dict:
    case, env = world.open(name, layout, text, max_checks)
    t0 = time.time()
    err, info = None, {}
    try:
        info = method.run_case(world, env, case) or {}
    except Exception:                                       # a crash is a failed case, never a lost run
        err = traceback.format_exc()[-1500:]
        if not env.done:
            env.close("hand_over", "method crashed")
    sc = env.score()
    hid = env.hid
    trap = (hid.get("trap") or {}).get("target_node")
    audit = world.rules.audit(env, trap)
    guess = info.get("final_guess")
    leak = env.leak
    row = {"method": method.name, "case": name, "suite": hid["suite"], "layout": layout, "k": len(env.layout),
           "trigger": hid["trigger"], "q_lps": round(leak["mean_lps"], 2) if leak else 0.0,
           "surfaced": bool(leak and leak.get("surfaced")), **sc,
           "dry_on_no_leak": sc["n_digs"] if not leak else 0,
           "guess_error_m": (round(env.distance_to_leak(guess), 1) if (leak and guess) else None),
           "regime": ",".join(sorted(regime.active())), "unsafe": audit["unsafe_total"], "violations": audit["violations_total"], "audit": audit,
           "tool_calls": info.get("tool_calls", 0), "tool_errors": info.get("tool_errors", 0),
           "llm_calls": info.get("llm_calls", 0), "llm_cached": info.get("llm_cached", 0),
           "llm_seconds": round(info.get("llm_seconds", 0.0), 1), "llm_wait_s": round(info.get("llm_wait_s", 0.0), 1), "tokens": info.get("tokens", 0),
           "llm_throttled": info.get("llm_throttled", 0), "reader": info.get("reader"), "reader_errors": info.get("reader_errors", 0),
           "reader_messages": info.get("reader_messages", 0), "steps": info.get("steps", 0), "wall_s": round(time.time() - t0, 1), "error": err}
    row["trace"] = info.get("trace")
    return row
