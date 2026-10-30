"""Every method of the evaluation, by its proposal name (Sections 10-11). Each has run_case(world, env, case)."""
from __future__ import annotations

from typing import Any

from .agent.toolbox import Toolbox
from .reader.rules import RuleReader


class NullReader:
    """No language understanding at all (the no-AI ablation, ADR-017): a message is a message, its text is never interpreted. Real
    leak-finding software does not read citizen text (proposal Sections 4-5); a regex written against this benchmark's own templates would
    give code a reading ability it has nowhere else."""
    name = "none"

    def read_many(self, reports):
        from .reader.schema import repair_read
        return {r["report_id"]: repair_read(None) for r in reports}

    def read(self, report):
        return self.read_many([report])[report["report_id"]]

    def read_replies(self, items):
        from .reader.schema import EMPTY_REPLY
        return [dict(EMPTY_REPLY) for _ in items]


def make_reader(kind: str, world, llm=None):
    if kind == "none":
        return NullReader()
    if kind == "rules":
        return RuleReader(world.town.places)
    if kind == "llm":
        from .reader.llm_reader import LLMReader
        return LLMReader(llm)
    if kind == "local":
        from .reader.local_reader import shared_local_reader
        return shared_local_reader()
    raise ValueError(kind)


class ToolboxMethod:
    """A brain on the AquaMind toolbox (the LLM agent, its ablations, or the code-only planner)."""

    def __init__(self, name: str, brain_factory, reader: str = "llm", search_kw: dict | None = None,
                 llm_factory=None, **tb_kw):
        self.name, self.brain_factory, self.reader_kind = name, brain_factory, reader
        self.search_kw, self.llm_factory, self.tb_kw = search_kw or {}, llm_factory, tb_kw

    def run_case(self, world, env, case) -> dict:
        llm = self.llm_factory() if self.llm_factory else None
        reader_llm = llm
        reader = make_reader(self.reader_kind, world, reader_llm)
        tb = Toolbox(env, case, world.search(**self.search_kw), reader, world.docs, world.rules, **self.tb_kw)
        brain = self.brain_factory(llm)
        brain.run(tb)
        if not (tb.done or env.done):
            tb.close("hand_over", "no decision", force=True)
        u = llm.usage if llm else None
        calls = [t for t in tb.trace if t["tool"] not in ("open",)]
        bad_reads = sum(bool(r.get("reader_error")) for r in tb.reads.values())          # a reader that failed leaves an EMPTY read, never a guess
        return {"final_guess": tb.final_guess(), "reader": getattr(reader, "name", self.reader_kind), "reader_errors": bad_reads,
                "reader_messages": len(tb.reads), "trace": tb.trace + getattr(brain, "turns", []),
                "tool_calls": len(calls), "tool_errors": sum(not t["result"].get("ok", True) for t in calls),
                "steps": len(getattr(brain, "turns", [])) or len(calls),
                "llm_calls": u.calls if u else 0, "llm_cached": u.cached if u else 0,
                "llm_seconds": u.seconds if u else 0.0, "llm_wait_s": u.waited if u else 0.0, "tokens": (u.prompt_tokens + u.output_tokens) if u else 0,
                "llm_throttled": u.throttled if u else 0}


def registry(agent_model: str | None = None, reader: str = "rules") -> dict[str, Any]:
    """Every method of the evaluation by name. `reader` (rules | llm | local) is the same for every method of a run (fair comparison).

    Imports are NOT guarded: a broken module must stop the run, not silently drop a method from the comparison."""
    from .agent.crew import Crew
    from .ablations.registry import get_ablation_registry
    from .baselines.registry import get_baseline_registry
    from .llm.client import AGENT_MODEL, LLM
    model = agent_model or AGENT_MODEL
    llm = lambda: LLM(model, max_tokens=1200)

    m: dict[str, Any] = {}
    m.update(get_baseline_registry(llm))                                   # proposal Section 10
    m.update(get_ablation_registry(llm, reader=reader))                    # proposal Section 11 + crew-role ablations

    # AquaMind itself (ADR-014/015/016/020): every agentic method searches ONLY hypothesised junctions (no 782-junction table, no planner)
    H = {"hypothesis_mode": True, "reply_confirms": False}
    kw = {"reader": reader, "llm_factory": llm, "search_kw": H, "use_planner": False, "min_checks_close": 3}
    m["agentix_crew"] = ToolboxMethod("agentix_crew", lambda l: Crew(l), **kw)
    m["agentix_crew_no_override"] = ToolboxMethod("agentix_crew_no_override", lambda l: Crew(l, chance_policy=False), **kw)
    return m


def get_method_info(name: str) -> dict[str, Any]:
    """Return category, proposal section, and LLM requirement for any method."""
    from .baselines.registry import BASELINE_METADATA
    from .ablations.registry import ABLATION_METADATA

    if name in BASELINE_METADATA:
        return BASELINE_METADATA[name]
    if name in ABLATION_METADATA:
        return ABLATION_METADATA[name]
    for k, v in ABLATION_METADATA.items():
        if v.get("alias") == name:
            return v
    if name in ("agentix_crew", "agentix_crew_no_override", "agentix_crew_v2"):
        return {
            "category": "agentix_proposed",
            "description": "AquaMind multi-agent dialectical investigation system behind code gate",
            "needs_llm": True,
            "proposal_section": "7 (Proposed Idea & Architecture)",
            "sensor_configs": ["spread_k3", "spread_k5", "spread_k10"],
        }
    return {
        "category": "other",
        "description": name,
        "needs_llm": False,
        "proposal_section": "Unknown",
        "sensor_configs": ["spread_k5"],
    }
