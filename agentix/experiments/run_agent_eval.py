"""Phase 3: run investigation 'brains' (LLM agent, rule brain, baselines, ablations) on the same cases.

Every method gets the same case, sensor layout (info_k5), budget (1.6 crew-units), operator and rules.
Results are appended to <out>/rows.jsonl (one line per case x method; the run can be stopped and resumed) and
the full step log of every run goes to <out>/traces/<method>/<case>.json.

usage:
  python -I scripts/experiments/run_agent_eval.py --split dev --per-suite 4 --methods rule_agent,agentix_llm \
      --out runs/phase3/dev [--docs docs_heldout] [--suites S5_flush_doc,S7_traps] [--workers 3]
The chosen case list is saved to <out>/cases.json the first time and reused afterwards (frozen test set).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SUITES = ["S1_standard", "S2_silent", "S3_misleading", "S4_noleak", "S7_traps", "S5_flush_doc", "S5_flush_fire",
          "S5_valve_doc", "S5_fault_doc", "S5_fault_undoc", "S5_leak_fault_doc"]
LLM_METHODS = {"agentix_llm", "agentix_llm_nocritic", "agentix_llm_noabduction", "agentix_llm_nogate",
               "agentix_llm_nokeyword",
               "react_llm", "oneshot_llm"}
_W = {}


def make_brain(name: str):
    """method name -> (brain, extra options for opening the case)."""
    from agentix.active.policies import GreedyVOI
    from agentix.agent.llm_agent import LLMInvestigator
    from agentix.agent.llm_baselines import OneShotLLM, ReActLLM
    from agentix.agent.rule_agent import RuleInvestigator

    class Greedy:                                   # the Phase 2 policy, unchanged (no taint, no documents)
        name = "greedy_voi"

        def run(self, tb):
            GreedyVOI().run(tb.inv, tb.case, tb.env)
            return []

    llm = _W.get("llm")
    table = {
        "greedy_voi": (Greedy(), {"taint": False}),
        "rule_agent": (RuleInvestigator(), {}),
        "rule_agent_notaint": (RuleInvestigator(), {"taint": False}),
        "agentix_llm": (LLMInvestigator(llm), {}),
        "agentix_llm_nocritic": (LLMInvestigator(llm, use_critic=False, name="agentix_llm_nocritic"), {}),
        "agentix_llm_noabduction": (LLMInvestigator(llm, use_hypotheses=False, name="agentix_llm_noabduction"),
                                    {"allow_alternatives": False}),
        "agentix_llm_nogate": (LLMInvestigator(llm, name="agentix_llm_nogate"), {"use_gate": False}),
        "agentix_llm_nokeyword": (LLMInvestigator(llm, use_keyword_scan=False, name="agentix_llm_nokeyword"), {}),
        "react_llm": (ReActLLM(llm), {}),
        "oneshot_llm": (OneShotLLM(llm), {}),
    }
    brain, kw = table[name]
    brain.name = name
    return brain, kw


def _init(bench, need_llm):
    from agentix.agent.runner import World
    _W["world"] = World(bench)
    if need_llm:
        from agentix.llm.client import LLM
        _W["llm"] = LLM()


def _run(job):
    from agentix.agent.runner import run_brain
    cid, method, docs, out = job
    brain, kw = make_brain(method)
    row, trace = run_brain(_W["world"], brain, cid, "info_k5", docs_dir=docs, **kw)
    row.update({"method": method, "docs": docs})
    os.makedirs(f"{out}/traces/{method}", exist_ok=True)
    tag = "" if docs == "docs" else f"__{docs}"
    with open(f"{out}/traces/{method}/{cid}{tag}.json", "w") as f:
        json.dump({"row": row, "trace": trace}, f, default=str)
    return row


def pick_cases(bench: str, split: str, per_suite: int, suites: list[str]) -> list[str]:
    from agentix.env.loader import list_cases
    keep = (0, 1, 2) if split == "dev" else (3, 4, 5, 6, 7, 8, 9)
    allc = [c for c in list_cases(bench) if int(c[-4:]) % 10 in keep]
    out = []
    for s in suites:
        out += sorted(c for c in allc if f"-{s}-" in c)[:per_suite]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark/ltown_v2")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--per-suite", type=int, default=4)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--methods", required=True)
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    case_file = f"{a.out}/cases.json"
    if os.path.exists(case_file):
        cases = json.load(open(case_file))                  # frozen once chosen
    else:
        cases = pick_cases(a.bench, a.split, a.per_suite, a.suites.split(","))
        json.dump(cases, open(case_file, "w"), indent=1)
    methods = a.methods.split(",")
    done = set()
    rows_path = f"{a.out}/rows.jsonl"
    if os.path.exists(rows_path):
        keep = []
        for line in open(rows_path):
            r = json.loads(line)
            if (r.get("llm") or {}).get("errors", 0) > 0 or r.get("error"):   # LLM call failed or run crashed:
                continue                                                           # run it again
            done.add((r["case"], r["method"], r["docs"]))
            keep.append(line)
        open(rows_path, "w").writelines(keep)
    jobs = [(c, m, a.docs, a.out) for m in methods for c in cases if (c, m, a.docs) not in done]
    print(f"{len(cases)} cases x {len(methods)} methods; {len(jobs)} runs to do", flush=True)
    need_llm = any(m in LLM_METHODS for m in methods)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.bench, need_llm)) as ex, \
            open(rows_path, "a") as f:
        for k, r in enumerate(ex.map(_run, jobs)):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            print(f"{k + 1}/{len(jobs)} {r['method']:24s} {r['case']:28s} {r['outcome']:12s} wrong={r['wrong_digs']} "
                  f"unsafe={r['audit']['unsafe_total']} {r['sec']:.0f}s  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
