"""RAG evaluation: how well AquaMind's document search (aquamind/rag/retriever.py) and its uses in the toolbox find the right documents.

No LLM and no API calls. Ground truth comes from the benchmark's hidden files (every citizen message knows whether it comes from a recurring
false-alarm spot, its true cause and where the water was seen) and from the machine-checkable SOP (configs/rules.yaml via RuleBook).
The production functions are called as they are (Toolbox.records_near, Toolbox.sop_context, retriever.clean), so the numbers describe the
system that ran in the evaluation.

Four parts:
  A. Archive retrieval for a message (records_near): does the search return the past false-alarm tickets of the place a message names?
     Queried with the gold place names, so reader mistakes do not count against retrieval. Also the raw BM25 ranking (recall@k, MRR).
  B. SOP retrieval for a dig (sop_context): are the SOP clauses that apply at the dig site among the clauses retrieved for the Auditor?
  C. Citation accuracy (from the evaluation traces): the archive records the agents cited - accepted / rejected by the code check, and
     whether accepted false-alarm citations were really about a non-leak message; failing-main tickets near the true leak.
  D. Injection guard (retriever.clean): does it remove instruction-like text, and does it leave normal text alone?

usage: .venv/bin/python -I scripts/eval_rag.py [--bench data/benchmark/agentix_master] [--out logs/evaluation/extra]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import types
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from aquamind.agent.toolbox import Toolbox  # noqa: E402
from aquamind.eval.runner import World  # noqa: E402
from aquamind.rag.retriever import clean  # noqa: E402

ALWAYS_DIG = {"SOP-1", "SOP-2", "SOP-8", "SOP-10"}      # apply to every dig: approval, two-dig limit, tell reporters, field check first


def load_truth(bench: str) -> dict:
    """Hidden ground truth of every case.

    Args: bench = benchmark folder.
    Returns: {case name: hidden dict (leak, reports with gold + truth, ...)}.
    """
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{bench}/hidden/*-*.json")}


def mrr(ranked: list[str], rel: set[str]) -> float:
    """Args: ranked = retrieved ids, best first; rel = relevant ids.
    Returns: 1 / rank of the first relevant id (0 if none).
    """
    return next((1.0 / (i + 1) for i, x in enumerate(ranked) if x in rel), 0.0)


# ---------------------------------------------------------------------------------------------------------------- A. archive retrieval
def part_a(world, truth: dict) -> dict:
    """Archive retrieval for citizen messages, with M4 (false-alarm tickets only) on and off.

    Args: world = World; truth = load_truth().
    Returns: metrics dict.
    """
    P, net, docs = world.town.places, world.town.net, world.docs
    recs = {c.id: c.text for c in docs.chunks if c.provenance == "record"}
    fa = {k for k, v in recs.items() if Toolbox.is_false_alarm_ticket(v)}
    out = {}
    for m4 in (True, False):
        ns = types.SimpleNamespace(use_rag=True, places=P, docs=docs, m4_ticket_type=m4, reads={},
                                   is_false_alarm_ticket=Toolbox.is_false_alarm_ticket)
        hot = defaultdict(list)
        leak_misleading = leak_n = noise_n = noise_ret = 0
        for case, h in truth.items():
            for r in h["reports"]:
                places = (r.get("gold") or {}).get("places") or []
                if not places:
                    continue
                ns.reads = {"q": {"places": places}}
                got = [c.id for c in Toolbox.records_near(ns, "q")]
                t = r["truth"]
                if t.get("hotspot"):
                    d = P.describe(net.index[t["seen_node"]])
                    rel = {k for k in fa if d in recs[k]}
                    hot["hit"].append(bool(set(got) & rel))
                    hot["recall"].append(len(set(got) & rel) / max(len(rel), 1))
                    hot["precision"].append(len(set(got) & rel) / len(got) if got else 0.0)
                    hot["mrr"].append(mrr(got, rel))
                elif t.get("kind") == "leak":
                    leak_n += 1
                    leak_misleading += any(g in fa for g in got)
                else:
                    noise_n += 1
                    noise_ret += bool(got)
        out["m4_on" if m4 else "m4_off"] = {
            "hotspot_messages": len(hot["hit"]),
            "hit_rate@3": round(float(np.mean(hot["hit"])), 3), "recall@3": round(float(np.mean(hot["recall"])), 3),
            "precision@3": round(float(np.mean(hot["precision"])), 3), "mrr": round(float(np.mean(hot["mrr"])), 3),
            "leak_messages": leak_n, "leak_messages_shown_a_false_alarm_ticket": round(leak_misleading / max(leak_n, 1), 3),
            "other_messages": noise_n, "other_messages_with_any_ticket": round(noise_ret / max(noise_n, 1), 3)}
    # raw BM25 ranking over the whole archive (before the place-name filter): the search engine itself
    rk = defaultdict(list)
    for case, h in truth.items():
        for r in h["reports"]:
            t = r["truth"]
            places = (r.get("gold") or {}).get("places") or []
            if not t.get("hotspot") or not places:
                continue
            names = [P.places[ids[0]]["name"] for p in places if (ids := P.resolve(p["text"], p.get("type")))]
            if not names:
                continue
            d = P.describe(net.index[t["seen_node"]])
            rel = {k for k in fa if d in recs[k]}
            ranked = [c.id for c, _ in docs.search(" ".join(names), k=10, provenance="record")]
            for k in (1, 3, 6, 10):
                rk[f"recall@{k}"].append(len(set(ranked[:k]) & rel) / max(len(rel), 1))
                rk[f"hit@{k}"].append(bool(set(ranked[:k]) & rel))
            rk["mrr"].append(mrr(ranked, rel))
    out["bm25_ranking"] = {k: round(float(np.mean(v)), 3) for k, v in rk.items()} | {"queries": len(rk["mrr"])}
    out["archive"] = {"tickets": len(recs), "false_alarm_tickets": len(fa), "repair_tickets": len(recs) - len(fa)}
    return out


# ---------------------------------------------------------------------------------------------------------------- B. SOP retrieval
def part_b(world, case_name: str = "dev-brown-000") -> dict:
    """SOP clauses retrieved for the Auditor at every junction (every possible dig site), with M5 (all streets at a junction) on and off.

    Args: world = World; case_name = any case (only the map and SOP are used).
    Returns: metrics dict.
    """
    from aquamind.reader.rules import RuleReader
    case, env = world.open(case_name, "spread_k5")
    tb = Toolbox(env, case, world.search(hypothesis_mode=True), RuleReader(world.town.places), world.docs, world.rules)
    net, rules = world.town.net, world.rules
    when = dt.datetime(2018, 7, 2, 10, 0)                 # a weekday during school hours: every location rule is active
    out = {}
    for m5 in (True, False):
        tb.m5_corner_permits = m5
        sit, alls, prec, rr, situ_n, kinds, every = [], [], [], [], 0, Counter(), Counter()
        for j in range(len(net.junctions)):
            need = {x["sop"] for x in rules.requirements("dig", [net.junctions[j]], when)}
            got = [ln.strip().split(":")[0] for ln in tb.sop_context(j) if ln.strip().startswith("SOP-")]
            rel = need | ALWAYS_DIG
            every.update(got)
            if need:
                situ_n += 1
                sit.append(len(need & set(got)) / len(need))
                kinds.update(f"{s}_{'found' if s in got else 'missed'}" for s in need)
            alls.append(len(rel & set(got)) / len(rel))
            prec.append(len(rel & set(got)) / max(len(got), 1))
            rr.append(mrr(got, rel))
        out["m5_on" if m5 else "m5_off"] = {
            "dig_sites": len(alls), "sites_with_location_rules": situ_n,
            "location_rule_recall@6": round(float(np.mean(sit)), 3) if sit else None,
            "all_needed_recall@6": round(float(np.mean(alls)), 3), "precision@6": round(float(np.mean(prec)), 3),
            "mrr": round(float(np.mean(rr)), 3), "by_rule": dict(sorted(kinds.items())),
            "share_of_sites_retrieving": {k: round(v / len(alls), 3) for k, v in sorted(every.items())}}
    out["sop_clauses"] = sum(1 for c in world.docs.chunks if c.provenance == "policy")
    return out, tb


# ---------------------------------------------------------------------------------------------------------------- C. citation accuracy
def part_c(tb, truth: dict, traces: dict[str, str]) -> dict:
    """Archive citations the agents made in the evaluation runs (from trace files).

    Args: tb = a Toolbox (for record footprints); truth = load_truth(); traces = {label: trace jsonl path}.
    Returns: {label: metrics}.
    """
    net = tb.net
    recs = {c.id: c.text for c in tb.docs.chunks if c.provenance == "record"}
    out = {}
    for label, path in traces.items():
        if not os.path.exists(path):
            continue
        c = Counter()
        seen = set()
        for line in open(path):
            s = json.loads(line)
            tool, case = s.get("tool"), s.get("case")
            if tool not in ("use_record", "use_repair_record") or case not in truth:
                continue
            a, res = s.get("args") or {}, s.get("result") or {}
            key = (case, tool, json.dumps(a, sort_keys=True))
            if key in seen:                                        # the same citation repeated in a later turn counts once
                continue
            seen.add(key)
            ok = bool(res.get("ok"))
            if tool == "use_record":
                c["false_alarm_citations"] += 1
                c["false_alarm_accepted"] += ok
                rep = next((r for r in truth[case]["reports"] if r.get("report_id") == a.get("report_id")), None)
                txt = recs.get(str(a.get("record_id")))
                if ok and txt is not None:
                    c["accepted_citing_a_repair_ticket"] += not Toolbox.is_false_alarm_ticket(txt)     # the M4 failure mode
                if ok and rep is not None:
                    c["accepted_on_real_leak_message"] += rep["truth"].get("kind") == "leak"
                    c["accepted_on_non_leak_message"] += rep["truth"].get("kind") != "leak"
            else:
                c["repair_citations"] += 1
                c["repair_accepted"] += ok
                txt = recs.get(str(a.get("record_id")))
                lk = truth[case].get("leak")
                if ok and txt and lk:
                    w = tb._record_footprint(txt)
                    if w is not None:
                        d = float(net.pipe_distance[int(np.argmax(w)), net.index[lk["nearest_node"]]])
                        c["accepted_repair_with_leak_case"] += 1
                        c["accepted_repair_within_300m_of_leak"] += d <= 300
        acc = c["false_alarm_accepted"]
        out[label] = {**dict(c),
                      "false_alarm_acceptance_rate": round(acc / max(c["false_alarm_citations"], 1), 3),
                      "accepted_false_alarm_precision": round(c["accepted_on_non_leak_message"] / max(acc, 1), 3),
                      "accepted_citing_repair_ticket_share": round(c["accepted_citing_a_repair_ticket"] / max(acc, 1), 3),
                      "repair_acceptance_rate": round(c["repair_accepted"] / max(c["repair_citations"], 1), 3),
                      "accepted_repair_near_leak_rate": round(c["accepted_repair_within_300m_of_leak"] / max(c["accepted_repair_with_leak_case"], 1), 3)}
    return out


# ---------------------------------------------------------------------------------------------------------------- D. injection guard
def part_d(truth: dict) -> dict:
    """The prompt-injection filter applied to every citizen message as a classifier.

    Args: truth = load_truth().
    Returns: detection rate on messages with instructions, false-positive rate on the others.
    """
    pos = neg = tp = fp = 0
    for h in truth.values():
        for r in h["reports"]:
            inj = bool((r.get("gold") or {}).get("contains_instructions")) or r["truth"].get("kind") == "injection"
            changed = clean(r["text"]) != r["text"]
            if inj:
                pos += 1
                tp += changed
            else:
                neg += 1
                fp += changed
    return {"messages_with_instructions": pos, "removed": tp, "detection_rate": round(tp / max(pos, 1), 3),
            "normal_messages": neg, "wrongly_changed": fp, "false_positive_rate": round(fp / max(neg, 1), 4)}


def to_md(res: dict) -> str:
    """Args: res = all results.
    Returns: a Markdown report of the tables.
    """
    a, b, c, d = res["A_archive_retrieval"], res["B_sop_retrieval"], res["C_citations"], res["D_injection_guard"]
    L = ["# RAG evaluation", "", f"Archive: {a['archive']['tickets']} tickets ({a['archive']['false_alarm_tickets']} false-alarm visits, "
         f"{a['archive']['repair_tickets']} repairs); SOP: {b['sop_clauses']} clauses. No LLM involved; ground truth from the benchmark.", "",
         "## A. Archive retrieval for a citizen message (records_near, 3 tickets shown)", "",
         "| | hotspot msgs | hit@3 | recall@3 | precision@3 | MRR | leak msgs shown a false-alarm ticket | other msgs with a ticket |",
         "|---|---|---|---|---|---|---|---|"]
    for k in ("m4_on", "m4_off"):
        x = a[k]
        L.append(f"| {k.replace('_', ' ')} | {x['hotspot_messages']} | {x['hit_rate@3']} | {x['recall@3']} | {x['precision@3']} | {x['mrr']} | "
                 f"{x['leak_messages_shown_a_false_alarm_ticket']} | "
                 f"{x['other_messages_with_any_ticket']} |")
    r = a["bm25_ranking"]
    L += ["", f"Raw BM25 ranking ({r['queries']} queries): recall@1 {r['recall@1']}, @3 {r['recall@3']}, @6 {r['recall@6']}, @10 {r['recall@10']}; "
          f"hit@1 {r['hit@1']}, hit@3 {r['hit@3']}; MRR {r['mrr']}.", "",
          "## B. SOP clauses retrieved for the Auditor (sop_context, 6 clauses) at every possible dig site", "",
          "| | dig sites | sites with location rules | location-rule recall@6 | all-needed recall@6 | precision@6 | MRR |", "|---|---|---|---|---|---|---|"]
    for k in ("m5_on", "m5_off"):
        x = b[k]
        L.append(f"| {k.replace('_', ' ')} | {x['dig_sites']} | {x['sites_with_location_rules']} | {x['location_rule_recall@6']} | "
                 f"{x['all_needed_recall@6']} | {x['precision@6']} | {x['mrr']} |")
    L += ["", "By rule (M5 on): " + ", ".join(f"{k} {v}" for k, v in b["m5_on"]["by_rule"].items()),
          "Share of dig sites whose retrieved clauses include each SOP: " + ", ".join(f"{k} {v}" for k, v in b["m5_on"]["share_of_sites_retrieving"].items()), "",
          "## C. Archive citations made by the agents (evaluation traces)", "",
          "| run | false-alarm citations | accepted | accepted that cite a repair ticket (M4 target) | accepted on a real-leak message | "
          "precision of accepted | repair citations | accepted | accepted repairs within 300 m of the leak |", "|---|---|---|---|---|---|---|---|---|"]
    for k, x in c.items():
        L.append(f"| {k} | {x.get('false_alarm_citations', 0)} | {x['false_alarm_acceptance_rate']} | {x['accepted_citing_repair_ticket_share']} | "
                 f"{x.get('accepted_on_real_leak_message', 0)} | "
                 f"{x['accepted_false_alarm_precision']} | {x.get('repair_citations', 0)} | {x['repair_acceptance_rate']} | {x['accepted_repair_near_leak_rate']} |")
    L += ["", "## D. Prompt-injection guard (retriever.clean) on citizen messages", "",
          f"Messages with instructions: {d['messages_with_instructions']}, removed {d['removed']} (detection {d['detection_rate']}); "
          f"normal messages wrongly changed: {d['wrongly_changed']} of {d['normal_messages']} (false positives {d['false_positive_rate']}).", ""]
    return "\n".join(L)


def main():
    """Run parts A-D and write rag_eval.json and rag_eval.md."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark/agentix_master")
    ap.add_argument("--out", default="logs/evaluation/extra")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    world, truth = World(a.bench), load_truth(a.bench)
    res = {"A_archive_retrieval": part_a(world, truth)}
    res["B_sop_retrieval"], tb = part_b(world)
    T = "logs/evaluation/agentix/crew"
    traces = {f"{v} {k}": f"{T}/{k}/{v}_{k}_agentix_crew_trace.jsonl" for k in ("k5", "k3", "k10") for v in ("v1", "v2")}
    res["C_citations"] = {k.replace("v1", "mid").replace("v2", "final"): v for k, v in part_c(tb, truth, traces).items()}
    res["D_injection_guard"] = part_d(truth)
    json.dump(res, open(f"{a.out}/rag_eval.json", "w"), indent=2)
    open(f"{a.out}/rag_eval.md", "w").write(to_md(res))
    print(to_md(res))


if __name__ == "__main__":
    main()
