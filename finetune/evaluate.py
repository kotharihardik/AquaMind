"""Evaluate complaint readers on a reader test set (proposal metric: field F1 for place, symptom and time).

usage: python -I finetune/evaluate.py <test.jsonl> <reader: rules|llm|local> [--out results.json] [--limit N]
Metrics (reads): JSON valid, category accuracy, symptoms micro-F1, places F1 (exact span, and relaxed: token
overlap >= 0.5), source accuracy, private-property accuracy, since exact match, instruction flag precision/recall,
seconds per message. Replies: answered / source / not_network accuracy, places + flows_from F1.
"""
import argparse
import json
import re
import time


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def span_f1(pred: list[str], gold: list[str], relaxed: bool) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in pred:
        for i, g in enumerate(gold):
            if i in used:
                continue
            hit = (p.strip().lower() == g.strip().lower()) if not relaxed else (
                len(toks(p) & toks(g)) / max(len(toks(g)), 1) >= 0.5)
            if hit:
                used.add(i)
                tp += 1
                break
    return tp, len(pred), len(gold)


def f1(tp, np_, ng):
    p = tp / np_ if np_ else 1.0
    r = tp / ng if ng else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(rows, reader) -> dict:
    from aquamind.reader.schema import valid_read, valid_reply
    acc = {k: [] for k in ("valid", "category", "source", "private", "since", "r_answered", "r_source", "r_cause")}
    counts = {k: [0, 0, 0] for k in ("symptoms", "places_exact", "places_relaxed", "r_places")}
    inj = [0, 0, 0]
    secs = []
    for row in rows:
        m = row["messages"]
        user, gold = json.loads(m[1]["content"]), json.loads(m[2]["content"])
        t0 = time.time()
        if row["meta"]["task"] == "read":
            pred = reader.read({"report_id": "R", "channel": user.get("channel"), "text": user["text"]})
            secs.append(time.time() - t0)
            acc["valid"].append(valid_read(pred))
            acc["category"].append(pred["category"] == gold["category"])
            acc["source"].append(pred["source"] == gold["source"])
            acc["private"].append(pred["on_private_property"] == gold["on_private_property"])
            acc["since"].append((pred["since"] or "").strip().lower() == (gold["since"] or "").strip().lower())
            ps, gs = set(pred["symptoms"]), set(gold["symptoms"])
            c = counts["symptoms"]
            c[0] += len(ps & gs); c[1] += len(ps); c[2] += len(gs)
            for key, rel in (("places_exact", False), ("places_relaxed", True)):
                tp, a, b = span_f1([p["text"] for p in pred["places"]], [p["text"] for p in gold["places"]], rel)
                counts[key][0] += tp; counts[key][1] += a; counts[key][2] += b
            inj[0] += pred["contains_instructions"] and gold["contains_instructions"]
            inj[1] += pred["contains_instructions"]
            inj[2] += gold["contains_instructions"]
        else:
            pred = reader.read_replies([(user["question"], user["answer"])])[0]
            secs.append(time.time() - t0)
            acc["valid"].append(valid_reply(pred))
            acc["r_answered"].append(pred["answered"] == gold["answered"])
            acc["r_source"].append(pred["source"] == gold["source"])
            acc["r_cause"].append(pred["not_network"] == gold["not_network"])
            tp, a, b = span_f1(pred["places"] + pred["flows_from"], gold["places"] + gold["flows_from"], True)
            c = counts["r_places"]
            c[0] += tp; c[1] += a; c[2] += b
    out = {k: round(sum(v) / len(v), 3) for k, v in acc.items() if v}
    out.update({f"{k}_f1": round(f1(*v), 3) for k, v in counts.items()})
    out["instruction_precision"] = round(inj[0] / inj[1], 3) if inj[1] else None
    out["instruction_recall"] = round(inj[0] / inj[2], 3) if inj[2] else None
    out["sec_per_item"] = round(sum(secs) / max(len(secs), 1), 2)
    out["n"] = len(rows)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("reader")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.test)]
    if a.limit:
        rows = rows[: a.limit]
    from aquamind.town.assets import Town
    if a.reader == "rules":
        from aquamind.reader.rules import RuleReader
        reader = RuleReader(Town().places)
    elif a.reader == "llm":
        from aquamind.llm.client import LLM
        from aquamind.reader.llm_reader import LLMReader
        reader = LLMReader(LLM(), batch=1)
    else:
        from aquamind.reader.local_reader import LocalReader
        reader = LocalReader()
    res = {"reader": a.reader, "test": a.test, **evaluate(rows, reader)}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
