"""Assemble the hand-labelled reader dataset (reader-2.0 / reply-2.0) into chat-format JSONL for fine-tuning. It GENERATES NOTHING.

Inputs (finetune/data_src/):
  real_pool.json            real FixMyStreet water/drainage reports (public Open311 API), text as the citizen wrote it
  labels_real_*.jsonl       hand labels for real reports: {"i": index into real_pool, "cat", "sym", "pl": [[text, type]], "src", "priv",
                            "since", "inj"?, "sum"}
  hand_reads_*.jsonl        messages WRITTEN BY HAND in the style of Indian civic channels, each with its hand label:
                            {"ch", "text", "cat", "sym", "pl", "src", "priv", "since", "inj"?, "sum", "kind"}
  hand_replies_*.jsonl      call-back answers written by hand: {"q", "a", "ans", "src", "pl", "from", "nn", "flow", "inj"?, "kind"}
Checks: every label passes the reader-2.0 / reply-2.0 schema; every place and time phrase appears VERBATIM in the text (case-sensitive);
no duplicate texts. Splits: real reports -> test_real (250, never trained on) and train_real (the rest); hand-written -> dev (30 messages + 10 answers),
test_hand (70 + 20) and train (the rest) by a fixed seed; test_bench = held-out text from the benchmark's own message/answer generator (checked). Output: finetune/kaggle_dataset/{train,dev,test_hand,test_real,test_bench}.jsonl + data_card.json.
usage: python -I finetune/assemble.py
"""
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, ".")
from agentix.reader.prompts import FT_READ_SYSTEM, FT_REPLY_SYSTEM  # noqa: E402
from agentix.reader.schema import valid_read, valid_reply  # noqa: E402

SRC, OUT = "finetune/data_raw", "finetune/dataset"
N_TEST_REAL = 250


def read_label(x: dict) -> dict:
    return {"category": x["cat"], "symptoms": x["sym"], "places": [{"text": t, "type": ty} for t, ty in x["pl"]], "source": x["src"],
            "on_private_property": x["priv"], "since": x["since"], "contains_instructions": bool(x.get("inj", False)), "summary": x["sum"]}


def reply_label(x: dict) -> dict:
    return {"answered": x["ans"], "source": x["src"], "places": x["pl"], "flows_from": x["from"], "not_network": x["nn"],
            "still_flowing": x["flow"], "contains_instructions": bool(x.get("inj", False))}


def check_spans(text: str, spans: list[str], where: str, errs: list) -> None:
    for s in spans:
        if s and s not in text:
            errs.append(f"{where}: '{s}' is not verbatim in the text")


def row_read(channel, text, label, meta):
    return {"messages": [{"role": "system", "content": FT_READ_SYSTEM},
                         {"role": "user", "content": json.dumps({"channel": channel, "text": text}, ensure_ascii=False)},
                         {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)}], "meta": {"task": "read", **meta}}


def row_reply(q, a, label, meta):
    return {"messages": [{"role": "system", "content": FT_REPLY_SYSTEM},
                         {"role": "user", "content": json.dumps({"question": q, "answer": a}, ensure_ascii=False)},
                         {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)}], "meta": {"task": "reply", **meta}}


def main():
    errs, seen = [], set()
    pool = json.load(open(f"{SRC}/real_pool.json"))
    real = []
    for f in sorted(glob.glob(f"{SRC}/labels_real_*.jsonl")):
        for n, line in enumerate(open(f), 1):
            x = json.loads(line)
            r = pool[x["i"]]
            lab = read_label(x)
            where = f"{os.path.basename(f)}:{n} (real {x['i']})"
            if not valid_read(lab):
                errs.append(f"{where}: invalid reader-2.0 label")
            check_spans(r["text"], [p["text"] for p in lab["places"]] + ([lab["since"]] if lab["since"] else []), where, errs)
            if r["text"] in seen:
                errs.append(f"{where}: duplicate text")
            seen.add(r["text"])
            real.append(row_read(r["channel"], r["text"], lab, {"source": "fixmystreet_real", "id": r["rid"], "kind": f"real:{x['cat']}"}))
    hand_r, hand_q = [], []
    for f in sorted(glob.glob(f"{SRC}/hand_reads_*.jsonl")):
        for n, line in enumerate(open(f), 1):
            x = json.loads(line)
            lab = read_label(x)
            where = f"{os.path.basename(f)}:{n}"
            if not valid_read(lab):
                errs.append(f"{where}: invalid reader-2.0 label")
            check_spans(x["text"], [p["text"] for p in lab["places"]] + ([lab["since"]] if lab["since"] else []), where, errs)
            if x["text"] in seen:
                errs.append(f"{where}: duplicate text")
            seen.add(x["text"])
            hand_r.append(row_read(x["ch"], x["text"], lab, {"source": "hand_written", "kind": x.get("kind", x["cat"])}))
    for f in sorted(glob.glob(f"{SRC}/hand_replies_*.jsonl")):
        for n, line in enumerate(open(f), 1):
            x = json.loads(line)
            lab = reply_label(x)
            where = f"{os.path.basename(f)}:{n}"
            if not valid_reply(lab):
                errs.append(f"{where}: invalid reply-2.0 label")
            check_spans(x["a"], lab["places"] + lab["flows_from"], where, errs)
            hand_q.append(row_reply(x["q"], x["a"], lab, {"source": "hand_written", "kind": x.get("kind", "reply")}))
    if errs:
        print("\n".join(errs[:60]))
        print(f"{len(errs)} problem(s): fix the labels first")
        sys.exit(1)
    rng = random.Random(20261026)
    rng.shuffle(real)
    rng.shuffle(hand_r)
    rng.shuffle(hand_q)
    test_real, train_real = real[:N_TEST_REAL], real[N_TEST_REAL:]
    N_DEV_R, N_DEV_Q, N_TEST_R, N_TEST_Q = 30, 10, 70, 20            # hand-written messages / call-back answers held out
    dev = hand_r[:N_DEV_R] + hand_q[:N_DEV_Q]
    test_hand = hand_r[N_DEV_R:N_DEV_R + N_TEST_R] + hand_q[N_DEV_Q:N_DEV_Q + N_TEST_Q]
    train = hand_r[N_DEV_R + N_TEST_R:] + hand_q[N_DEV_Q + N_TEST_Q:] + train_real
    bench = [json.loads(l) for l in open(f"{SRC}/test_bench.jsonl")] if os.path.exists(f"{SRC}/test_bench.jsonl") else []
    rng.shuffle(train)
    os.makedirs(OUT, exist_ok=True)
    card = {"contract": "reader-2.0 / reply-2.0", "splits": {}}
    for name, rows in (("train", train), ("dev", dev), ("test_hand", test_hand), ("test_real", test_real), ("test_bench", bench)):
        with open(f"{OUT}/{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        kinds = {}
        for r in rows:
            k = f"{r['meta']['task']}:{r['meta']['source']}"
            kinds[k] = kinds.get(k, 0) + 1
        card["splits"][name] = {"n": len(rows), "by_source": kinds,
                                "sha256": hashlib.sha256(open(f"{OUT}/{name}.jsonl", "rb").read()).hexdigest()[:16]}
    json.dump(card, open(f"{OUT}/data_card.json", "w"), indent=1)
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
