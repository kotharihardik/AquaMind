"""Build the complaint reader's fine-tuning data (reader-2.0 and reply-2.0), gold-first.

1. GOLD FIRST: the simulator's message generator (agentix/sim/reports.py) decides the facts and the exact
   reader-2.0 label, then renders text, on 4 naming skins of the town (different street/landmark names).
2. RE-WORDING: part of the messages are re-written by writer LLMs with personas (round-trip checked: every gold
   place and time phrase must stay verbatim, else the rule-made text is kept).
3. REPLIES: call-back questions (many phrasings) answered by the simulated citizen, with exact reply-2.0 labels.
Splits (no leakage between them):
   train/dev     skins ltown, town2, town3; writers: Gemini 3.1 flash-lite and Gemini flash-lite-latest (+ rule text)
   test_synth    skin town4 (never seen) and writer Gemini 3.5 flash-lite (never seen)
   test_real     REAL messages (finetune/real/), hand-checked labels: never used for training
Output: data/finetune/{train,dev,test_synth}.jsonl in chat format ({"messages": [...], "meta": {...}}).
usage: python -I finetune/build_dataset.py [n_train_reads] [--no-llm]
"""
import json
import os
import random
import sys

import numpy as np

from agentix.reader.prompts import FT_READ_SYSTEM as READ_SYSTEM, FT_REPLY_SYSTEM as REPLY_SYSTEM
from agentix.sim.citizen import Citizen
from agentix.sim.reports import ReportGenerator, TownContext
from agentix.sim.terrain import Terrain
from agentix.sim.writer import PERSONAS, SYSTEM, ok
from agentix.town.assets import Town

OUT = os.environ.get("AGENTIX_FT_OUT", "data/finetune")
QUESTIONS = [
    "Is the water coming out of the road itself, or flowing from somewhere else?",
    "Where exactly does the water come out? Which street and nearest landmark?",
    "Does the water smell bad or look dirty?", "Is there a garden, a hose or someone watering plants nearby?",
    "Could it be from a house pipe or an overflowing tank in a compound?", "Did it rain there recently?",
    "Is the water still flowing right now?", "Which side does the water come down from?",
    "Is anyone washing a car or a shop front there?", "Can you tell me the road name and a shop or temple near it?",
    "Water kahan se aa raha hai - road se ya kisi ghar se?", "Abhi bhi paani beh raha hai kya?",
    "Is it clean water or drain water? Any smell?", "Since when, and is it still coming out?",
]


def town_ctx(skin: str, base_town: Town, terrain, geometry, hot):
    gaz = json.load(open(f"{base_town.dir}/gazetteer/gazetteer_{skin}.json"))
    return TownContext(base_town.net, terrain, gaz, geometry, {"hotspots": hot}), gaz


def sample_read(gen: ReportGenerator, ctx, rng) -> dict:
    J = len(ctx.net.junctions)
    j = int(rng.integers(J))
    t = float(rng.uniform(8, 10) * 86400)
    u = rng.random()
    if u < 0.45:                                                    # a real leak sighting
        sym = [str(rng.choice(["water_coming_out", "burst", "pooling", "wet_ground", "water_coming_out", "pooling"]))]
        if rng.random() < 0.08:
            sym.append("road_damage")
        src = "here" if sym[0] in ("water_coming_out", "burst") and rng.random() < 0.7 else (
            "elsewhere" if sym[0] == "pooling" and rng.random() < 0.4 else "unknown")
        return gen._message(t, j, kind="leak", symptoms=sym, category="leak", source=src, private=None,
                            onset_s=t - float(rng.uniform(1, 60)) * 3600)
    if u < 0.75:
        return gen.false_report(t)
    if u < 0.88:
        sym = [str(rng.choice(["low_pressure", "no_water"]))]
        return gen._message(t, j, kind="supply", symptoms=sym, category="supply", source="unknown", private=None,
                            onset_s=t - float(rng.uniform(1, 30)) * 3600)
    if u < 0.94:
        return gen._message(t, j, kind="quality", symptoms=["dirty_water"], category="quality", source="unknown",
                            private=None)
    return gen.trap(t, j, 0)[0]


def chat(system, user, gold, meta):
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                         {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)}], "meta": meta}


def reword(items, writer, rng):
    """Re-write a batch with the writer LLM; keep the rule text when the round-trip check fails."""
    if writer is None:
        return items
    batch = [{"id": str(k), "text": it["text"], "persona": rng.choice(PERSONAS),
              "must_keep": [p["text"] for p in it["gold"]["places"]] + ([it["gold"]["since"]] if it["gold"]["since"] else [])}
             for k, it in enumerate(items)]
    d = writer.json(SYSTEM, json.dumps(batch, ensure_ascii=False), ["texts"], tag="ft_writer")
    got = d.get("texts") if isinstance(d.get("texts"), dict) else {}
    for b, it in zip(batch, items):
        t = str(got.get(b["id"], "")).strip()
        if ok(t, b["must_keep"]):
            it["text"], it["writer"] = t, writer.model
    return items


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2400
    use_llm = "--no-llm" not in sys.argv
    from agentix.llm.client import LLM
    town = Town()
    terrain = Terrain(town.net)
    geometry = json.load(open(f"{town.dir}/gazetteer/geometry.json"))
    hot = json.load(open(f"{town.dir}/truth/hotspots.json"))
    writers_train = [LLM("gemini:gemini-3.1-flash-lite", 0.8, max_tokens=3000),
                     LLM("gemini:gemini-flash-lite-latest", 0.8, max_tokens=3000)] if use_llm else [None]
    writer_test = LLM("gemini:gemini-3.5-flash-lite", 0.8, max_tokens=3000) if use_llm else None
    plan = [("train", ["ltown", "town2", "town3"], n), ("dev", ["ltown", "town2", "town3"], n // 12),
            ("test_synth", ["town4"], max(300, n // 8))]
    os.makedirs(OUT, exist_ok=True)
    for split, skins, size in plan:
        path = f"{OUT}/{split}.jsonl"
        if os.path.exists(path):
            continue
        rng = np.random.default_rng({"train": 1, "dev": 2, "test_synth": 3}[split])
        prng = random.Random(split)
        rows = []
        for skin in skins:
            ctx, gaz = town_ctx(skin, town, terrain, geometry, hot)
            gen = ReportGenerator(ctx, rng, __import__("datetime").datetime(2026, 3, 2))
            k = size // len(skins)
            items = [sample_read(gen, ctx, rng) for _ in range(k)]
            for it in items:
                it["writer"] = "rules"
            idx = [i for i in range(k) if prng.random() < 0.7 and items[i]["truth"]["kind"] != "injection"]
            for b in range(0, len(idx), 8):
                w = writer_test if split == "test_synth" else writers_train[(b // 8) % len(writers_train)]
                chunk = [items[i] for i in idx[b:b + 8]]
                reword(chunk, w, prng)
            for it in items:
                user = json.dumps({"channel": it["channel"], "text": it["text"]}, ensure_ascii=False)
                rows.append(chat(READ_SYSTEM, user, it["gold"], {"task": "read", "skin": skin, "writer": it["writer"],
                                                                  "kind": it["truth"]["kind"], "split": split}))
            # replies to call-backs
            cit = Citizen(town.net, None, ctx.street_of, ctx.nearest_landmark, None)
            for it in items[: k // 2]:
                if it["truth"]["kind"] in ("supply", "quality"):
                    continue
                q = prng.choice(QUESTIONS)
                if prng.random() < 0.4:
                    q = q + " " + prng.choice(QUESTIONS)
                text, gold = cit.answer_with_gold(it, q, rng)
                if text is None:
                    continue
                user = json.dumps({"question": q, "answer": text}, ensure_ascii=False)
                rows.append(chat(REPLY_SYSTEM, user, gold, {"task": "reply", "skin": skin, "split": split,
                                                           "kind": it["truth"]["kind"]}))
        prng.shuffle(rows)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows), "examples", flush=True)
