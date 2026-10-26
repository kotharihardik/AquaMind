"""Re-word every simulated citizen message the way real people write (the "realistic" text condition).

The writer is an LLM of ANOTHER family than the agent (Gemini flash-lite; agents use NVIDIA nemotron), so the agent
is not reading its own style. Each message gets a persona (age, mood, channel habits, English / Hinglish mix).
Round-trip check: every gold place word and time phrase must still appear verbatim, and no extra cause may be
added; otherwise the rule-made text is kept. Attack messages (instruction + coordinated fakes) are kept as written.
Messages known at case opening go to cases/<id>/messages_realistic.json; ALL messages (later arrivals too) go to
hidden/<id>.realistic.json, so methods can never read a message before it arrives.

usage: python -I scripts/write_messages.py <bench_dir> [dev|test|all]
"""
import json
import os
import random
import sys

from agentix.llm.client import LLM, WRITER_MODEL
from agentix.sim.writer import PERSONAS, SYSTEM, must_keep, ok

if __name__ == "__main__":
    bench = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "all"
    llm = LLM(WRITER_MODEL, temperature=0.8, max_tokens=3000)
    names = sorted(n[:-5] for n in os.listdir(f"{bench}/hidden") if n.endswith(".json") and not n.endswith("realistic.json"))
    if split != "all":
        names = [n for n in names if n.startswith(split + "-")]
    stats = {"rewritten": 0, "kept": 0}
    def process(name):
        out_h = f"{bench}/hidden/{name}.realistic.json"
        if os.path.exists(out_h):
            return
        hid = json.load(open(f"{bench}/hidden/{name}.json"))
        rng = random.Random(name)
        todo = [r for r in hid["reports"] if r["truth"]["kind"] not in ("injection", "fake")]
        texts = {r["report_id"]: r["text"] for r in hid["reports"]}
        for i in range(0, len(todo), 8):
            chunk = todo[i:i + 8]
            items = [{"id": r["report_id"], "text": r["text"], "persona": rng.choice(PERSONAS),
                      "must_keep": must_keep(r["gold"])} for r in chunk]
            d = llm.json(SYSTEM, json.dumps(items, ensure_ascii=False), ["texts"], tag="writer")
            if "_error" in d:                      # the free tier failed: leave this case for the next pass (no fallback text)
                print(name, "LLM error, will retry on the next pass", flush=True)
                return
            got = d.get("texts") if isinstance(d.get("texts"), dict) else {}
            for it in items:
                t = str(got.get(it["id"], "")).strip()
                if ok(t, it["must_keep"]):
                    texts[it["id"]] = t
                    stats["rewritten"] += 1
                else:
                    stats["kept"] += 1
        json.dump(texts, open(out_h, "w"), indent=1, ensure_ascii=False)
        opened = {r["report_id"] for r in hid["reports"] if r["received_s"] <= hid["open_s"]}
        json.dump({k: v for k, v in texts.items() if k in opened},
                  open(f"{bench}/cases/{name}/messages_realistic.json", "w"), indent=1, ensure_ascii=False)
        print(name, stats, f"llm calls {llm.usage.calls}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(int(os.environ.get("WRITER_THREADS", "8"))) as ex:
        list(ex.map(process, names))
