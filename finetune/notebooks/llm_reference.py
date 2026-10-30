"""Reference scores of the API reader (the LLM the crew uses) on the same test files, with the SAME scoring code as the Kaggle notebook (its cells are executed).
Writes finetune/kaggle_dataset/llm_reference.json: {split: {metric: value}}; upload it next to the dataset and the notebook prints a third column.
A fixed subset per split keeps the API cost small: usage: python -I finetune/kaggle/llm_reference.py"""
import json
import random
import sys
import time

sys.path.insert(0, ".")
from aquamind.llm.client import LLM  # noqa: E402
from aquamind.reader.llm_reader import LLMReader  # noqa: E402

nb = json.load(open("finetune/notebooks/agentix_reader_finetune.ipynb"))
codes = ["\n".join(l for l in "".join(c["source"]).splitlines() if not l.strip().startswith("!")) for c in nb["cells"] if c["cell_type"] == "code"]
g = {}
setup = codes[1].replace("import torch", "import types as _t; torch = _t.SimpleNamespace(manual_seed=lambda s: None, cuda=_t.SimpleNamespace(is_available=lambda: False))")
setup = setup.replace('DATA = os.path.dirname(cand[0]) if cand else "/kaggle/input/agentix-reader-data"', 'DATA = "finetune/dataset"')
exec(setup, g)                                      # config cell with the GPU parts stubbed (loads the splits)
exec(codes[2], g)                                   # contract, repair and validity helpers (code cell 2)
exec(codes[3], g)                                   # metrics (code cell 3)
N = {"test_hand": 90, "test_real": 100, "test_bench": 120}
reader = LLMReader(LLM(), batch=4, parallel=2)
out = {}
for split, n in N.items():
    rows = [json.loads(l) for l in open(f"finetune/dataset/{split}.jsonl")]
    random.Random(7).shuffle(rows)
    rows = rows[:n]
    t0 = time.time()
    reads = [r for r in rows if r["meta"]["task"] == "read"]
    reps = [r for r in rows if r["meta"]["task"] == "reply"]
    preds = {}
    rd = reader.read_many([{"report_id": f"{split}-{i}", "channel": json.loads(r["messages"][1]["content"]).get("channel"),
                            "text": json.loads(r["messages"][1]["content"])["text"]} for i, r in enumerate(reads)])
    for i, r in enumerate(reads):
        preds[id(r)] = (True, rd[f"{split}-{i}"])
    for k in range(0, len(reps), 4):
        chunk = reps[k:k + 4]
        items = [(json.loads(r["messages"][1]["content"])["question"], json.loads(r["messages"][1]["content"])["answer"]) for r in chunk]
        for r, p in zip(chunk, reader.read_replies(items)):
            preds[id(r)] = (True, p)
    ordered = [preds[id(r)] for r in rows]
    s, kinds, conf = g["score"](rows, ordered)
    s["sec_per_item"] = round((time.time() - t0) / len(rows), 2)
    out[split] = s
    print(split, json.dumps(s), flush=True)
    json.dump(out, open("finetune/dataset/llm_reference.json", "w"), indent=1)
