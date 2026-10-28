"""Write the Kaggle notebook for the complaint-reader fine-tune (finetune/kaggle/agentix_reader_finetune.ipynb).
usage: python -I finetune/kaggle/build_notebook.py"""
import json
import os
import sys

sys.path.insert(0, ".")
from agentix.reader.prompts import FT_READ_SYSTEM, FT_REPLY_SYSTEM, READ_SYSTEM, REPLY_SYSTEM  # noqa: E402

OUT = "finetune/notebooks/agentix_reader_finetune.ipynb"
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# AgentiX complaint reader: QLoRA fine-tune of Qwen2.5-3B-Instruct (reader-2.0 / reply-2.0)

**What this trains.** One citizen message to a water utility (WhatsApp, X, SMS, web form, email, call-centre note; English, Hinglish, Hindi,
Kannada, Marathi, Tamil, typos) -> one JSON object (`reader-2.0`): category, symptoms, places copied verbatim, source (here / elsewhere),
private property, time phrase, instruction-injection flag, summary. Plus `reply-2.0` for a citizen's answer to a call-back question.

**Data** (dataset `agentix-reader-data`, attached under `/kaggle/input/`):
| split | what | used for |
|---|---|---|
| `train.jsonl` (702) | hand-written messages and call-back replies (Indian civic channels), hand-labelled, + 200 REAL FixMyStreet water/drainage reports, hand-labelled | training |
| `dev.jsonl` (40) | hand-written, held out | monitoring (loss per epoch) |
| `test_hand.jsonl` (90) | hand-written, held out | test, same style as train |
| `test_real.jsonl` (250) | REAL FixMyStreet reports (UK), hand-labelled, never trained on | test, real-world transfer |
| `test_bench.jsonl` (347) | text from the benchmark's own message and call-back-answer generator (held-out town and writer), gold from the generator | test, what the agent crew will actually receive |

No example was produced by a template script. Every label was written and checked by hand, and every place and time span is verbatim in its text
(checked by `finetune/assemble.py`).

**Settings.** Accelerator GPU T4 x2 or P100 (the notebook uses ONE GPU), Internet ON (to download the base model), Persistence off.
**Time.** Training about 15 min (4 epochs). Evaluation about 60-70 min (the base-model zero-shot baseline on test_hand and test_real, the fine-tuned model on all four test sets). Total about 1.5-2 h.
**First run with `SMOKE = True` (cell 3): about 5 min, same code on 24 training rows and 8 test rows. If it finishes, set `SMOKE = False` and Run All.**
""")

code("""
!pip install -q -U "transformers>=4.44,<4.50" "peft>=0.12,<0.15" "bitsandbytes>=0.43" "accelerate>=0.33" jsonschema matplotlib
""")

code(f"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # one GPU: avoids model sharding / DataParallel surprises on a T4 x2 machine
import json, re, time, math, random, glob
import torch
from collections import Counter, defaultdict

SMOKE = False      # True: a 5-minute dry run on a few rows to check the notebook runs end to end; then set False and Run All

SEED = 20261026
random.seed(SEED); torch.manual_seed(SEED)
BASE = "Qwen/Qwen2.5-3B-Instruct"          # Apache-2.0, multilingual (Hindi etc.), strong at JSON; fits a T4 in 4-bit
cand = glob.glob("/kaggle/input/*/train.jsonl") + glob.glob("/kaggle/input/*/*/train.jsonl")
DATA = os.path.dirname(cand[0]) if cand else "/kaggle/input/agentix-reader-data"
if os.path.exists("/kaggle/input/datasets/hardik300/agentix-reader-data/train.jsonl"):
    DATA = "/kaggle/input/datasets/hardik300/agentix-reader-data"      # the owner's dataset path (found on Kaggle)
OUT = "/kaggle/working"
print("data:", DATA, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

FT_READ_SYSTEM = {json.dumps(FT_READ_SYSTEM)}
FT_REPLY_SYSTEM = {json.dumps(FT_REPLY_SYSTEM)}
READ_SYSTEM = {json.dumps(READ_SYSTEM)}      # full instructions: used only for the BASE model zero-shot baseline
REPLY_SYSTEM = {json.dumps(REPLY_SYSTEM)}

def load(name):
    p = f"{{DATA}}/{{name}}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

splits = {{k: load(k) for k in ("train", "dev", "test_hand", "test_real", "test_bench")}}
if SMOKE:
    splits = {{k: v[:24] if k == "train" else v[:8] for k, v in splits.items()}}
for k, v in splits.items():
    print(k, len(v), Counter(r["meta"]["task"] + ":" + r["meta"]["source"] for r in v))
""")

md("## The label contract (reader-2.0 / reply-2.0) and the safe repair every prediction goes through")
code("""
CATEGORIES = ["leak", "supply", "quality", "sewer_drain", "other"]
SYMPTOMS = ["water_coming_out", "burst", "pooling", "wet_ground", "sound_underground", "road_damage", "low_pressure", "no_water", "dirty_water"]
PLACE_TYPES = ["street", "landmark", "area", "address", "other"]
SOURCES = ["here", "elsewhere", "unknown"]
NOT_NETWORK = ["private_pipe", "garden_or_hose", "sewer_drain", "rain", "washing", "other"]
READ_KEYS = ["category", "symptoms", "places", "source", "on_private_property", "since", "contains_instructions", "summary"]
REPLY_KEYS = ["answered", "source", "places", "flows_from", "not_network", "still_flowing", "contains_instructions"]

def parse_json(text):
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return None

def valid_read(d):
    return (isinstance(d, dict) and set(d) == set(READ_KEYS) and d["category"] in CATEGORIES and isinstance(d["symptoms"], list)
            and all(s in SYMPTOMS for s in d["symptoms"]) and isinstance(d["places"], list)
            and all(isinstance(p, dict) and set(p) == {"text", "type"} and p["type"] in PLACE_TYPES for p in d["places"])
            and d["source"] in SOURCES and d["on_private_property"] in (True, False, None) and (d["since"] is None or isinstance(d["since"], str))
            and isinstance(d["contains_instructions"], bool) and isinstance(d["summary"], str) and len(d["summary"]) <= 160)

def valid_reply(d):
    return (isinstance(d, dict) and set(d) == set(REPLY_KEYS) and isinstance(d["answered"], bool) and d["source"] in SOURCES
            and all(isinstance(x, str) for x in d["places"] + d["flows_from"]) and d["not_network"] in [None] + NOT_NETWORK
            and d["still_flowing"] in (True, False, None) and isinstance(d["contains_instructions"], bool))

def as_list(x):
    if isinstance(x, str):
        return [x] if x.strip() else []
    return list(x) if isinstance(x, (list, tuple)) else []

def repair_read(d):
    out = {"category": "other", "symptoms": [], "places": [], "source": "unknown", "on_private_property": None, "since": None,
           "contains_instructions": False, "summary": ""}
    if not isinstance(d, dict):
        return out
    if d.get("category") in CATEGORIES: out["category"] = d["category"]
    out["symptoms"] = list(dict.fromkeys(s for s in as_list(d.get("symptoms")) if s in SYMPTOMS))
    for p in as_list(d.get("places")):
        if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
            out["places"].append({"text": p["text"].strip()[:80], "type": p.get("type") if p.get("type") in PLACE_TYPES else "other"})
    out["source"] = d.get("source") if d.get("source") in SOURCES else "unknown"
    out["on_private_property"] = d.get("on_private_property") if isinstance(d.get("on_private_property"), bool) else None
    out["since"] = d["since"][:60] if isinstance(d.get("since"), str) and d["since"].strip() else None
    out["contains_instructions"] = bool(d.get("contains_instructions"))
    out["summary"] = str(d.get("summary") or "")[:160]
    return out

def repair_reply(d):
    out = {"answered": False, "source": "unknown", "places": [], "flows_from": [], "not_network": None, "still_flowing": None,
           "contains_instructions": False}
    if not isinstance(d, dict):
        return out
    out["answered"] = bool(d.get("answered"))
    out["source"] = d.get("source") if d.get("source") in SOURCES else "unknown"
    for k in ("places", "flows_from"):
        out[k] = [str(x)[:80] for x in as_list(d.get(k)) if isinstance(x, str) and x.strip()]
    out["not_network"] = d.get("not_network") if d.get("not_network") in NOT_NETWORK else None
    out["still_flowing"] = d.get("still_flowing") if isinstance(d.get("still_flowing"), bool) else None
    out["contains_instructions"] = bool(d.get("contains_instructions"))
    return out

assert all(valid_read(json.loads(r["messages"][2]["content"])) for s in splits.values() for r in s if r["meta"]["task"] == "read")
assert all(valid_reply(json.loads(r["messages"][2]["content"])) for s in splits.values() for r in s if r["meta"]["task"] == "reply")
print("all gold labels valid")
""")

md("## Metrics (the proposal's field F1, plus everything the agent depends on)")
code("""
def toks(s):
    return set(re.findall(r"[a-z0-9\\u0900-\\u0dff]+", s.lower()))

def span_counts(pred, gold, relaxed):
    used, tp = set(), 0
    for p in pred:
        for i, g in enumerate(gold):
            if i in used: continue
            hit = (p.strip().lower() == g.strip().lower()) if not relaxed else (len(toks(p) & toks(g)) / max(len(toks(g)), 1) >= 0.5)
            if hit:
                used.add(i); tp += 1; break
    return tp, len(pred), len(gold)

def f1(tp, np_, ng):
    p = tp / np_ if np_ else 1.0
    r = tp / ng if ng else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0

def score(rows, preds):
    acc = defaultdict(list); cnt = defaultdict(lambda: [0, 0, 0]); inj = [0, 0, 0]
    by_kind = defaultdict(lambda: defaultdict(list)); conf = Counter()
    for row, (raw_ok, pred) in zip(rows, preds):
        gold = json.loads(row["messages"][2]["content"]); kind = row["meta"].get("kind", "?")
        if row["meta"]["task"] == "read":
            acc["json_valid"].append(raw_ok)
            for k, v in (("category", pred["category"] == gold["category"]), ("source", pred["source"] == gold["source"]),
                         ("private", pred["on_private_property"] == gold["on_private_property"]),
                         ("since_exact", (pred["since"] or "").strip().lower() == (gold["since"] or "").strip().lower()),
                         ("since_presence", bool(pred["since"]) == bool(gold["since"]))):
                acc[k].append(v); by_kind[kind][k].append(v)
            conf[(gold["category"], pred["category"])] += 1
            ps, gs = set(pred["symptoms"]), set(gold["symptoms"])
            c = cnt["symptoms"]; c[0] += len(ps & gs); c[1] += len(ps); c[2] += len(gs)
            for key, rel in (("places_exact", False), ("places_relaxed", True)):
                tp, a, b = span_counts([p["text"] for p in pred["places"]], [p["text"] for p in gold["places"]], rel)
                cnt[key][0] += tp; cnt[key][1] += a; cnt[key][2] += b
            tp, a, b = span_counts([pred["since"]] if pred["since"] else [], [gold["since"]] if gold["since"] else [], True)
            cnt["since_relaxed"][0] += tp; cnt["since_relaxed"][1] += a; cnt["since_relaxed"][2] += b
            inj[0] += pred["contains_instructions"] and gold["contains_instructions"]; inj[1] += pred["contains_instructions"]; inj[2] += gold["contains_instructions"]
        else:
            acc["reply_json_valid"].append(raw_ok)
            for k, v in (("reply_answered", pred["answered"] == gold["answered"]), ("reply_source", pred["source"] == gold["source"]),
                         ("reply_cause", pred["not_network"] == gold["not_network"]), ("reply_still_flowing", pred["still_flowing"] == gold["still_flowing"])):
                acc[k].append(v)
            tp, a, b = span_counts(pred["places"] + pred["flows_from"], gold["places"] + gold["flows_from"], True)
            cnt["reply_places"][0] += tp; cnt["reply_places"][1] += a; cnt["reply_places"][2] += b
    out = {k: round(sum(v) / len(v), 3) for k, v in acc.items() if v}
    out.update({f"{k}_f1": round(f1(*v), 3) for k, v in cnt.items()})
    out["instruction_precision"] = round(inj[0] / inj[1], 3) if inj[1] else None
    out["instruction_recall"] = round(inj[0] / inj[2], 3) if inj[2] else None
    out["n"] = len(rows)
    kinds = {k: {m: round(sum(v) / len(v), 3) for m, v in d.items()} | {"n": len(d["category"])} for k, d in by_kind.items()}
    return out, kinds, conf
""")

md("## Load the base model in 4-bit and set up generation")
code("""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16)
print(model.get_memory_footprint() / 1e9, "GB")

def prompt_of(row, finetuned):
    sys_msg = row["messages"][0]["content"] if finetuned else (READ_SYSTEM if row["meta"]["task"] == "read" else REPLY_SYSTEM)
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": row["messages"][1]["content"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def predict(m, rows, finetuned, bs=8, max_new=256):
    m.eval(); out = []; t0 = time.time()
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        enc = tok([prompt_of(r, finetuned) for r in batch], return_tensors="pt", padding=True).to(m.device)
        gen = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            text = tok.decode(g, skip_special_tokens=True)
            d = parse_json(text)
            if r["meta"]["task"] == "read":
                out.append((valid_read(d), repair_read(d)))
            else:
                out.append((valid_reply(d), repair_reply(d)))
    sec = (time.time() - t0) / max(len(rows), 1)
    return out, sec
""")

md("## Baseline: the base model, zero-shot, with the full written instructions")
code("""
results = {}
for name in ("test_hand", "test_real"):
    preds, sec = predict(model, splits[name], finetuned=False)
    s, kinds, conf = score(splits[name], preds)
    s["sec_per_item"] = round(sec, 2)
    results[f"base_zeroshot/{name}"] = {"overall": s, "by_kind": kinds}
    print(name, json.dumps(s))
""")

md("## QLoRA fine-tune (loss on the assistant answer only)")
code("""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

MAX_LEN = 1024
def encode(row):
    msgs = row["messages"]
    prompt = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
    answer = msgs[2]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:MAX_LEN]
    labels = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels}

train_enc = [encode(r) for r in splits["train"]]
dev_enc = [encode(r) for r in splits["dev"]]
print("max tokens:", max(len(e["input_ids"]) for e in train_enc))

class DS(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id); lab = torch.full((len(batch), n), -100); att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):                   # right padding for training
        L = len(b["input_ids"]); ids[i, :L] = torch.tensor(b["input_ids"]); lab[i, :L] = torch.tensor(b["labels"]); att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.config.use_cache = False

args = TrainingArguments(output_dir=f"{OUT}/run", num_train_epochs=4, max_steps=3 if SMOKE else -1, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                         learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0, fp16=True,
                         logging_steps=10, eval_strategy="epoch", save_strategy="no", per_device_eval_batch_size=4,
                         report_to="none", seed=SEED, gradient_checkpointing=True, optim="paged_adamw_8bit")
trainer = Trainer(model=model, args=args, train_dataset=DS(train_enc), eval_dataset=DS(dev_enc), data_collator=collate)
t0 = time.time(); trainer.train(); print("train minutes:", round((time.time() - t0) / 60, 1))
log = [x for x in trainer.state.log_history]
json.dump(log, open(f"{OUT}/train_log.json", "w"), indent=1)
""")

md("## Evaluate the fine-tuned reader on the same splits")
code("""
tok.padding_side = "left"
model.config.use_cache = True
pred_dump = {}
for name in ("dev", "test_hand", "test_real", "test_bench"):
    preds, sec = predict(model, splits[name], finetuned=True)
    s, kinds, conf = score(splits[name], preds)
    s["sec_per_item"] = round(sec, 2)
    results[f"finetuned/{name}"] = {"overall": s, "by_kind": kinds, "category_confusion": {f"{g}->{p}": n for (g, p), n in conf.items()}}
    pred_dump[name] = [{"id": r["meta"].get("id"), "text": json.loads(r["messages"][1]["content"]).get("text") or json.loads(r["messages"][1]["content"]),
                        "gold": json.loads(r["messages"][2]["content"]), "pred": p, "json_valid": ok} for r, (ok, p) in zip(splits[name], preds)]
    print(name, json.dumps(s))
json.dump(results, open(f"{OUT}/reader_results.json", "w"), indent=1)
with open(f"{OUT}/predictions.jsonl", "w") as f:
    for name, rows in pred_dump.items():
        for r in rows:
            f.write(json.dumps({"split": name, **r}, ensure_ascii=False) + "\\n")
""")

md("## Comparison table and chart: base zero-shot vs fine-tuned")
code("""
import matplotlib.pyplot as plt
fields = ["json_valid", "category", "symptoms_f1", "places_exact_f1", "places_relaxed_f1", "source", "private", "since_exact", "since_relaxed_f1",
          "reply_source", "reply_cause", "reply_still_flowing", "reply_places_f1"]
ref_path = f"{DATA}/llm_reference.json"        # optional: the same metrics for the API reader (computed outside Kaggle), same test files
ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
for name in ("test_hand", "test_real", "test_bench"):
    b = results.get(f"base_zeroshot/{name}", {}).get("overall", {}); f_ = results[f"finetuned/{name}"]["overall"]; a = ref.get(name, {})
    print(f"\\n{name}: {'field':22s} {'base 0-shot':>12s} {'fine-tuned':>12s} {'API reader':>12s}")
    for k in fields + ["instruction_precision", "instruction_recall", "sec_per_item"]:
        if k in b or k in f_ or k in a:
            print(f"{'':10s} {k:22s} {str(b.get(k)):>12s} {str(f_.get(k)):>12s} {str(a.get(k)):>12s}")
fig, axes = plt.subplots(1, 3, figsize=(20, 4), sharey=True)
for ax, name in zip(axes, ("test_hand", "test_real", "test_bench")):
    ks = [k for k in fields if k in results[f"finetuned/{name}"]["overall"]]
    x = range(len(ks))
    ax.bar([i - 0.27 for i in x], [results.get(f"base_zeroshot/{name}", {}).get("overall", {}).get(k, 0) for k in ks], 0.27, label="base zero-shot")
    ax.bar([i for i in x], [results[f"finetuned/{name}"]["overall"].get(k, 0) for k in ks], 0.27, label="fine-tuned")
    if ref.get(name):
        ax.bar([i + 0.27 for i in x], [ref[name].get(k, 0) for k in ks], 0.27, label="API reader")
    ax.set_xticks(list(x)); ax.set_xticklabels(ks, rotation=60, ha="right"); ax.set_title(name); ax.set_ylim(0, 1)
axes[0].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/reader_comparison.png", dpi=130); plt.show()
# acceptance gate for replacing the API reader in AgentiX (proposal Section 6: field F1 for place, symptom, time)
for name in ("test_hand", "test_bench", "test_real"):
    f_ = results[f"finetuned/{name}"]["overall"]
    print(name, {k: f_.get(k) for k in ("json_valid", "category", "symptoms_f1", "places_relaxed_f1", "since_relaxed_f1", "source", "reply_cause")})
print("\\nper kind (fine-tuned, test_hand):")
for k, v in sorted(results["finetuned/test_hand"]["by_kind"].items()):
    print(f"  {k:22s} n={v['n']:3d} category={v.get('category')} source={v.get('source')} since={v.get('since_exact')}")
""")

md("## Save the adapter (plug into AgentiX as `--reader local`)")
code("""
model.save_pretrained(f"{OUT}/reader_adapter"); tok.save_pretrained(f"{OUT}/reader_adapter")
json.dump({"base_model": BASE, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": "all attention + MLP projections"},
           "epochs": 4, "lr": 2e-4, "effective_batch": 16, "max_len": MAX_LEN, "quantization": "4-bit NF4 (QLoRA)",
           "prompt_read": FT_READ_SYSTEM, "prompt_reply": FT_REPLY_SYSTEM, "contract": "reader-2.0 / reply-2.0"},
          open(f"{OUT}/reader_adapter/agentix_reader.json", "w"), indent=1)
import shutil; shutil.make_archive(f"{OUT}/reader_adapter", "zip", f"{OUT}/reader_adapter")
print(os.listdir(OUT))
""")

md("## Merge the adapter into the base model (needed to run the reader locally on the Mac; takes about 5 min and writes about 6 GB)")
code("""
MERGE = True       # writes a merged fp16 model (about 6 GB) to /kaggle/working/reader_merged; set False to keep only the small adapter
if MERGE and not SMOKE:
    try:
        del model, trainer
        import gc; gc.collect(); torch.cuda.empty_cache()
        from peft import PeftModel
        base16 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map={"": 0})
        merged = PeftModel.from_pretrained(base16, f"{OUT}/reader_adapter").merge_and_unload()
        merged.save_pretrained(f"{OUT}/reader_merged", safe_serialization=True); tok.save_pretrained(f"{OUT}/reader_merged")
        print("merged model saved")
    except Exception as e:
        print("merge skipped:", repr(e))
""")

md("## Pack everything into ONE file: `/kaggle/working/agentix_reader_outputs.zip` (download this single file and send it back)")
code("""
import zipfile, shutil
KEEP_LOOSE = False       # True keeps the separate files next to the zip; False leaves only the zip in Output (smaller, one download)
shutil.rmtree(f"{OUT}/run", ignore_errors=True)         # training checkpoints are not needed
ZIP = f"{OUT}/agentix_reader_outputs.zip"
items = ["reader_results.json", "predictions.jsonl", "reader_comparison.png", "train_log.json", "reader_adapter", "reader_merged"]
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED, allowZip64=True) as z:       # stored: model weights do not compress, so this is fast
    for it in items:
        path = f"{OUT}/{it}"
        if os.path.isfile(path):
            z.write(path, it)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, OUT))
        else:
            print("not found (skipped):", it)
print("zip written:", ZIP, round(os.path.getsize(ZIP) / 1e9, 2), "GB")
with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    print("integrity:", "ok" if bad is None else f"BAD member {bad}", "| members:", len(z.namelist()))
if not KEEP_LOOSE:
    for it in items + ["reader_adapter.zip"]:
        path = f"{OUT}/{it}"
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else (os.remove(path) if os.path.exists(path) else None)
print(os.listdir(OUT))
""")

md("""
## How to use the result in AgentiX
1. From the notebook's Output tab download the single file `agentix_reader_outputs.zip` (results, predictions, chart, training log, adapter and the merged model, about 6 GB) and unzip it.
2. Put the merged folder at `finetune/models/reader_qwen3b_hf/` in the project, then on the Mac: `zsh finetune/convert_reader.sh`
   (4-bit MLX model, about 2 GB) and `zsh finetune/serve_reader.sh` (OpenAI-compatible server on 127.0.0.1:8471).
3. Run the evaluation with `export AGENTIX_READER_URL=http://127.0.0.1:8471 AGENTIX_READER_TUNED=1` and `--reader local`.
4. Report the numbers in `docs/READER_FINETUNE.md`: base zero-shot vs fine-tuned vs API reader, field by field, on `test_hand`, `test_real` and `test_bench`.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT, len(cells), "cells")
