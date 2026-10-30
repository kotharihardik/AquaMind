# Complaint reader: data, training, evaluation

Full guide: `docs/READER_FINETUNE.md`. Contract: `aquamind/reader/schema.py` (reader-2.0 / reply-2.0).

```bash
# 1. data (synthetic, gold first; LLM re-wording with Gemini personas; held-out town + writer for test_synth)
.venv/bin/python -I finetune/build_dataset.py 2400            # -> data/finetune/{train,dev,test_synth}.jsonl
# 2. real test messages (slow and polite; never used for training, never redistributed)
.venv/bin/python -I finetune/real/fetch_fixmystreet.py 300   # -> data/finetune/real/fixmystreet_raw.jsonl
# 3. train on the laptop (MLX LoRA, Qwen2.5-1.5B-Instruct 4-bit) -> finetune/adapters/reader
zsh finetune/train_mlx.sh 600
# 4. evaluate readers (rules / llm = large prompted model / local = fine-tuned)
.venv/bin/python -I finetune/evaluate.py data/finetune/test_synth.jsonl local --out finetune/results_local.json
```

GPU alternative (Colab T4, 3B): load `data/finetune/train.jsonl` (chat format) with Unsloth `FastLanguageModel`
(`unsloth/Qwen2.5-3B-Instruct-bnb-4bit`), LoRA r=16 / alpha=32 on all projections, 2 epochs, lr 2e-4, max length
768, train on responses only. Export the adapter, or merge and convert to GGUF Q4_K_M for llama.cpp.
