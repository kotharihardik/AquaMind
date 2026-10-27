#!/bin/zsh
# LoRA fine-tuning of the complaint reader on a laptop (Apple silicon, MLX). Loss only on the answer (--mask-prompt).
# Base model: Qwen2.5-1.5B-Instruct (4-bit) by default; set BASE to change it. GPU / Colab (QLoRA 3B): README.md.
# usage: zsh finetune/train_mlx.sh [iters]
set -e
cd "$(dirname "$0")/.."
BASE=${BASE:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}
ITERS=${1:-600}
D=data/finetune/mlx
mkdir -p $D finetune/adapters
.venv/bin/python -I -c "
import json
for src, dst in (('data/finetune/train.jsonl', '$D/train.jsonl'), ('data/finetune/dev.jsonl', '$D/valid.jsonl')):
    with open(dst, 'w') as f:
        for l in open(src):
            f.write(json.dumps({'messages': json.loads(l)['messages']}, ensure_ascii=False) + '\n')
"
.venv/bin/python -I -m mlx_lm lora --model $BASE --train --data $D --iters $ITERS --batch-size 2 \
  --num-layers 16 --learning-rate 1e-4 --mask-prompt --max-seq-length 768 --grad-checkpoint \
  --steps-per-eval 100 --val-batches 20 --save-every 200 --adapter-path finetune/adapters/reader
