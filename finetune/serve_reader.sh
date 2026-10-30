#!/bin/zsh
# Serve the fine-tuned reader locally (OpenAI-compatible API on 127.0.0.1:8471): one copy of the model in memory, shared by all evaluation workers.
# aquamind/reader/local_reader.py talks to it when AGENTIX_READER_URL is set (export AGENTIX_READER_TUNED=1 too: the model learned the short prompt).
cd "$(dirname "$0")/.."
MODEL=${MODEL:-finetune/models/reader_qwen3b_mlx}
exec .venv/bin/python -I -m mlx_lm server --model $MODEL --host 127.0.0.1 --port 8471
