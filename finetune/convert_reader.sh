#!/bin/zsh
# Convert the merged Kaggle model (Hugging Face format) to a 4-bit MLX model for the Mac. Run once after downloading finetune/models/reader_qwen3b_hf/.
cd "$(dirname "$0")/.."
exec .venv/bin/python -I -m mlx_lm convert --hf-path finetune/models/reader_qwen3b_hf --mlx-path finetune/models/reader_qwen3b_mlx -q
