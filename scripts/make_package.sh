#!/bin/zsh
# Build ONE zip for another PC:   zsh scripts/make_package.sh              -> dist/agentix_full_pack.zip  (everything needed to run, incl. the fine-tuned reader, ~1.7 GB)
#                                 zsh scripts/make_package.sh --no-reader  -> the same without the reader model (~130 MB; the PC then needs the model from elsewhere)
# Contents: the whole codebase, tests, configs, docs, the benchmark data/benchmark/agentix_master (the SAME 210 cases incl. the frozen test list), the town assets,
# the two EPANET models, the fine-tuned reader (MLX, Apple silicon). NOT included: .env (API keys: every PC uses its own), .venv, .git, logs, runs, .cache, __pycache__, raw SCADA csv, pdf/zip files.
cd "$(dirname "$0")/.."; ROOT=$(pwd)
STAGE=${STAGE:-$(mktemp -d)/agentix_pack}
OUT=dist
mkdir -p $OUT $STAGE
EXC=(--exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='*.zip' --exclude='*.pdf' --exclude='.cache' --exclude='/en??????')

# ---- code, tests, configs, docs
rsync -a $EXC agentix configs scripts tests docs pyproject.toml CLAUDE.md README.md $STAGE/
mkdir -p $STAGE/finetune && cp finetune/*.sh finetune/*.py finetune/README.md $STAGE/finetune/
# ---- data: the benchmark of this evaluation, the town assets, the two EPANET models
mkdir -p $STAGE/data/benchmark $STAGE/data/raw/ltown
rsync -a $EXC data/benchmark/agentix_master $STAGE/data/benchmark/
rsync -a $EXC data/ltown $STAGE/data/
cp data/raw/ltown/L-TOWN.inp data/raw/ltown/L-TOWN_Real.inp $STAGE/data/raw/ltown/
mkdir -p $STAGE/logs $STAGE/runs
if [ "$1" != "--no-reader" ]; then
  mkdir -p $STAGE/finetune/models && rsync -a finetune/models/reader_qwen3b_mlx $STAGE/finetune/models/ || { echo "reader model missing: finetune/models/reader_qwen3b_mlx"; exit 1; }
fi

# ---- environment: exact versions of the environment this was built and tested in
.venv/bin/python -m pip freeze 2>/dev/null | grep -v '^-e \|^agentix\|file://' > $STAGE/requirements.txt
grep -viE '^(mlx|mlx-lm|mlx-metal)==' $STAGE/requirements.txt > $STAGE/requirements-core.txt
cat > $STAGE/.env.example <<'EOF'
# Copy to .env and fill in THIS PC's OWN keys (never the same key on two PCs: they would share one rate limit). Never commit or share .env.
NVIDIA_API_KEY_1=
NVIDIA_API_KEY_2=
NVIDIA_API_KEY_3=
EOF

cat > $STAGE/setup_pc.sh <<'EOF'
#!/bin/zsh
# One-time setup of this PC: python environment with the exact versions the evaluation was built with.
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}; command -v $PY >/dev/null || PY=python3
$PY -m venv .venv || exit 1
.venv/bin/python -m pip install -q --upgrade pip
if [ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ]; then REQ=requirements.txt; else REQ=requirements-core.txt; echo "not an Apple-silicon Mac: the local reader (MLX) is unavailable here"; fi
.venv/bin/python -m pip install -q -r $REQ && .venv/bin/python -m pip install -q -e . --no-deps && echo "setup done ($REQ)"
[ -f .env ] || { cp .env.example .env; echo "-> now put YOUR OWN NVIDIA keys into .env"; }
EOF

cat > $STAGE/verify_pack.sh <<'EOF'
#!/bin/zsh
# Checks that this PC has exactly the files (code, configs, benchmark cases, frozen test list) the main PC evaluates with.
cd "$(dirname "$0")"
if command -v shasum >/dev/null; then shasum -a 256 -c MANIFEST.sha256 --quiet && echo "files: all $(wc -l < MANIFEST.sha256) files identical"; else sha256sum -c MANIFEST.sha256 --quiet && echo "files: identical"; fi || { echo "FILES DIFFER"; exit 1; }
.venv/bin/python -I - <<'PY'
import hashlib, json
f = json.load(open("data/benchmark/agentix_master/test_frozen.json"))
ok = hashlib.sha256("\n".join(f["cases"]).encode()).hexdigest() == f["sha256"]
dev = json.load(open("data/benchmark/agentix_master/dev.json"))["cases"]
print(f"benchmark: {len(dev)} dev + {len(f['cases'])} frozen test = {len(dev) + len(f['cases'])} cases; frozen-test hash {'OK' if ok else 'MISMATCH'}")
import os
print("reader model:", "present" if os.path.exists("finetune/models/reader_qwen3b_mlx/model.safetensors") else "MISSING (needed for --reader local)")
from aquamind.methods import registry
r = registry(None, "local")
print(f"methods registered: {len(r)}; agentix_crew {'yes' if 'agentix_crew' in r else 'MISSING'}")
import os
keys = [k for k in os.environ if k.startswith("NVIDIA_API_KEY")] or [l.split("=")[0] for l in open(".env") if l.startswith("NVIDIA_API_KEY") and l.split("=", 1)[1].strip()]
print(f"NVIDIA keys in .env: {len(keys)} (use your own, not the main PC's)")
PY
EOF
chmod +x $STAGE/setup_pc.sh $STAGE/verify_pack.sh $STAGE/scripts/*.sh 2>/dev/null

cat > $STAGE/SETUP_AND_RUN.md <<'EOF'
# AquaMind evaluation pack: set up and run on another PC

Same code, same prompts, same 210 benchmark cases (60 dev + 150 frozen test) as the main PC. Results of all PCs are merged afterwards.

1. **Unzip** `agentix_full_pack.zip` (anywhere), `cd agentix_pack`.
2. **Setup**:  `zsh setup_pc.sh`   (needs Python 3.12; creates `.venv` with the exact package versions). Put YOUR OWN 3 NVIDIA keys into `.env`.
3. **Reader** (the fine-tuned reader is already inside: `finetune/models/reader_qwen3b_mlx`; Apple-silicon Mac only): `zsh finetune/serve_reader.sh &`
   (leave it running; the run scripts start it if it is not up).
4. **Check**:  `zsh verify_pack.sh`   -> must print: files identical, 210 cases / frozen-test hash OK, reader model present, agentix_crew registered, your keys.
5. **Preview and smoke test**:  `DRY=1 zsh scripts/final_eval.sh pcA`  (prints the commands),  `LIMIT=2 zsh scripts/final_eval.sh pcA`  (2 cases).
   Smoke-test results are in `runs/final_pcA/` and `logs/evaluation/`: delete both folders before the real run.
6. **Run your profile** (power plugged in, lid open, no sleep; ~25 h; it can be stopped and restarted: finished cases are skipped, logs append):

   | PC | command | runs |
   |---|---|---|
   | PC-2 | `zsh scripts/final_eval.sh pcA` | react_agent, ablation_no_reports, plain_llm, then roles no_forensic, no_skeptic |
   | PC-3 | `zsh scripts/final_eval.sh pcB` | ablation_no_temp_sensor, ablation_no_rag, ablation_one_shot, then roles no_strategist, no_auditor |
   | PC-4 | `zsh scripts/final_eval.sh pcC` | ablation_no_gate, agentix_crew_no_override, then roles no_sop, no_triangulation |

   Do NOT run the `main` profile (that is the main PC). `STRIDE_ROLES=2` runs the role ablations on every 2nd case (about 6 h less).
7. **Send back** `runs/final_<profile>/` (the `*.jsonl` result files) and `logs/evaluation/`. The log folders per method and layout are described in `docs/EVALUATION_LOGS.md`.
8. If the run stops: just run the same command again.
EOF

# ---- manifest of every file (code, configs, benchmark, assets): the other PC verifies it is identical
(cd $STAGE && find . -type f ! -name MANIFEST.sha256 ! -name '.env' | sort | xargs shasum -a 256 > MANIFEST.sha256)
rm -f $OUT/agentix_full_pack.zip
(cd $STAGE/.. && zip -1 -qr -X "$ROOT/$OUT/agentix_full_pack.zip" agentix_pack)          # one zip, top folder agentix_pack/
echo "files in pack: $(wc -l < $STAGE/MANIFEST.sha256) | stage: $(du -sh $STAGE | cut -f1) | zip: $(du -sh $OUT/agentix_full_pack.zip | cut -f1)"
ls -la $OUT
echo "stage dir (can be deleted): $STAGE"
