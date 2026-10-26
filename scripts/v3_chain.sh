#!/bin/zsh
# Benchmark v3 (ADR-010) evaluation without LLM calls: value of complaints vs the field-check budget, per kind of evidence.
# usage: [BENCH=data/benchmark/agentix_v4 OUT=runs/v4] zsh scripts/v3_chain.sh dev|val [template|realistic]
cd "$(dirname "$0")/.."
SPLIT=${1:-dev}; TEXTV=${2:-template}; B=${BENCH:-data/benchmark/agentix_v3}; O=${OUT:-runs/v3}; mkdir -p $O
R=(.venv/bin/python -I -u scripts/run_eval.py --bench $B --split $SPLIT --text $TEXTV --reader rules --workers 3)
M=agentix_code,agentix_no_reports_code,agentix_code_brown,agentix_code_brown_nowork,fixed_rules,crew_practice
for k in 5 3 10; do $R --methods $M --layouts spread_k$k --out $O/${SPLIT}_rows.jsonl >> $O/${SPLIT}.log 2>&1; done
for c in 3 2; do $R --methods agentix_code,agentix_no_reports_code,agentix_code_brown --layouts spread_k5 --max-checks $c --out $O/${SPLIT}_rows.jsonl >> $O/${SPLIT}.log 2>&1; done
echo done >> $O/${SPLIT}.log
