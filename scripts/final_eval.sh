#!/bin/zsh
# Final evaluation split over 4 PCs. Every profile runs ALL 210 cases of benchmark agentix_master (150 frozen test + 60 dev), resumable.
# Only the first agent version (aquamind/agent/crew.py) is evaluated; agent_v2 is not part of any profile.
# usage:   zsh scripts/final_eval.sh <profile>      DRY=1 prints the commands only    WORKERS=3 (one per API key)    LIMIT=n smoke test    STRIDE_ROLES=2 role ablations on every 2nd case
#
#   main    PC-1 (the main PC): agentix_crew on spread_k5, spread_k3, spread_k10          (runs/final_crew/run_crew.sh does the same)
#   pcA     PC-2: [core] react_agent, ablation_no_reports, plain_llm                     then [roles] no_forensic, no_skeptic          (spread_k5)
#   pcB     PC-3: [core] ablation_no_temp_sensor, ablation_no_rag, ablation_one_shot     then [roles] no_strategist, no_auditor        (spread_k5)
#   pcC     PC-4: [core] ablation_no_gate, agentix_crew_no_override                      then [roles] no_sop, no_triangulation         (spread_k5)
#   pcA_rev the main PC after its k10 run: the SAME methods as pcA's core, in reverse order (plain_llm, no_reports, react_agent); results are merged by (method, case), duplicates removed
#   pcB_rev the main PC after pcA_rev: the SAME methods as pcB's core, in reverse order (one_shot, no_rag, no_temp_sensor); merged by (method, case)
#   pcC_roles the main PC: only pcC's two role ablations (no_sop, no_triangulation), without pcC's core list
#   nollm   every method without an LLM on spread_k3/k5/k10 + RQ1 one-look on all 12 layouts   (fast; any PC)
#
# Order inside pcA/pcB/pcC: core methods on test then dev, THEN the role ablations on test then dev (roles last: they answer "what does each agent add").
# Needs on every PC: the same project folder (same code and prompts), data/benchmark/agentix_master, its OWN NVIDIA keys in .env
# (NVIDIA_API_KEY_1.. : never the same key on two PCs), the fine-tuned reader server (zsh finetune/serve_reader.sh, Apple Silicon), power + no sleep.
# Results: runs/final_<profile>/*.jsonl -> copy to the main PC -> scripts/make_metrics.py --rows <all files>.
# Use ONE name per ablation: ablation_X and agentix_crew_X are the same method (the report merges them).
cd "$(dirname "$0")/.."
P=${1:?profile: main|pcA|pcB|pcC|pcA_rev|pcB_rev|pcC_roles|nollm}
W=${WORKERS:-3}
B=${BENCH:-data/benchmark/agentix_master}
OUT=runs/final_$P
mkdir -p $OUT
export AGENTIX_READER_URL=${AGENTIX_READER_URL:-http://127.0.0.1:8471} AGENTIX_READER_TUNED=1

LAYOUTS=(spread_k5); CORE=""; ROLES=""
case $P in
  main)  CORE=agentix_crew; LAYOUTS=(spread_k5 spread_k3 spread_k10);;
  pcA)   CORE=react_agent,ablation_no_reports,plain_llm;                       ROLES=ablation_no_forensic,ablation_no_skeptic;;
  pcB)   CORE=ablation_no_temp_sensor,ablation_no_rag,ablation_one_shot;       ROLES=ablation_no_strategist,ablation_no_auditor;;
  pcC)   CORE=ablation_no_gate,agentix_crew_no_override;                       ROLES=ablation_no_sop,ablation_no_triangulation;;
  pcA_rev) CORE=plain_llm,ablation_no_reports,react_agent;                           ROLES=;;      # the main PC, after k10: pcA's core list from the END, so it meets the PC that runs pcA from the front
  pcB_rev) CORE=ablation_one_shot,ablation_no_rag,ablation_no_temp_sensor;           ROLES=;;      # the main PC after pcA_rev: pcB's core list from the END
  pcC_roles) CORE=; ROLES=ablation_no_sop,ablation_no_triangulation;;
  nollm) CORE=agentix_code,ablation_no_ai,fixed_rules,crew_practice,classical_bruteforce,classical_direct_inversion
         LAYOUTS=(spread_k5 spread_k3 spread_k10);;
  *) echo "unknown profile $P"; exit 2;;
esac

if [ -z "$DRY" ] && ! curl -s -m 5 $AGENTIX_READER_URL/v1/models >/dev/null; then
  if [[ $AGENTIX_READER_URL == http://127.0.0.1* ]]; then
    echo "starting the reader server"; nohup zsh finetune/serve_reader.sh > $OUT/reader_server.log 2>&1 &
    until curl -s -m 5 $AGENTIX_READER_URL/v1/models >/dev/null; do sleep 5; done
  else
    echo "reader server $AGENTIX_READER_URL is not answering"; exit 1
  fi
fi

# stage <name> <methods> <stride>
# ONE growing log per stage: logs/evaluation/<...>/v1_<layouts>_<methods>.txt. A restart appends to it (cases finished earlier are merged in from every
# $OUT/<profile>_<stage>_*.jsonl); a brand-new run of the stage (no results file yet) starts the next version (v2_, v3_, ...).
stage() {
  local name=$1 methods=$2 stride=${3:-1} k split
  local STR=(); [ "$stride" != 1 ] && STR=(--stride $stride)
  local FRESH=""; [ -z "$(ls $OUT/${P}_${name}_*.jsonl 2>/dev/null)" ] && FRESH=--fresh-log
  [ -z "$methods" ] && return
  for k in $LAYOUTS; do
    for split in test dev; do
      echo "=== $(date +%T) profile=$P stage=$name layout=$k split=$split workers=$W"
      CMD=(.venv/bin/python -I -u scripts/run_eval.py --bench $B --split $split --methods $methods --reader local
           --layouts $k --log-rows "$OUT/${P}_${name}_*.jsonl" $FRESH --workers $W ${LIMIT:+--limit $LIMIT} ${STR[@]}
           --out $OUT/${P}_${name}_${k}_${split}.jsonl --title "$P $name $split $k local reader")
      if [ -n "$DRY" ]; then echo "${CMD[@]}"; else "${CMD[@]}" >> $OUT/run_${name}_${k}_${split}.log 2>&1; fi
      FRESH=""
    done
  done
}

stage core "$CORE" 1
stage roles "$ROLES" ${STRIDE_ROLES:-1}

if [ "$P" = nollm ]; then
  for split in test dev; do
    CMD=(.venv/bin/python -I scripts/run_one_look.py --bench $B --split $split --reader local --out $OUT/one_look_${split}.jsonl)
    if [ -n "$DRY" ]; then echo "${CMD[@]}"; else "${CMD[@]}" > $OUT/one_look_${split}.log 2>&1; fi
  done
fi
echo "ALL DONE $(date +%T)"
