#!/bin/zsh
# Final evaluation on the FROZEN test set of benchmark v2 (ADR-007, ADR-008). Resumable: finished rows are skipped.
# usage: zsh scripts/final_chain.sh [out_dir]       (default runs/final)
# Needs: test cases + realistic messages built; LLM keys in .env.
#   READER=rules|llm|local   one reader for every method of the run (default rules; local = the fine-tuned reader,
#                            served by `zsh finetune/serve_reader.sh`)
#   TEXTV=realistic|template  realistic needs the Gemini re-wording of the test messages (scripts/write_messages.py <bench> test)
#   REACT_LIMIT=60           ReAct needs ~15 LLM calls per case; on the free tier we run it on the first N test cases
# The free NVIDIA tier sustains ~5-8 successful calls a minute, so the LLM parts take hours. Do not start a second
# copy while one is running (workers that outlive their parent eat the rate limit).
cd "$(dirname "$0")/.."
F=${1:-runs/final}
B=${BENCH:-data/benchmark/agentix_v4}
READER=${READER:-rules}
TEXTV=${TEXTV:-realistic}
REACT_LIMIT=${REACT_LIMIT:-60}
mkdir -p $F
R=(.venv/bin/python -I -u scripts/run_eval.py --bench $B --split test --text $TEXTV --reader $READER)
CODE=agentix_code,agentix_no_reports_code,fixed_rules,crew_practice
EXPLORE=agentix_code_canvass,agentix_no_reports_canvass,agentix_code_brown,agentix_code_brown_nowork
ABL=agentix_one_shot,agentix_no_rag,agentix_no_gate,agentix_no_fieldchecks,agentix_no_reports

# RQ1: one look, 3/5/10 sensors x 4 layouts (code only, same reader as the agent), by kind of evidence
[ -s $F/one_look.jsonl ] || .venv/bin/python -I scripts/run_one_look.py --bench $B --split test --text $TEXTV --reader $READER \
  --out $F/one_look.jsonl > $F/one_look.log 2>&1

# RQ2 + main system: every cheap method on 3 / 5 / 10 sensors (spread layouts), all test cases
for k in 5 3 10; do
  $R --methods $CODE,$EXPLORE --layouts spread_k$k --workers 2 --out $F/rq2.jsonl >> $F/rq2.log 2>&1
done
# AgentiX (about 1 LLM call per case)
for k in 5 3 10; do
  $R --methods agentix --layouts spread_k$k --workers 3 --out $F/rq2.jsonl >> $F/rq2.log 2>&1
done
# plain LLM (1 call per case) and ReAct (up to 15 calls per case, first REACT_LIMIT cases, 5 sensors)
$R --methods plain_llm --layouts spread_k5 --workers 3 --out $F/rq2.jsonl >> $F/rq2.log 2>&1
$R --methods react_agent --layouts spread_k5 --limit $REACT_LIMIT --workers 3 --out $F/rq2_react.jsonl >> $F/rq2.log 2>&1

# Value of evidence: the same cheap methods with 3 and 2 field checks (ADR-010)
for c in 3 2; do
  $R --methods agentix_code,agentix_no_reports_code,agentix_code_brown --layouts spread_k5 --max-checks $c --workers 2 --out $F/rq2.jsonl >> $F/rq2.log 2>&1
done

# Ablations (proposal Section 11), 5 sensors
$R --methods $ABL --layouts spread_k5 --workers 3 --out $F/ablations.jsonl >> $F/ablations.log 2>&1

.venv/bin/python -I scripts/make_report.py docs/RESULTS.md $F/rq2.jsonl $F/rq2_react.jsonl $F/ablations.jsonl --one-look $F/one_look.jsonl
.venv/bin/python -I scripts/make_budget_report.py docs/RESULTS_BUDGET.md $F/rq2.jsonl
echo done > $F/CHAIN_DONE
