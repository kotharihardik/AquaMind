#!/bin/zsh
# Phase 3 test runs AFTER the recheck double-counting fix (same frozen cases). Methods whose code path is not
# touched by the fix (react_llm, oneshot_llm) are copied from the first test run.
cd /Users/hardikkothari/Pictures/Resume_project/Water_Pipeline
while pgrep -f "run_agent_eval.py --split test" > /dev/null || pgrep -f "out runs/phase3/dev_v5" > /dev/null; do sleep 15; done
export LLM_MIN_INTERVAL_S=6
run_eval() { .venv/bin/python -I -u scripts/experiments/run_agent_eval.py --split test "$@"; }
mkdir -p runs/phase3/test_v2/traces
cp runs/phase3/test/cases.json runs/phase3/test_v2/cases.json
grep -E '"method": "(react_llm|oneshot_llm)"' runs/phase3/test/rows.jsonl > runs/phase3/test_v2/rows.jsonl
cp -R runs/phase3/test/traces/react_llm runs/phase3/test/traces/oneshot_llm runs/phase3/test_v2/traces/ 2>/dev/null
run_eval --methods greedy_voi,rule_agent,rule_agent_notaint --out runs/phase3/test_v2 --workers 4 > runs/phase3/test_v2/stage0_code.log 2>&1
for pass in 1 2; do
  run_eval --methods agentix_llm,react_llm,oneshot_llm --out runs/phase3/test_v2 --workers 5 > runs/phase3/test_v2/stage1_llm_p$pass.log 2>&1
done
for pass in 1 2; do
  run_eval --methods agentix_llm_nogate,agentix_llm_nocritic,agentix_llm_noabduction --out runs/phase3/test_v2 --workers 5 > runs/phase3/test_v2/stage2_ablations_p$pass.log 2>&1
done
for docs in docs docs_heldout; do
  for pass in 1 2; do
    run_eval --methods rule_agent,agentix_llm,agentix_llm_nokeyword,react_llm --docs $docs --out runs/phase3/test_heldout --workers 5 > runs/phase3/test_heldout/stage3_${docs}_p$pass.log 2>&1
  done
done
echo CHAIN_DONE > runs/phase3/test_v2/CHAIN_DONE
