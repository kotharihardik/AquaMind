# ADR-011: A fair evaluation, with every number taken from logged rows

**Status:** in force

## Problem
A comparison is only worth something if every method gets the same chances and the numbers can be checked later.

## Decision
- **Same conditions for every method:** the same 210 cases, sensor layouts, field budget (ADR-003), reader (ADR-004) and LLM
  (`nvidia/nemotron-3-ultra-550b-a55b` for every agent and every LLM baseline). The same random luck too: seeds come from the case
  name (crc32), so the same question to the same person gets the same answer in every method.
- **One LLM client** with a shared rate limiter and a disk cache (`.cache/llm`): a re-run with the same prompts costs nothing
  and gives the same answers.
- **Methods compared:**
  - Ours: `agentix_crew`.
  - Baselines: `react_agent` (standard ReAct, same LLM and tools), `plain_llm` (one LLM call), `fixed_rules`, `crew_practice`
    (what crews do today), classical model inversion (EPANET), and one-look methods for RQ1.
  - Ablations: remove one part at a time (no messages, no step test, no RAG, no gate, one shot, no AI, code only) and one agent at
    a time (ADR-013).
- **Main metrics:** cases solved, correct digs (leak within 100 m), median distance error, dry holes, unsafe actions, rule
  violations, traps handled, tool-call accuracy.
- **Every number comes from the result rows** in `logs/evaluation/`. Each run is a new log version: v1 = mid report, the newest =
  final. Old versions are never overwritten.

## Why
So that differences come from the methods themselves, and anyone can re-compute every table from the logs.

## Costs and limits
One simulated town and one LLM. The free API tier is slow, so a full evaluation takes many hours.

## Where in the code
`scripts/run_eval.py`, `aquamind/eval/` (runner, metrics, logger, splits), `aquamind/llm/client.py`, `aquamind/baselines/`, `aquamind/ablations/`.
