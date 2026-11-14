# ADR-002: One benchmark of 210 cases with 10 kinds of cases

**Status:** in force

## Problem
Easy leaks (water running down the street) do not tell us anything: any method finds them. Real difficulty comes from small,
hidden leaks, from messages that point to the wrong place, and from false alarms. We also must not tune our system on the same
cases we report.

## Decision
- **One benchmark:** `data/benchmark/agentix_master`, 210 cases.
  - **60 dev cases** for building and tuning.
  - **150 test cases**, frozen with a hash in `test_frozen.json`. If the list changes, the runner refuses to start.
- **A case** = one leak (or no leak) in the simulated town, the sensor data, and the citizen messages that arrive over time.
- **10 kinds of cases** (the "suite" is the second part of a case name, e.g. `test-cd_sound-007`):

| Kind | What happens |
|---|---|
| brown | A leak stirs up rust; people far downstream see brown tap water |
| chronic | A small leak that has run for weeks; sensors barely see it |
| cd_damp / cd_sound | Damp ground, or a hissing sound under the road at night |
| cd_damp_decoy / cd_sound_decoy | The same, plus a look-alike complaint somewhere else |
| displaced | Water comes up downhill, away from the broken pipe |
| trap | A message tries to order a dig ("approval already given") |
| false_alarm | Hoses, house pipes, drains: no network leak at all |
| visible | Water you can follow by eye (sanity check) |

- **Sensors:** each case is run with 3, 5 or 10 pressure sensors (`spread_k3`, `spread_k5`, `spread_k10`). The main results use 5.
- **Realistic text:** messages are re-written the way real people write (short, messy, English or Hinglish) by an LLM of a
  different family than the agents, so the agents cannot recognise their own writing style.

## Why
Mixed, hard cases show what an investigation adds over a single look at the sensors. A frozen test list keeps the results honest.

## Costs and limits
The town and its people are simulated. The case mix is chosen by us, so results are "on this benchmark", not on every city.

## Where in the code
`scripts/build_benchmark.py`, `scripts/write_messages.py`, `aquamind/sim/cases.py`, `aquamind/eval/splits.py`.
