# ADR-009: Code checks every claim an agent makes

**Status:** in force

## Problem
LLMs sound confident even when they are wrong: they cite a record that does not fit, say water "runs down" from a place that is
lower, or pick a field check that has little chance to help. If these claims went straight onto the board, one mistake would spread.

## Decision
Every claim passes a code check before it counts:

| Agent claim | Code check |
|---|---|
| "Record REC-x shows this complaint is a known false alarm" | The record must exist, be a non-utility ticket, and be at the message's place |
| "Record REC-x shows an old, failing main here" | The record must be a utility repair; then the failure chance of that stretch is raised (x4) |
| "Water runs down from caller A to callers B, C" | Each downhill caller must really be lower than A and within reach along the pipes |
| "Hypothesis H, supported by messages R-1, R-2" | Each cited message must exist and point within 300 m of the place |
| "Run field check M4 next" (Strategist) | **Chance guard:** if M4's chance of finding the leak is below 60 % of the best option, code runs the best option instead, unless M4 separates the hypotheses clearly better |

Refused claims are logged and shown to the agents as "rejected citations", so they can correct themselves.

## Why
The agents bring language and judgement (what to ask, what to suspect). Code makes sure that judgement never turns into a wrong number.

## Costs and limits
The guard makes the Strategist's freedom smaller. The ablation `agentix_crew_no_override` turns the guard off to measure it.
Strict checks also mean more refused tool calls in the logs; that is the checks working.

## Where in the code
`aquamind/agent/toolbox.py` (`use_record`, `use_repair_record`, `triangulate`), `aquamind/search/hypotheses.py`, `aquamind/agent/crew.py` (`_guard`, `chance`).
