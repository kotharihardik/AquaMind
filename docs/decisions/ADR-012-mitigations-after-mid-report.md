# ADR-012: Five fixes after the mid report (M1-M5)

**Status:** in force (all five on by default)

## Problem
The mid-report logs showed five repeating failures of the crew: weak clues were thrown away, the agents ran out of turns too early,
the last dig was sent without a final check, agents cited the wrong kind of archive ticket, and permits were missed at street corners.

## Decision
Each fix targets one failure seen in the logs:

| Fix | Failure seen | What changed |
|---|---|---|
| **M1 Weak leads** | A step test with only weak extra flow, or a quiet listening check, was treated as "nothing here" | Weak-flow areas can be tested again and dug; a quiet area that still holds much of the map can be listened to again |
| **M2 Turn budget** | The Forensic used its turns before the field results came in | More Forensic turns (up to 6) and LLM calls (up to 20); the Skeptic gets one turn after the first quiet check, to question *where* the leak is |
| **M3 Confirm dig** | The last allowed dig was sent on weak support | Before the last dig, code listens at the spot first; an Auditor objection with no checks left hands the case over instead of digging |
| **M4 Ticket type** | A utility repair ticket was cited as proof that a complaint is a false alarm | Only non-utility tickets count as false alarms; refused citations are shown to the agents |
| **M5 Corner permits** | A dig at a corner checked only one street for a permit | Every street at the dig junction is checked |

Each fix can be switched off (`AGENTIX_MITIGATIONS`, or the `m1..m5` switches) to measure it.

## Why
A fix is kept only if it helps on the cases without hurting others. On 5 sensors the crew went from 163 solved cases (mid) to 172 (final).

## Costs and limits
The mid-to-final gain also includes other changes (bug fixes, the fine-tuned reader). Case by case, the difference is not
statistically significant on its own (McNemar p = 0.09), so we report it as an improvement, not as proof.

## Where in the code
`aquamind/agent/crew.py` and `aquamind/agent/toolbox.py` (look for `m1_weak_leads` ... `m5_corner_permits`), `tests/test_mitigations.py`.
