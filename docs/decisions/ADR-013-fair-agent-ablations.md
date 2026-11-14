# ADR-013: Fair agent ablations: when an agent is removed, code must not do its job

**Status:** in force (13 Nov 2026). Changes the "No Forensic" ablation only; the full crew and all other methods are unchanged.

## Problem
An agent ablation asks: **"What happens if this agent is missing?"** To answer honestly, the agent must really be missing.

In the first final run, the crew without the Forensic solved 171 of 210 cases, almost the same as the full crew (172). At mid time,
the same ablation found only 110 of 185 leaks. The reason was not the agents but three pieces of code that did the Forensic's job:

1. **A ready-made map sweep.** The Forensic's job is to say *where to look*. The wide "exploration" step test was offered even
   with no hypothesis, so code handed the Strategist a "look here" option (used 351 times).
2. **An extra agent.** Fix M2 called the Skeptic to challenge the Forensic's hypotheses, even when there was no Forensic (168 turns).
3. **Dig areas made by code.** Fix M1 turned weak-flow areas into dig areas, which in the full crew is the Forensic re-thinking.

## Decision
When the Forensic is switched off, these three helpers are switched off too:

| Helper | Full crew | Without the Forensic |
|---|---|---|
| Exploration step test in the menu | offered | not offered; the Strategist must propose its own |
| Skeptic turn after a quiet check (M2) | yes | no |
| Weak-flow areas become dig areas (M1) | yes | no |

The fixed fallback still names where to look (as at mid time), and the Strategist, gate, Auditor and fixes M3-M5 still work.
The other agent ablations were checked and needed no change: nothing does the Skeptic's, Auditor's or triangulation's job, and
without the Strategist only a simple rule picks the checks.

## Why
Otherwise the ablation measures what code can do in the agent's place, not what the agent adds.

## Costs and limits
The "No Forensic" number goes down compared with the first final run. That is the honest number.

## How it was checked
A one-case test (no sweep, no Skeptic turn without the Forensic), then a full re-run of all 210 cases as log version v4 in
`logs/evaluation/ablations/no_forensic/k5/` (v3, the first run, is kept).

## Result (5 sensors, 210 cases)

| Run | Solved | Correct digs | Median error | Dry-hole rate | Traps |
|---|---|---|---|---|---|
| Full crew | **172** | **79.5 %** | 23 m | 11.4 % | 13/13 |
| No Forensic, first final run (unfair, v3) | 171 | 79.5 % | 23 m | 9.8 % | 13/13 |
| **No Forensic, fair (v4)** | **116** | **50.3 %** | 74 m | 18.4 % | 11/13 |
| No Forensic, 29 Oct mid re-run (for reference) | 115 | 49.2 % | 75 m | 20.2 % | 11/13 |

The fair run matches the mid-time re-run almost exactly. Without the Forensic, the crew solves 56 fewer cases: naming where to
look, and asking people the right questions, is the most valuable job in the crew.

## Where in the code
`aquamind/agent/crew.py` (`Crew.menu`, `Crew.run`), `aquamind/agent/toolbox.py` (`weak_dig_areas`), `aquamind/ablations/roles.py`.
