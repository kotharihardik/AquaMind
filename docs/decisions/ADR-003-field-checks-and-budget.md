# ADR-003: The same small field budget for every case

**Status:** in force

## Problem
With unlimited field work, any method finds every leak (just test everywhere). A real utility has a few crews, a few nights and
strict rules about digging up roads. The interesting question is how well a method uses a **small** budget.

## Decision
Every method, in every case, gets the same budget and the same tools (`configs/field.yaml`):

| Item | Limit | What it is |
|---|---|---|
| Field checks | **5** | A night step test, a listening survey or a temporary logger each uses one |
| Digs | **2** | SOP-2: at most two excavations per case |
| Time | **5 days** | The case must be closed in time |
| Call-backs | free, at most 2 per message | 75 % of people answer, after 0.5-3 hours |

The field tools behave like real ones, with errors:
- **Night step test:** close the valves of up to 6 sections one by one and watch the extra night flow. A valve can fail (8 %).
- **Listening survey:** within 250 m. A small leak is often not heard; traffic is sometimes mistaken for a leak (5 %).
- **Dig:** finds the leak only if it is within 100 m of pipe.

## Why
This turns leak finding into a real decision problem: which check next, where, and when is the evidence strong enough to dig.

## Costs and limits
The numbers (noise, hearing chance, answer rate) are reasonable guesses from practice, not measured in one real city.

## Where in the code
`configs/field.yaml`, `aquamind/sim/field.py` (truth), `aquamind/search/tool_models.py` (the methods' own, rougher idea of the tools).
