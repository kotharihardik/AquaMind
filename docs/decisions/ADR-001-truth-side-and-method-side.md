# ADR-001: The simulated world and the methods are kept apart

**Status:** in force

## Problem
We test leak-finding methods in a simulated town. If a method could look at the simulator's hidden truth (where the leak really
is, the exact pipe model that made the data), it would look far better than it ever could in a real city. This mistake is called an
"inverse crime".

## Decision
The code is split into two sides that never mix:

| Truth side (`aquamind/sim/`) | Method side (everything else) |
|---|---|
| The real town: `L-TOWN_Real.inp` (BattLeDIM 2020) | The utility's own, rougher map: `L-TOWN.inp` (nominal model) |
| The hidden leak, real hydraulics, real sensor data | Only what a utility would see: sensor readings, messages |
| Citizens who answer calls, field crews, the operator | Tools that ask the truth side for a result |

- A method may **not** import `aquamind/sim`. It reads a case only through `aquamind/case.py`.
- A method gets results from the field only through the case environment (a call, a step test, a dig), exactly like a utility
  would get them from people.
- Hidden facts of a case (for example the true leak) are stored apart, in `hidden/`, and only the scorer reads them.

## Why
So that every number we report is something a real utility could reach with the same information.

## Costs and limits
The methods' physics is always a little wrong (rougher pipe model, unknown demands). That is on purpose: it is what real
utilities live with.

## Where in the code
`aquamind/sim/` (truth), `aquamind/case.py` (the only door to a case), `configs/ltown.yaml` (which model each side uses).
