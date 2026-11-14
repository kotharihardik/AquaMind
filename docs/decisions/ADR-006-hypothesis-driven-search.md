# ADR-006: Simulate only the places someone suspects

**Status:** in force

## Problem
A simple approach simulates a leak at every one of the 782 junctions and picks the best match. That is slow, and it hides the
reasoning: nobody had to think about *where* the leak might be. Real engineers start from a suspicion and test it.

## Decision
- The search works with **hypotheses**: "a leak in section S05", "these complaints are private pipes", "there is no leak".
  At most 4 at a time.
- When a hypothesis names a place, code **simulates a leak there** on the hydraulic model (WNTR, 3 L/s at each junction of the area)
  and compares the predicted pressure changes with the real sensors.
- Code gives each hypothesis a **mass**: the share of the leak map inside it.
- A dig is allowed **only at a junction that was hypothesised** (simulated). Until someone suspects a place, P(success) is 0.
- Physical proof is always added: a section that lights up in a step test, or a heard leak, becomes a candidate even if no agent
  named it.

## Why
It makes reasoning visible and testable: guess, predict with physics, check against data, keep or drop the guess.
It is also faster: only the suspected places are simulated, and results are cached.

## Costs and limits
If nobody suspects the right area, the search can miss it. The wide "exploration" step test and the physical-proof rule are there
to catch that.

## Where in the code
`aquamind/search/hypotheses.py`, `aquamind/search/physics_tool.py`, `aquamind/search/investigation.py` (`add_candidates`).
