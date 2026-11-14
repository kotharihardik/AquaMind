# ADR-005: One evidence board, computed by code

**Status:** in force

## Problem
Evidence comes in many forms: pressure drops, flow changes, messages, call-back answers, old repair tickets, step tests, listening.
An LLM is bad at adding these up into a trustworthy number, and a wrong number can send a crew to dig in the wrong street.

## Decision
Code keeps one **evidence board**: a leak map with one probability for each of the 782 junctions.

- Each junction's score = its prior (pipe length, failure history) + how well the pressure data fits a leak there
  + what the messages say + what the field checks found.
- Each message keeps its own weight, including the **chance it is a false alarm**. A call-back answer or a record can change that
  weight later. Several messages about the same spot are not counted twice in full.
- Two numbers drive every decision:
  - **P(leak):** is there a leak at all?
  - **P(success):** if we dig at the best spot now, will we hit the leak?
- The board changes **only** when a tool returns a result. An agent's opinion never changes it directly.

## Why
Numbers that decide where to dig must be correct, repeatable and explainable. Code gives the same answer every time, and every
change on the board can be traced to one piece of evidence.

## Costs and limits
The board is only as good as its models of sensors, people and tools. Those are simple on purpose (ADR-003).

## Where in the code
`aquamind/search/board.py`, `aquamind/search/pressure.py`, `aquamind/search/report_lik.py`, `aquamind/search/investigation.py`.
