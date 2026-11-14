# ADR-007: A crew of four LLM agents with clear jobs

**Status:** in force

## Problem
One LLM that does everything (read, guess, question, plan, check the rules) mixes up its jobs, trusts its own first idea and forgets
the safety steps. We saw this in the standard ReAct agent.

## Decision
Four agents, each one LLM call that returns one JSON object. They work on the evidence board (ADR-005).

| Agent | Its job | What it returns |
|---|---|---|
| **Forensic Advocate** | Read the case, ask callers its own questions, cite records, name hypotheses | questions, records, triangulation, hypotheses, stance |
| **Critical Skeptic** | Attack the leading hypothesis with the physical data | up to 3 challenges, verdict "stands" or "weak" |
| **Hydraulic Strategist** | Choose the next field check from a menu built by code | one option (or its own proposal) |
| **Safety & SOP Auditor** | Before a dig: try to refute it and apply the SOP | approve / object / hold, and the paperwork |

**Order of work:** Forensic first look, then Skeptic attacks, then Forensic answers. Then a loop (at most 14 rounds): new messages go
to the Forensic; if the dig gate is open, the Auditor checks and the dig is sent; otherwise the Strategist picks a check, the result
lands on the board, and the Forensic re-thinks when something important changed.

**Budgets:** at most 20 LLM calls per case, at most 6 Forensic turns, at most 6 Strategist turns.

## Why
Separate roles make the agents argue with each other (advocate against skeptic) and keep planning and safety from being forgotten.
Each role gets only the context it needs, which keeps prompts short.

## Costs and limits
More LLM calls than a single agent (about 12 per case). The role ablations (ADR-013) measure what each agent really adds.

## Where in the code
`aquamind/agent/crew.py` (prompts, roles, the loop), `aquamind/methods.py` (`agentix_crew`).
