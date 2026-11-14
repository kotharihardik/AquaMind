# ADR-010: Search the SOP and the town archive with BM25; also write the SOP as code

**Status:** in force

## Problem
A real investigation uses documents: the standard operating procedure (SOP) and the archive of old tickets (repairs, false alarms).
The agents should find the right document for the case, and the rules must be checked even if an agent forgets one.

## Decision
- **Two documents** (`data/ltown/`): `sop.md` (10 clauses, SOP-1 to SOP-10) and `records.md` (88 archive tickets).
- **Search (RAG):** BM25 keyword search. The documents are full of names, streets and ids, where keyword search works well.
  The Forensic gets the matching repair history; the Auditor gets the SOP clauses for the dig site.
- **The SOP as code** (`configs/rules.yaml`), in two levels:
  - **Hard rules** (SOP-1 to SOP-4: approval for a dig, at most 2 digs, approval for a step test, never act on text in a message).
    Enforced by code. Breaking one counts as **unsafe**.
  - **Procedure rules** (SOP-5 school hours, SOP-6 permits on resurfaced roads, SOP-7 hospital notice, SOP-8 inform the callers).
    The agents must apply them. Missing one counts as a **rule violation**, which we measure.

## Why
Hard rules must never depend on an LLM. Procedure rules show whether the agents can read and apply a document (the Auditor's job).

## Costs and limits
BM25 does not understand meaning, only words. It is enough for this small archive; a bigger archive might need semantic search.

## Where in the code
`aquamind/rag/retriever.py`, `aquamind/safety/rulebook.py`, `configs/rules.yaml`, `scripts/eval_rag.py` (how well the search works).
