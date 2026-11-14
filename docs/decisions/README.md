# Architecture Decision Records (ADRs)

An ADR is a short note that explains **one important decision**: what the problem was, what we chose, why, and what it costs.
They are written in simple language so that anyone (a student, an examiner, a new developer) can follow how AquaMind is built.

This is the set for the **final version** of the project (14 Nov 2026). It replaces all earlier ADR notes.
Code comments point to these numbers, for example `# (ADR-006)`.

| ADR | Decision in one line |
|---|---|
| [001](ADR-001-truth-side-and-method-side.md) | The simulated "real world" and the methods are kept apart; methods use a rougher map of the town. |
| [002](ADR-002-benchmark-and-case-types.md) | One benchmark of 210 cases (60 for tuning, 150 frozen for testing) with 10 kinds of cases. |
| [003](ADR-003-field-checks-and-budget.md) | Every case gets the same small budget: 5 field checks, 2 digs, 5 days. |
| [004](ADR-004-message-reader.md) | A small fine-tuned model turns each citizen message into fixed fields; agents never see the raw text. |
| [005](ADR-005-evidence-board.md) | All evidence is combined by code into one leak map with two key numbers: P(leak) and P(success). |
| [006](ADR-006-hypothesis-driven-search.md) | Physics is simulated only for the places someone suspects; a dig is allowed only there. |
| [007](ADR-007-four-agent-crew.md) | Four LLM agents with clear jobs: Forensic, Skeptic, Strategist and Auditor. |
| [008](ADR-008-tools-gates-and-operator.md) | Agents act only through checked tools; two code gates and a one-time operator token protect every dig. |
| [009](ADR-009-code-checks-agent-claims.md) | Code checks every claim an agent makes (records, downhill water, the chosen check) before it counts. |
| [010](ADR-010-documents-rag-and-sop.md) | The SOP and the town archive are searched with BM25; the SOP is also written as code rules. |
| [011](ADR-011-fair-evaluation.md) | Every method gets the same cases, budget, LLM, reader and luck; results come only from logged rows. |
| [012](ADR-012-mitigations-after-mid-report.md) | Five fixes (M1-M5) made after the mid report, each aimed at a failure seen in the logs. |
| [013](ADR-013-fair-agent-ablations.md) | When an agent is switched off in an ablation, code must not secretly do its job. |

How to add a new ADR: copy the shape of an existing one (Problem, Decision, Why, Costs and limits, Where in the code), give it the
next number, keep it short, and add one line to this table.
