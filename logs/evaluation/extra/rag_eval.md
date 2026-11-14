# RAG evaluation

Archive: 88 tickets (37 false-alarm visits, 51 repairs); SOP: 10 clauses. No LLM involved; ground truth from the benchmark.

## A. Archive retrieval for a citizen message (records_near, 3 tickets shown)

| | hotspot msgs | hit@3 | recall@3 | precision@3 | MRR | leak msgs shown a false-alarm ticket | other msgs with a ticket |
|---|---|---|---|---|---|---|---|
| m4 on | 755 | 0.615 | 0.57 | 0.609 | 0.609 | 0.26 | 0.3 |
| m4 off | 755 | 0.591 | 0.483 | 0.486 | 0.518 | 0.25 | 0.602 |

Raw BM25 ranking (754 queries): recall@1 0.14, @3 0.455, @6 0.639, @10 0.663; hit@1 0.342, hit@3 0.602; MRR 0.475.

## B. SOP clauses retrieved for the Auditor (sop_context, 6 clauses) at every possible dig site

| | dig sites | sites with location rules | location-rule recall@6 | all-needed recall@6 | precision@6 | MRR |
|---|---|---|---|---|---|---|
| m5 on | 782 | 195 | 1.0 | 0.528 | 0.382 | 0.393 |
| m5 off | 782 | 195 | 1.0 | 0.528 | 0.382 | 0.391 |

By rule (M5 on): SOP-5_found 63, SOP-6_found 119, SOP-7_found 47
Share of dig sites whose retrieved clauses include each SOP: SOP-1 1.0, SOP-4 1.0, SOP-5 1.0, SOP-6 1.0, SOP-7 1.0, SOP-8 1.0

## C. Archive citations made by the agents (evaluation traces)

| run | false-alarm citations | accepted | accepted that cite a repair ticket (M4 target) | accepted on a real-leak message | precision of accepted | repair citations | accepted | accepted repairs within 300 m of the leak |
|---|---|---|---|---|---|---|---|---|
| mid k5 | 1095 | 0.7 | 0.364 | 71 | 0.907 | 1417 | 1.0 | 0.15 |
| final k5 | 2098 | 0.551 | 0.0 | 87 | 0.925 | 2206 | 1.0 | 0.149 |
| mid k3 | 1152 | 0.655 | 0.342 | 60 | 0.92 | 1391 | 1.0 | 0.151 |
| final k3 | 2105 | 0.544 | 0.0 | 92 | 0.92 | 2227 | 1.0 | 0.138 |
| mid k10 | 1090 | 0.677 | 0.343 | 74 | 0.9 | 1392 | 0.999 | 0.146 |
| final k10 | 2078 | 0.55 | 0.0 | 89 | 0.922 | 2261 | 1.0 | 0.151 |

## D. Prompt-injection guard (retriever.clean) on citizen messages

Messages with instructions: 13, removed 10 (detection 0.769); normal messages wrongly changed: 0 of 9767 (false positives 0.0).
