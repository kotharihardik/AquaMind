# AquaMind

**Finding water leaks with few sensors, citizen messages and an investigation crew.**

A water utility has only 3, 5 or 10 pressure sensors in a town of 782 junctions. A small leak barely moves
them. Citizens do notice things: damp ground, a hissing sound at night, a pool that keeps coming back. AquaMind
reads those messages, combines them with the sensors in a physics model, and investigates the case step by step
before anyone digs.

## How it works

1. **Read.** A fine-tuned small model (Qwen2.5-3B + QLoRA, runs locally) turns each raw message into fixed fields:
   what was seen, where, when. The agents never see raw citizen text, only these code-built summaries.
2. **Search.** Code (WNTR + numpy) scores the places the agents name against the sensors and the messages and
   keeps an evidence board: P(leak) per place, P(success) of a dig, and the value of each field check.
3. **Investigate.** Four LLM agents work on that board:
   - **Forensic Advocate:** calls people back with its own questions and names hypotheses.
   - **Critical Skeptic:** attacks the leading hypothesis.
   - **Hydraulic Strategist:** picks the next field check (night step test or listening survey).
   - **Safety & SOP Auditor:** challenges the dig and does the SOP paperwork, using BM25 search (RAG) over the
     town archive and the SOP.
4. **Act safely.** A dig needs the code gate (P(success) ≥ 0.80 plus field support) and a one-time operator
   token for that exact spot. A case gets at most 5 field checks and 2 digs.

The LLM never produces a number that a decision depends on. All physics and probability are computed in code.

## Results (final, 5 sensors, 210 cases)

| Method | Solved | Correct digs | Within 100 m | Median error | Dry holes | Unsafe | Rule violations |
|---|---|---|---|---|---|---|---|
| **AquaMind crew** | **172** | **79.5 %** | **86.5 %** | **23 m** | **11.4 %** | **0 %** | **2.4 %** |
| ReAct agent (same LLM, tools and budget) | 66 | 26.5 % | 32.4 % | 371 m | 53.3 % | 41.0 % | 56.2 % |
| Crews' current practice | 70 | 27.6 % | 34.1 % | 409 m | 44.6 % | 0 % | 41.0 % |
| Plain LLM | 29 | 11.9 % | 11.9 % | 557 m | 88.5 % | 0.5 % | 91.0 % |
| Classical inversion (EPANET) | 33 | 7.6 % | 8.1 % | 1305 m | 76.3 % | 0 % | 28.1 % |
| Fixed rules | 21 | 3.2 % | 3.2 % | 710 m | 95.9 % | 0 % | 69.5 % |

*Solved* = a leak dug within 100 m, or a no-leak case closed without a dig. *Correct digs* = leak cases (185) that
end with a dig within 100 m of the leak. *Dry holes* = share of digs more than 100 m from any leak. The crew handles
all 13 prompt-injection traps; with 3 and 10 sensors it digs correctly in 75.1 % and 77.3 % of leak cases.

- **Final report:** [`docs/AquaMind_Final_Report.pdf`](docs/AquaMind_Final_Report.pdf) (RQ1 and RQ2, all 12
  ablations, reader and RAG metrics). **Poster:** [`docs/AquaMind_Final_Poster.pptx`](docs/AquaMind_Final_Poster.pptx).
- **Demo:** `.venv/bin/python demo_ui` (see [`demo_ui/README.md`](demo_ui/README.md)): replays real runs of the
  crew, compares it with ReAct on the same case, and recomputes every result table live from the logs.
- Every number comes from the result rows in `logs/evaluation/` (one row per case; the newest log version of a run
  is the final one, `v1` is the mid-report run).

## Repository map

```
aquamind/
  agent/       the crew (crew.py: switches + investigation loop), its toolbox with the code gate and operator token, code-only policy
    roles/     one module per role: forensic, skeptic, strategist, auditor, field_checks (menu + chance), base, prompts
  search/      leak search, all maths: pressure, messages, field checks -> evidence board
  reader/      message reader: contract (schema.py), rules, prompted LLM, fine-tuned local model
  rag/         BM25 retriever over the town archive and the SOP
  safety/      the SOP as code (requirements and audit)
  baselines/   ReAct agent, plain LLM, fixed rules, crews' practice, classical inversion, RQ1 one-look methods
  ablations/   the proposal ablations and the role ablations
  town/        public map knowledge: network, place names, step-test sections
  sim/         the simulated town (truth side: leaks, hydraulics, citizens, field crews, operator)
  eval/        runner, metrics, logger, frozen splits
  llm/         one LLM client with a shared rate limiter and a disk cache
  case.py      the only way a method reads a case
  methods.py   every method by name
configs/       ltown.yaml (network), field.yaml (field checks, budget), rules.yaml (SOP rules), costs.yaml
data/ltown/    built town assets; records.md (archive, 88 tickets) and sop.md (10 clauses) are the RAG documents
data/benchmark/agentix_master/   the 210 cases: 60 dev + 150 test (frozen with a hash in test_frozen.json)
finetune/      reader dataset builder, MLX training, evaluation, metrics
scripts/       build assets and benchmark, run evaluations, make metric tables
tests/         fast checks
demo_ui/       the presentation demo (reads the code and the logs, changes nothing)
docs/decisions/  the ADRs: why AquaMind is built this way, one short note per decision (start with its README)
docs/final_report/  sources of the final report and poster (HTML, figure and poster scripts)
logs/evaluation/    result rows and traces of every evaluation run
```

Methods may not import `aquamind/sim`. The truth side uses the official `L-TOWN_Real.inp`; every method uses the
nominal `L-TOWN.inp`, so there is no inverse crime.

## Setup

Python 3.12 on macOS (Apple Silicon is needed for the local fine-tuned reader).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e . wntr numpy scipy networkx pyyaml jsonschema pytest mlx-lm
```

1. **Network files.** Download L-Town from the BattLeDIM 2020 release
   ([Zenodo 4017659](https://doi.org/10.5281/zenodo.4017659)) and put `L-TOWN.inp` and `L-TOWN_Real.inp` in
   `data/raw/ltown/`. They are not in git (`L-TOWN_Real.inp` is over GitHub's 100 MB limit).
2. **API keys.** Create `.env` in the repo root with `NVIDIA_API_KEY` (agents, NVIDIA Nemotron-3 Ultra with
   thinking off) and `GEMINI_API_1` (message re-wording only). Several keys work as `NVIDIA_API_KEY_1`,
   `NVIDIA_API_KEY_2`, ...; one key per parallel worker is best. Never commit `.env`.
3. **Fine-tuned reader (optional).** The weights are too large for git (`finetune/models/` is ignored). With the
   model in `finetune/models/reader_qwen3b_mlx`, start the server with `zsh finetune/serve_reader.sh`. Without it,
   use `--reader rules`.

Check the install:

```bash
.venv/bin/python -I -m pytest tests -q
```

## Running

Evaluate methods on the benchmark (one result row per case, with a full trace):

```bash
.venv/bin/python -I scripts/run_eval.py --bench data/benchmark/agentix_master --split dev \
    --methods agentix_crew,react_agent --layouts spread_k5 --reader local --workers 3
```

- `--layouts spread_k3|spread_k5|spread_k10` sets the number of sensors.
- `--reader rules|llm|local` is the same reader for every method in a run.
- `--limit 5` runs a quick smoke test.
- Logs are written to `logs/evaluation/<category>/<method>/<layout>/`.

RQ1 one-look comparison (no field checks, no LLM):

```bash
.venv/bin/python -I scripts/run_one_look.py --bench data/benchmark/agentix_master --split dev --reader local \
    --out runs/one_look_dev.jsonl
```

The full evaluation (all methods, 3/5/10 sensors) is `zsh scripts/final_eval.sh <profile>`. Profiles are listed
at the top of the script; `DRY=1` prints the commands without running them.

### Reproduce the report numbers

Everything below reads only the logged result rows (no LLM call):

```bash
.venv/bin/python -I scripts/eval_agent_extra.py          # agent behaviour metrics -> logs/evaluation/extra/agent_extra.*
.venv/bin/python -I scripts/eval_rag.py                  # retrieval and citation metrics -> logs/evaluation/extra/rag_eval.*
.venv/bin/python -I docs/final_report/build_figures.py          # report figures (SVG) -> docs/final_report/figs/
.venv/bin/python -I docs/final_report/build_poster_figures.py   # poster figures (PNG, drawn at print size)
python3 -I docs/final_report/build_poster_template.py          # fills the provided poster template -> docs/AquaMind_Final_Poster.pptx
```

The RQ1 one-look rows are not in git (`runs/` is ignored); rebuild them (no LLM) with

```bash
for s in dev test; do .venv/bin/python -I scripts/run_one_look.py --bench data/benchmark/agentix_master --split $s \
    --reader local --out runs/final_queue/one_look_$s.jsonl; done
```

The report PDF is printed from `docs/final_report/report.html` with any Chromium browser (`--print-to-pdf`). The
demo's Results view (`.venv/bin/python demo_ui`, view 6) recomputes every table with 95 % intervals and McNemar tests.

### Method names

| Group | Names |
|---|---|
| Ours | `agentix_crew` |
| Baselines | `react_agent`, `plain_llm`, `fixed_rules`, `crew_practice`, `classical_direct_inversion`, `classical_bruteforce` |
| Ablations | `ablation_no_reports`, `ablation_no_temp_sensor` (no step test), `ablation_no_rag`, `ablation_no_gate`, `ablation_one_shot`, `ablation_no_ai` |
| Role ablations | `ablation_no_forensic`, `ablation_no_skeptic`, `ablation_no_strategist`, `ablation_no_auditor`, `ablation_no_sop`, `ablation_no_triangulation` |

## Rules we keep

- **Fair comparison:** every method gets the same cases, sensor layouts, field budget, LLM, reader and random
  luck (crc32 seeds).
- **Untrusted text:** citizen messages never control tool calls; only the reader sees raw text.
- **Dev for tuning only:** the 150 test cases are frozen with a hash.
- **Free API limits:** the free NVIDIA tier allows only a few calls a minute. All LLM calls go through one rate
  limiter and are cached in `.cache/llm`, so a re-run with the same prompts costs nothing. Use at most 3 workers.

## References

- Vrachimis et al., BattLeDIM 2020 dataset: <https://doi.org/10.5281/zenodo.4017659>
- WNTR, Water Network Tool for Resilience: <https://github.com/USEPA/WNTR>
- Yao et al., ReAct (2023): <https://arxiv.org/abs/2210.03629>
- Dettmers et al., QLoRA (2023): <https://arxiv.org/abs/2305.14314>
- Qwen2.5 technical report (2024): <https://arxiv.org/abs/2412.15115>
