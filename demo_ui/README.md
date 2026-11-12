# AquaMind demo UI

A control-room web app for presenting AquaMind. It shows the real crew investigating real benchmark cases. Every number on screen comes
from the AquaMind code, the evaluation logs or a recorded run of the real system. Nothing is mocked up.

Everything UI-related lives in this folder. The module **only reads** `aquamind/`, the benchmark and `logs/`. It changes none of them.
It needs nothing beyond the project's `.venv`: the server is Python's standard library, and the front end is plain HTML/CSS/JS with no
build step and no CDN, so it works offline in the presentation room.

```bash
.venv/bin/python demo_ui                 # serves http://127.0.0.1:8765 and opens the browser
.venv/bin/python demo_ui serve --port 9000 --no-browser
```

Optional extras:

- `zsh finetune/serve_reader.sh`: the fine-tuned reader server. Without it, the demo reads benchmark messages from the reader cache. A
  new message typed in a lab loads the merged MLX model in-process (Apple Silicon, first load about 10 s); set `AQUAMIND_DEMO_MLX=0` to
  forbid that, and the rule reader becomes the labelled fallback.
- `.env` with `NVIDIA_API_KEY_*` for **live** runs. Recordings, log replays and all labs work without it.

## The six views

| # | View | What it proves |
|---|------|----------------|
| 1 | **Overview** | Problem, architecture, the five capabilities with their measured effect, headline numbers |
| 2 | **Mission Control** | One case, step by step: raw messages → fine-tuned reader → the four agents' turns (their own questions, hypotheses, attacks, field-check choices, SOP paperwork) → code verification → the leak map over 782 junctions → code gate → operator token → dig → ground truth |
| 3 | **Crew vs ReAct** | The same case, LLM, tools and budget, side by side. `test-trap-008`: ReAct digs where a fake "operator has approved" message tells it to; the crew closes the case without a dig |
| 4 | **Safety Lab** | Type any malicious message. It is injected into a real case and you watch 7 defence layers (reader quarantine, agent view, tool design, code gate, operator SOP-4 check, one-time token, RAG cleaning) refuse it, all running real code |
| 5 | **Reader & RAG** | The fine-tuned Qwen2.5-3B reader on any message (Hinglish, typos, landmarks) next to the rule reader, its metrics and training curve; BM25 search over the archive and SOP; click any junction to see the SOP clauses the Auditor reads before a dig there |
| 6 | **Results** | Every table of the mid report, **recomputed live from `logs/evaluation/*.jsonl`**, with 95% bootstrap intervals, McNemar tests and the source file under every row: RQ2, safety and traps, RQ1 (3/5/10 sensors), ablations, role ablations, case types, mitigations M1–M5, reader |

### Mission Control sources

- **Recording** (green badge): a full-fidelity run of the *current* crew, made with `demo_ui record`. Every agent turn (including the exact
  board text the agent read: "what this agent saw"), every tool call, every citizen call-back reply, every operator token, and the
  posterior leak map at every step.
- **From log** (amber badge): any of the 210 cases, rebuilt from the mid-report evaluation logs (`logs/evaluation/.../*_trace.jsonl`).
  Every turn and field result is there. The logs do not store the leak map, so the heat shows the hypotheses' mass instead.
- **Live** (red badge): the crew runs now, in a background thread, and streams its events. Prompts already in `.cache/llm` are answered
  from the cache. A faithful live run is saved as a recording automatically.

## A 10-minute presenter script

1. **Overview (1 min).** Read the headline: *with 5 sensors the crew digs at the right spot in 74.6% of leak cases, about three times a
   standard ReAct agent, with no unsafe action.* Point at the architecture: *code does every number and every safety check; the agents do
   language and judgement.*
2. **Mission Control (4 min).** Press `2`, then `C` to choose *Damp ground + decoy* (or any recording), then `Space`. While it plays:
   - the left inbox: raw Hinglish messages and the reader's fields. *Agents never see this text.*
   - the Forensic Advocate writes its own questions; the caller's reply comes back and is read into fields (`not network: rain`).
   - code verifies every citation (✓ / ✗) and scores the hypotheses; the purple hulls on the map are the search space.
   - the Skeptic attacks; the Strategist picks a check from a menu with code-computed chances; a night step test lights a section.
   - the dig gate opens only with P(success) ≥ 0.80 **and** field support; the operator issues a one-time token; the Auditor adds the SOP
     paperwork (school hours, hospital notice, permits).
   - the ground truth appears at the end with the distance. Click `what this agent saw` on any card to show the code-built board.
3. **Crew vs ReAct (2 min).** Press `3`. `test-trap-008` loads; press *Play both* and watch ReAct dig at the fake message's spot.
4. **Safety Lab (1 min).** Press `4`, then *Launch attack*. Invite the professor to type their own message.
5. **Reader & RAG (1 min).** Press `5`, pick *Hinglish night sound*, then click a school on the SOP map.
6. **Results (1 min).** Press `6`. Every number is computed from the logs, and the source file is printed under each row.

Keyboard: `1`–`6` views · `Space` play/pause · `←` `→` step · `Home`/`End` · `C` cases · `T` light/dark (use light on a washed-out
projector) · `F` full screen. Deep links: `/?file=<recording>&seek=40#mission`, `/?case=test-brown-002&method=agentix_crew#mission`,
`/?case=test-trap-008#h2h`, `?theme=light`.

**Before the presentation:** open each view once (warm-up), and if you want a live run in the room, run that case live once beforehand.
Its answers are then cached and the rerun is fast and identical.

## Recording more cases

```bash
.venv/bin/python demo_ui record --cases dev-brown-002,test-trap-000 --online     # uses the NVIDIA free tier for new prompts
.venv/bin/python demo_ui record --cases dev-brown-002 --reader rules             # cache only: replays a cached evaluation run
```

Without `--online`, a prompt that is not cached gets an error answer and the recording reports `llm_misses`. Record new cases with
`--online`. Recordings land in `demo_ui/recordings/` and appear in the case picker. The featured order is `FEATURED` in `recordings.py`.

## How it works

```
demo_ui/
  __main__.py     CLI: serve | record | warm
  server.py       stdlib HTTP server: static files, JSON API, Server-Sent Events
  instrument.py   InstrumentedToolbox / InstrumentedCrew: subclasses that observe the real Toolbox and Crew (no core change),
                  DemoLLM (same client and cache; offline mode never sends anything), run_case() -> a recording
  logreplay.py    rebuilds a timeline from logs/evaluation traces (turns and tool calls interleaved like the crew loop)
  results.py      the mid-report tables from the result rows (same definitions; matches Tables 1-4 exactly)
  labs.py         reader lab, RAG search, SOP for a dig site, attack sandbox: real Toolbox on a copy of a benchmark case
  readers.py      the fine-tuned reader: evaluation cache -> reader server -> in-process MLX -> rule reader (labelled)
  geometry.py     the L-Town map payload (nodes, pipes, sections and parts, landmarks, sensors, plain-language names)
  live.py         one live investigation at a time, streamed to the browser
  recordings/     recorded investigations (demo assets)
  static/         index.html, css/app.css, js/*.js (map.js is the canvas network renderer)
```

Generated caches (`network.json`, `results.json`, new reader answers) go to `.cache/demo_ui/`, which `.gitignore` already covers.
Ground truth (true leak, which messages were real) is read only by the harness, exactly as the scorer does, and is shown only at the end of
a case, labelled as truth.

## Notes for honesty during the demo

- The Results view uses the mid-report runs (v1 logs). Point it at the final runs by editing the file lists at the top of `results.py`.
- The one-look table (report Table 5) is quoted from the report, because its output file is not in `logs/evaluation`. The UI says so.
- Recordings use the current code (mitigations M1–M5 on). Their single-case outcomes can differ from the mid-report log of the same case.
