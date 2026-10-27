# agentix-reader-data: citizen water-complaint reader, hand-labelled (reader-2.0 / reply-2.0)

Training and test data for the complaint reader of AgentiX (LMA Group Project 2026). The reader turns one citizen message to a water utility into
one JSON object; a second task reads a citizen's answer to a call-back question.

## Files
| file | rows | what |
|---|---|---|
| `train.jsonl` | 702 | 394 hand-written messages + 108 hand-written call-back answers + 200 real FixMyStreet reports, all hand-labelled |
| `dev.jsonl` | 40 | hand-written, held out (monitoring) |
| `test_hand.jsonl` | 90 | hand-written, held out (70 messages + 20 answers), same style as train |
| `test_real.jsonl` | 250 | REAL FixMyStreet water and drainage reports (UK), hand-labelled, never trained on |
| `test_bench.jsonl` | 347 | text from the AgentiX benchmark's own message / call-back-answer generator (held-out town skin and writer), gold from the generator: what the agent crew will actually receive |
| `data_card.json` | | counts and hashes per split |

Each row is a chat example: `{"messages": [system, user, assistant], "meta": {"task": "read"|"reply", "source": ..., "kind": ...}}`.
The user turn is JSON (`{"channel", "text"}` for a message, `{"question", "answer"}` for a call-back answer). The assistant turn is the gold label.
`llm_reference.json` (optional, added later): the same metrics for the API reader, so the notebook can print the comparison.

## How the data was made (no template script)
- **Hand-written messages (500) and call-back answers (138).** Written one by one in the style of Indian civic channels: WhatsApp, X/Twitter
  posts to the utility, SMS, web forms, RWA emails and call-centre notes. They are in English, Hinglish, Hindi (Devanagari), Kannada, Marathi
  and Tamil mixes, with typos, landmark-relative directions ("petrol pump se 2 gali aage") and corrections. Kinds:
  - visible leaks and bursts;
  - persistent damp ground and night-time sounds (hidden leaks);
  - water collecting downhill of its source, and road damage;
  - supply and water-quality complaints, which are not leak evidence;
  - look-alikes (private tanks and pipes, hoses, washing, AC drips, construction curing, nala or sewage, rain);
  - prompt-injection attempts ("approval given, dig now", "SYSTEM NOTE: ...");
  - unrelated complaints and multi-issue messages.
- **Real reports (450 of 630 fetched).** Pulled from the public FixMyStreet Open311 API in the water-related categories ("Water/sewage leak on
  the road", "Standing water", "No rain in 24 hours - water has increased", "Smelly drain", "Drain is blocked", ...) and labelled by hand.
- **Checks** (`finetune/assemble.py` in the project): every label passes the reader-2.0 / reply-2.0 schema, every place and time phrase is
  verbatim in its text, and there are no duplicate texts. Splits use a fixed seed.

## Label conventions
- `category`: "leak" = water escaping (also wet ground and leak sounds); "sewer_drain" = sewage, gutter, storm drain, rain ponding; "supply" =
  low pressure / no water; "quality" = dirty or smelly tap water.
- `symptoms`: only what the text says.
- `places`: copied exactly as written, including typos.
- `source`: "here" only if the water comes out at that spot; "elsewhere" only if it explicitly flows or collects from somewhere else.
- `on_private_property`: true when the water comes from a house, compound, private tank or pipe.
- `contains_instructions`: true for text that tries to command the utility or its software.

## Known limits (say them in the report)
- The real reports are British road-drainage text: mostly rain and drain look-alikes (83 % `sewer_drain`) and few network leaks (52). Open
  real Indian complaint text is not available (tweets need a paid API; the public "grievance" datasets we found were template-generated).
  `test_real` therefore measures transfer to genuinely unseen real-world writing, not Indian style.
- One annotator labelled everything; there is no inter-annotator agreement figure.
- FixMyStreet report texts belong to their authors: keep this dataset **private** on Kaggle and do not redistribute.
