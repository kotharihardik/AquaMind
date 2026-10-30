"""Persona re-writing of citizen messages (environment side; shared by the benchmark and the reader data)."""

PERSONAS = ["a retired teacher, polite, full sentences", "a busy shopkeeper, short, few commas",
            "a college student typing on a phone, lowercase, abbreviations (pls, nr, opp, rd)",
            "an angry resident, some CAPITALS and !!!", "an auto-rickshaw driver, Hinglish, very short",
            "a housewife, worried, Hinglish mixed with English", "an office worker, formal but brief",
            "an elderly man, rambling, repeats himself a little", "a security guard, simple English, small typos",
            "a social media user tagging the utility, emojis allowed"]
SYSTEM = """You rewrite short messages that citizens send to a water utility so they sound like real people wrote them.
For each message you get its text, a persona and MUST-KEEP phrases.
Rules: keep the meaning and every fact; keep every MUST-KEEP phrase exactly as given (same spelling and case);
do not add any new place, time, cause or symptom; do not explain anything; 1-3 sentences; you may add greetings,
filler, small typos in other words, Hinglish if the original is Hinglish. Never reveal a cause the original does
not state. Return {"texts": {"<id>": "<rewritten message>", ...}}."""


def must_keep(gold: dict) -> list[str]:
    keep = [p["text"] for p in gold.get("places") or []]
    if gold.get("since"):
        keep.append(gold["since"])
    return keep


def ok(text: str, keep: list[str]) -> bool:
    return bool(text) and all(k.lower() in text.lower() for k in keep) and len(text) < 600
