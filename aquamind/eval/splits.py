"""Dev/test case lists. The test list is frozen (with its hash) before any final run; a changed list is refused."""
import hashlib
import json
import sys


def cases_for(bench: str, split: str) -> list[str]:
    if split == "test":
        f = json.load(open(f"{bench}/test_frozen.json"))
        if hashlib.sha256("\n".join(f["cases"]).encode()).hexdigest() != f["sha256"]:
            sys.exit("test_frozen.json was modified: refusing to run")
        return f["cases"]
    return json.load(open(f"{bench}/{split}.json"))["cases"]      # dev | val (both are tuning sets)
