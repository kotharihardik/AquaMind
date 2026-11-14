"""Dev/test case lists. The test list is frozen (with its hash) before any final run; a changed list is refused."""
import hashlib
import json
import sys


def cases_for(bench: str, split: str) -> list[str]:
    """Case names of a split.

    Args: bench = benchmark folder; split = 'dev' | 'test' | 'val' | 'all' (dev + test) | comma list like 'dev,test'.
    Returns: list of case names. Exits if the frozen test list was modified.
    """
    if "," in split:
        out = []
        for s in split.split(","):
            out += cases_for(bench, s.strip())
        return out
    if split == "all":
        return cases_for(bench, "dev") + cases_for(bench, "test")
    if split == "test":
        f = json.load(open(f"{bench}/test_frozen.json"))
        if hashlib.sha256("\n".join(f["cases"]).encode()).hexdigest() != f["sha256"]:
            sys.exit("test_frozen.json was modified: refusing to run")
        return f["cases"]
    return json.load(open(f"{bench}/{split}.json"))["cases"]      # dev | val (both are tuning sets)
