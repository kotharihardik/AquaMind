"""Fetch REAL citizen reports about water problems from FixMyStreet (UK) through its public Open311 API.

Polite on purpose (one request every 45 s, backs off on HTTP 429). Saves title/description/category/date only.
The raw text stays in data/finetune/real/ (git-ignored): it is used as a hand-labelled TEST set for the reader,
never for training, and is not redistributed (keep only ids + this script in the repo).
usage: python -I finetune/real/fetch_fixmystreet.py [max_reports]
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://www.fixmystreet.com/open311/v2"
H = {"User-Agent": "AquaMind-research/0.1 (university course project; polite crawler)", "Accept": "application/json"}
OUT = "data/finetune/real/fixmystreet_raw.jsonl"
WATER = re.compile(r"water|leak|burst|hydrant|flood|drain|sewer|puddle", re.I)


def get(url, tries=8):
    wait = 60
    for _ in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(wait)
            wait = min(wait * 2, 900)
    return None


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seen = {json.loads(l)["service_request_id"] for l in open(OUT)} if os.path.exists(OUT) else set()
    sv = get(f"{BASE}/services.json?jurisdiction_id=fixmystreet.com")
    codes = sorted({s["service_code"] for s in sv["services"] if WATER.search(s["service_name"])})
    print(len(codes), "water-related categories", flush=True)
    windows = [(f"20{y}-{m:02d}-01", f"20{y}-{m:02d}-28") for y in (24, 25, 26) for m in range(1, 13)]
    with open(OUT, "a") as f:
        for code in codes:
            for a, b in windows:
                if len(seen) >= limit:
                    sys.exit(0)
                time.sleep(45)
                q = urllib.parse.urlencode({"jurisdiction_id": "fixmystreet.com", "service_code": code,
                                            "start_date": a + "T00:00:00Z", "end_date": b + "T23:59:59Z"})
                d = get(f"{BASE}/requests.json?{q}")
                for r in (d or {}).get("service_requests", []) if isinstance(d, dict) else (d or []):
                    rid = str(r.get("service_request_id"))
                    if rid in seen or not r.get("description"):
                        continue
                    seen.add(rid)
                    f.write(json.dumps({"service_request_id": rid, "category": r.get("service_name"),
                                        "title": r.get("title"), "description": r.get("description"),
                                        "requested": r.get("requested_datetime")}) + "\n")
                    f.flush()
                print(code, a, len(seen), flush=True)
