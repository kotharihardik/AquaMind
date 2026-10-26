"""Quick per-suite summary of a rows.jsonl file (development aid; the final tables come from make_report.py)."""
import collections
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows:
    by[(r["method"], r["layout"])].append(r)
for (m, l), rs in sorted(by.items()):
    leak = [r for r in rs if r["has_leak"]]
    nol = [r for r in rs if not r["has_leak"]]
    f = sum(r["found"] for r in leak)
    print(f"{m} {l}: found {f}/{len(leak)} = {f / max(len(leak), 1):.0%} | dry digs on leak cases "
          f"{sum(r['wrong_digs'] for r in leak)} | crews sent on no-leak {sum(r['n_digs'] > 0 for r in nol)}/{len(nol)} | "
          f"unsafe {sum(r['unsafe'] for r in rs)} | errors {sum(bool(r['error']) for r in rs)} | "
          f"mean {sum(r['wall_s'] for r in rs) / len(rs):.0f}s, {sum(r['llm_calls'] for r in rs) / len(rs):.1f} llm calls")
    s = collections.defaultdict(lambda: [0, 0])
    for r in leak:
        s[r["suite"]][0] += r["found"]
        s[r["suite"]][1] += 1
    print("   ", {k: f"{a}/{b}" for k, (a, b) in sorted(s.items())},
          "checks", collections.Counter(r["checks_used"] for r in leak).most_common())
