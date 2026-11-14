"""Entry point.

    .venv/bin/python demo_ui                         serve the demo (http://127.0.0.1:8765) and open the browser
    .venv/bin/python demo_ui serve --port 9000 --no-browser
    .venv/bin/python demo_ui record --cases dev-brown-002,test-trap-000 [--online] [--reader finetuned|rules] [--layout spread_k5]
    .venv/bin/python demo_ui warm                    precompute the map and the result tables

`record` runs the real AquaMind crew. Without --online it only replays answers already in the LLM cache (nothing is sent); a case whose
prompts are not cached then shows LLM errors, so record new cases with --online (NVIDIA API keys from .env, free tier).
"""
from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):                         # `python demo_ui` or `python demo_ui/__main__.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "demo_ui"

from demo_ui.config import setup_paths  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="demo_ui", description="AquaMind demo UI")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    r = sub.add_parser("record")
    r.add_argument("--cases", required=True)
    r.add_argument("--layout", default="spread_k5")
    r.add_argument("--online", action="store_true", help="allow LLM API calls for prompts that are not cached")
    r.add_argument("--reader", default="finetuned", choices=["finetuned", "rules"])
    sub.add_parser("warm")
    a = ap.parse_args(argv)
    setup_paths()

    if a.cmd == "record":
        from demo_ui import instrument, recordings
        for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
            print(f"recording {c} ({a.layout}, reader={a.reader}, {'online' if a.online else 'cache only'}) ...", flush=True)
            rec = instrument.run_case(c, a.layout, allow_network=a.online, reader=a.reader,
                                      sink=lambda ev: print(f"  [{ev['i']:3d}] {ev['kind']:10s} {ev.get('tool') or ev.get('role') or ''}", flush=True)
                                      if ev["kind"] in ("tool", "turn") else None)
            r = rec["result"]
            name = recordings.save(rec)
            print(f"  -> {name}: {r['outcome']} | error {r['guess_error_m']} m | digs {r['n_digs']} | checks {r['checks_used']} | "
                  f"LLM calls {r['llm_calls']} new, {r['llm_cached']} cached, {r['llm_misses']} missing | {r['wall_s']} s"
                  + (f"\n  ERROR: {r['error']}" if r.get("error") else ""), flush=True)
        return
    if a.cmd == "warm":
        from demo_ui import geometry, results
        geometry.payload()
        results.payload(refresh=True)
        print("map and result tables cached in .cache/demo_ui/")
        return
    from demo_ui import server
    host = getattr(a, "host", "127.0.0.1")
    port = getattr(a, "port", 8765)
    server.serve(host, port, open_browser=not getattr(a, "no_browser", False))


if __name__ == "__main__":
    main()
