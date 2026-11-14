"""Recorded investigations (demo_ui/recordings/*.json): list, load, save."""
from __future__ import annotations

import json
import re

from .config import RECORDINGS, SUITES, suite_of

# the curated order shown first in Mission Control (missing files are simply skipped)
FEATURED = ["dev-chronic-010", "test-trap-000", "dev-displaced-003", "dev-cd_damp_decoy-000", "dev-cd_sound_decoy-002", "dev-brown-002", "dev-trap-003", "dev-false_alarm-002", "dev-cd_damp_decoy-002", "dev-cd_sound_decoy-003"]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)


def path_for(case: str, layout: str):
    return RECORDINGS / f"{_safe(case)}__{_safe(layout)}.json"


def save(rec: dict) -> str:
    m = rec["meta"]
    p = path_for(m["case"], m["layout"])
    p.write_text(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
    return p.name


def load(name: str) -> dict | None:
    p = RECORDINGS / _safe(name)
    if not p.suffix:
        p = p.with_suffix(".json")
    if not p.exists() or p.parent != RECORDINGS:
        return None
    return json.loads(p.read_text())


def listing() -> list[dict]:
    out = []
    for p in sorted(RECORDINGS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        m, r = d.get("meta", {}), d.get("result", {})
        s = m.get("suite") or suite_of(m.get("case", ""))
        out.append({"file": p.name, "case": m.get("case"), "layout": m.get("layout"), "suite": s, "suite_name": SUITES.get(s, (s, ""))[0],
                    "suite_blurb": SUITES.get(s, ("", ""))[1], "reader": m.get("reader"), "recorded_at": m.get("recorded_at"),
                    "outcome": r.get("outcome"), "found": r.get("found"), "error_m": r.get("guess_error_m"), "digs": r.get("n_digs"),
                    "checks": r.get("checks_used"), "llm_misses": r.get("llm_misses", 0), "events": len(d.get("events", [])),
                    "has_leak": r.get("has_leak")})
    rank = {c: i for i, c in enumerate(FEATURED)}
    out.sort(key=lambda x: (rank.get(x["case"], 999), x["case"] or ""))
    return out
