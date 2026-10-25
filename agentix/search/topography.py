"""Cross-witness topology: where do the callers stand relative to each other on the hill and along the pipes? (ADR-015)

One leak often produces several reports at different places: a hiss heard high up, a soaked verge along the trench, water pooling at the bottom of the
slope. Treated one by one they look like three leaks. Code states the facts (elevation of the place each caller names, the drop between callers and the
pipe distance), the Forensic agent reads them as a story ("acoustic origin -> underground path -> surface terminus"), and code VERIFIES the story before it
touches the leak map: the claimed origin must lie uphill of every claimed downstream caller and within reach along the pipes.
"""
from __future__ import annotations

import numpy as np

MAX_REACH_M = 900.0         # a trench or a slope carries water this far at most
MIN_DROP_M = 1.0            # downstream must be at least this much lower than the origin


def anchor(tb, rid: str) -> int | None:
    m = tb.board.messages.get(rid)
    return int(np.argmax(m.w)) if (m is not None and m.w is not None) else None


def view(tb, max_pairs: int = 8) -> list[str]:
    """Code-built lines for the agents: elevation of each usable complaint and the strongest uphill/downhill relations between them."""
    net = tb.net
    usable = [(rid, anchor(tb, rid)) for rid, m in tb.board.messages.items() if m.usable()]
    usable = [(r, j) for r, j in usable if j is not None]
    if not usable:
        return []
    lines = ["TOPOGRAPHY OF THE COMPLAINTS (code; elevation of the place each caller names):"]
    for rid, j in usable:
        lines.append(f"  {rid}: {tb.places.describe(j)} | elevation {net.elevation[j]:.1f} m")
    pairs = []
    for a, ja in usable:
        for b, jb in usable:
            d = float(net.pipe_distance[ja, jb])
            drop = float(net.elevation[ja] - net.elevation[jb])
            if a != b and drop >= MIN_DROP_M and d <= MAX_REACH_M:
                pairs.append((drop, a, b, d))
    if pairs:
        lines.append("DOWNHILL RELATIONS (a caller higher up on the same pipes can be the source of what a lower caller sees):")
        for drop, a, b, d in sorted(pairs, reverse=True)[:max_pairs]:
            lines.append(f"  {b} is {drop:.1f} m BELOW {a}, {d:.0f} m along the pipes")
    return lines


def triangulate(tb, claim) -> tuple[bool, str]:
    """Verify an 'origin / downstream' story and, if it holds, tell the leak map that the downstream callers see water that runs down from the origin.
    Effect: their source becomes 'elsewhere' and their kernel is multiplied by where the origin caller points (the same mechanism a caller's own
    'it comes down from X' uses)."""
    if not isinstance(claim, dict):
        return False, "give {origin, downstream}"
    origin, down = str(claim.get("origin", "")), [str(x) for x in (claim.get("downstream") or [])][:4]
    jo = anchor(tb, origin)
    if jo is None:
        return False, f"{origin!r} is not a usable complaint with a place"
    ok_ids = []
    for rid in down:
        jd = anchor(tb, rid)
        if jd is None or rid == origin:
            continue
        if tb.net.elevation[jo] - tb.net.elevation[jd] < MIN_DROP_M:
            continue
        if tb.net.pipe_distance[jo, jd] > MAX_REACH_M:
            continue
        ok_ids.append(rid)
    if not ok_ids:
        return False, "no claimed downstream caller is lower than the origin and within reach along the pipes"
    mo = tb.board.messages[origin]
    for rid in ok_ids:
        m = tb.board.messages[rid]
        m.source = "elsewhere"
        m.w_up = mo.w
        m.notes.append(f"triangulation: water runs down from {origin}'s place")
    tb.board.touch()
    return True, f"{', '.join(ok_ids)} placed downhill of {origin}"
