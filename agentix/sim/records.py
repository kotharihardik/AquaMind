"""The utility's own records (built once per town): complaint history and repair history.

Real utilities keep these and they matter: the same garden hose, car wash or overflowing private tank produces
"leak" complaints again and again. The proposal's RAG component retrieves exactly this ("past repair history, such as
streets with repeated false alarms"). The public part is free text (`records.md`, searched by RAG); the structured
list of recurring spots is truth-side (the report generator uses it) and is never read by a method.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

CAUSE_WORDS = {
    "garden_or_hose": ["gardener's hose left running", "garden watering overflowing onto the road",
                       "sprinkler of the park running onto the footpath"],
    "washing": ["shop washing its front every morning", "car washing at the service point"],
    "private_pipe": ["leaking pipe inside a house compound", "overhead tank of a house overflowing",
                     "private service pipe leaking at the compound wall"],
    "sewer_drain": ["blocked storm drain overflowing", "sewer chamber overflow (not drinking water)"],
    "rain": ["rain water collecting at the low point; drain choked"],
}
LANDMARK_CAUSE = {"park": "garden_or_hose", "market": "washing", "kirana_shop": "washing",
                  "petrol_pump": "washing", "community_hall": "private_pipe", "school": "garden_or_hose"}


def build_records(net, places, terrain, seed: int = 0, n_hotspots: int = 10, start=dt.date(2025, 6, 1)) -> dict:
    rng = np.random.default_rng(seed)
    lms = [p for p in places.places.values() if p["type"] == "landmark"]
    rng.shuffle(lms)
    hotspots, used = [], set()
    for lm in lms:                                        # recurring nuisance spots next to parks, shops, halls...
        cause = LANDMARK_CAUSE.get(lm.get("subtype"))
        if cause and lm["geometry"]["node"] not in used:
            hotspots.append({"node": lm["geometry"]["node"], "cause": cause, "landmark": lm["name"]})
            used.add(lm["geometry"]["node"])
        if len(hotspots) >= n_hotspots - 3:
            break
    sinks = [j for j in np.where(terrain.is_sink)[0] if net.junctions[j] not in used]
    for j in rng.choice(sinks, size=min(2, len(sinks)), replace=False):     # low points that flood after rain
        hotspots.append({"node": net.junctions[int(j)], "cause": str(rng.choice(["rain", "sewer_drain"])), "landmark": None})
    res = [j for j in np.argsort(-net.base_demand)[:150] if net.junctions[j] not in used]
    j = int(rng.choice(res))
    hotspots.append({"node": net.junctions[j], "cause": "private_pipe", "landmark": None})

    lines = []
    def when():
        return (start + dt.timedelta(days=int(rng.integers(0, 480)))).isoformat()
    for h in hotspots:
        j = net.index[h["node"]]
        where = places.describe(j)
        n = int(rng.integers(2, 5))
        for k in range(n):
            what = str(rng.choice(["water flowing on road", "water leaking on the road", "pipe leak", "water logging"]))
            lines.append((when(), f"Complaint '{what}' {where}. Crew visit: {rng.choice(CAUSE_WORDS[h['cause']])}. "
                                  f"No utility pipe leak found."))
    for _ in range(18):                                   # genuine repairs (realistic background)
        j = int(rng.integers(len(net.junctions)))
        kind = str(rng.choice(["main burst repaired", "service connection leak repaired", "joint leak repaired",
                               "valve gland leak fixed"]))
        lines.append((when(), f"{kind[0].upper() + kind[1:]} {places.describe(j)}. Closed."))
    for _ in range(8):
        j = int(rng.integers(len(net.junctions)))
        lines.append((when(), f"Low pressure complaint {places.describe(j)}; customer-side issue (private pump). Closed."))
    lines.sort()
    text = ["# Complaint and repair history (utility records)", "",
            "One line per closed ticket. Use: places with repeated non-utility causes are usually false alarms again.", ""]
    text += [f"- REC-{i + 1:03d} | {d} | {t}" for i, (d, t) in enumerate(lines)]
    return {"hotspots": hotspots, "markdown": "\n".join(text) + "\n"}
