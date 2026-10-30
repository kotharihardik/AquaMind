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


# ---------------------------------------------------------------------------- failure history (benchmark v4h, ADR-018)
REPAIR_NOTES = [
    "Main burst repaired {where}. Old CI main badly corroded; clamp fitted. Recommend replacement of this stretch.",
    "Leak on distribution main repaired {where}. Pipe wall thin, heavy tuberculation found. Second repair on this stretch this year.",
    "Joint failure repaired {where}. Crew note: same stretch repaired before, lead joints perished, replacement needed.",
    "Emergency repair {where}: circumferential crack on the main. Surrounding pipe in poor condition.",
    "Pipe burst {where} repaired after night call-out. Corrosion pitting along the main; added to renewal list.",
    "Repeat leak {where}: new clamp next to an earlier repair clamp. Main is old cast iron, recommend renewal.",
]


def weak_stretches(net, rng, n: int = 12, radius_m: float = 150.0) -> list[list[str]]:
    """Old, corroded stretches of main (truth side): a seed pipe and the pipes within `radius_m`, drawn per km of pipe."""
    pipes = net.pipes()
    L = np.array([net.wn.get_link(p).length for p in pipes])
    out, taken = [], set()
    D = net.pipe_distance
    for p in rng.choice(pipes, size=n * 3, replace=False, p=L / L.sum()):
        u, v = net.pipe_endpoints(p)
        if u not in net.index or p in taken:
            continue
        c = net.index[u]
        group = [q for q in pipes if q not in taken and all(x in net.index for x in net.pipe_endpoints(q))
                 and min(D[c, net.index[x]] for x in net.pipe_endpoints(q)) <= radius_m]
        if len(group) < 3:
            continue
        out.append(group)
        taken |= set(group)
        if len(out) >= n:
            break
    return out


def build_records_h(net, places, terrain, seed: int = 0, start=dt.date(2025, 6, 1)) -> dict:
    """The ordinary archive (same false-alarm history as build_records) + free-text repair history on the weak stretches: 2-4 repairs each in
    the last 16 months. Leaks of benchmark v4h happen 4x more often per km on these stretches (pipe failure history predicts failures)."""
    base = build_records(net, places, terrain, seed=seed, start=start)
    rng = np.random.default_rng(seed + 77)
    weak = weak_stretches(net, rng)
    head, body = base["markdown"].split("\n- REC-", 1)
    old = [("- REC-" + body).split("\n")[i] for i in range(len(("- REC-" + body).strip().split("\n")))]
    lines = [(l.split(" | ")[1], l.split(" | ", 2)[2]) for l in old if l.startswith("- REC-")]
    for g in weak:
        u, v = net.pipe_endpoints(g[0])
        for _ in range(int(rng.integers(2, 5))):
            p = g[int(rng.integers(len(g)))]
            a, b = net.pipe_endpoints(p)
            j = net.index[a] if a in net.index else net.index[b]
            when = (start + dt.timedelta(days=int(rng.integers(0, 480)))).isoformat()
            lines.append((when, str(rng.choice(REPAIR_NOTES)).format(where=places.describe(j))))
    lines.sort()
    text = ["# Complaint and repair history (utility records)", "",
            "One line per closed ticket. Repeated repairs on the same stretch of main mean old, failing pipe; places with repeated "
            "non-utility causes are usually false alarms again.", ""]
    text += [f"- REC-{i + 1:03d} | {d} | {t}" for i, (d, t) in enumerate(lines)]
    return {"markdown": "\n".join(text) + "\n", "weak_pipes": sorted({p for g in weak for p in g}), "stretches": weak}
