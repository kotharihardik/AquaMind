"""Step-test sections: the areas a utility can isolate one by one at night by closing boundary valves.

A real step test closes the valves of one section at a time during the night (01:00-05:00) and watches the drop in
the zone inflow after each closure: the drop is the section's night use plus any leak inside it. We cut each
pressure zone into contiguous sections of about 55 junctions (about 3 km of main) with a graph Voronoi partition
(every junction joins the nearest seed along the pipes, so every section is connected). L-Town's nominal model has
no isolation valves, so this partition stands in for the utility's valve map.

Built once by scripts/build_assets.py -> data/ltown/sections.json. Public knowledge (methods may use it).
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def build_sections(net, places, target_size: int = 55, night_factor: dict | None = None) -> dict:
    D = net.pipe_distance
    J = len(net.junctions)
    sec = np.full(J, -1)
    sid = 0
    for zone in sorted(set(net.zone_of.values())):
        idx = np.array([i for i, n in enumerate(net.junctions) if net.zone_of[n] == zone])
        k = max(1, round(len(idx) / target_size))
        seeds = [int(idx[np.argmin(net.xy[idx, 0] + net.xy[idx, 1])])]
        for _ in range(k - 1):                                   # farthest-point seeds, then k-medoids refinement
            seeds.append(int(idx[np.argmax(D[np.ix_(seeds, idx)].min(0))]))
        for _ in range(30):
            a = np.argmin(D[np.ix_(seeds, idx)], 0)
            new = []
            for c in range(k):
                m = idx[a == c]
                new.append(int(m[np.argmin(D[np.ix_(m, m)].sum(1))]))
            if new == seeds:
                break
            seeds = new
        sec[idx] = np.argmin(D[np.ix_(seeds, idx)], 0) + sid
        sid += k
    nf = night_factor or {}
    out = []
    for s in range(sid):
        members = np.where(sec == s)[0]
        streets = Counter(p["name"] for j in members if (p := places.node_street(int(j))))
        lms = [places.node_landmark(int(j), 60) for j in members]
        lm_names = Counter(l["name"] for l in lms if l)
        main = streets.most_common(1)[0][0] if streets else f"zone {net.zone_of[net.junctions[members[0]]]}"
        label = f"S{s + 1:02d} ({main}" + (f", {lm_names.most_common(1)[0][0]}" if lm_names else "") + ")"
        night_lps = float(sum(net.base_demand[j] * 1000 * nf.get(net.junctions[j], 0.25) for j in members))
        out.append({"id": f"S{s + 1:02d}", "label": label, "zone": net.zone_of[net.junctions[members[0]]],
                    "junctions": [net.junctions[j] for j in members], "pipe_km": round(float(
                        net.pipe_length_at_node[members].sum()) / 1000, 2),
                    "expected_night_use_lps": round(night_lps, 2)})
    return {"note": "valve-bounded step-test sections (graph Voronoi on the pipe network)", "sections": out}


def split_section(net, members: list[int], target: int = 14) -> list[list[int]]:
    """Cut one section into parts of about `target` junctions (closing internal valves on a second night)."""
    D = net.pipe_distance
    idx = np.array(members)
    k = max(2, round(len(idx) / target))
    seeds = [int(idx[0])]
    for _ in range(k - 1):
        seeds.append(int(idx[np.argmax(D[np.ix_(seeds, idx)].min(0))]))
    for _ in range(20):
        a = np.argmin(D[np.ix_(seeds, idx)], 0)
        new = [int(idx[a == c][np.argmin(D[np.ix_(idx[a == c], idx[a == c])].sum(1))]) for c in range(k)]
        if new == seeds:
            break
        seeds = new
    a = np.argmin(D[np.ix_(seeds, idx)], 0)
    return [idx[a == c].tolist() for c in range(k)]


class Sections:
    """Lookup helper: section of each junction, member masks; 'areas' = sections and their parts (S03.2)."""

    def __init__(self, data: dict, net):
        self.items = data["sections"]
        self.ids = [s["id"] for s in self.items]
        self.by_id = {s["id"]: s for s in self.items}
        self.of = np.zeros(len(net.junctions), dtype=int)
        for k, s in enumerate(self.items):
            for n in s["junctions"]:
                self.of[net.index[n]] = k
        self.mask = np.zeros((len(self.items), len(net.junctions)), dtype=bool)
        self.mask[self.of, np.arange(len(net.junctions))] = True
        self.expected_night = np.array([s["expected_night_use_lps"] for s in self.items])
        self.night_sd = np.array([s.get("night_use_sd_lps", 0.18 * s["expected_night_use_lps"]) for s in self.items])
        self.areas = {}
        for k, s in enumerate(self.items):
            self.areas[s["id"]] = {"mask": self.mask[k], "expected": float(self.expected_night[k]),
                                   "sd": float(self.night_sd[k]), "parent": None, "junctions": s["junctions"]}
            for p in s.get("parts", []):
                m = np.zeros(len(net.junctions), bool)
                m[[net.index[n] for n in p["junctions"]]] = True
                self.areas[p["id"]] = {"mask": m, "expected": p["expected_night_use_lps"], "sd": p["night_use_sd_lps"],
                                       "parent": s["id"], "junctions": p["junctions"]}
                self.by_id[p["id"]] = p

    def parts(self, sid: str) -> list[str]:
        return [a for a, v in self.areas.items() if v["parent"] == sid]

    def area_mass(self, aid: str, post: np.ndarray) -> float:
        return float(post[self.areas[aid]["mask"]].sum())

    @classmethod
    def load(cls, path: str, net) -> "Sections":
        return cls(json.load(open(path)), net)

    def mass(self, post: np.ndarray) -> np.ndarray:
        """Probability mass of each section under a distribution over junctions."""
        return np.bincount(self.of, weights=post, minlength=len(self.items))

    def index(self, sid: str) -> int:
        return self.ids.index(sid)
