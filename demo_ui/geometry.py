"""The town map for the UI: nodes, pipes, step-test sections, landmarks, sensors, plain-language names (public map knowledge only)."""
from __future__ import annotations

import json

import numpy as np

from .config import CACHE, LAYOUTS

_VERSION = 3


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def build() -> dict:
    from .world import town, world
    t = town()
    net, places, secs = t.net, t.places, t.sections
    wn = net.wn
    J = len(net.junctions)

    # every node (junctions first, in the methods' order, so node index == junction index j for j < 782)
    names = list(net.junctions) + [n for n in wn.node_name_list if n not in net.index]
    nidx = {n: i for i, n in enumerate(names)}
    nodes = []
    for n in names:
        nd = wn.get_node(n)
        x, y = nd.coordinates
        kind = "J" if n in net.index else ("T" if n in wn.tank_name_list else "R")
        nodes.append([_r(x), _r(y), kind, _r(getattr(nd, "elevation", 0.0) or 0.0)])

    links = []
    for name, link in wn.links():
        a, b = nidx[link.start_node_name], nidx[link.end_node_name]
        kind = {"Pipe": "P", "Valve": "V", "Pump": "U"}.get(link.link_type, "P")
        d = float(getattr(link, "diameter", 0.1) or 0.1) if kind == "P" else 0.3
        links.append([a, b, kind, round(d * 1000)])           # diameter in mm

    sec_of = [int(k) for k in secs.of]
    sections = []
    for k, s in enumerate(secs.items):
        members = [net.index[n] for n in s["junctions"]]
        cx, cy = net.xy[members].mean(0)
        parts = [{"id": p["id"], "j": [net.index[n] for n in p["junctions"]]} for p in s.get("parts", [])]
        sections.append({"id": s["id"], "label": s["label"], "zone": s["zone"], "j": members, "c": [_r(cx), _r(cy)],
                         "night_lps": s["expected_night_use_lps"], "km": s.get("pipe_km"), "parts": parts})

    landmarks = []
    for p in places.landmarks:
        j = net.index[p["geometry"]["node"]]
        landmarks.append({"id": p["place_id"], "name": p["name"], "sub": p.get("subtype"), "j": j})
    streets = []
    for p in places.places.values():
        if p["type"] == "street":
            js = [net.index[n] for n in p["geometry"].get("nodes", []) if n in net.index]
            if js:
                streets.append({"name": p["name"], "j": js})
    localities = [{"name": p["name"]} for p in places.places.values() if p["type"] == "locality"]

    w = world()
    rules = w.rules
    resurfaced = sorted(getattr(rules, "resurfaced", []) or [])
    if not resurfaced:
        import yaml
        resurfaced = yaml.safe_load(open("configs/rules.yaml"))["procedure"]["resurfaced_permit"]["streets"]

    cand = [net.index[n] for n in t.cfg["pressure_candidates"]]
    layouts = {k: [net.index[n] for n in w.layouts[k]] for k in LAYOUTS if k in w.layouts}

    desc = [places.describe(j) for j in range(J)]
    xs, ys = net.xy[:, 0], net.xy[:, 1]
    return {"v": _VERSION, "nodes": nodes, "links": links, "n_junctions": J, "junction_ids": list(net.junctions),
            "zone": [net.zone_of[n] for n in net.junctions], "section_of": sec_of, "sections": sections,
            "landmarks": landmarks, "streets": streets, "localities": localities, "resurfaced": resurfaced,
            "sensor_candidates": cand, "layouts": layouts, "describe": desc,
            "bounds": [_r(xs.min()), _r(ys.min()), _r(xs.max()), _r(ys.max())],
            "stats": {"junctions": J, "pipes": int(wn.num_pipes), "km": _r(sum(l.length for _, l in wn.pipes()) / 1000, 1),
                      "sensors_total": len(cand), "elev": [_r(net.elevation.min()), _r(net.elevation.max())]}}


def payload() -> dict:
    p = CACHE / "network.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if d.get("v") == _VERSION:
                return d
        except ValueError:
            pass
    d = build()
    p.write_text(json.dumps(d, separators=(",", ":")))
    return d


def junction_of_weights(w) -> int | None:
    return int(np.argmax(w)) if w is not None and float(np.sum(w)) > 0 else None
