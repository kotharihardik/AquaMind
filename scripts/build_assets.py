"""Build the town's assets once (idempotent: existing files are kept).

  data/ltown/pipe_distance.npy         all-pairs pipe distance on the nominal network
  data/ltown/gazetteer/                streets, landmarks, localities (+ 3 extra naming skins for the reader)
  data/ltown/sections.json             valve-bounded step-test sections (public)
  data/ltown/records.md                complaint + repair history (public, searched by RAG)
  data/ltown/truth/hotspots.json       recurring false-alarm spots behind those records (truth side)
  data/ltown/truth/real_truth.npz      parsed official L-TOWN_Real.inp (truth side)
  data/ltown/signatures.npz            leak fingerprints on the nominal model (public)

usage: python -I scripts/build_assets.py [configs/ltown.yaml]
"""
import json
import os
import sys

import numpy as np

from agentix.town.assets import Town

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/ltown.yaml"
    town = Town(config)
    cfg, net, d = town.cfg, town.net, town.dir
    print(f"pipe distance: {net.pipe_distance.shape}")

    if not os.path.exists(f"{d}/gazetteer/gazetteer_ltown.json"):
        from agentix.town.gazetteer import build_gazetteer, save, skin, strip_geometry
        os.makedirs(f"{d}/gazetteer", exist_ok=True)
        geo = build_gazetteer(net, seed=0)
        save(geo, f"{d}/gazetteer/geometry.json")
        for t in ["ltown", "town2", "town3", "town4"]:
            g = skin(geo, t)
            save(g, f"{d}/gazetteer/gazetteer_{t}.json")
            save(strip_geometry(g), f"{d}/gazetteer/gazetteer_{t}_names_only.json")
        town = Town(config)

    if not os.path.exists(f"{d}/truth/real_truth.npz"):
        from agentix.sim.truth_model import parse_real_inp, save_cache
        os.makedirs(f"{d}/truth", exist_ok=True)
        save_cache(parse_real_inp(cfg["real_inp"]), f"{d}/truth/real_truth.npz")

    if not os.path.exists(f"{d}/signatures.npz"):
        from agentix.search.signatures import build_library
        build_library({**cfg, "signatures": cfg["signatures"]}, f"{d}/signatures.npz", workers=4)

    if not os.path.exists(f"{d}/sections.json"):
        from agentix.town.sections import build_sections
        wn = net.wn
        nf = {}                                         # nominal share of mean use during 01-05 h (billing estimate)
        for j in net.junctions:
            num = den = 0.0
            for dm in wn.get_node(j).demand_timeseries_list:
                pat = np.asarray(wn.get_pattern(dm.pattern_name).multipliers) if dm.pattern_name else np.ones(288)
                pat = np.resize(pat, max(288, len(pat) // 288 * 288))
                per_day = pat.reshape(-1, 288)
                num += dm.base_value * per_day[:, 12:60].mean()
                den += dm.base_value
            nf[j] = num / den if den else 0.25
        sec = build_sections(net, town.places, night_factor=nf)
        json.dump(sec, open(f"{d}/sections.json", "w"), indent=1)
        sizes = [len(s["junctions"]) for s in sec["sections"]]
        print(f"sections: {len(sizes)}, junctions per section {min(sizes)}-{max(sizes)}, expected night use "
              f"{min(s['expected_night_use_lps'] for s in sec['sections'])}-"
              f"{max(s['expected_night_use_lps'] for s in sec['sections'])} L/s")

    if not os.path.exists(f"{d}/records.md"):
        from agentix.sim.records import build_records
        from agentix.sim.terrain import Terrain
        rec = build_records(net, town.places, Terrain(net), seed=0)
        open(f"{d}/records.md", "w").write(rec["markdown"])
        os.makedirs(f"{d}/truth", exist_ok=True)
        json.dump(rec["hotspots"], open(f"{d}/truth/hotspots.json", "w"), indent=1)
        print(f"records: {rec['markdown'].count('REC-')} lines, {len(rec['hotspots'])} recurring false-alarm spots")
    print("assets ready in", d)
