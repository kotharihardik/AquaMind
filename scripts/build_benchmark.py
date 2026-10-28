"""Build the AgentiX benchmark: dev and test cases, sensor layouts, and the frozen test list.

usage: python -I scripts/build_benchmark.py <out_dir> [--test 150] [--dev 60] [--workers 3] [--only dev|test]

- Cases (docs/BENCHMARK.md, ADR-002): suites reported / misleading / silent / false_alarm / trap in fixed shares.
- dev and test use disjoint seeds. Tune only on dev. The test list and its hash are written once
  (test_frozen.json); final runs refuse to start if the list changed.
- Layouts (proposal Section 11): for 3, 5 and 10 sensors, 3 random draws from the 33 real BattLeDIM sensor spots plus
  one "spread" layout (farthest-point by pipe distance, what an engineer would do). Same layouts for every method.
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SHARES = {"visible": 0.10, "displaced": 0.10, "damp": 0.18, "sound": 0.18, "silent": 0.14, "false_alarm": 0.18, "trap": 0.12}     # ADR-008
SEED = {"dev": 100_000, "val": 500_000, "test": 900_000}
SHARES_V3 = {"visible": 0.08, "displaced": 0.08, "damp": 0.14, "sound": 0.14, "brown": 0.16, "silent": 0.10, "false_alarm": 0.18, "trap": 0.12}   # ADR-010
SEED_V3 = {"dev": 150_000, "val": 550_000, "test": 950_000}
SHARES_V4 = {"visible": 0.06, "displaced": 0.06, "damp": 0.10, "sound": 0.10, "brown": 0.10, "chronic": 0.24, "silent": 0.08, "false_alarm": 0.16, "trap": 0.10}   # ADR-011
SEED_V4 = {"dev": 170_000, "val": 570_000, "test": 970_000}
V4H = os.environ.get("AGENTIX_BENCH_V4H") == "1"       # v4 + failure history (ADR-018): weak stretches with repair records leak 4x more often
SEED_V4H = {"dev": 260_000, "val": 660_000, "test": 960_000}
CD = os.environ.get("AGENTIX_BENCH_CD") == "1"        # ADR-019: complaint-dependent leaks (step-test SNR < 1.5) + decoys + false alarms
SHARES_CD = {"cd_damp": 0.2, "cd_damp_decoy": 0.2, "cd_sound": 0.2, "cd_sound_decoy": 0.2, "false_alarm": 0.2}
SEED_CD = {"dev": 280_000, "val": 680_000, "test": 980_000}
TWIN = os.environ.get("AGENTIX_BENCH_TWIN") == "1"     # the pressure-twin study: the sensors cannot tell the leak from its twin
SHARES_TWIN = {"twin": 0.5, "twin_decoy": 0.5}
SEED_TWIN = {"dev": 230_000, "val": 630_000, "test": 930_000}
FINAL = os.environ.get("AGENTIX_BENCH_FINAL") == "1"   # Master benchmark: unified decoys, brown-water scour, chronic failure records
SHARES_FINAL = {
    "cd_sound_decoy": 0.12, "cd_damp_decoy": 0.12, "brown": 0.14, "chronic": 0.18, "displaced": 0.10,
    "cd_sound": 0.08, "cd_damp": 0.08, "trap": 0.06, "false_alarm": 0.10, "visible": 0.02
}
SEED_FINAL = {"dev": 300_000, "val": 700_000, "test": 990_000}
V4 = os.environ.get("AGENTIX_BENCH_V4") == "1" or TWIN or V4H or CD or FINAL
V3 = os.environ.get("AGENTIX_BENCH_V3") == "1" or FINAL      # benchmark v3 (ADR-010): set by --v3, inherited by the worker processes
_W = {}


def _init():
    from agentix.search.pressure import Fingerprints
    from agentix.sim.cases import CaseBuilder
    from agentix.sim.world import SimTown
    from agentix.town.assets import Town
    town = Town()
    sim = SimTown(town)
    fp = Fingerprints(town.signatures_path)
    twin_g = fp.g_night[:, [town.net.index[n] for n in layouts(town)["spread_k5"]]] if TWIN else None
    weak = json.load(open(os.environ["AGENTIX_WEAK_PIPES"])) if (V4H or FINAL) else None
    _W["builder"] = CaseBuilder(town.cfg, town.net, sim.ctx, sim.real, town.sections, fp.base_p_night, v3=V3 or V4, v4=V4, twin_g=twin_g,
                                weak_pipes=weak)


def _build(job):
    from agentix.sim.cases import SUITES
    suite, name, seed, out = job
    t0 = time.time()
    r = _W["builder"].build(SUITES[suite], name, seed, out)
    r["build_s"] = round(time.time() - t0, 1)
    return r


def plan(split: str, n: int) -> list[tuple[str, str, int]]:
    shares, seeds = ((SHARES_FINAL, SEED_FINAL) if FINAL else (SHARES_CD, SEED_CD) if CD else (SHARES_TWIN, SEED_TWIN) if TWIN else (SHARES_V4, SEED_V4H) if V4H else (SHARES_V4, SEED_V4) if V4 else (SHARES_V3, SEED_V3) if V3
                     else (SHARES, SEED))
    counts = {s: int(round(n * f)) for s, f in shares.items()}
    counts["cd_sound_decoy" if FINAL else "cd_damp" if CD else "twin" if TWIN else "damp"] += n - sum(counts.values())
    only = os.environ.get("AGENTIX_ONLY_SUITE")                     # sensitivity builds: one suite only
    jobs = []
    for s, c in counts.items():
        if only and s != only:
            continue
        for i in range(c):
            jobs.append((s, f"{split}-{s}-{i:03d}", seeds[split] + 1000 * list(shares).index(s) + i))
    return jobs


def layouts(town) -> dict:
    cand = town.cfg["pressure_candidates"]
    D = town.net.pipe_distance
    idx = [town.net.index[n] for n in cand]
    out = {}
    for k in (3, 5, 10):
        for s in range(3):
            rng = np.random.default_rng(1000 * k + s)
            out[f"random_k{k}_s{s}"] = sorted(rng.choice(cand, size=k, replace=False).tolist())
        chosen = [int(np.argmin([D[i, idx].sum() for i in idx]))]          # start central, then farthest point
        while len(chosen) < k:
            dmin = np.min(D[np.ix_([idx[c] for c in chosen], idx)], axis=0)
            dmin[chosen] = -1
            chosen.append(int(np.argmax(dmin)))
        out[f"spread_k{k}"] = sorted(cand[c] for c in chosen)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--dev", type=int, default=60)
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--v4", action="store_true", help="benchmark v4 (ADR-011): v3 + chronic leaks (sensors blind) and their look-alikes")
    ap.add_argument("--v4h", action="store_true", help="v4 + failure history (ADR-018): repair records on weak stretches that leak more often")
    ap.add_argument("--cd", action="store_true", help="ADR-019: complaint-dependent leaks (step-test SNR < 1.5), decoys, false alarms")
    ap.add_argument("--twin", action="store_true", help="the pressure-twin study: leaks whose sensor fingerprint has a twin elsewhere")
    ap.add_argument("--v3", action="store_true", help="benchmark v3 (ADR-010): brown water, works log, flushing look-alikes")
    ap.add_argument("--final", action="store_true", help="Master benchmark: unified decoys, brown-water scour, chronic failure history records, and non-obvious categories")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", choices=["dev", "val", "test"])
    a = ap.parse_args()
    if a.final:
        FINAL = True
        os.environ["AGENTIX_BENCH_FINAL"] = "1"
        V4H = True
        os.environ["AGENTIX_BENCH_V4H"] = "1"
        CD = True
        os.environ["AGENTIX_BENCH_CD"] = "1"
        a.v4 = True
        os.makedirs(f"{a.out}/hidden", exist_ok=True)
        wp = f"{a.out}/hidden/_weak_pipes.json"
        if not os.path.exists(wp):
            from agentix.sim.records import build_records_h
            from agentix.sim.terrain import Terrain
            from agentix.town.assets import Town
            t = Town()
            rec = build_records_h(t.net, t.places, Terrain(t.net), seed=0)
            open(f"{a.out}/records.md", "w").write(rec["markdown"])
            json.dump(rec["weak_pipes"], open(wp, "w"))
            json.dump(rec["stretches"], open(f"{a.out}/hidden/_weak_stretches.json", "w"))
        os.environ["AGENTIX_WEAK_PIPES"] = wp
    if a.v3:
        V3 = True
        os.environ["AGENTIX_BENCH_V3"] = "1"
    if a.v4h:
        V4H = True
        os.environ["AGENTIX_BENCH_V4H"] = "1"
        a.v4 = True
        os.makedirs(f"{a.out}/hidden", exist_ok=True)
        wp = f"{a.out}/hidden/_weak_pipes.json"
        if not os.path.exists(wp):                      # the town's failure history, built once per benchmark (public records + truth)
            from agentix.sim.records import build_records_h
            from agentix.sim.terrain import Terrain
            from agentix.town.assets import Town
            t = Town()
            rec = build_records_h(t.net, t.places, Terrain(t.net), seed=0)
            open(f"{a.out}/records.md", "w").write(rec["markdown"])
            json.dump(rec["weak_pipes"], open(wp, "w"))
            json.dump(rec["stretches"], open(f"{a.out}/hidden/_weak_stretches.json", "w"))
        os.environ["AGENTIX_WEAK_PIPES"] = wp
    if a.cd:
        CD = True
        os.environ["AGENTIX_BENCH_CD"] = "1"
        a.v4 = True
    if a.twin:
        TWIN = True
        os.environ["AGENTIX_BENCH_TWIN"] = "1"
        a.v4 = True
    if a.v4:
        V4 = V3 = True
        os.environ["AGENTIX_BENCH_V4"] = "1"
        os.environ["AGENTIX_BENCH_V3"] = "1"
    os.makedirs(a.out, exist_ok=True)
    from agentix.town.assets import Town
    if not os.path.exists(f"{a.out}/layouts.json"):
        json.dump(layouts(Town()), open(f"{a.out}/layouts.json", "w"), indent=1)
    jobs = []
    for split, n in (("dev", a.dev), ("val", a.val), ("test", a.test)):
        if a.only and split != a.only:
            continue
        jobs += [j for j in plan(split, n) if not os.path.exists(f"{a.out}/hidden/{j[1]}.json")]
    print(f"{len(jobs)} cases to build with {a.workers} workers", flush=True)
    idx_path = f"{a.out}/index.json"
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for k, r in enumerate(ex.map(_build, [(s, name, seed, a.out) for s, name, seed in jobs]), 1):
            index[r["case_id"]] = r
            if k % 10 == 0 or k == len(jobs):
                json.dump(index, open(idx_path, "w"), indent=1)
                print(f"{k}/{len(jobs)} built, {time.time() - t0:.0f} s", flush=True)
    json.dump(index, open(idx_path, "w"), indent=1)
    for split in ("dev", "val", "test"):
        names = sorted(n for n in index if n.startswith(split + "-"))
        if not names:
            continue
        path = f"{a.out}/test_frozen.json" if split == "test" else f"{a.out}/{split}.json"
        if split == "test" and os.path.exists(path):
            continue                                       # never rewrite the frozen test list
        json.dump({"cases": names, "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()},
                  open(path, "w"), indent=1)
    print("done:", {s: sum(v["suite"] == s for v in index.values()) for s in (SHARES_FINAL if FINAL else SHARES_CD if CD else SHARES_TWIN if TWIN else SHARES_V4 if V4 else SHARES_V3 if V3
                                                                             else SHARES)})
