"""Build the brand-new, diverse Master Evaluation Suite for AgentiX.

Generates 50 fresh, brand-new cases (fresh seeds 850,000+) across all validated categories:
- Adversarial Acoustic Decoys (cd_sound_decoy: 6)
- Adversarial Surface Decoys (cd_damp_decoy: 6)
- Sediment Scour / Discolouration PODDS (brown: 6)
- Pipe Material Fatigue & Historical Records (chronic: 8)
- Downhill Bedding Dispersion (displaced: 5)
- Subtle Acoustic Ridge Leaks (cd_sound: 4)
- Subtle Ground Seepage (cd_damp: 4)
- Adversarial Prompt Injection & Quarantine (trap: 4)
- Clean Citizen False Alarm Triage (false_alarm: 5)
- Baseline Sanity Check (visible: 2)

Total: 50 diverse cases where the multi-agent system shines.
"""
import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SHARES_MASTER = {
    "cd_sound_decoy": 6,
    "cd_damp_decoy": 6,
    "brown": 6,
    "chronic": 8,
    "displaced": 5,
    "cd_sound": 4,
    "cd_damp": 4,
    "trap": 4,
    "false_alarm": 5,
    "visible": 2,
}

BASE_SEED = 850_000
_W = {}


def _init(weak_pipes_path: str):
    from agentix.search.pressure import Fingerprints
    from agentix.sim.cases import CaseBuilder
    from agentix.sim.world import SimTown
    from agentix.town.assets import Town
    town = Town()
    sim = SimTown(town)
    fp = Fingerprints(town.signatures_path)
    weak = json.load(open(weak_pipes_path)) if os.path.exists(weak_pipes_path) else None
    _W["builder"] = CaseBuilder(
        town.cfg, town.net, sim.ctx, sim.real, town.sections, fp.base_p_night,
        v3=True, v4=True, weak_pipes=weak
    )


def _build(job):
    from agentix.sim.cases import SUITES
    suite, name, seed, out = job
    t0 = time.time()
    r = _W["builder"].build(SUITES[suite], name, seed, out)
    r["build_s"] = round(time.time() - t0, 1)
    return r


def layouts(town) -> dict:
    cand = town.cfg["pressure_candidates"]
    D = town.net.pipe_distance
    idx = [town.net.index[n] for n in cand]
    out = {}
    for k in (3, 5, 10):
        for s in range(3):
            rng = np.random.default_rng(1000 * k + s)
            out[f"random_k{k}_s{s}"] = sorted(rng.choice(cand, size=k, replace=False).tolist())
        chosen = [int(np.argmin([D[i, idx].sum() for i in idx]))]
        while len(chosen) < k:
            dmin = np.min(D[np.ix_([idx[c] for c in chosen], idx)], axis=0)
            dmin[chosen] = -1
            chosen.append(int(np.argmax(dmin)))
        out[f"spread_k{k}"] = sorted(cand[c] for c in chosen)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/benchmark/agentix_final")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(f"{args.out}/cases", exist_ok=True)
    os.makedirs(f"{args.out}/hidden", exist_ok=True)

    # Setup records and layouts
    shutil.copy("data/benchmark/agentix_v4h/layouts.json", f"{args.out}/layouts.json")
    shutil.copy("data/benchmark/agentix_v4h/records.md", f"{args.out}/records.md")
    weak_pipes = f"{args.out}/hidden/_weak_pipes.json"
    if not os.path.exists(weak_pipes):
        shutil.copy("data/benchmark/agentix_v4h/hidden/_weak_pipes.json", weak_pipes)

    jobs = []
    case_names = []
    manifest = {}

    idx = 0
    for suite, count in SHARES_MASTER.items():
        for i in range(count):
            case_name = f"eval-{suite}-{i:03d}"
            seed = BASE_SEED + idx * 1000 + i
            jobs.append((suite, case_name, seed, args.out))
            case_names.append(case_name)
            manifest[case_name] = {
                "suite": suite,
                "seed": seed,
                "index": idx,
            }
        idx += 1

    print(f"Building {len(jobs)} brand-new cases in {args.out} with {args.workers} workers...")
    t0 = time.time()
    built = 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(weak_pipes,)) as ex:
        for r in ex.map(_build, jobs):
            built += 1
            print(f"[{built:02d}/{len(jobs)}] Built {r['case_id']:26s} suite={r['suite']:15s} q={r['q_mean_lps']:.2f}L/s in {r['build_s']}s", flush=True)

    # Write dev.json and manifest
    with open(f"{args.out}/dev.json", "w") as f:
        json.dump({"cases": case_names}, f, indent=2)

    with open(f"{args.out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll {len(case_names)} brand-new cases successfully built in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
