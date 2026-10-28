"""Verification test script for AgentiX v2."""
import json
import time

from agentix.eval.runner import World, run_case
from agentix.methods import registry

TEST_CASES = [
    ("data/benchmark/agentix_ltown", "dev-silent-000", "Hidden SCADA Leak"),
    ("data/benchmark/agentix_master", "dev-cd_sound-001", "Citizen Acoustic Report"),
    ("data/benchmark/agentix_master", "dev-brown-000", "Sediment Scour Feeder"),
    ("data/benchmark/agentix_master", "dev-chronic-000", "Pipe Material Fatigue"),
]

if __name__ == "__main__":
    methods = registry()
    crew2 = methods["agentix_crew_v2"]

    print(f"{'Case ID':<22} | {'Category':<24} | {'Found':<6} | {'Outcome':<10} | {'Checks':<6} | {'Digs':<5} | {'WallTime'}")
    print("-" * 88)

    for bench, case_id, label in TEST_CASES:
        world = World(bench)
        t0 = time.time()
        res = run_case(world, crew2, case_id, "spread_k5", "realistic", 5)
        dt = time.time() - t0
        print(f"{case_id:<22} | {label:<24} | {str(res['found']):<6} | {res['outcome']:<10} | {res['checks_used']:<6} | {res['n_digs']:<5} | {dt:.1f}s")
