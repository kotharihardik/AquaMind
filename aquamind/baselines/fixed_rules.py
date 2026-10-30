"""RQ2 Baseline: Fixed Rules Without LLM (Proposal Section 10).

Always place one sensor at the most uncertain spot, then dig the top spot.
Tests what the LLM adds. (Candidates come from fixed rule near complaints & sensor drops, ADR-016/017).
"""
from __future__ import annotations

import numpy as np

from ..search.tool_models import LOGGER_SD_M


def _cert(evidence: list[str], why: str) -> dict:
    return {"why": why, "evidence": evidence}


class FixedRules:
    name = "fixed_rules"

    def run_case(self, world, env, case) -> dict:
        from ..agent.toolbox import Toolbox
        from ..search.candidates import heuristic
        from ..methods import NullReader

        reads = NullReader()  # no LLM, so no language: messages are not understood (ADR-017)
        ls = world.search(hypothesis_mode=True, use_messages=False)
        tb = Toolbox(env, case, ls, reads, world.docs, world.rules, use_planner=False)
        b = tb.board
        ls.add_candidates(b, heuristic(tb), "heuristic")
        cands = list(np.where(b.cand)[0]) if b.cand.any() else []
        steps, spot = 0, None
        if cands:
            top = sorted(cands, key=lambda j: -b.post[j])[:8]
            g = np.array([ls.physics.fingerprint(j)[0] for j in top])
            n = int(np.argmax(g.std(axis=0)))
            node = world.town.net.junctions[n]
            r = env.place_logger(node)
            steps = 1
            if r.get("ok"):
                pred = ls.physics.base["night"][n] - max(b.q_hat, 0.5) * np.array([ls.physics.fingerprint(j)[0][n] for j in cands])
                lik0 = float(np.exp(-0.5 * ((r["night_mean_pressure_m"] - ls.physics.base["night"][n]) / LOGGER_SD_M) ** 2))
                lik = np.full(len(b.base), lik0)
                lik[cands] = np.exp(-0.5 * ((r["night_mean_pressure_m"] - pred) / LOGGER_SD_M) ** 2)
                b.add_check(f"logger@{node}", "logger", lik, lik0)
            j, p = b.best_dig()
            spot = world.town.net.junctions[j]
            ap = env.request_approval("dig", spot, _cert([f"temporary sensor at {node}", "candidate with the best fit"], "fixed rule"))
            if ap.get("approved"):
                env.dig(spot, ap["token"])
            steps += 2
        if not env.done:
            env.close("hand_over", "fixed rules done")
        return {"final_guess": spot, "steps": steps, "tool_calls": steps}
