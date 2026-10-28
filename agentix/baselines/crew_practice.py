"""Supplementary Baseline: Crews' Current Practice (Human Utility Heuristic).

What municipal crews do today without decision support:
Listen where people report (and near the sensor that dropped most),
walk outward, and dig where leak noise is heard.
"""
from __future__ import annotations

import numpy as np

from ..reader.rules import RuleReader
from ..search.pressure import build_evidence


def _cert(evidence: list[str], why: str) -> dict:
    return {"why": why, "evidence": evidence}


class CrewPractice:
    name = "crew_practice"

    def run_case(self, world, env, case) -> dict:
        net, places = world.town.net, world.town.places
        reader = RuleReader(places)
        reads = reader.read_many(case.reports)
        rep = world.search(hypothesis_mode=True).rep
        leads = []
        for r in sorted(case.reports, key=lambda r: r["received_at"], reverse=True):
            rd = reads[r["report_id"]]
            if rd.get("category") == "leak" and not rd.get("contains_instructions"):
                w = rep.place_weights(rd.get("places"))
                if w is not None:
                    leads.append(int(np.argmax(w)))
        ev = build_evidence(case.time_s, case.pressure, case.flow, case.tank_level, case.case_open_s)
        if case.trigger == "scada_alarm" or not leads:
            s = case.pressure_nodes[int(np.argmax(ev.z))]
            leads.append(net.index[s])
        D = net.pipe_distance
        ring = []
        for c in leads:
            around = [int(j) for j in np.where((D[c] > 200) & (D[c] < 450))[0]]
            pick = []
            for j in sorted(around, key=lambda j: -net.elevation[j]):
                if all(D[j, k] > 300 for k in pick):
                    pick.append(j)
            ring += pick[:4]
        steps, guess, tried = 0, None, []
        for c in leads + ring:
            if env.done or env.checks_left() <= 0:
                break
            if any(net.pipe_distance[c, t] < 150 for t in tried):
                continue
            tried.append(c)
            r = env.listen(net.junctions[c], 250.0)
            steps += 1
            guess = guess if guess is not None else net.junctions[c]
            if r.get("ok") and r.get("heard"):
                spot = r["pinpointed_at"]
                guess = spot
                ap = env.request_approval("dig", spot, _cert(["citizen report", "leak noise heard"], "crew heard a leak"))
                if ap.get("approved"):
                    env.dig(spot, ap["token"])
                    steps += 1
        if not env.done:
            env.close("hand_over", "crews found nothing")
        return {"final_guess": guess, "steps": steps, "tool_calls": steps}
