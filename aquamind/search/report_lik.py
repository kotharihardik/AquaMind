"""Citizen messages -> evidence about where the leak is (code only, independent of the message GENERATOR).

  P(message | leak at j) = (1 - eps) * K_src(j, .) . w  +  eps * B . w
    w      where the message points: the reader's place words matched to the map (ambiguous name -> mixture;
           two place words that overlap -> intersection)
    K_src  where water from a leak at j is SEEN: a near-source kernel (exp(-d/40 m)) mixed with a downhill kernel
           (steepest descent from j, exp(-path/100 m)); the mix depends on whether the person says the water comes
           out right there ("here") or only collects there ("elsewhere")
    B      where unrelated messages come from (people ~ water use)
    eps    the chance the message is unrelated to a network leak (false alarm), from the reader's fields
Different structure and parameters from aquamind/sim/reports.py (no inverse crime).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SOURCE_MIX = {"here": 0.9, "elsewhere": 0.15, "unknown": 0.6}
LEAK_SYMPTOMS = {"water_coming_out", "burst", "pooling", "wet_ground", "sound_underground", "road_damage"}
VISIBLE = {"water_coming_out", "burst", "pooling", "road_damage"}
# chance that a message of this kind is NOT about a utility leak (damp patches and noises have many other causes)
EPS_BY_KIND = {"visible": None, "wet_ground": 0.65, "sound_underground": 0.6, "road_damage": 0.7}


class ReportModel:
    def __init__(self, net, places, lambda_near_m: float = 60.0, lambda_down_m: float = 220.0,
                 max_down_m: float = 1200.0, surface_radius_m: float = 60.0, lambda_damp_m: float = 90.0,
                 lambda_sound_m: float = 70.0, lambda_road_m: float = 60.0):
        self.net, self.places = net, places
        D = net.pipe_distance                      # damp patches and pipe noise travel along the pipe and its trench
        self.k_damp = np.exp(-D / lambda_damp_m)
        self.k_damp /= self.k_damp.sum(1, keepdims=True)
        self.k_road = np.exp(-D / lambda_road_m)
        self.k_road /= self.k_road.sum(1, keepdims=True)
        self.k_sound = np.exp(-D / lambda_sound_m)
        self.k_sound /= self.k_sound.sum(1, keepdims=True)
        xy, elev = net.xy, net.elevation
        J = len(xy)
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        self.k_near = np.exp(-d / lambda_near_m)
        self.k_near /= self.k_near.sum(1, keepdims=True)
        nbrs = [set(js) - {i} for i, js in enumerate(cKDTree(xy).query_ball_point(xy, r=surface_radius_m))]
        for u, v in net.pipe_graph.edges:
            iu, iv = net.index.get(u), net.index.get(v)
            if iu is not None and iv is not None:
                nbrs[iu].add(iv)
                nbrs[iv].add(iu)
        nxt = np.full(J, -1)
        for i in range(J):
            cand = [j for j in nbrs[i] if elev[j] < elev[i] - 0.05]
            if cand:
                nxt[i] = max(cand, key=lambda j: (elev[i] - elev[j]) / max(d[i, j], 1.0))
        spread = np.exp(-0.5 * (d / 30.0) ** 2)
        k_down = np.zeros((J, J))
        for j in range(J):
            cur, dist, row = j, 0.0, np.zeros(J)
            row[j] = 1.0
            while nxt[cur] >= 0 and dist < max_down_m:
                dist += d[cur, nxt[cur]]
                cur = nxt[cur]
                row[cur] += np.exp(-dist / lambda_down_m)
            k_down[j] = row @ spread
        self.k_down = k_down / k_down.sum(1, keepdims=True)
        dem = net.base_demand / net.base_demand.mean()
        self.background = (0.3 + dem) / (0.3 + dem).sum()

    def place_weights(self, place_list: list) -> np.ndarray | None:
        """Reader place words (dicts {text, type} or strings) -> weights over junctions, or None."""
        ws = []
        for p in place_list:
            text, typ = (p["text"], p.get("type")) if isinstance(p, dict) else (p, None)
            w = self.places.mention_nodes(text, typ if typ in ("street", "landmark", "area") else None)
            if w is not None and w.sum() > 0:
                ws.append(w)
        if not ws:
            return None
        w = ws[0]
        for w2 in ws[1:]:
            cos = float(w @ w2 / np.sqrt((w @ w) * (w2 @ w2) + 1e-300))
            w = w * w2 if cos > 0.05 else 0.5 * (w + w2)
            w = w / w.sum()
        return w

    @staticmethod
    def kind(read: dict) -> str:
        sym = set(read.get("symptoms") or [])
        if read.get("category") == "quality" and "dirty_water" in sym and not sym & VISIBLE:
            return "discolour"
        if sym & (VISIBLE - {"road_damage"}):
            return "visible"
        if "road_damage" in sym and not sym & {"wet_ground", "sound_underground"}:
            return "road_damage"
        return "wet_ground" if "wet_ground" in sym else "sound_underground"

    @staticmethod
    def credibility(read: dict, base_eps: float, age_h: float = 0.0, tau_age_h: float = 24.0,
                    hour: int | None = None, eps_kind: dict | None = None) -> float | None:
        """Chance the message is unrelated to a network leak; None = not leak evidence at all."""
        if read.get("contains_instructions"):
            return None                                    # quarantined: never used as evidence (SOP-4)
        if ReportModel.kind(read) == "discolour":          # brown / turbid tap water: evidence via the hydraulic footprint (ADR-010)
            return 0.95 if read.get("on_private_property") is True else 0.0
        if read.get("category") != "leak" or not (set(read.get("symptoms") or []) & LEAK_SYMPTOMS):
            return None                                    # supply, quality, sewer/drain, other
        if read.get("on_private_property") is True:
            return 0.95
        eps = (eps_kind or EPS_BY_KIND).get(ReportModel.kind(read)) or base_eps
        if ReportModel.kind(read) == "sound_underground" and hour is not None:
            eps = eps * (0.85 if (hour >= 21 or hour < 6) else 1.3)    # traffic and machines mask / mimic a leak by day
            eps = min(eps, 0.95)
        if age_h > 0 and ReportModel.kind(read) == "visible":      # damp patches and noises persist: no age decay
            eps = 1 - (1 - eps) * float(np.exp(-age_h / tau_age_h))
        return eps

    def kernel(self, w: np.ndarray, source: str, pooling: bool = False, kind: str = "visible") -> np.ndarray:
        if kind == "road_damage":                 # a void / sinking above the pipe: along the pipe, a bit tighter than damp ground
            return self.k_road @ w
        if kind == "wet_ground":                  # persistent damp, no stream: soaks the ground near the pipe
            return 0.85 * (self.k_damp @ w) + 0.15 * (self.k_down @ w)
        if kind == "sound_underground":           # heard along the pipe, nothing visible
            return self.k_sound @ w
        a = SOURCE_MIX.get(source, 0.6)
        if pooling and source == "unknown":
            a = 0.3                        # people who see collected water mostly stand downhill of the leak
        return a * (self.k_near @ w) + (1 - a) * (self.k_down @ w)
