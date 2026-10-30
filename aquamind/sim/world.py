"""Truth-side town context, loaded once: the official Real model, terrain, and the citizens' habits."""
from __future__ import annotations

import json

from .reports import ReportParams, TownContext
from .terrain import Terrain
from .truth_model import RealTruth


class SimTown:
    def __init__(self, town, params: ReportParams | None = None):
        self.town = town
        d = town.dir
        self.real = RealTruth(f"{d}/truth/real_truth.npz")
        self.terrain = Terrain(town.net)
        geometry = json.load(open(f"{d}/gazetteer/geometry.json"))
        records = {"hotspots": json.load(open(f"{d}/truth/hotspots.json"))}
        self.ctx = TownContext(town.net, self.terrain, town.gazetteer, geometry, records, params)
