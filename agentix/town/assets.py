"""Load the town's public assets once: config, nominal network, place index, step-test sections, documents."""
from __future__ import annotations

import json
import os

import yaml

from .network import NetworkInfo, load_model
from .places import PlaceIndex
from .sections import Sections


class Town:
    def __init__(self, config: str = "configs/ltown.yaml", skin: str = "ltown"):
        self.cfg = yaml.safe_load(open(config))
        self.dir = self.cfg["assets"]
        self.net = NetworkInfo(load_model(self.cfg["nominal_inp"]), self.cfg["network"], cache_dir=self.dir)
        self.gazetteer = json.load(open(f"{self.dir}/gazetteer/gazetteer_{skin}.json"))
        self.places = PlaceIndex(self.gazetteer, self.net)
        sp = f"{self.dir}/sections.json"
        self.sections = Sections.load(sp, self.net) if os.path.exists(sp) else None
        self.sop_path = f"{self.dir}/sop.md"
        self.records_path = f"{self.dir}/records.md"
        self.signatures_path = f"{self.dir}/signatures.npz"
