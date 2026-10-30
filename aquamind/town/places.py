"""Place index over the gazetteer (streets, landmarks, localities): text <-> network junctions.

Public map knowledge (a utility knows its own streets and landmarks). Used by the code that turns a reader's place
words into a distribution over junctions, by the rule reader, and to describe junctions to people in plain names.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

_SHORT = {"road": "rd", "street": "st", "rd.": "rd", "st.": "st", "lane": "ln"}
_LONG = {"rd": "road", "st": "street", "ln": "lane"}


def norm(s: str) -> str:
    s = s.lower().replace("'", "").replace(",", " ")
    s = re.sub(r"[^a-z0-9ऀ-෿ ]+", " ", s)
    return " ".join(_SHORT.get(t, t) for t in s.split())


class PlaceIndex:
    def __init__(self, gazetteer: dict, net):
        self.places = {p["place_id"]: p for p in gazetteer["places"]}
        self.net = net
        self.alias_map: dict[str, list[str]] = {}
        for pid, p in self.places.items():
            for a in [p["name"]] + list(p.get("aliases", [])):
                k = norm(a)
                if len(k) >= 3 and pid not in self.alias_map.setdefault(k, []):
                    self.alias_map[k].append(pid)
        self.alias_keys = sorted(self.alias_map, key=len, reverse=True)     # longest first
        self._node_cache: dict[str, np.ndarray] = {}
        self._street_by_node: dict[str, dict] = {}
        for p in self.places.values():
            if p["type"] == "street":
                for n in p["geometry"].get("nodes", []):
                    self._street_by_node.setdefault(n, p)
        self.landmarks = [p for p in self.places.values() if p["type"] == "landmark"]
        self._lm_xy = np.array([net.xy[net.index[p["geometry"]["node"]]] for p in self.landmarks])

    # ------------------------------------------------------------------ text -> mentions
    def find_mentions(self, text: str) -> list[tuple[str, int, int, list[str]]]:
        """(surface text, start, end, place_ids) for every gazetteer name in raw text (non-overlapping)."""
        low = text.lower()
        taken = np.zeros(len(low) + 1, dtype=bool)
        out = []
        for key in self.alias_keys:
            alts = [f"(?:{re.escape(w)}|{_LONG[w]})" if w in _LONG else re.escape(w) for w in key.split()]
            for m in re.finditer(r"\b" + r"[\s,\.']*".join(alts) + r"\.?\b", low):
                if not taken[m.start():m.end()].any():
                    taken[m.start():m.end()] = True
                    out.append((text[m.start():m.end()], m.start(), m.end(), list(self.alias_map[key])))
        return sorted(out, key=lambda x: x[1])

    # ------------------------------------------------------------------ mention -> places
    def resolve(self, mention: str, mention_type: str | None = None, fuzzy: float = 0.84) -> list[str]:
        k = norm(mention)
        if k in self.alias_map:
            ids = self.alias_map[k]
        else:
            best, ids = 0.0, []
            for key in self.alias_keys:
                r = SequenceMatcher(None, k, key).ratio()
                if r > best:
                    best, ids = r, list(self.alias_map[key])
            if best < fuzzy:
                return []
        if mention_type in ("street", "landmark", "area", "locality"):
            want = "locality" if mention_type == "area" else mention_type
            ids = [i for i in ids if self.places[i]["type"] == want] or ids
        return ids

    def candidates(self, mention: str, k: int = 8) -> list[dict]:
        """The k gazetteer places whose names look most like a mention (for model-based linking)."""
        m = norm(mention)
        scored = {}
        for key, pids in self.alias_map.items():
            r = SequenceMatcher(None, m, key).ratio() + (0.3 if m and (m in key or key in m) else 0.0)
            for pid in pids:
                scored[pid] = max(scored.get(pid, 0.0), r)
        best = sorted(scored, key=lambda p: -scored[p])[:k]
        return [{"place_id": p, "name": self.places[p]["name"], "type": self.places[p]["type"],
                 "aliases": self.places[p].get("aliases", [])[:4]} for p in best]

    # ------------------------------------------------------------------ places -> junction weights
    def place_nodes(self, place_id: str, landmark_sigma_m: float = 60.0) -> np.ndarray:
        """Normalised weight over junctions for 'somewhere at this place'."""
        if place_id not in self._node_cache:
            g = self.places[place_id].get("geometry", {})
            w = np.zeros(len(self.net.junctions))
            if "node" in g:                       # landmark: soft disc around it (people say "near")
                d = np.hypot(*(self.net.xy - self.net.xy[self.net.index[g["node"]]]).T)
                w = np.exp(-0.5 * (d / landmark_sigma_m) ** 2)
            else:
                for n in g.get("nodes", []):
                    if n in self.net.index:
                        w[self.net.index[n]] = 1.0
            self._node_cache[place_id] = w / w.sum() if w.sum() > 0 else w
        return self._node_cache[place_id]

    def mention_nodes(self, mention: str, mention_type: str | None = None) -> np.ndarray | None:
        ids = self.resolve(mention, mention_type)
        if not ids:
            return None
        return sum(self.place_nodes(i) for i in ids) / len(ids)     # an ambiguous name -> mixture of all matches

    # ------------------------------------------------------------------ junction -> names
    def node_street(self, j: int) -> dict | None:
        return self._street_by_node.get(self.net.junctions[j])

    def node_landmark(self, j: int, max_m: float = 150.0) -> dict | None:
        d = np.hypot(*(self._lm_xy - self.net.xy[j]).T)
        i = int(np.argmin(d))
        return self.landmarks[i] if d[i] <= max_m else None

    def describe(self, j: int) -> str:
        """Plain-language name of a junction, e.g. 'near Hanuman Temple on Station Road'."""
        lm, st = self.node_landmark(j), self.node_street(j)
        if lm and st:
            return f"near {lm['name']} on {st['name']}"
        if lm:
            return f"near {lm['name']}"
        if st:
            return f"on {st['name']}"
        return f"junction {self.net.junctions[j]}"
