"""B0: rule-based complaint reader (keywords + place-name matching) -> reader-2.0 / reply-2.0.

The simple baseline the fine-tuned reader must beat (proposal Section 6, "field F1"). It never follows instructions
in the text; it only flags them.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..town.places import norm as norm_key

SYMPTOMS = [
    ("burst", r"burst|gushing|fountain|shooting|phat gay"),
    ("water_coming_out", r"coming out|leak|oozing|bubbl|flowing (out|from)|nikal|zameen se|road se paani|beh raha|upar aa"),
    ("pooling", r"collect|puddle|standing|stagnant|accumulat|jama|bhara|ruka|logging"),
    ("wet_ground", r"\bwet\b|geel|damp|moist|soggy|never dries|sookh(ta|ti) hi nahi|lush|green(er)? (patch|strip)|moss|algae|kaai|seeps?|manhole|cable pit|service pit|hari rehti|standing inside"),
    ("road_damage", r"sinking|sinks|caved|cave in|dhans|pothole|gaddha|sagging|a dip is forming|cracks? keep"),
    ("sound_underground", r"sound of|awaaz|hiss|sssss|gurgl|water running|running water|noise (of|from) (the )?(water|ground)|"
                          r"hear(ing)? water|can hear"),
    ("low_pressure", r"low (water )?pressure|pressure (is )?(very )?(weak|low|kam)|slowly in the tap|dhaar|weak flow"),
    ("no_water", r"no water|supply stopped|paani nahi aa|not getting water"),
    ("dirty_water", r"dirty|muddy|brown|rusty|sandy|turbid|cloudy|milky|yellowish|ganda paani|colou?red water|bhura|zang|mitti|doodh jaisa"),
]
SEWER = r"gutter|sewer|sewage|drainage|naali|naala|badbu|bad smell|\brain|baarish|barsaat|storm drain"
PRIVATE = r"house pipe|ghar ki pipe|compound|inside (the|our) house|private|tanki|overhead tank|ghar ka pipe"
OTHER_CAUSE = r"watering|\bhose\b|paudhon|washing|dhulai|car wash"
INJECTION = (r"system( note)?:|ignore (previous|all|the) (rules|instructions)|approval (already )?(given|granted)|"
             r"operator has approved|no need for checks|auto-?approval|skip the evidence|commissioner office|\[admin\]")
SINCE = (r"(since [a-z0-9: ]{2,20}|for \d+ (hours?|days?)|\d+ (ghante|din) se|(kal raat|kal shaam|subah|dopahar|kal) se|"
         r"from (last night|yesterday|morning))")
NOT_VISIBLE = (r"(?:no|nothing|not)\s+(?:any\s+)?(?:water\s+)?(?:is\s+|being\s+)?(?:visible|seen|showing)|no visible (?:water|leak)|"
               r"nothing (?:is )?(?:visible|to see|on the (?:road|surface))|kuch (?:dikh|dikhai) nahi|dikh nahi")
STOP = {"the", "a", "an", "near", "road", "street", "behind", "opposite", "opp", "next", "to", "in", "on", "at", "of", "and",
        "area", "from", "our", "my", "is", "are", "water", "sound", "since", "for", "that", "this", "with"}
ELSEWHERE = r"collect|stagnant|jama|ruka|flowing down|runs? down|from somewhere|kahin aur|taraf se"


class RuleReader:
    name = "rules_b0"

    def __init__(self, places):
        self.places = places

    def _mentions(self, text: str):
        found = self.places.find_mentions(text)
        taken = [(a, b) for _, a, b, _ in found]
        found = list(found) + self._fuzzy(text, taken)
        for surface, a, b, pids in sorted(found, key=lambda x: x[1]):
            t = self.places.places[pids[0]]["type"]
            yield surface, a, "area" if t == "locality" else t

    def _fuzzy(self, text: str, taken: list, thr: float = 0.84):
        """Places written with a typo ('hosptial', 'BusS top'): compare word windows with gazetteer names."""
        pl = self.places
        by_len: dict[int, list[str]] = {}
        for key in pl.alias_keys:
            by_len.setdefault(len(key.split()), []).append(key)
        toks = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z'\.]*", text)]
        out, used = [], list(taken)
        for L in (4, 3, 2, 1):
            for i in range(len(toks) - L + 1):
                a, b = toks[i][0], toks[i + L - 1][1]
                while b > a and text[b - 1] in ".'":
                    b -= 1
                if any(a < tb and ta < b for ta, tb in used):
                    continue
                surface = text[a:b]
                words = [w.lower() for w in surface.split()]
                if all(w in STOP for w in words) or len(surface) < 5:
                    continue
                k = norm_key(surface)
                best, bk = 0.0, None
                for key in by_len.get(L, []):
                    if abs(len(key) - len(k)) > 2 or key[0] != k[:1]:
                        continue
                    r = SequenceMatcher(None, k, key).ratio()
                    if r > best:
                        best, bk = r, key
                if best >= thr:
                    used.append((a, b))
                    out.append((surface, a, b, list(pl.alias_map[bk])))
        return out

    def read(self, report: dict) -> dict:
        text = report["text"]
        low = re.sub(NOT_VISIBLE, " ", text.lower())              # "no water is visible" is not "no water at the tap"
        sym = [s for s, pat in SYMPTOMS if re.search(pat, low)]
        if re.search(SEWER, low):
            cat, sym = "sewer_drain", []
        elif {"low_pressure", "no_water"} & set(sym) and not {"water_coming_out", "burst", "pooling"} & set(sym):
            cat = "supply"
        elif sym == ["dirty_water"]:
            cat = "quality"
        elif re.search(OTHER_CAUSE, low):
            cat, sym = "other", []
        else:
            cat = "leak" if sym else "other"
        if cat == "leak":
            sym = [s for s in sym if s not in ("low_pressure", "no_water", "dirty_water")]
        src = "elsewhere" if re.search(ELSEWHERE, low) else ("here" if {"water_coming_out", "burst"} & set(sym) else "unknown")
        m = re.search(SINCE, low)
        places = [{"text": s, "type": t} for s, _, t in self._mentions(text)]
        names = [self.places.places[ids[0]]["name"] for p in places if (ids := self.places.resolve(p["text"]))]
        return {"category": cat, "symptoms": sym, "places": places, "source": src,
                "on_private_property": True if re.search(PRIVATE, low) else None,
                "since": text[m.start():m.end()] if m else None, "contains_instructions": bool(re.search(INJECTION, low)),
                "summary": (f"{cat}: {', '.join(sym) or 'no clear symptom'}" +
                            (f"; place: {', '.join(names)}" if names else ""))[:160]}

    def read_reply(self, question: str, reply: str | None) -> dict:
        out = {"answered": bool(reply), "source": "unknown", "places": [], "flows_from": [], "not_network": None,
               "still_flowing": None, "contains_instructions": False}
        if not reply:
            return out
        low = reply.lower()
        out["contains_instructions"] = bool(re.search(INJECTION, low) or re.search(r"just dig|khod", low))
        if re.search(r"heavy trucks|truck traffic|trak|old road|purani sadak", low):
            out["not_network"] = "other"
        elif re.search(r"only the hot|just the hot|hot water tap|only one tap|just one tap|sirf .{0,30}garam|bas ek nal", low) and \
                not re.search(r"all the taps|every tap|saare nal|har nal", low):
            out["not_network"] = "private_pipe"
        elif re.search(r"flushing|flush kar|water board|hydrant|looks like milk|doodh jaisa dikhta|bubbles, i think", low):
            out["not_network"] = "other"
        elif re.search(r"borewell|generator|\bmotor\b|\bpump\b|engine", low) and \
                not re.search(r"no (pump|machine|generator)|koi pump", low):
            out["not_network"] = "other"
        elif re.search(r"house|compound|ghar|tanki|\btank\b", low) and not re.search(r"not (from )?(any|a) house|kisi ghar se nahi", low):
            out["not_network"] = "private_pipe"
        elif re.search(r"hose|gardener|maali|watering|sprinkler", low) and not re.search(r"no garden|no hose|nobody is watering|koi garden", low):
            out["not_network"] = "garden_or_hose"
        elif re.search(r"smell|sewage|gutter|ganda|badbu", low) and not re.search(r"no smell|koi badbu nahi", low):
            out["not_network"] = "sewer_drain"
        elif re.search(r"it rained|baarish hui|drain is overflowing|naala bhar", low):
            out["not_network"] = "rain"
        elif re.search(r"wash|dhulai", low) and not re.search(r"nobody is washing|koi dhulai nahi", low):
            out["not_network"] = "washing"
        if re.search(r"coming out|from the road itself|zameen se|road ke andar|from below", low) and \
                not re.search(r"not coming out|nahi nikal", low):
            out["source"] = "here"
        elif re.search(r"not coming out|only collected|just standing|sirf jama|nahi nikal|somewhere else|kahin aur", low):
            out["source"] = "elsewhere"
        if re.search(r"still|never stops|abhi bhi|ruk hi nahi|keeps coming back|sank again|phir dhans|never dries|always damp|sookhta nahi|goes on all night|"
                     r"steady hissing|continuous|raat bhar", low):
            out["still_flowing"] = True
        elif re.search(r"stopped|ruk gaya|drying|sookh|it dries", low):
            out["still_flowing"] = False
        up = re.search(r"(down from|from) (the side of )?[^.]+|[^.]+ (ki taraf se|wali side se)", reply, re.I)
        for surface, a, t in self._mentions(reply):
            if up and up.start() <= a < up.end() and out["source"] != "here":
                out["flows_from"].append(surface)
            elif t != "area":
                out["places"].append(surface)
        return out

    def read_many(self, reports: list[dict]) -> dict[str, dict]:
        return {r["report_id"]: self.read(r) for r in reports}

    def read_replies(self, items: list[tuple[str, str | None]]) -> list[dict]:
        return [self.read_reply(q, a) for q, a in items]
