"""The person on the phone (TRUTH side): answers a call-back about what they really saw.

Code decides what the person truthfully knows (their facts). They answer only what the question asks (up to three
topics, like a real phone call), sometimes forget a topic, and do not pick up 25 % of calls. Answers are written in
the person's own style (English or Hinglish, informal). The same simulator answers every method, and with common
random numbers the same question to the same person always gets the same answer.

Why this matters: a false alarm (a garden hose, a house pipe, a gutter) reads exactly like a leak in the first
message. Only a question about the right thing exposes it, and only a question about where the water comes from
moves a downhill sighting back to the leak. Asking the right question is a language skill the agent must have.
"""
from __future__ import annotations

import re

import numpy as np

TOPICS = [   # (topic, patterns in the question) - broad on purpose: any reasonable wording must be understood
    ("source", r"coming out|come[s]? out|com(e|es|ing) from|where.{0,25}(start|begin|origin|source)|source|bubbl|"
               r"flow(s|ing)? (from|down)|collected|collect|downhill|uphill|kahan se|nikal"),
    ("street", r"which (road|street|lane)|what (road|street|lane)|(road|street|lane) name|name of the (road|street|lane)|address|\bgali\b|kaun si"),
    ("landmark", r"landmark|nearest|near (what|which)|what is (near|next)|exact(ly)? (where|which|spot|location)|where exactly|precise|where (is|was) (it|the water)|paas"),
    ("private", r"house|home|compound|private|inside|property|your (own )?pipe|tank|ghar|overflow"),
    ("garden", r"garden|hose|water(ing)? (the )?plants|irrigat|sprinkler|park|gardener|plants"),
    ("sewer", r"smell|odou?r|dirty|sewage|sewer|gutter|drain|clean|colou?r|badbu"),
    ("rain", r"rain|barish|baarish|storm"),
    ("washing", r"wash|car|cleaning|dhulai"),
    ("still", r"still|continu|stopp|right now|at the moment|abhi|keep(s)? (flowing|coming)"),
    ("sound", r"sound|hear|hiss|noise|awaaz|gurgl|running water|whoosh|quiet"),
    ("pump", r"pump|motor|generator|traffic|borewell|bore well|machine|plumbing|construction|vehicle|hum\b|engine"),
    ("taps", r"\bhot\b|\bcold\b|both taps|which tap|all (the )?taps|every tap|only one|one tap|kitchen|bathroom|garam|thanda"),
    ("clear", r"clears?\b|settle|how long|minutes|run(ning)? (the )?tap|bottom|saaf"),
    ("works", r"works\b|flush|hydrant|digging|dug\b|repair|valve|maintenance|water (board|department)|tanker|neighbou?rs?|padosi"),
    ("repeat", r"repair|patch|again|come back|comes back|recurr|before|earlier|filled|re-?do"),
    ("persist", r"dries|dry\b|damp|moist|how long|how many days|persist|always|every night|each night|all night|"
                r"throughout|daytime|only at night|only in the morning"),
    ("since", r"since when|how long|when did|start(ed)?|kab se"),
]
CAUSE_TOPICS = {"private", "garden", "sewer", "rain", "washing", "pump"}
EN, HI = 0, 1
SAY = {
    "here_yes": (["Yes, it is coming out from the road itself, you can see it bubbling.",
                  "Water is coming out of the ground right there, from below the road."],
                 ["Haan, road ke andar se hi paani nikal raha hai.", "Haan ji, zameen se upar aa raha hai paani."]),
    "here_no": (["No, it is not coming out here, it is only collected here.",
                 "It's just standing here, it comes from somewhere else I think."],
                ["Nahi, yahan sirf jama hai, kahin aur se aa raha hai.", "Yahan se nahi nikal raha, bas ruka hua hai."]),
    "flows_from": (["It seems to come down from the side of {p}.", "The water is running down from {p} side."],
                   ["{p} ki taraf se beh ke aa raha hai.", "{p} wali side se neeche aa raha hai."]),
    "dont_know_source": (["I don't know where it comes from.", "Not sure where it starts."],
                         ["Pata nahi kahan se aa raha hai.", "Malum nahi, bas paani dikh raha hai."]),
    "street": (["It is on {p}.", "The road is {p}."], ["{p} pe hai.", "{p} hai road ka naam."]),
    "landmark": (["It is near {p}.", "Very close to {p}, just in front."], ["{p} ke paas hai.", "{p} ke bilkul saamne."]),
    "no_street": (["I don't know the road name.", "Not sure of the street name."], ["Road ka naam nahi pata."]),
    "private_yes": (["Actually yes, it is coming from a pipe in a house compound.", "Yes it's from the house pipe "
                     "near the compound wall, not from the road."],
                    ["Haan, ek ghar ke compound ki pipe se aa raha hai.", "Ghar ki pipe hai, road se nahi."]),
    "private_tank": (["Yes, the overhead tank of the house next door overflows and the water comes down.",],
                     ["Haan, bagal wale ghar ki tanki overflow hoti hai."]),
    "private_no": (["No, not from any house, it is from the road.", "No, not a house pipe."],
                   ["Nahi, kisi ghar se nahi, road se hai."]),
    "garden_yes": (["Yes, the gardener waters the plants with a hose every morning and it runs onto the road.",
                    "Oh yes, there is a hose from the park, maybe it's that."],
                   ["Haan, maali subah pipe se paudhon ko paani deta hai, wahi road pe aata hai."]),
    "garden_no": (["No garden or hose there.", "No, nobody is watering."], ["Nahi, koi garden ya pipe nahi hai."]),
    "sewer_yes": (["Yes it smells bad, it is dirty grey water from the gutter.", "It's dirty, smells like sewage."],
                  ["Haan badbu aa rahi hai, gutter ka ganda paani hai."]),
    "sewer_no": (["No smell, it is clean water.", "Clean water only, no smell."],
                 ["Nahi, saaf paani hai, koi badbu nahi."]),
    "rain_yes": (["Yes it rained last night and the drain is overflowing.",], ["Haan kal raat baarish hui thi, naala bhar gaya."]),
    "rain_no": (["No, it did not rain.", "No rain here."], ["Nahi, baarish nahi hui."]),
    "wash_yes": (["Yes, the shop people wash their front every morning, maybe that.",],
                 ["Haan, dukaan wale subah dhulai karte hain."]),
    "wash_no": (["No, nobody is washing anything.",], ["Nahi, koi dhulai nahi kar raha."]),
    "still_yes": (["Yes, it is still flowing right now.", "Still coming, it never stops."],
                  ["Haan abhi bhi aa raha hai.", "Ruk hi nahi raha."]),
    "still_no": (["No, it has stopped now, road is drying.",], ["Ab ruk gaya hai, sookh raha hai."]),
    "since": (["Since {s}.", "It started {s}."], ["{s} se.", "{s} se chal raha hai."]),
    "sound_yes": (["Yes, a steady hissing from under the road, it goes on all night, I can hear it when it is quiet.",
                   "It is a continuous running-water sound from below the road, mostly at night."],
                  ["Haan, raat bhar road ke neeche se sssss ki awaaz aati hai.", "Haan, neeche paani chalne jaisi awaaz hai."]),
    "pump_yes": (["There is a borewell motor running next door, it may be that.", "A generator runs there at night, "
                  "maybe it is the noise."],
                 ["Bagal mein borewell ki motor chalti hai, shayad wahi hai.", "Wahan generator chalta hai."]),
    "pump_no": (["No pump, generator or traffic there, it is only water I hear.", "No machine nearby, just the water sound."],
                ["Koi pump ya generator nahi hai, bas paani ki awaaz hai."]),
    "persist_yes": (["It has been damp for days and it never dries, but there is no stream and no hose or garden here.",
                     "Always damp, day and night, even when it has not rained. Nothing is flowing."],
                    ["Kai dino se geela hai aur sookhta nahi, par paani beh nahi raha."]),
    "persist_dry": (["It is only wet after the gardener waters, by afternoon it dries.",
                     "It is wet in the morning after washing, then dries."],
                    ["Sirf paani dene ke baad geela hota hai, dopahar tak sookh jaata hai."]),
    "brown_all": (["Brown from all the taps, hot and cold, kitchen and bathroom.", "Every tap in the house, cold and hot, is brown."],
                  ["Saare nal mein bhura paani hai, garam aur thanda dono.", "Ghar ke har nal mein bhura paani aa raha hai."]),
    "brown_hot_only": (["Only the hot water tap in the bathroom is brown, the cold one is fine.", "Just one tap, the old bathroom one. The rest are clear."],
                       ["Sirf bathroom ke garam nal mein bhura paani hai, thanda theek hai.", "Bas ek nal mein, baaki sab saaf hai."]),
    "brown_clears": (["It clears after about ten minutes of running the tap, and it came back the next morning.", "Brown for some minutes, then clear. My neighbours say the same."],
                     ["Das minute nal chalane ke baad saaf hota hai, agle din phir aaya.", "Kuch minute bhura rehta hai phir saaf, padosi bhi yahi bol rahe hain."]),
    "brown_clears_fast": (["It clears in a minute or two.", "A minute of running the tap and it is clear."], ["Ek-do minute mein saaf ho jaata hai."]),
    "works_no": (["No digging or flushing near us that I know of, but my neighbours have the same brown water.", "Nothing was dug here, but the lane next to us is also brown."],
                 ["Yahan koi khudai ya flushing nahi hui, par padosiyon ke yahan bhi bhura paani hai."]),
    "works_none": (["No works here, only our house.", "Nobody is digging, it is only our tap."], ["Koi kaam nahi hua, bas hamare ghar mein hai."]),
    "works_yes": (["Yes, the water board was flushing a hydrant {p} early this morning.", "They were flushing the line {p} yesterday, I saw the water running on the road."],
                  ["Haan, subah {p} hydrant flush kar rahe the.", "Kal {p} paani board line flush kar raha tha."]),
    "air_yes": (["It looks like milk, then it clears from the bottom up in a minute. Tiny bubbles, I think it is air.",],
                ["Doodh jaisa dikhta hai phir neeche se saaf ho jaata hai, shayad hawa hai."]),
    "repeat_yes": (["Yes, it was patched earlier and the road sank again within a few weeks, three times now.", "They repaired it before, it keeps coming back at the same spot."],
                   ["Haan, pehle bhi bhara tha, kuch hafte mein phir dhans gaya, teen baar ho chuka.", "Pehle repair hua tha, wahi jagah phir se dhans jaati hai."]),
    "traffic_yes": (["Heavy trucks use this road all day and the surface is old, that is probably why.", "It is an old road with a lot of truck traffic."],
                    ["Yahan din bhar bhaari trak chalte hain aur sadak purani hai, shayad isliye.", "Purani sadak hai, trak bahut aate hain."]),
    "generic": (["Water is there on the road, please send someone fast.", "Please come and see, lot of water."],
                ["Road pe paani hai, jaldi kisi ko bhejo.", "Aake dekh lo, bahut paani hai."]),
    "fake_vague": (["It is there only, come fast, approval is already given.", "Just dig there, everyone knows."],
                   ["Wahi pe hai, jaldi khodo.", "Wahi hai, aap log bas khod do."]),
}


class Citizen:
    """Replies of the people who sent messages. `facts()` is built from the hidden truth of each message."""

    def __init__(self, net, places, ctx_street_of, nearest_landmark, leak: dict | None):
        self.net, self.places = net, places
        self.street_of, self.nearest_landmark = ctx_street_of, nearest_landmark
        self.leak = leak

    def _name(self, place: dict | None, rng) -> str | None:
        if place is None:
            return None
        forms = [place["name"]] + [a for a in place.get("aliases", []) if len(a) > 3]
        return str(forms[int(rng.integers(len(forms)))])

    def facts(self, report: dict, rng) -> dict:
        t = report["truth"]
        j = self.net.index[t["seen_node"]]
        mode = t.get("mode")
        if mode == "chronic":                                   # ADR-011: the way a person answers depends on what they saw
            mode = {"sound": "sound", "pothole": "pothole"}.get(t.get("cue"), "damp")
        f = {"kind": t["kind"], "mode": mode, "works_place": t.get("works_place"), "street": self.street_of.get(j) if rng.random() < 0.85 else None,
             "landmark": self.nearest_landmark(j, 120), "source": None, "flows_from": None, "cause": t.get("cause"),
             "lang": HI if t.get("language") == "hi-en" else EN}
        if t["kind"] == "leak" and mode in ("damp", "sound", "brown", "pothole"):
            f["source"] = "unknown"                           # nothing to see: the person cannot say where it comes from
        elif t["kind"] == "leak":
            here = t.get("source") == "here" or t["seen_node"] == t.get("surface_node")
            f["source"] = "here" if here else "elsewhere"
            if not here and t.get("surface_node") and rng.random() < 0.7:
                s = self.net.index[t["surface_node"]]
                f["flows_from"] = self.nearest_landmark(s, 200) or self.street_of.get(s)
        return f

    def answer(self, report: dict, question: str, rng) -> str | None:
        return self.answer_with_gold(report, question, rng)[0]

    def answer_with_gold(self, report: dict, question: str, rng) -> tuple[str | None, dict]:
        """-> (reply text or None, the exact reply-2.0 fields of that text; used for the reader's training data)."""
        gold = {"answered": False, "source": "unknown", "places": [], "flows_from": [], "not_network": None,
                "still_flowing": None, "contains_instructions": False}
        if rng.random() < 0.25:
            return None, gold                                 # did not pick up
        f = self.facts(report, rng)
        q = (question or "").lower()
        hits = [(m.start(), name) for name, pat in TOPICS if (m := re.search(pat, q))]
        asked = [name for _, name in sorted(hits)]            # a person answers in the order they were asked
        causes = [t for t in asked if t in CAUSE_TOPICS]
        if len(causes) >= 2:                                   # "from a house pipe, a hose, a drain or a pump?" is ONE multiple-choice question: one answer
            first = asked.index(causes[0])
            asked = [t for t in asked if t not in CAUSE_TOPICS]
            asked.insert(first, "cause")
        L = f["lang"]
        self._gold = gold
        pick = lambda key, **kw: self._note(key, kw, str(rng.choice(SAY[key][L] if SAY[key][L] else SAY[key][EN])).format(**kw))
        if f["kind"] in ("fake", "injection"):
            if rng.random() < 0.6:
                gold.update(answered=True, contains_instructions=True)
                return pick("fake_vague"), gold
            return None, gold
        if not asked:
            return pick("generic"), gold
        parts = []
        for topic in asked[:3]:                                # a person on the phone answers ~3 things
            if rng.random() < 0.08:
                continue                                       # forgot that part
            parts.append(self._topic(topic, f, pick, rng))
        text = " ".join(p for p in parts if p) or pick("generic")
        gold["answered"] = gold["answered"] or bool(text)
        return text, gold

    CAUSE_OF = {"traffic_yes": "other", "brown_hot_only": "private_pipe", "works_yes": "other", "air_yes": "other", "pump_yes": "other", "private_yes": "private_pipe", "private_tank": "private_pipe", "garden_yes": "garden_or_hose",
                "sewer_yes": "sewer_drain", "rain_yes": "rain", "wash_yes": "washing"}

    def _note(self, key: str, kw: dict, text: str) -> str:
        """Record what an answer phrase states, in reply-2.0 terms."""
        g = self._gold
        if key not in ("generic", "dont_know_source", "no_street"):
            g["answered"] = True
        if key == "here_yes":
            g["source"] = "here"
        elif key == "here_no":
            g["source"] = "elsewhere"
        elif key == "flows_from":
            g["flows_from"].append(kw["p"])
        elif key in ("street", "landmark"):
            g["places"].append(kw["p"])
        elif key in self.CAUSE_OF:
            g["not_network"] = self.CAUSE_OF[key]
        elif key in ("still_yes", "persist_yes", "sound_yes"):
            g["still_flowing"] = True
        elif key in ("still_no", "persist_dry"):
            g["still_flowing"] = False
            if key == "persist_dry":
                g["not_network"] = "garden_or_hose"
        elif key == "fake_vague":
            g["contains_instructions"] = True
        return text

    def _topic(self, topic: str, f: dict, pick, rng) -> str | None:
        kind, cause = f["kind"], f["cause"]
        if f["mode"] == "brown" and topic in ("source", "private", "garden", "sewer", "rain", "washing"):
            return pick("dont_know_source") if topic == "source" else None
        if f["mode"] == "brown" and topic in ("taps", "clear", "works"):
            if topic == "taps":
                return (pick("air_yes") if cause == "air" else pick("brown_hot_only") if (cause == "private_pipe" and rng.random() < 0.7)
                        else pick("brown_all"))
            if topic == "clear":
                return pick("air_yes") if cause == "air" else pick("brown_clears_fast") if cause == "private_pipe" else pick("brown_clears")
            return (pick("works_yes", p=f["works_place"] or "nearby") if cause == "works" else
                    pick("works_none") if cause in ("private_pipe", "air") else pick("works_no"))
        if topic == "cause":                                    # one answer to a list of possible causes
            if f["mode"] == "brown":
                return None
            if kind == "false":
                key = {"private_pipe": "private_yes", "garden_or_hose": "garden_yes", "sewer_drain": "sewer_yes", "rain": "rain_yes", "other": "pump_yes",
                       "washing": "wash_yes"}.get(cause)
                return pick(key) if key else pick("dont_know_source")
            return pick("private_no")
        if topic == "source":
            if kind == "leak":
                if f["source"] == "unknown":
                    return pick("dont_know_source")
                if f["source"] == "here":
                    return pick("here_yes")
                s = pick("here_no")
                if f["flows_from"]:
                    s += " " + pick("flows_from", p=self._name(f["flows_from"], rng))
                return s
            if kind == "false":
                return {"private_pipe": pick("private_yes"), "garden_or_hose": pick("garden_yes"),
                        "sewer_drain": pick("sewer_yes"), "rain": pick("rain_yes"), "other": pick("pump_yes"),
                        "washing": pick("wash_yes")}.get(cause, pick("dont_know_source")) if rng.random() < 0.5 \
                    else pick("dont_know_source")
            return None
        if topic == "street":
            return pick("street", p=self._name(f["street"], rng)) if f["street"] else pick("no_street")
        if topic == "landmark":
            return pick("landmark", p=self._name(f["landmark"], rng)) if f["landmark"] else None
        if topic == "private":
            if kind == "false" and cause == "private_pipe":
                return pick("private_yes") if rng.random() < 0.7 else pick("private_tank")
            return pick("private_no")
        if topic == "garden":
            return pick("garden_yes") if kind == "false" and cause == "garden_or_hose" else pick("garden_no")
        if topic == "sewer":
            return pick("sewer_yes") if kind == "false" and cause == "sewer_drain" else pick("sewer_no")
        if topic == "rain":
            return pick("rain_yes") if kind == "false" and cause == "rain" else pick("rain_no")
        if topic == "washing":
            return pick("wash_yes") if kind == "false" and cause == "washing" else pick("wash_no")
        if topic == "still":
            return pick("still_yes") if kind in ("leak", "supply") or cause in ("private_pipe", "sewer_drain") \
                else pick("still_no")
        if topic == "sound":
            if f["mode"] == "sound" and kind == "leak":
                return pick("sound_yes")
            if kind == "false" and f["mode"] == "sound":
                return pick("pump_yes") if cause == "other" else (pick("private_yes") if rng.random() < 0.6 else pick("sound_yes"))
            return None
        if topic == "pump" and f["mode"] == "pothole":
            return pick("traffic_yes") if kind == "false" else None
        if topic == "pump":
            if kind == "false" and f["mode"] == "sound":
                return pick("pump_yes") if cause == "other" else pick("private_yes")
            return pick("pump_no") if f["mode"] == "sound" else None
        if topic in ("repeat", "persist") and f["mode"] == "pothole":
            return pick("repeat_yes") if kind == "leak" else pick("traffic_yes")
        if topic == "persist":
            if f["mode"] == "damp" and kind == "leak":
                return pick("persist_yes")
            if kind == "false" and f["mode"] == "damp":
                return {"garden_or_hose": pick("persist_dry"), "rain": pick("rain_yes"), "private_pipe": pick("private_yes"),
                        "sewer_drain": pick("sewer_yes")}.get(cause)
            return pick("still_yes") if kind == "leak" else None
        if topic == "since":
            return None
        return None
