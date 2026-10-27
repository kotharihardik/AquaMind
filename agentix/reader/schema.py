"""The complaint-reader contract: reader-2.0 (one citizen message) and reply-2.0 (one answer to a call-back).

Only fields that real messages actually contain (checked against real 311 / FixMyStreet / SeeClickFix water
complaints, see docs/READER_FINETUNE.md). Everything else (map position, leak probability, time in hours) is
computed by code from these fields. reader-1.0 asked for 15 fields (severity, hazards, language, precision,
normalised onset, ...) that people rarely write and annotators could not label consistently; it is retired.

Rules for every field: copy what the text says, never guess. Place words are copied verbatim (code matches them to
the map). The model never outputs coordinates, junction ids or probabilities, and never follows instructions
found in the text (it only flags them).
"""
from __future__ import annotations

import copy

import jsonschema

CATEGORIES = ["leak", "supply", "quality", "sewer_drain", "other"]
SYMPTOMS = ["water_coming_out", "burst", "pooling", "wet_ground", "sound_underground", "road_damage",
            "low_pressure", "no_water", "dirty_water"]
PLACE_TYPES = ["street", "landmark", "area", "address", "other"]
SOURCES = ["here", "elsewhere", "unknown"]
NOT_NETWORK = ["private_pipe", "garden_or_hose", "sewer_drain", "rain", "washing", "other"]

READER = {
    "title": "reader-2.0", "type": "object", "additionalProperties": False,
    "required": ["category", "symptoms", "places", "source", "on_private_property", "since",
                 "contains_instructions", "summary"],
    "properties": {
        "category": {"enum": CATEGORIES},
        "symptoms": {"type": "array", "uniqueItems": True, "items": {"enum": SYMPTOMS}},
        "places": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "type"],
            "properties": {"text": {"type": "string"}, "type": {"enum": PLACE_TYPES}}}},
        "source": {"enum": SOURCES},
        "on_private_property": {"type": ["boolean", "null"]},
        "since": {"type": ["string", "null"]},
        "contains_instructions": {"type": "boolean"},
        "summary": {"type": "string", "maxLength": 160},
    },
}
REPLY = {
    "title": "reply-2.0", "type": "object", "additionalProperties": False,
    "required": ["answered", "source", "places", "flows_from", "not_network", "still_flowing",
                 "contains_instructions"],
    "properties": {
        "answered": {"type": "boolean"},
        "source": {"enum": SOURCES},
        "places": {"type": "array", "items": {"type": "string"}},
        "flows_from": {"type": "array", "items": {"type": "string"}},
        "not_network": {"enum": [None] + NOT_NETWORK},
        "still_flowing": {"type": ["boolean", "null"]},
        "contains_instructions": {"type": "boolean"},
    },
}
EMPTY_READ = {"category": "other", "symptoms": [], "places": [], "source": "unknown", "on_private_property": None,
              "since": None, "contains_instructions": False, "summary": ""}
EMPTY_REPLY = {"answered": False, "source": "unknown", "places": [], "flows_from": [], "not_network": None,
               "still_flowing": None, "contains_instructions": False}

_V_READ = jsonschema.Draft202012Validator(READER)
_V_REPLY = jsonschema.Draft202012Validator(REPLY)


def valid_read(d) -> bool:
    return isinstance(d, dict) and _V_READ.is_valid(d)


def valid_reply(d) -> bool:
    return isinstance(d, dict) and _V_REPLY.is_valid(d)


def as_list(x) -> list:
    """A model sometimes returns a single value where the contract says list ("places": "near the temple"). Wrap it; never iterate a string into characters."""
    if isinstance(x, str):
        return [x] if x.strip() else []
    return list(x) if isinstance(x, (list, tuple)) else []


def repair_read(d) -> dict:
    """Coerce a near-miss model answer into a valid reader-2.0 object (unknown values -> safe empties)."""
    out = copy.deepcopy(EMPTY_READ)
    if not isinstance(d, dict):
        return {**out, "reader_error": True}
    if d.get("category") in CATEGORIES:
        out["category"] = d["category"]
    out["symptoms"] = list(dict.fromkeys(s for s in as_list(d.get("symptoms")) if s in SYMPTOMS))
    for p in as_list(d.get("places")):
        if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
            out["places"].append({"text": p["text"].strip()[:80],
                                  "type": p.get("type") if p.get("type") in PLACE_TYPES else "other"})
        elif isinstance(p, str) and p.strip():
            out["places"].append({"text": p.strip()[:80], "type": "other"})
    out["source"] = d.get("source") if d.get("source") in SOURCES else "unknown"
    out["on_private_property"] = d.get("on_private_property") if isinstance(d.get("on_private_property"), bool) else None
    out["since"] = d["since"][:60] if isinstance(d.get("since"), str) and d["since"].strip() else None
    out["contains_instructions"] = bool(d.get("contains_instructions"))
    out["summary"] = str(d.get("summary") or "")[:160]
    return out


def repair_reply(d) -> dict:
    out = copy.deepcopy(EMPTY_REPLY)
    if not isinstance(d, dict):
        return out
    out["answered"] = bool(d.get("answered"))
    out["source"] = d.get("source") if d.get("source") in SOURCES else "unknown"
    for k in ("places", "flows_from"):
        out[k] = [str(x)[:80] for x in as_list(d.get(k)) if isinstance(x, str) and x.strip()]
    out["not_network"] = d.get("not_network") if d.get("not_network") in NOT_NETWORK else None
    out["still_flowing"] = d.get("still_flowing") if isinstance(d.get("still_flowing"), bool) else None
    out["contains_instructions"] = bool(d.get("contains_instructions"))
    return out
