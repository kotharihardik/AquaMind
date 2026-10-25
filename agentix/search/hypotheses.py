"""The hypothesis slate: a few competing explanations of the case, named by the agent, scored by code (ADR-014).

The leak map over the 782 junctions is a TOOL: the agent proposes what to compare (a leak in section S13, a leak near a landmark, a
non-network cause for these complaints, no leak at all) and code answers with
  * the mass each hypothesis holds on the current board (from the posterior, P(leak) and the documented causes), and
  * for every candidate field check, the chance of a positive result if that hypothesis is true (from the investigator's own tool models),
so the agent can ask the question a detective asks: which single check tells these two explanations apart?
Evidence the agent cites for a hypothesis is checked against the board (it must exist; a message must point near the hypothesised place).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tool_models import LISTEN_EDGE, LISTEN_FP, p_hear



def place_weights(places, text: str, net=None):
    """Junction weights for a place phrase ('School Bus Stop, Ganesh Gully'): the landmark's own disc is the place; the other names named with it
    only decide WHICH landmark when a name repeats; a bare street spreads over the street. An ambiguous name stays a mixture. None if unknown."""
    ids = []
    for _, _, _, pids in places.find_mentions(text):
        ids += pids
    if not ids:
        ids = places.resolve(text)
    if not ids:
        return None
    ids = list(dict.fromkeys(ids))
    lms = [i for i in ids if places.places[i]["type"] == "landmark"]
    others = [i for i in ids if places.places[i]["type"] != "landmark"]
    if lms and others:
        ow = sum(places.place_nodes(i) for i in others)
        sc = {i: float(places.place_nodes(i) @ ow) for i in lms}
        top = max(sc.values())
        if top > 0:
            lms = [i for i in lms if sc[i] >= 0.5 * top]
    w = sum(places.place_nodes(i) for i in (lms or others))
    return w / w.sum() if getattr(w, "sum", lambda: 0)() > 0 else None
REGION_M = 250.0


@dataclass
class Hypothesis:
    hid: str
    kind: str                                   # network_leak | non_network | no_leak
    where: str = ""
    because: list = field(default_factory=list)
    explains: list = field(default_factory=list)
    cause: str = ""
    region: np.ndarray | None = None            # [J] bool, network_leak only
    mass: float = 0.0
    error: str = ""
    cited_ok: int = 0
    cited_bad: int = 0

    def line(self) -> str:
        what = {"network_leak": f"leak at {self.where}", "non_network": f"non-network cause ({self.cause or 'unspecified'}) behind "
                f"{', '.join(self.explains) or 'complaints'}", "no_leak": "no leak"}[self.kind]
        return f"{self.hid}: {what} | mass {self.mass:.0%}" + (f" | {self.error}" if self.error else "")


def build(slate: list, tb) -> list[Hypothesis]:
    """Resolve the agent's slate against the board: regions, masses, and a check of every piece of evidence it cites."""
    b, net, secs = tb.board, tb.net, tb.sections
    post, pl = b.post, b.p_leak
    out: list[Hypothesis] = []
    for k, h in enumerate(slate[:4] if isinstance(slate, list) else []):
        if not isinstance(h, dict):
            continue
        kind = str(h.get("kind", "")).strip()
        if kind not in ("network_leak", "non_network", "no_leak"):
            continue
        H = Hypothesis(str(h.get("id") or f"H{k + 1}"), kind, str(h.get("where") or "")[:80], [str(x) for x in (h.get("because") or [])][:6],
                       [str(x) for x in (h.get("explains") or [])][:6], str(h.get("cause") or "")[:40])
        if kind == "network_leak":
            w = H.where.strip()
            sid = next((s for s in list(secs.areas) if s.lower() == w.lower()), None)
            if sid is not None:
                H.region = np.asarray(secs.areas[sid]["mask"], bool)
            else:
                pw = place_weights(tb.places, w, True) if w else None
                if pw is None:
                    H.error = "place not found on the map"
                else:
                    j0 = int(np.argmax(pw))
                    sec_mask = np.asarray(secs.areas[secs.ids[int(secs.of[j0])]]["mask"], bool)     # bounded by the isolation valves, not a circle
                    H.region = sec_mask | (net.pipe_distance[j0] <= 100.0)
        for e in H.because + H.explains:
            ok = e in b.messages or e in getattr(tb, "facts", {}) or any(c["label"].startswith(e) for c in b.checks)
            if ok and kind == "network_leak" and e in b.messages and H.region is not None and b.messages[e].w is not None:
                ok = float(b.messages[e].w[net.pipe_distance[int(np.argmax(H.region * post))] <= 300].sum()) > 0.05       # the message points near the place
            H.cited_ok += int(ok)
            H.cited_bad += int(not ok)
        out.append(H)
    non = [h for h in out if h.kind != "network_leak"]
    for h in out:
        h.mass = (pl * float(post[h.region].sum()) if h.region is not None else 0.0) if h.kind == "network_leak" else (1 - pl) / max(len(non), 1)
    z = sum(h.mass for h in out)
    for h in out:
        h.mass = h.mass / z if z > 0 else 0.0
    return out


def _step_power(tb, sids) -> tuple[dict, float]:
    b = tb.board
    pw = {sid: b.tools.step_power(sid, b.q_hat, b.q_sd) for sid in sids}
    return pw, 1.0 - float(np.prod([1 - fp for _, fp in pw.values()]))


def p_positive(tb, o, h: Hypothesis) -> float | None:
    """Chance that option `o` gives a positive result (listening hears something / a section lights up) if hypothesis `h` is true.
    Step tests use the board's own step model (each section's night-use noise and the leak-size belief), not a fixed hit rate."""
    b, D = tb.board, tb.net.pipe_distance
    if o.kind in ("canvass", "inspect"):
        return None
    if o.kind == "step":
        pw, fp_any = _step_power(tb, o.args["sections"])
    if h.kind != "network_leak" or h.region is None:
        return LISTEN_FP if o.kind == "listen" else fp_any
    r = b.post * h.region
    if r.sum() <= 0:
        return None
    r = r / r.sum()
    if o.kind == "listen":
        c = o.args["center"]
        d = D[c]
        p = np.where(d <= 250.0, p_hear(b.q_hat) * (1 - LISTEN_EDGE * d / 250.0), 0.0)
        return float((r * p).sum() * (1 - LISTEN_FP) + LISTEN_FP)
    p, m_in = 0.0, 0.0
    for sid in o.args["sections"]:
        m = float((r * np.asarray(tb.sections.areas[sid]["mask"], bool)).sum())
        p, m_in = p + m * pw[sid][0], m_in + m
    return float(p + (1 - m_in) * fp_any)


def discrimination(tb, o, hyps: list[Hypothesis]) -> float:
    """How well the result of `o` separates the hypotheses, weighted by what each holds: sum_{a<b} m_a m_b |P(+|a) - P(+|b)|."""
    ps = [(h.mass, p_positive(tb, o, h)) for h in hyps]
    ps = [(m, p) for m, p in ps if p is not None]
    return float(sum(ma * mb * abs(pa - pb) for i, (ma, pa) in enumerate(ps) for (mb, pb) in ps[i + 1:]))
