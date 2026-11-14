"""The field checks the Strategist chooses from, all built and scored by code: the menu from the hypotheses, each option's chance of
finding the leak, the guard that keeps a pick near the best chance (ADR-009), the Strategist's own check, and the M3 confirmation listen."""
from __future__ import annotations

import numpy as np

from ...search import hypotheses as hyp
from ...search.tool_models import LISTEN_EDGE, p_hear


class FieldCheckMixin:
    """Menu, chance and guard of the field checks (code only, no LLM)."""

    def menu(self, tb) -> None:
        """Field checks built from the hypotheses (no look-ahead planner): a step test of the hypotheses' sections (or the parts of a lit
        section), listening at each hypothesis' best-fitting junction, and one exploratory step test of the map's largest untested sections."""
        b, secs = tb.board, tb.sections
        tb.options = {}
        tested = {c["label"].split(":")[-1] for c in b.checks if c["kind"] == "step"}
        lit = [c["label"].split(":")[-1] for c in b.checks if c["kind"] == "step" and c.get("support") is not None]
        if self.m1_weak_leads:
            suggestive = []
            for t in tb.trace:
                if t.get("tool") == "field_check" and t.get("result", {}).get("kind") == "step test":
                    suggestive += t["result"].get("suggestive", [])
            sec_to_expand = list(dict.fromkeys(lit + suggestive))
        else:
            sec_to_expand = lit
        k = 1
        parts = [a for sid in sec_to_expand for a in secs.parts(sid) if a not in tested]
        if parts:
            label_sec = "the lit section" if not (self.m1_weak_leads and not lit) else "section with extra flow"
            tb.add_option(f"M{k}", "step", {"sections": parts[:6]}, f"night step test of the parts {', '.join(parts[:6])} of {label_sec}"); k += 1
        hs = [h for h in self.slate if h.kind == "network_leak" and h.region is not None and h.region.any()]
        sids = []
        for h in hs:
            for sid in secs.ids:
                if sid not in tested and sid not in sids and (np.asarray(secs.areas[sid]["mask"], bool) & h.region).any():
                    sids.append(sid)
        if sids:
            tb.add_option(f"M{k}", "step", {"sections": sids[:6]}, f"night step test of sections {', '.join(sids[:6])} (the hypotheses)"); k += 1
        quiet = [c["center"] for c in b.checks if "listen" in c.get("kind", "")]
        for h in hs[:3]:
            jj = np.where(h.region & (b.cand if getattr(b, "cand", None) is not None else True))[0]
            unquiet_jj = [j for j in jj if all(tb.net.pipe_distance[j, q] > 120 for q in quiet)]
            if unquiet_jj:
                j = int(max(unquiet_jj, key=lambda x: b.post[x]))
                tb.add_option(f"M{k}", "listen", {"center": j}, f"listen within 250 m of {tb.describe_node(j)} ({h.hid})"); k += 1
            elif self.m1_weak_leads and len(jj) > 0 and h.mass >= 0.15:
                # M1: Allow a re-check while the area still holds much of the leak map
                near_quiet = [q for q in quiet if any(tb.net.pipe_distance[j, q] <= 120 for j in jj)]
                if len(near_quiet) < 2:
                    j = int(max(jj, key=lambda x: b.post[x]))
                    tb.add_option(f"M{k}", "listen", {"center": j}, f"listen within 250 m of {tb.describe_node(j)} ({h.hid} - re-check: area holds {h.mass:.0%} of map)"); k += 1
        # Exploration step test of the map's largest untested sections (up to 6, as in the mid evaluation). Offered always, also while a lead
        # exists: the agents' hypotheses are often wrong when water surfaces away from the leak, and in the mid run this sweep lit the leak's
        # section in 72 of 210 cases. The 2-section/needs-a-hypothesis rule (28 Oct) and M2's "no sweep while a lead exists" removed it and
        # cost leak finds (final run of 12 Nov), so both were reverted on 13 Nov.
        mass = secs.mass(b.post)
        rest = [secs.ids[i] for i in np.argsort(-mass) if secs.ids[i] not in tested and secs.ids[i] not in sids][:6]
        # Fair role ablation (ADR-013, as in the 29 Oct rerun): without the Forensic nobody names where to look, so code does not hand the Strategist
        # a ready-made map sweep either; it must propose its own step test (custom option) or use the listening anchors below.
        if rest and self.use_forensic:
            tb.add_option(f"M{k}", "step", {"sections": rest}, f"night step test of sections {', '.join(rest)} (exploration: largest untested share of the map)"); k += 1
        # listening anchored on what the witnesses said: where a caller says the water comes from, and the credible complaint places
        D = tb.net.pipe_distance
        used = [o.args["center"] for o in tb.options.values() if o.kind == "listen"] + quiet
        anchors = []
        for rid, m in b.messages.items():
            if m.usable() and m.w_up is not None:
                anchors.append((2.0, int(np.argmax(m.w_up)), f"where {rid}'s caller says the water comes from"))
        ranked = sorted(((float(b.post[D[int(np.argmax(m.w))] <= 250].sum()) * (1 - m.eps), int(np.argmax(m.w)), rid)
                         for rid, m in b.messages.items() if m.usable()), reverse=True)
        anchors += [(sc, j, f"the place named in {rid}") for sc, j, rid in ranked[:2]]
        for _, j, why in sorted(anchors, key=lambda a: -a[0]):
            if all(D[j, u] > 150 for u in used):
                tb.add_option(f"M{k}", "listen", {"center": j}, f"listen within 250 m of {tb.describe_node(j)} ({why})"); k += 1
                used.append(j)
        if self.m3_confirm_dig and tb.env.checks_left() == 1:
            j_cand, p_cand = b.best_dig()
            jc = self._confirm_spot(tb, j_cand)            # a spot near the dig candidate that no earlier listening covered
            if p_cand < 0.80 and jc is not None and not any(o.kind == "listen" and o.args.get("center") == jc for o in tb.options.values()):
                tb.add_option(f"M{k}", "listen", {"center": jc}, f"listen within 250 m of {tb.describe_node(jc)} (confirm dig spot before excavation)"); k += 1
        for o in tb.options.values():                     # one honest number per option, comparable across kinds (code)
            o.label += f" | chance it finds the leak {self.chance(tb, o):.0%}"

    @staticmethod
    def _confirm_spot(tb, j: int) -> int | None:
        """M3: where to listen to confirm a dig at junction j without repeating an earlier listening check.

        Args: tb = Toolbox; j = dig candidate junction index.
        Returns: j itself if no earlier listening was within 120 m of it; else the most likely junction within 250 m of j that is more than
                 120 m from every earlier listening centre; None if there is no such junction.
        """
        D, post = tb.net.pipe_distance, tb.board.post
        heard = [c.get("center", c.get("at")) for c in tb.board.checks if "listen" in c.get("kind", "")]
        heard = [int(q) for q in heard if q is not None]
        if all(D[j, q] > 120 for q in heard):
            return int(j)
        near = [int(x) for x in np.where(D[j] <= 250)[0] if all(D[x, q] > 120 for q in heard)]
        return max(near, key=lambda x: post[x]) if near else None

    def _confirm_before_last_dig(self, tb, j: int) -> bool:
        """M3: after a dry dig the next dig is the last one allowed (SOP-2); listen near its spot first unless listening already covered it.

        Args: tb = Toolbox; j = junction the gate would dig at.
        Returns: True if a confirmation listening check was run (the caller re-evaluates the gate), else False.
        """
        if not self.m3_confirm_dig or tb.digs < 1 or tb.env.checks_left() <= 0 or j in self.confirmed:
            return False
        self.confirmed.add(j)
        D = tb.net.pipe_distance
        if any("listen" in c.get("kind", "") and c.get("center", c.get("at")) is not None and D[int(c.get("center", c.get("at"))), j] <= 120
               for c in tb.board.checks):
            return False                                   # a listening check already covered this spot
        oid = tb.add_option(f"M{len(tb.options) + 1}", "listen", {"center": int(j)},
                            f"listen within 250 m of {tb.describe_node(int(j))} (confirm the spot before the last allowed dig)")
        if oid is None:
            return False
        tb.field_check(oid)
        return True

    def _best_by_chance(self, tb):
        """Pick the step/listen option with the highest computed chance of finding the leak.

        Args: tb = Toolbox.
        Returns: option id (e.g. 'M2') or None.
        """
        ch = {oid: self.chance(tb, o) for oid, o in tb.options.items() if o.kind in ("step", "listen")}
        return max(ch, key=ch.get) if ch else None

    def _guard(self, tb, pick):
        """ADR-009: the Strategist's pick may not be far worse than the best listed option. The chance is code's (the map x the check's power); the
        options and the map's credibility terms come from the agents. Logged on the turn."""
        if pick not in tb.options:
            return pick
        ch = {oid: self.chance(tb, o) for oid, o in tb.options.items() if o.kind in ("step", "listen")}
        if pick not in ch or not ch:
            return pick
        best = max(ch, key=ch.get)
        if ch[pick] < self.override_ratio * ch[best]:
            # Respect strategist choice if pick has substantially higher hypothesis discrimination (Value of Information)
            if self.slate and len(self.slate) >= 2:
                disc_pick = hyp.discrimination(tb, tb.options[pick], self.slate)
                disc_best = hyp.discrimination(tb, tb.options[best], self.slate)
                if disc_pick > disc_best + 0.04:
                    return pick
            self.turns[-1]["override"] = {"chosen": pick, "chance": round(ch[pick], 3), "best": best, "best_chance": round(ch[best], 3)}
            return best
        return pick

    @staticmethod
    def chance(tb, o) -> float:
        """P(this check hears / lights the leak) from the current map: listening = P(leak) x sum over the 250 m disc of P(leak there) x
        P(hear | there); a step test = P(leak) x sum over its sections of P(leak in it) x P(it lights | leak in it)."""
        b = tb.board
        post, pl = b.post, b.p_leak
        if o.kind == "listen":
            d = tb.net.pipe_distance[o.args["center"]]
            inside = d <= 250.0
            return float(pl * (post[inside] * p_hear(b.q_hat) * (1 - LISTEN_EDGE * d[inside] / 250.0)).sum())
        if o.kind == "step":
            return float(pl * sum(tb.sections.area_mass(sid, post) * b.tools.step_power(sid, b.q_hat, b.q_sd)[0] for sid in o.args["sections"]))
        return 0.0

    def _custom(self, tb, c) -> str | None:
        """Add the Strategist's own check (a step test of sections or a listen at a place) as option 'C1'.

        Args: tb = Toolbox; c = {'kind': 'step', 'sections': [...]} or {'kind': 'listen', 'at': place or junction}.
        Returns: option id or None if invalid.
        """
        if not isinstance(c, dict):
            return None
        if c.get("kind") == "step":
            sids = [str(x) for x in (c.get("sections") or []) if str(x) in tb.sections.areas]
            tested = {chk["label"].split(":")[-1] for chk in tb.board.checks if chk["kind"] == "step"}
            sids = [s for s in sids if s not in tested]
            return tb.add_option("C1", "step", {"sections": sids}, "night step test (Strategist's own)") if sids else None
        if c.get("kind") == "listen":
            at = str(c.get("at", ""))
            j = tb.net.index.get(at)
            if j is None:
                w = hyp.place_weights(tb.places, at, True)
                j = int(np.argmax(w)) if w is not None else None
            if j is not None:
                quiet = [chk["center"] for chk in tb.board.checks if "listen" in chk.get("kind", "")]
                if any(tb.net.pipe_distance[j, q] <= 120 for q in quiet):
                    if not (self.m1_weak_leads and float(tb.board.post[tb.net.pipe_distance[j] <= 250].sum()) >= 0.15):
                        return None
                return tb.add_option("C1", "listen", {"center": int(j)}, f"listen within 250 m of {tb.describe_node(int(j))} (Strategist's own)")
        return None
