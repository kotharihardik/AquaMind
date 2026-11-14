"""FORENSIC ADVOCATE: what does the evidence mean? Writes its own questions for the callers, reads the town archive (RAG), reads the
callers as one story (triangulation) and names the competing hypotheses, which become the simulated search space (ADR-006)."""
from __future__ import annotations

import numpy as np

from ...search import topography as topo
from .prompts import DOMAIN, FORENSIC


class ForensicMixin:
    """Forensic Advocate turn and the code that turns its hypotheses and the witnesses into places to simulate."""

    def forensic(self, tb, why: str) -> dict:
        """Forensic Advocate turn: cite records and failing mains, triangulate, call citizens, name hypotheses.

        Args: tb = Toolbox; why = reason for this turn.
        Returns: the turn record.
        """
        self.n_forensic += 1
        d = self._ask("forensic", FORENSIC.format(domain=DOMAIN), self._text(tb, why, topography=True, challenges=True)
                      + ("\n" + "\n".join(self.open_questions(tb)) if self.open_questions(tb) else "") + "\n\nYour analysis?", why)
        rec = self.turns[-1]
        self.stance = "continue"
        if "_error" in d:
            return rec
        for r in d.get("records") or []:
            if isinstance(r, dict):
                rec_id = str(r.get("record_id"))
                rep_id = str(r.get("report_id"))
                res = tb.use_record(rep_id, rec_id)
                if not res.get("ok") and self.m4_ticket_type:
                    self.rejected_citations.append({"citation": rec_id, "reason": res.get("error", "invalid record")})
        for rec_id in [str(x) for x in (d.get("failing_mains") or []) if isinstance(x, (str, int))][:4]:
            r = tb.use_repair_record(rec_id)
            rec.setdefault("failing_mains", []).append({"id": rec_id, "ok": r.get("ok"), "where": r.get("where")})
            if not r.get("ok") and self.m4_ticket_type:
                self.rejected_citations.append({"citation": rec_id, "reason": r.get("error", "invalid repair ticket")})
        for h in self.slate if self.m4_ticket_type else []:
            if h.cited_bad > 0 and h.error:
                self.rejected_citations.append({"citation": f"hypothesis {h.hid}", "reason": h.error})
        if self.use_triangulation and isinstance(d.get("triangulate"), dict):
            ok, msg = topo.triangulate(tb, d["triangulate"])
            rec["triangulate"] = {**{k: d["triangulate"].get(k) for k in ("origin", "downstream")}, "ok": ok, "msg": msg}
            tb._log("triangulate", d["triangulate"], {"ok": ok, "msg": msg})
        calls = [{"report_id": str(c.get("report_id")), "question": str(c.get("question", ""))[:300]}
                 for c in (d.get("questions") or []) if isinstance(c, dict) and tb.can_call(str(c.get("report_id"))) and str(c.get("question", "")).strip()
                 and not (self.calls_unexplained_only and tb.board.messages[str(c.get("report_id"))].eps >= 0.9)][:4]     # no call about a complaint a record already explains
        if calls:
            rec["questions"] = calls
            tb.call_back(calls)
        self._set_slate(tb, d.get("hypotheses"))
        self._hypothesise(tb)
        self._witness_candidates(tb)
        self._refresh_slate(tb)
        st = str(d.get("stance") or "continue").strip().lower()
        self.stance = st if st in ("continue", "close_no_leak", "hand_over") else "continue"
        rec["stance"] = self.stance
        return rec

    @staticmethod
    def open_questions(tb) -> list[str]:
        """What is still UNKNOWN about each usable complaint (code reads the reply-2.0 fields; the Advocate writes the actual questions).
        The single most valuable gap: water that only COLLECTS at a place (source 'elsewhere') without a known origin; the caller can usually
        say where it runs down from, and that origin is where the leak is."""
        out = []
        for rid, m in tb.board.messages.items():
            if not m.usable() or not tb.can_call(rid):
                continue
            kind = tb.search.rep.kind(m.read)
            gaps = []
            if m.source == "elsewhere" and m.w_up is None:
                gaps.append("WHERE THE WATER COMES FROM: it only collects there; ask which way it runs and from which street or landmark "
                            "(the origin locates the leak)")
            elif m.source == "unknown" and kind not in ("damp", "sound", "discolour") and m.w_up is None:
                gaps.append("whether the water comes out right there or flows in from somewhere else, and from where")
            if m.calls == 0:
                gaps.append("look-alike causes (house pipe or tank, hose or garden, drain, pump, washing)")
            if gaps:
                out.append(f"  {rid}: still unknown: " + "; ".join(gaps))
        return (["OPEN QUESTIONS (what the callers can still tell us):"] + out) if out else []

    def _witness_candidates(self, tb) -> None:
        """Hypothesise the places the witnesses point to: where a caller says the water comes from, and the places of the complaints that are
        still credible (simulated like any other hypothesis)."""
        b = tb.board
        if getattr(b, "cand", None) is None:
            return
        D, js = tb.net.pipe_distance, []
        for m in b.messages.values():
            if not m.usable():
                continue
            if m.w_up is not None:
                js += list(np.where(D[int(np.argmax(m.w_up))] <= 250)[0])
            if m.eps < 0.5:
                js += list(np.where(D[int(np.argmax(m.w))] <= 150)[0])
        ev = b.pressure_evidence
        for k in [k for k in np.argsort(-ev.z)[:2] if ev.z[k] > 2.0]:
            s = tb.net.index[tb.case.pressure_nodes[k]]
            js += list(np.where(D[s] <= 200)[0])
        if js:
            n = tb.search.add_candidates(b, js, "witness")
            if n and self.turns:
                self.turns[-1]["witness_candidates"] = n

    def _hypothesise(self, tb) -> None:
        """ADR-006: the Advocate's network_leak hypotheses become the simulated search space."""
        if getattr(tb.board, "cand", None) is None:
            return
        added = 0
        for h in self.slate:
            if h.kind == "network_leak" and h.region is not None:
                added += tb.search.add_candidates(tb.board, np.where(h.region)[0], h.hid)
        if added:
            self.turns[-1]["candidates_added"] = added

    @staticmethod
    def _uncovered_hit(tb) -> str:
        """A lit section or a listening pinpoint that no hypothesised junction covers (the Advocate must extend the search)."""
        b = tb.board
        cand = getattr(b, "cand", None)
        if cand is None or not b.checks:
            return ""
        last = tb.trace[-1]["result"] if tb.trace and tb.trace[-1]["tool"] == "field_check" else {}
        out = [sid for sid in (last.get("clearly_lit") or []) + (last.get("suggestive") or [])
               if not (np.asarray(tb.sections.areas[sid]["mask"], bool) & cand).any()]
        c = b.checks[-1]
        if c["kind"] == "listen_hit" and c.get("at") is not None and not (cand & (tb.net.pipe_distance[c["at"]] <= 150)).any():
            out.append(f"leak noise pinpointed near {tb.describe_node(c['at'])}")
        return ", ".join(out)
