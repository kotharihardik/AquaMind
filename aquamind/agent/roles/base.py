"""Shared plumbing of the crew roles: one LLM turn, the board text an agent reads, and the hypothesis slate.

The attributes used here (llm, turns, n_llm, slate, challenges, rejected_citations, m4_ticket_type, _view) are set in Crew.__init__.
"""
from __future__ import annotations

import numpy as np

from ...search import hypotheses as hyp
from ...search import topography as topo
from ..toolbox import next_seq


class CrewBase:
    """LLM turn, board text and slate bookkeeping shared by every role."""

    def _ask(self, role: str, system: str, user: str, why: str) -> dict:
        """Send one prompt to the LLM for a role and record the turn.

        Args: role = 'forensic' | 'skeptic' | 'strategist' | 'auditor'; system = role prompt; user = board text; why = reason.
        Returns: the parsed JSON answer (dict; has '_error' on failure).
        """
        self.n_llm += 1
        d = self.llm.json(system, user, [], tag=f"crew_{role}")
        self.turns.append({"turn": len(self.turns), "role": role, "why": why, "thought": d.get("thought"), "error": d.get("_error"),
                           "seq": next_seq()})                               # seq: replay order between turns and tool calls
        return d

    def _text(self, tb, why: str, options_table: bool = False, topography: bool = False, challenges: bool = False) -> str:
        """Board text plus crew extras: topography, hypotheses, options table, Skeptic challenges, rejected citations (M4).

        Args: tb = Toolbox; why = reason; options_table, topography, challenges = which extras to add.
        Returns: prompt text.
        """
        t = self._view.board_text(tb, why)
        extra = []
        if topography:
            extra += topo.view(tb)
        if self.slate:
            extra += ["HYPOTHESES (named by the Forensic Advocate, scored by code):"] + ["  " + h.line() for h in self.slate]
            if getattr(tb.board, "cand", None) is not None:
                post = tb.board.post
                for h in self.slate:
                    if h.region is not None and h.region.any():
                        jj = np.where(h.region & tb.board.cand)[0]
                        if len(jj):
                            jb = int(jj[np.argmax(post[jj])])
                            extra.append(f"  physics of {h.hid}: {len(jj)} junctions simulated; best fit {tb.describe_node(jb)} holds {post[jb]:.1%} of the map")
            if options_table and tb.options:
                extra.append("OPTION PREDICTIONS (chance of a positive result under each hypothesis; 'discriminates' = how well it tells them apart):")
                for oid, o in tb.options.items():
                    ps = {h.hid: hyp.p_positive(tb, o, h) for h in self.slate}
                    head = f"  {oid}: {o.label} | leak inside {o.p_inside:.0%}" if not tb.use_planner else f"  {oid}: value {o.value:.2f}"
                    extra.append(head + " | " + ", ".join(f"{k} {v:.2f}" for k, v in ps.items() if v is not None)
                                 + f" | discriminates {hyp.discrimination(tb, o, self.slate):.2f}")
        if challenges and self.challenges:
            extra.append("SKEPTIC'S CHALLENGES (answer them: revise your hypotheses or explain why the case stands):")
            extra += [f"  against {c['target']}: {c['contradiction']} (evidence: {', '.join(c['evidence']) or '-'}; suggests {c['check']})" for c in self.challenges]
        if self.m4_ticket_type and self.rejected_citations:          # M4: show each agent the citations code refused
            extra.append("REJECTED CITATIONS (code checks failed; do not rely on these):")
            for rej in self.rejected_citations[-4:]:
                extra.append(f"  {rej['citation']}: {rej['reason']}")
        return t + ("\n" + "\n".join(extra) if extra else "")

    def _set_slate(self, tb, raw) -> None:
        """Turn the Advocate's raw hypotheses into scored Hypothesis objects and log them.

        Args: tb = Toolbox; raw = list of hypothesis dicts from the LLM.
        """
        if not self.use_slate or not isinstance(raw, list) or not raw:
            return
        self.slate = hyp.build(raw, tb)
        self.turns[-1]["slate"] = [{"id": h.hid, "kind": h.kind, "where": h.where, "cause": h.cause, "because": h.because, "explains": h.explains,
                                    "mass": round(h.mass, 3), "cited_ok": h.cited_ok, "cited_bad": h.cited_bad, "error": h.error,
                                    "region": [int(i) for i in h.region.nonzero()[0]] if h.region is not None else None} for h in self.slate]

    def _refresh_slate(self, tb) -> None:
        """Code re-scores the agent's own hypotheses as evidence arrives (the regions stay; the masses move)."""
        if not self.slate:
            return
        raw = [{"id": h.hid, "kind": h.kind, "where": h.where, "because": h.because, "explains": h.explains, "cause": h.cause} for h in self.slate]
        self.slate = hyp.build(raw, tb)
