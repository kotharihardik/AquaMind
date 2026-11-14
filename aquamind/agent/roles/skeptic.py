"""CRITICAL SKEPTIC: attacks the Advocate's leading hypothesis with the physical data; code counts which cited evidence exists."""
from __future__ import annotations

from .prompts import SKEPTIC


class SkepticMixin:
    """Critical Skeptic turn."""

    def skeptic(self, tb, why: str) -> None:
        """Critical Skeptic turn: attack the hypotheses with physical facts; challenges are checked by code.

        Args: tb = Toolbox; why = reason for this turn.
        """
        self._refresh_slate(tb)
        d = self._ask("skeptic", SKEPTIC, self._text(tb, why, topography=True) + "\n\nYour challenges?", why)
        rec = self.turns[-1]
        known = set(tb.board.messages) | {"SENSORS"} | {h.hid for h in self.slate} | set(tb.docs.by_id)
        self.challenges = []
        for c in (d.get("challenges") or [])[:3]:
            if isinstance(c, dict) and str(c.get("contradiction", "")).strip():
                ev = [str(x) for x in (c.get("evidence") or [])][:5]
                self.challenges.append({"target": str(c.get("target", "H1"))[:6], "contradiction": str(c["contradiction"])[:220], "evidence": ev,
                                        "check": str(c.get("check", "none")).lower() if str(c.get("check", "none")).lower() in ("listen", "step") else "none",
                                        "cited_ok": sum(e in known for e in ev), "cited_bad": sum(e not in known for e in ev)})
        rec["verdict"] = str(d.get("verdict", ""))[:10]
        rec["challenges"] = self.challenges
