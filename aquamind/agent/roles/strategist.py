"""HYDRAULIC STRATEGIST: chooses the next field check from the menu (or proposes its own)."""
from __future__ import annotations

from ...search import hypotheses as hyp
from .prompts import STRATEGIST


class StrategistMixin:
    """Hydraulic Strategist turn."""

    def decision_open(self, tb) -> bool:
        """Is the choice of the next check genuinely open? (two options of similar value, or two hypotheses that each hold real mass)"""
        opts = tb.plan()
        close = len(opts) >= 2 and opts[0].value - opts[1].value < 0.08
        split = sum(h.mass >= 0.2 for h in self.slate) >= 2
        return close or split

    def strategist(self, tb, why: str):
        """Hydraulic Strategist turn: choose the next field check from the menu (or a custom one).

        Args: tb = Toolbox; why = reason for this turn.
        Returns: option id to run, or None.
        """
        self._refresh_slate(tb)
        if not tb.use_planner:
            self.menu(tb)
        d = self._ask("strategist", STRATEGIST, self._text(tb, why, options_table=True, challenges=True) + "\n\nWhich check?", why)
        rec = self.turns[-1]
        pick = str(d.get("choose") or "").strip().upper()
        if d.get("custom"):
            c = self._custom(tb, d.get("custom"))
            if c:
                rec["choose"], rec["custom"] = c, d.get("custom")
                return c
        if pick in tb.options:
            rec["choose"] = pick
            rec["discriminates"] = round(hyp.discrimination(tb, tb.options[pick], self.slate), 3) if self.slate else None
            return pick
        rec["note"] = "no valid choice"
        return None
