"""SAFETY & SOP AUDITOR: tries to refute the case before a crew is sent and applies the municipal SOP (retrieved from sop.md); it can
delay or hold a dig, never open a gate."""
from __future__ import annotations

from .prompts import AUDITOR


class AuditorMixin:
    """Safety & SOP Auditor turn."""

    def audit(self, tb, why: str) -> str:
        """-> 'approve' | 'delay' (one more check was run) | 'hold'. Also fills self.sop_actions for the dig."""
        j, p = tb.board.best_dig()
        self._refresh_slate(tb)
        text = self._text(tb, why, options_table=bool(tb.options), challenges=True) + "\n" + ("\n".join(tb.sop_context(j)) if self.use_sop else "")
        d = self._ask("auditor", AUDITOR, text + f"\n\nDIG PROPOSED at {tb.describe_node(j)} with P(success) {p:.2f}. Your verdict?", why)
        rec = self.turns[-1]
        rec["verdict"], rec["objections"] = d.get("verdict"), d.get("objections")
        self.sop_actions = [str(x) for x in (d.get("sop_actions") or [])][:6] if self.use_sop else []
        rec["sop_actions"] = self.sop_actions
        b = tb.board
        physical = bool(b.area_support(j)) or any(c["kind"] == "listen_hit" and c.get("at") is not None and tb.net.pipe_distance[c["at"], j] <= 150
                                                  for c in b.checks)
        if str(d.get("verdict")).lower() == "hold" and physical:
            rec["note"] = "hold ignored: the dig has physical support (a lit section or a listening pinpoint); a narrative cannot veto it"
        elif str(d.get("verdict")).lower() == "hold" and self.use_hold and not self.held and p < self.hold_below:
            ob = "; ".join(str(x) for x in (d.get("objections") or []))[:200] or str(d.get("thought") or "")[:200]
            if ob.strip():
                self.held = True
                rec["note"] = "HOLD: the dig is not sent; the case is handed to the supervisor"
                tb.close("hand_over", f"auditor hold: {ob}", force=True)
                return "hold"
        if str(d.get("verdict")).lower() != "object" or self.delays >= 2:
            return "approve"
        if not tb.use_planner:
            self.menu(tb)
        opts = tb.plan() if tb.use_planner else list(tb.options.values())
        if not opts or tb.env.checks_left() <= 0:
            if self.m3_confirm_dig and p < tb.planner.dig_bar and not physical:      # like the hold rule: a narrative cannot veto physical support
                ob = "; ".join(str(x) for x in (d.get("objections") or []))[:200] or str(d.get("thought") or "")[:200]
                self.held = True
                rec["note"] = f"HOLD: Auditor objected to weak dig (P={p:.2f} < {tb.planner.dig_bar:.2f}) with no checks left; handing over to supervisor to avoid dry dig"
                tb.close("hand_over", f"auditor objection: {ob}", force=True)
                return "hold"
            rec["note"] = "objection ignored: no field check left"
            return "approve"
        need = str(d.get("needs") or "none").lower()
        cand = [oid for oid, o in tb.options.items() if o.kind == need] or ([] if need == "none" else list(tb.options))
        if need == "none" or not cand:
            if self.m3_confirm_dig and tb.env.checks_left() > 0:
                conf_oid = tb.add_option(f"M{len(tb.options)+1}", "listen", {"center": j}, f"listen within 250 m of {tb.describe_node(j)} (Auditor confirmation)")
                cand = [conf_oid] if conf_oid else []
            if not cand:
                rec["note"] = "objection ignored: it names no concrete check"
                return "approve"
        self.delays += 1
        rec["ran"] = cand[0]
        tb.field_check(cand[0])
        return "delay"
