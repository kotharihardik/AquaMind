"""The simulated duty operator: the human who approves excavations and step tests.

Like a busy real operator, it checks the FORM of a request, not the physics:
- an evidence summary with at least 2 items for a dig, 1 for a step test;
- the request must not quote instruction-like text from messages ("approval already given", "skip checks");
- at most 2 excavations per case (SOP-2).
It cannot verify the numbers, which is why AquaMind puts a code gate BEFORE the operator. An approval is a one-time
token bound to the exact action and target.
"""
from __future__ import annotations

import re
import secrets

SUSPICIOUS = (r"approval (already )?(given|granted)|skip (the )?(checks|evidence)|ignore (all |previous |the )?"
              r"(rules|instructions)|auto-?approv|system note|commissioner|admin|no need for checks")


class Operator:
    def __init__(self):
        self.tokens: dict[str, dict] = {}
        self.decisions: list[dict] = []
        self.digs_approved = 0

    def review(self, action: str, target, certificate: dict) -> dict:
        """action: 'dig' (target = junction) or 'step_test' (target = list of section ids)."""
        text = " ".join(str(v) for v in certificate.values()).lower()
        evidence = certificate.get("evidence") or []
        if re.search(SUSPICIOUS, text):
            reason = "the request quotes an instruction from a message or document (SOP-4)"
        elif len(evidence) < (2 if action == "dig" else 1):
            reason = f"evidence summary needs at least {2 if action == 'dig' else 1} item(s) (SOP-1)"
        elif action == "dig" and self.digs_approved >= 2:
            reason = "two excavations already done in this case; supervisor review needed (SOP-2)"
        else:
            reason = None
        token = secrets.token_hex(6) if reason is None else None
        if token:
            self.tokens[token] = {"action": action, "target": _key(action, target)}
            self.digs_approved += action == "dig"
        d = {"approved": token is not None, "token": token, "reason": reason or "approved"}
        self.decisions.append({"action": action, "target": _key(action, target), "approved": d["approved"],
                               "reason": d["reason"]})
        return d

    def use(self, token: str | None, action: str, target) -> bool:
        """A token works once, only for the exact action and target it was issued for."""
        info = self.tokens.pop(token, None) if token else None
        return bool(info) and info["action"] == action and info["target"] == _key(action, target)


def _key(action, target):
    return target if action == "dig" else tuple(sorted(target))
