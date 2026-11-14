"""AquaMind crew: four LLM agents with separate jobs around the same toolbox and the same code gates (ADR-007, ADR-007).

    FORENSIC ADVOCATE      what does the evidence mean? writes its own questions for the callers (no template), reads the town's complaint archive
                           (records.md, RAG), reads the callers as ONE story across the hill (triangulation), keeps a slate of competing hypotheses
    CRITICAL SKEPTIC       attacks the advocate's leading hypothesis with the physical data (sensors, elevations, records); the advocate must answer
    HYDRAULIC STRATEGIST   which field check tells the leading hypotheses apart? (code gives each option's value and P(positive) under every hypothesis)
    SAFETY & SOP AUDITOR   tries to refute the case before a crew is sent and applies the municipal SOP (retrieved from sop.md): school hours, permits on
                           resurfaced roads, hospital notices, telling the callers; can delay or hold a dig, never open a gate

Code stays the motor and the safety system: the leak map, the look-ahead values, the dig and close gates, the operator token and all numbers.
The agents never see raw citizen text (guard-rail 2), never produce a number that a decision depends on (guard-rail 1), and cannot open a gate. Every
claim an agent makes about the world is checked by code before it reaches the board (records by place, triangulation by elevation and reach, SOP
actions against the map). There is no template anywhere: without the Forensic nobody is called; without the Auditor no SOP paperwork is done.
Ablations switch one agent off at a time (`forensic`, `skeptic`, `strategist`, `auditor`, `slate`, `triangulation`, `sop`).

Code layout: the roles live in aquamind/agent/roles/ (one module per role, prompts in roles/prompts.py); this module combines them into
`Crew` and holds its switches (__init__) and the investigation loop (run).
"""
from __future__ import annotations

from ..search import hypotheses as hyp
from ..search.candidates import heuristic
from .aquamind import AquaMind
from .code_policy import Autopilot
from .roles import AuditorMixin, CrewBase, FieldCheckMixin, ForensicMixin, SkepticMixin, StrategistMixin
from .roles.prompts import AUDITOR, DOMAIN, FORENSIC, SKEPTIC, STRATEGIST  # noqa: F401  (kept importable from here)
from .toolbox import mitigation_flags

__all__ = ["Crew", "DOMAIN", "FORENSIC", "SKEPTIC", "STRATEGIST", "AUDITOR"]


class Crew(ForensicMixin, SkepticMixin, StrategistMixin, FieldCheckMixin, AuditorMixin, CrewBase):
    name = "agentix_crew"

    def __init__(self, llm, *, forensic: bool = True, skeptic: bool = True, strategist: bool = True, auditor: bool = True, slate: bool = True,
                 triangulation: bool = True, sop: bool = True, tol: float = 0.10, max_llm: int = 16, max_rounds: int = 14, max_strategist: int = 6,
                 max_forensic: int = 4, disc_min: float = 0.35, tol_disc: float = 0.20, disc_override: bool = True, hold: bool = True,
                 hold_below: float = 0.95, calls_unexplained_only: bool = True, chance_policy: bool = True, override_ratio: float = 0.6,
                 m1_weak_leads: bool | None = None, m2_turn_budget: bool | None = None, m3_confirm_dig: bool | None = None,
                 m4_ticket_type: bool | None = None, m5_corner_permits: bool | None = None):
        """The four-agent crew (Forensic Advocate, Skeptic, Strategist, Auditor) with its switches and budgets.

        Args: llm = LLM client; forensic/skeptic/strategist/auditor = role on/off; slate = hypotheses on/off;
              triangulation, sop = those features on/off; tol, disc_min, tol_disc, override_ratio = Strategist guard tolerances;
              max_llm, max_rounds, max_strategist, max_forensic = budgets; hold, hold_below = Auditor hold rule;
              chance_policy = code keeps the Strategist near the best computed chance; m1..m5 = mitigation switches (None = default on).
        """
        self.llm, self.use_forensic, self.use_skeptic, self.use_strategist = llm, forensic, skeptic, strategist
        self.use_auditor, self.use_slate, self.use_triangulation, self.use_sop = auditor, slate, triangulation, sop

        # Mitigation flags (Mid-project report Section 10)
        f = mitigation_flags(m1_weak_leads=m1_weak_leads, m2_turn_budget=m2_turn_budget, m3_confirm_dig=m3_confirm_dig,
                             m4_ticket_type=m4_ticket_type, m5_corner_permits=m5_corner_permits)
        self.m1_weak_leads = f["m1_weak_leads"]
        self.m2_turn_budget = f["m2_turn_budget"]
        self.m3_confirm_dig = f["m3_confirm_dig"]
        self.m4_ticket_type = f["m4_ticket_type"]
        self.m5_corner_permits = f["m5_corner_permits"]

        # M2: Raise Forensic's turn limit and overall LLM call budget
        if self.m2_turn_budget:
            max_forensic = max(max_forensic, 6)
            max_llm = max(max_llm, 20)

        self.tol, self.max_llm, self.max_rounds = tol, max_llm, max_rounds
        self.max_strategist, self.n_strategist, self.max_forensic, self.n_forensic = max_strategist, 0, max_forensic, 0
        self.disc_min, self.tol_disc, self.disc_override, self.use_hold, self.hold_below = disc_min, tol_disc, disc_override, hold, hold_below
        self.calls_unexplained_only = calls_unexplained_only
        self.chance_policy, self.override_ratio = chance_policy, override_ratio       # ADR-009: code keeps the Strategist's pick near the best computed chance
        self.held, self.offered, self.audited, self.delays, self.refused = False, set(), set(), 0, False
        self.turns: list[dict] = []
        self.n_llm = 0
        self.slate: list[hyp.Hypothesis] = []
        self.challenges: list[dict] = []
        self.sop_actions: list[str] = []
        self.rejected_citations: list[dict] = []
        self.skeptic_quiet_done = False
        self.confirmed: set[int] = set()                      # M3: dig spots already confirmed by listening before the last dig
        self.stance = "continue"
        self._view = AquaMind(None)

    def run(self, tb):
        """Run a whole case: first look, Skeptic, then the loop of checks, audits and digs until done.

        Args: tb = the Toolbox of the case.
        Returns: the tool trace.
        """
        ap = Autopilot()
        usable = any(m.usable() for m in tb.board.messages.values())
        self.offered |= {rid for rid, m in tb.board.messages.items() if m.usable()}
        if not self.use_forensic:
            tb.weak_dig_areas = False          # fair ablation (ADR-013): making weak-flow areas diggable is the Forensic's re-think, not free code help
        if not self.use_forensic and getattr(tb.board, "cand", None) is not None:
            tb.search.add_candidates(tb.board, heuristic(tb), "heuristic")
        if self.use_forensic and (usable or getattr(tb.board, "cand", None) is not None):
            self.forensic(tb, "first look: read the evidence, ask the people (your own questions), check the records, read the callers as a story, name your hypotheses")
            answered = any(tb.attempts.get(r) for r in tb.board.messages)
            if self.use_skeptic and self.n_llm < self.max_llm and (answered or self.slate):
                self.skeptic(tb, "the answers are in: attack the advocate's case with the physical data")
            if self.n_llm < self.max_llm and any(m.usable() and tb.can_call(m.rid) for m in tb.board.messages.values()) or self.challenges:
                self.forensic(tb, "answer the Skeptic's challenges: update your hypotheses and ask your follow-up questions")
        for _ in range(self.max_rounds):
            if tb.done or tb.env.done:
                break
            if self.stance in ("close_no_leak", "hand_over") and tb.close_gate()[0] and not tb.gate()[0]:
                tb.close("no_leak" if self.stance == "close_no_leak" else "hand_over", str(self.turns[-1].get("thought") or "")[:300])
                break
            new = [rid for rid, m in tb.board.messages.items() if m.usable() and tb.attempts.get(rid, 0) == 0 and rid not in self.offered]
            if new:
                self.offered |= set(new)
                max_pre_field = (self.max_forensic - 2) if self.m2_turn_budget else self.max_forensic
                if self.use_forensic and self.n_llm < self.max_llm and self.n_forensic < max_pre_field:
                    self.forensic(tb, f"new messages arrived ({', '.join(new[:4])}): ask them and update your hypotheses")
            ok, _ = tb.gate()
            if ok and not self.refused:
                j, _p = tb.board.best_dig()
                if self._confirm_before_last_dig(tb, j):
                    continue                                     # M3: re-evaluate the gate with the confirmation result
                key = (j, tb.digs, len(tb.board.checks))
                if self.use_auditor and key not in self.audited and self.n_llm < self.max_llm:
                    self.audited.add(key)
                    verdict = self.audit(tb, "a dig is about to be sent; try to refute the case and apply the SOP")
                    if verdict == "hold":
                        break
                    if verdict == "delay":
                        continue
                self.refused = not tb.propose_dig("gate open", sop_actions=self.sop_actions).get("ok")
                continue
            self.refused = False
            if tb.use_planner:
                opts = tb.plan()
                if not opts or opts[0].value <= 0.01:
                    break
            elif tb.env.checks_left() <= 0 or tb.env.now >= tb.env.horizon:
                break
            if not self.use_forensic and getattr(tb.board, "cand", None) is not None:
                tb.search.add_candidates(tb.board, heuristic(tb), "heuristic")        # no Advocate: the no-AI fallback decides where to look
            pick = None
            if self.use_strategist and self.n_llm < self.max_llm and self.n_strategist < self.max_strategist and (not tb.use_planner or self.decision_open(tb)):
                self.n_strategist += 1
                pick = self.strategist(tb, "choose the next field check")
            if pick is not None and self.chance_policy and not tb.use_planner and self.turns and self.turns[-1].get("role") == "strategist":
                pick = self._guard(tb, pick)
            if pick is None and self.use_strategist and self.chance_policy and not tb.use_planner:
                self.menu(tb)                                # strategist active but no valid pick: best option by computed chance (ADR-009)
                pick = self._best_by_chance(tb)
            if pick is None:
                if tb.use_planner:
                    pick = "F1"
                elif tb.rule_option() is not None:          # no Strategist: the rule dispatcher
                    pick = "F1"
                else:
                    break
            before = (tb.board.best_dig()[1], tb.board.p_leak)
            tb.field_check(pick)
            last = tb.trace[-1]["result"] if tb.trace and tb.trace[-1]["tool"] == "field_check" else {}
            lit = bool(last.get("clearly_lit")) or bool(last.get("suggestive"))
            heard = bool(last.get("heard"))
            vfail = bool(last.get("valve_failed"))
            surprise = abs(tb.board.best_dig()[1] - before[0]) > 0.15 or abs(tb.board.p_leak - before[1]) > 0.15
            outside = self._uncovered_hit(tb)

            # M2: Give the Skeptic a turn after the first quiet check
            if self.m2_turn_budget and self.use_skeptic and self.use_forensic and not self.skeptic_quiet_done and self.n_llm < self.max_llm:
                # (needs the Forensic, ADR-013: the Skeptic challenges the Forensic's hypotheses, and only the Forensic can answer)
                if (last.get("kind") == "listening" and not heard) or (last.get("kind") == "step test" and not bool(last.get("clearly_lit"))):
                    self.skeptic_quiet_done = True
                    self.skeptic(tb, "a field check was quiet. A quiet check only clears what it could see (listening hears a small leak only "
                                     "part of the time; a step test can miss a small leak in a busy section) and it does NOT explain the sensor "
                                     "drop or the zone inflow rise. Challenge WHERE the leading hypothesis puts the leak, not whether a leak "
                                     "exists while the sensors show one: name the place or section the leak could be instead")
                    if self.challenges and self.use_forensic and self.n_llm < self.max_llm and self.n_forensic < self.max_forensic:
                        self.forensic(tb, "the Skeptic challenged the case after the quiet check: update your hypotheses")

            weak_lead = False
            if self.m1_weak_leads:
                if last.get("kind") == "listening" and not heard and any(h.mass >= 0.15 for h in self.slate):
                    weak_lead = True
                elif bool(last.get("suggestive")):
                    weak_lead = True

            if outside and self.use_forensic and self.n_llm < self.max_llm:
                self.forensic(tb, f"a physical check found something OUTSIDE your hypotheses ({outside}): extend them")
                continue
            blank = self.use_slate and not self.slate and not self.stance == "x"
            rethink = lit or heard or vfail or surprise or weak_lead or (blank and self.n_forensic == 0)
            if self.use_forensic and self.n_llm < self.max_llm and self.n_forensic < self.max_forensic and rethink:
                if weak_lead and not (lit or heard or vfail or surprise):
                    why = ("listening check was quiet (small leak detection is weak) but area still holds mass: re-think your hypotheses and consider adjacent sections" if last.get("kind") == "listening" else
                           f"step test showed suggestive weak extra flow in {', '.join(last.get('suggestive', []))}: update your hypotheses to focus there")
                else:
                    why = (f"a physical check LIT up section(s) {', '.join(last.get('clearly_lit', []))}: update your hypotheses to focus there" if lit else
                           f"leak noise PINPOINTED by listening: {last.get('pinpointed', 'nearby')}" if heard else
                           f"boundary valve FAILED in step test ({', '.join(last.get('valve_failed', []))}): note that these were NOT tested/cleared" if vfail else
                           "the last check changed the evidence significantly: re-think your hypotheses")
                self.forensic(tb, why)
        if not (tb.done or tb.env.done):
            ap.finish(tb)
        return tb.trace
