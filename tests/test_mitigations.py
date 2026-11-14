"""Tests for the five mid-project mitigations (M1-M5) from Mid-Report Section 10:
- M1: Re-check weak leads, suggestive flow unlocks section parts, detection power displayed
- M2: Agent turn budget (raised forensic limit, turns kept for field, Skeptic after quiet check, no blind sweeps)
- M3: Confirm before dig, Auditor objections respected on weak last-chance digs
- M4: Ticket-type check (false alarms vs utility repairs) and rejected citation reporting
- M5: Multi-street corner permits at junctions
"""
import json
import os
import numpy as np
import pytest

from aquamind.town.assets import Town
from aquamind.rag.retriever import load_corpus
from aquamind.safety.rulebook import RuleBook
from aquamind.agent.toolbox import Toolbox
from aquamind.agent.crew import Crew
from aquamind.methods import registry

BENCH = "data/benchmark/agentix_master" if os.path.exists("data/benchmark/agentix_master/dev.json") else "data/benchmark/agentix_final"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{BENCH}/dev.json"), reason="benchmark not assembled")


@pytest.fixture(scope="module")
def world():
    """Fixture: one shared World for the benchmark (module scope)."""
    from aquamind.eval.runner import World
    return World(BENCH)


# ---------------------------------------------------------------------- M4 Tests
def test_m4_ticket_type_classification(world):
    """M4: Verify tickets in records.md are accurately partitioned into false alarms vs utility repairs."""
    # REC-001 is a garden watering complaint (no utility leak found)
    doc_fa = world.docs.by_id["REC-001"]
    assert Toolbox.is_false_alarm_ticket(doc_fa.text)

    # REC-002 is a service connection leak repaired (real utility repair)
    doc_rep = world.docs.by_id["REC-002"]
    assert not Toolbox.is_false_alarm_ticket(doc_rep.text)

    # REC-011 is a sprinkler complaint (no utility leak found)
    assert Toolbox.is_false_alarm_ticket(world.docs.by_id["REC-011"].text)

    # REC-008 is a main burst repaired (real utility repair)
    assert not Toolbox.is_false_alarm_ticket(world.docs.by_id["REC-008"].text)


def test_m4_record_check_rejects_repairs(world):
    """M4: _record_check rejects utility repair tickets as false alarms when m4_ticket_type=True."""
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)

    # Toolbox with M4 active
    tb_m4 = Toolbox(env, case, search, reader, world.docs, world.rules, m4_ticket_type=True)
    # Pick a message and check citation of REC-002 (repair ticket)
    rid = list(tb_m4.board.messages.keys())[0]
    ok, err = tb_m4._record_check(rid, "REC-002")
    assert not ok
    assert "utility repair ticket" in err

    # Toolbox with M4 disabled (mid-system behavior)
    tb_no_m4 = Toolbox(env, case, search, reader, world.docs, world.rules, m4_ticket_type=False)
    # With m4 disabled, it doesn't fail on ticket type
    ok_no, err_no = tb_no_m4._record_check(rid, "REC-002")
    assert "utility repair ticket" not in err_no


def test_m4_rejected_citations_tracking(world):
    """M4: Crew records and displays rejected citations to agents."""
    class DummyLLM:
        def json(self, system, user, *args, **kwargs):
            """Fake LLM answer: always 'continue'."""
            return {"thought": "test", "stance": "continue"}

    crew = Crew(DummyLLM(), m4_ticket_type=True)
    assert crew.rejected_citations == []

    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, search, reader, world.docs, world.rules)

    crew.rejected_citations.append({"citation": "REC-002", "reason": "utility repair ticket, not a false alarm"})
    t = crew._text(tb, "test")
    assert "REJECTED CITATIONS" in t
    assert "REC-002: utility repair ticket" in t


# ---------------------------------------------------------------------- M5 Tests
def test_m5_node_streets_corner_junctions(world):
    """M5: node_streets returns all intersecting streets at junction n344 (Old Mill Road and Patel Street)."""
    j = world.town.net.index["n344"]
    sts = world.town.places.node_streets(j)
    street_names = {s["name"] for s in sts}
    assert "Old Mill Road" in street_names
    assert "Patel Street" in street_names
    assert len(street_names) >= 2


def test_m5_rulebook_corner_permits(world):
    """M5: RuleBook recognizes resurfaced streets at corner junctions (e.g. Patel Street at n344)."""
    rb = world.rules
    when = world.open("dev-brown-000", "spread_k5")[1].clock()
    reqs = rb.requirements("dig", ["n344"], when)
    permit_reqs = [r for r in reqs if r["rule"] == "resurfaced_permit"]
    permit_streets = {r["street"] for r in permit_reqs}
    assert "Patel Street" in permit_streets


def test_m5_sop_context_lists_all_streets(world):
    """M5: Toolbox.sop_context lists all streets meeting at the junction."""
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, search, reader, world.docs, world.rules, m5_corner_permits=True)

    j = world.town.net.index["n344"]
    lines = tb.sop_context(j)
    site_line = next(l for l in lines if l.startswith("DIG SITE:"))
    assert "Old Mill Road" in site_line
    assert "Patel Street" in site_line


# ---------------------------------------------------------------------- M1 Tests
def test_m1_suggestive_flow_unlocks_section_parts(world):
    """M1: Suggestive/weak extra flow sections unlock testing of section parts in Crew.menu."""
    class DummyLLM:
        def json(self, system, user, *args, **kwargs):
            """Fake LLM answer: a thought only."""
            return {"thought": "test"}

    crew = Crew(DummyLLM(), m1_weak_leads=True)
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, search, reader, world.docs, world.rules, m1_weak_leads=True)

    # Simulate a step test trace entry with suggestive section 'S06' (which has parts S06.1, S06.2, etc.)
    tb.trace.append({
        "tool": "field_check",
        "result": {
            "kind": "step test",
            "clearly_lit": [],
            "suggestive": ["S06"]
        }
    })

    crew.menu(tb)
    # Check that parts of S06 are offered
    step_opts = [o for o in tb.options.values() if o.kind == "step" and "parts" in o.label]
    assert len(step_opts) > 0, "Expected step test of section parts to be offered for suggestive S06"
    assert any("S06" in s for o in step_opts for s in o.args["sections"])


def test_m1_detection_power_in_listening_result(world):
    """M1: Quiet listening check reports detection power."""
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, search, reader, world.docs, world.rules, m1_weak_leads=True)

    # Add a listening option at a node far from leak
    tb.options["L1"] = type("Opt", (), {"kind": "listen", "args": {"center": 0}, "label": "test listen"})()
    res = tb.field_check("L1")
    if not res.get("heard"):
        assert "detection_power" in res
        assert "strength" in res


# ---------------------------------------------------------------------- M2 Tests
def test_m2_turn_budget_initialization():
    """M2: Forensic turn budget is raised and turn pacing preserves field turns."""
    crew_m2 = Crew(None, m2_turn_budget=True)
    assert crew_m2.max_forensic >= 6
    assert crew_m2.max_llm >= 20

    crew_no_m2 = Crew(None, m2_turn_budget=False, max_forensic=4, max_llm=16)
    assert crew_no_m2.max_forensic == 4
    assert crew_no_m2.max_llm == 16


def test_exploration_sweep_offered_while_lead_exists(world):
    """The exploration step test (up to 6 largest untested sections) stays on the menu while a lead exists (reverted 13 Nov: banning it
    lost leak finds). Expected: one exploration option with 1..6 sections, none of them already in the hypothesis step test."""
    class DummyLLM:
        pass

    crew = Crew(DummyLLM(), m2_turn_budget=True)
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, search, reader, world.docs, world.rules)

    from aquamind.search import hypotheses as hyp
    crew.slate = hyp.build([{"id": "H1", "kind": "network_leak", "where": "S06", "because": []}], tb)

    crew.menu(tb)
    exploratory = [o for o in tb.options.values() if "exploration:" in o.label]
    assert len(exploratory) == 1
    secs = exploratory[0].args["sections"]
    assert 1 <= len(secs) <= 6
    hyp_steps = [o for o in tb.options.values() if "(the hypotheses)" in o.label]
    assert not hyp_steps or not set(secs) & set(hyp_steps[0].args["sections"])


# ---------------------------------------------------------------------- M3 Tests
def test_m3_confirmatory_check_offered_before_last_dig(world):
    """M3: When 1 check is left and P(success) < 0.80, confirmatory listening check at candidate spot is offered."""
    class DummyLLM:
        pass

    crew = Crew(DummyLLM(), m3_confirm_dig=True)
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, search, reader, world.docs, world.rules)

    # Set checks_left to 1
    env.checks_used = env.max_checks - 1
    assert env.checks_left() == 1

    crew.menu(tb)
    j_cand, _ = tb.board.best_dig()
    # Confirm that listening at j_cand is an available option
    cand_options = [o for o in tb.options.values() if o.kind == "listen" and o.args.get("center") == j_cand]
    assert len(cand_options) > 0, "Expected confirmatory listen at candidate dig spot"


def test_m3_auditor_objection_prevents_dry_dig(world):
    """M3: Auditor objection to weak dig (P < 0.80) with 0 checks left is respected and hands over."""
    class AuditorObjectLLM:
        def json(self, system, user, *args, **kwargs):
            """Fake Auditor answer: always objects to the dig."""
            return {"thought": "Not enough evidence; risk of dry hole", "verdict": "object", "objections": ["P is too low"]}

    crew = Crew(AuditorObjectLLM(), m3_confirm_dig=True)
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    search = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, search, reader, world.docs, world.rules)

    # Exhaust field checks
    env.checks_used = env.max_checks
    assert env.checks_left() == 0

    # Ensure P(success) is below 0.80
    j, p = tb.board.best_dig()
    assert p < 0.80

    verdict = crew.audit(tb, "dig proposed")
    assert verdict == "hold", "Auditor objection on weak dig with 0 checks must hold instead of approving"
    assert crew.held
    assert tb.done
    assert env.done
    assert env.outcome == "hand_over"


def test_m3_objection_cannot_veto_physical_support(world):
    """M3 fix (13 Nov): with 0 checks left an Auditor objection does NOT cancel a dig that has physical support (here the dig spot lies in
    a lit section); expected verdict 'approve' and the case stays open."""
    class AuditorObjectLLM:
        def json(self, system, user, *args, **kwargs):
            """Fake Auditor answer: always objects to the dig."""
            return {"thought": "hydrant flushing explains it", "verdict": "object", "objections": ["narrative doubt"]}

    crew = Crew(AuditorObjectLLM(), m3_confirm_dig=True)
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    tb = Toolbox(env, case, world.search(hypothesis_mode=True), RuleReader(world.town.places), world.docs, world.rules)
    env.checks_used = env.max_checks
    j, p = tb.board.best_dig()
    assert p < 0.80
    tb.board.area_support = lambda jj: ["step:S01"]          # a night step test lit the section that holds the dig spot
    verdict = crew.audit(tb, "dig proposed")
    assert verdict == "approve"
    assert not crew.held and not tb.done


# ---------------------------------------------------------------------- Methods & Env Switch Tests
def test_methods_registry_and_switches():
    """Verify methods registry includes agentix_crew, agentix_crew_mid, and individual mitigation ablations."""
    m = registry(reader="rules")
    assert "agentix_crew" in m
    assert "agentix_crew_mid" in m
    assert "agentix_crew_no_m1" in m
    assert "agentix_crew_no_m2" in m
    assert "agentix_crew_no_m3" in m
    assert "agentix_crew_no_m4" in m
    assert "agentix_crew_no_m5" in m


def test_env_mitigations_switch(monkeypatch):
    """Verify AGENTIX_MITIGATIONS environment variable toggles flags."""
    monkeypatch.setenv("AGENTIX_MITIGATIONS", "none")
    c_none = Crew(None)
    assert not c_none.m1_weak_leads
    assert not c_none.m2_turn_budget
    assert not c_none.m3_confirm_dig
    assert not c_none.m4_ticket_type
    assert not c_none.m5_corner_permits

    monkeypatch.setenv("AGENTIX_MITIGATIONS", "m4,m5")
    c_some = Crew(None)
    assert not c_some.m1_weak_leads
    assert not c_some.m2_turn_budget
    assert not c_some.m3_confirm_dig
    assert c_some.m4_ticket_type
    assert c_some.m5_corner_permits


# ---------------------------------------------------------------------- switch hygiene (final-evaluation fixes)
def test_explicit_switches_beat_environment(world, monkeypatch):
    """An explicit True/False always wins over AGENTIX_MITIGATIONS, so the mid baseline and the no_mX ablations cannot be flipped by the shell;
    switches left unset follow the variable and default to on."""
    from aquamind.agent.toolbox import MITIGATIONS, mitigation_flags
    monkeypatch.setenv("AGENTIX_MITIGATIONS", "m1,m2,m3,m4,m5")
    mid = Crew(None, **{k: False for k in MITIGATIONS})
    assert not any(getattr(mid, k) for k in MITIGATIONS)
    monkeypatch.setenv("AGENTIX_MITIGATIONS", "none")
    assert all(getattr(Crew(None, **{k: True for k in MITIGATIONS}), k) for k in MITIGATIONS)
    assert mitigation_flags(m1_weak_leads=None, m4_ticket_type=None) == {"m1_weak_leads": False, "m4_ticket_type": False}
    monkeypatch.setenv("AGENTIX_MITIGATIONS", "m1,m3")
    assert mitigation_flags(m1_weak_leads=None, m2_turn_budget=None, m3_confirm_dig=None) == \
        {"m1_weak_leads": True, "m2_turn_budget": False, "m3_confirm_dig": True}
    monkeypatch.delenv("AGENTIX_MITIGATIONS")
    assert all(getattr(Crew(None), k) for k in MITIGATIONS)


def test_m4_off_hides_rejected_citations(world):
    """With M4 off (mid system) the crew neither records nor shows rejected citations: that part belongs to M4."""
    case, env = world.open("dev-brown-000", "spread_k5")
    from aquamind.reader.rules import RuleReader
    tb = Toolbox(env, case, world.search(hypothesis_mode=True), RuleReader(world.town.places), world.docs, world.rules)
    crew = Crew(None, m4_ticket_type=False)
    crew.rejected_citations.append({"citation": "REC-002", "reason": "utility repair ticket, not a false alarm"})
    assert "REJECTED CITATIONS" not in crew._text(tb, "test")


def test_m4_repair_record_rejects_customer_side_tickets(world):
    """M4: a customer-side ticket (e.g. a private pump) cannot be cited as a failing utility main; without M4 it still can (mid behaviour)."""
    rec = next(c for c in world.docs.by_id.values() if c.provenance == "record" and "customer-side issue" in c.text)
    from aquamind.reader.rules import RuleReader
    reader = RuleReader(world.town.places)
    case, env = world.open("dev-brown-000", "spread_k5")
    tb = Toolbox(env, case, world.search(hypothesis_mode=True), reader, world.docs, world.rules, m4_ticket_type=True)
    r = tb.use_repair_record(rec.id)
    assert not r["ok"] and "customer-side" in r["error"]
    case, env = world.open("dev-brown-000", "spread_k5")
    tb_mid = Toolbox(env, case, world.search(hypothesis_mode=True), reader, world.docs, world.rules, m4_ticket_type=False)
    assert "customer-side" not in str(tb_mid.use_repair_record(rec.id).get("error", ""))
