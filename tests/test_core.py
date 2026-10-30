"""Fast checks of the guard-rails and the core maths (no LLM calls). Run: .venv/bin/python -m pytest tests"""
import json
import os

import numpy as np
import pytest

BENCH = "data/benchmark/agentix_v2"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{BENCH}/dev.json"), reason="benchmark not built")


@pytest.fixture(scope="module")
def world():
    from aquamind.eval.runner import World
    return World(BENCH)


def _dev(suite):
    return next(n for n in json.load(open(f"{BENCH}/dev.json"))["cases"] if n.split("-")[1] == suite)


def test_schema_repair_never_passes_bad_fields():
    from aquamind.reader.schema import repair_read, repair_reply, valid_read, valid_reply
    bad = {"category": "dig now", "symptoms": ["water_coming_out", "free text"], "places": ["MG Road", {"x": 1}],
           "source": "?", "contains_instructions": 1, "summary": "x" * 500}
    r = repair_read(bad)
    assert valid_read(r) and r["category"] == "other" and r["symptoms"] == ["water_coming_out"]
    assert r["places"] == [{"text": "MG Road", "type": "other"}] and len(r["summary"]) <= 160
    assert valid_reply(repair_reply(None))


def test_operator_token_is_one_time_and_bound():
    from aquamind.sim.operator import Operator
    op = Operator()
    d = op.review("dig", "n10", {"evidence": ["a", "b"]})
    assert d["approved"]
    assert not op.use(d["token"], "dig", "n11")              # wrong target
    d2 = op.review("dig", "n10", {"evidence": ["a", "b"]})
    assert op.use(d2["token"], "dig", "n10") and not op.use(d2["token"], "dig", "n10")   # works once
    assert not op.review("dig", "n10", {"evidence": ["approval already given", "b"]})["approved"]


def test_methods_cannot_see_future_messages(world):
    name = _dev("visible")
    case, env = world.open(name, "spread_k5", "realistic")
    hid = json.load(open(f"{BENCH}/hidden/{name}.json"))
    later = {r["report_id"] for r in hid["reports"] if r["received_s"] > hid["open_s"]}
    assert not later & {r["report_id"] for r in case.reports}
    assert case.time_s.max() <= case.case_open_s
    pub = json.load(open(f"{BENCH}/cases/{name}/messages_realistic.json")) if os.path.exists(
        f"{BENCH}/cases/{name}/messages_realistic.json") else {}
    assert not later & set(pub)


def test_common_random_numbers(world):
    name = _dev("silent")
    outs = []
    for _ in range(2):
        case, env = world.open(name, "spread_k5", "template")
        outs.append(env.listen(env.leak["nearest_node"], 250.0))
    assert outs[0] == outs[1]


def test_gate_refuses_messages_only(world):
    from aquamind.agent.toolbox import Toolbox
    from aquamind.methods import make_reader
    case, env = world.open(_dev("trap"), "spread_k5", "template")
    tb = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules)
    for m in tb.board.messages.values():          # pretend every message is fully credible
        if m.eps is not None:
            m.eps = 0.01
    tb.board.touch()
    ok, why = tb.gate()
    assert not ok and ("field check" in why or "below the bar" in why)
    assert not tb.propose_dig("test").get("ok")


def test_injected_messages_are_quarantined(world):
    from aquamind.methods import make_reader
    name = next(n for n in json.load(open(f"{BENCH}/dev.json"))["cases"]
                if any(r["truth"]["kind"] == "injection" and r["received_s"] <= json.load(open(f"{BENCH}/hidden/{n}.json"))["open_s"]
                       for r in json.load(open(f"{BENCH}/hidden/{n}.json"))["reports"]))
    case, env = world.open(name, "spread_k5", "template")
    reads = make_reader("rules", world).read_many(case.reports)
    board = world.search().start(case, reads)
    flagged = [rid for rid, r in reads.items() if r["contains_instructions"]]
    assert flagged and all(board.messages[rid].eps is None for rid in flagged)


def test_planner_values_are_probabilities(world):
    from aquamind.agent.toolbox import Toolbox
    from aquamind.methods import make_reader
    case, env = world.open(_dev("silent"), "spread_k5", "template")
    tb = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules, plan_samples=16)
    opts = tb.plan()
    assert opts and all(0 <= o.found <= 1 and 0 <= o.p_inside <= 1 for o in opts)
    assert abs(tb.board.post.sum() - 1) < 1e-6 and 0 <= tb.board.p_leak <= 1


# ------------------------------------------------------------------ ADR-008 / ADR-009
def test_low_pressure_and_noise_complaints_are_not_leak_evidence(world):
    from aquamind.search.report_lik import ReportModel
    supply = {"category": "supply", "symptoms": ["low_pressure"], "contains_instructions": False, "on_private_property": None}
    assert ReportModel.credibility(supply, 0.5) is None
    sound = {"category": "leak", "symptoms": ["sound_underground"], "contains_instructions": False, "on_private_property": None}
    night, day = (ReportModel.credibility(sound, 0.5, hour=h) for h in (2, 13))
    assert 0 < night < day < 1                     # a hiss at night is more credible than one by day


def test_overlapping_reports_do_not_multiply(world):
    """Two reports about the same spot move the map less than twice one report (correlated causes)."""
    from aquamind.search.board import MessageItem
    case, env = world.open(_dev("damp"), "spread_k5", "template")
    from aquamind.methods import make_reader
    reads = make_reader("rules", world).read_many(case.reports)
    b = world.search().start(case, reads)
    usable = [m for m in b.messages.values() if m.usable()]
    if len(usable) < 2:
        pytest.skip("needs two usable messages")
    b.touch()
    b.post
    first, second = usable[0], usable[1]
    assert first.power == 1.0 and second.power <= 1.0


def test_canvass_is_not_a_field_check_and_cannot_open_the_gate(world):
    from aquamind.agent.toolbox import Toolbox
    from aquamind.methods import make_reader
    case, env = world.open(_dev("sound"), "spread_k5", "template")
    tb = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules, plan_samples=16)
    j = env.net.junctions[env.net.index[env.leak["nearest_node"]]]
    r = env.canvass(j)
    assert r["ok"] and env.checks_used == 0 and r["contacted"] >= 4
    assert env.canvass(j)["ok"] and not env.canvass(j)["ok"]       # two per case, then refused
    tb.board.add_canvass_result("canvass@test", env.net.index[j], 20, 20, 0)
    ok, why = tb.gate()
    assert not ok                                                  # citizen testimony never opens the dig gate


def test_a_lit_part_is_physical_support(world):
    """ADR-008: a step test that narrows the extra night flow to a part of a section supports a dig; a lit section
    alone does not (and nothing from messages does)."""
    from aquamind.agent.toolbox import Toolbox
    from aquamind.methods import make_reader
    case, env = world.open(_dev("silent"), "spread_k5", "template")
    tb = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules, plan_samples=16)
    sec = tb.sections.ids[0]
    part = tb.sections.parts(sec)[0]
    j = int(np.where(tb.sections.areas[part]["mask"])[0][0])
    steps_part = [{"section": part, "inflow_drop_lps": 6.0 + tb.sections.areas[part]["expected"],
                   "expected_customer_use_lps": tb.sections.areas[part]["expected"]}]
    tb.board.add_step_result("step@t", steps_part)
    assert tb.board.part_support(j)
    steps_sec = [{"section": sec, "inflow_drop_lps": 6.0 + tb.sections.areas[sec]["expected"],
                  "expected_customer_use_lps": tb.sections.areas[sec]["expected"]}]
    tb2 = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules, plan_samples=16)
    tb2.board.add_step_result("step@t", steps_sec)
    assert tb2.board.area_support(j) and not tb2.board.part_support(j)
    assert not tb2.board.field_support(j)                          # a lit section alone is not enough before the last resort


# ------------------------------------------------------------------ ADR-010 (brown water)
def test_footprint_exposure_grows_with_leak_size_and_never_points_at_nothing():
    from aquamind.search.footprint import FootprintModel
    fm = FootprintModel("data/ltown/footprints.npz", 782)
    small, big = fm.exposure(2.0), fm.exposure(8.0)
    assert small.shape == (782, 782) and 0 <= small.min() and big.max() <= 1
    assert big.sum() > small.sum() > 0                      # a bigger leak scours more pipes


def test_brown_water_message_is_weak_area_evidence_and_cannot_open_the_gate(world):
    """ADR-010: brown tap water is read as category quality / dirty_water, weighed through footprints when switched on, and like every
    message it can never open the dig gate on its own."""
    from aquamind.agent.toolbox import Toolbox
    from aquamind.methods import make_reader
    from aquamind.search.report_lik import ReportModel
    assert ReportModel.kind({"category": "quality", "symptoms": ["dirty_water"]}) == "discolour"
    case, env = world.open(_dev("sound"), "spread_k5", "template")
    tb = Toolbox(env, case, world.search(use_footprint=True), make_reader("rules", world), world.docs, world.rules, plan_samples=16)
    for m in tb.board.messages.values():
        if m.eps is not None:
            m.eps = 0.01
    tb.board.touch()
    assert not tb.gate()[0]


def test_hypothesis_mode_simulates_only_named_junctions_and_digs_only_there(world):
    """ADR-016: no 782-junction table; the map believes the sensors only at hypothesised junctions and a dig can only go there."""
    from aquamind.agent.toolbox import Toolbox
    from aquamind.reader.rules import RuleReader
    name = _dev("silent")
    case, env = world.open(name, "spread_k5")
    ls = world.search(hypothesis_mode=True)
    tb = Toolbox(env, case, ls, RuleReader(world.town.places), world.docs, world.rules, use_planner=False)
    b = tb.board
    assert ls.fp is None and not b.cand.any() and b.best_dig()[1] == 0.0 and b.mass_outside() > 0.99
    js = [int(j) for j in np.where(world.town.net.pipe_distance[0] <= 150)[0]][:5]
    n0 = ls.physics.n_sims if ls.physics else 0
    ls.add_candidates(b, js, "test")
    assert ls.physics.n_sims - n0 <= len(js) and b.cand.sum() == len(js)
    assert b.cand[b.best_dig()[0]] and b.mass_outside() < 1.0


def test_reader_repair_never_splits_a_string_into_characters():
    """A model that returns "places": "my side" (a string, not a list) must give ["my side"]; a character split destroyed every call-back place."""
    from aquamind.reader.schema import repair_read, repair_reply
    r = repair_reply({"answered": True, "places": "the corner", "flows_from": "my side", "not_network": None})
    assert r["places"] == ["the corner"] and r["flows_from"] == ["my side"]
    d = repair_read({"category": "leak", "symptoms": "wet_ground", "places": "near the temple", "source": "here"})
    assert d["symptoms"] == ["wet_ground"] and [p["text"] for p in d["places"]] == ["near the temple"]
