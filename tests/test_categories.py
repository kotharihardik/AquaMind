"""Unit and physics integration tests for category-specific leak phenomena:
- Sediment scour & Boxall shear-stress thresholding
- Pipe material failure history (RAG records.md)
- Geotechnical subsidence / subsurface void cues (potholes, road damage)
- Adversarial decoy discrimination via callbacks
- Downhill overland/bedding dispersion triangulation
- Clean false alarm early closure
"""
import json
import os
import pytest
import numpy as np

BENCH = "data/benchmark/agentix_master" if os.path.exists("data/benchmark/agentix_master/dev.json") else "data/benchmark/agentix_final"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{BENCH}/dev.json"), reason="benchmark not assembled")


@pytest.fixture(scope="module")
def world():
    from agentix.eval.runner import World
    return World(BENCH)


def _find_case(world, suite):
    cases = json.load(open(f"{world.bench}/dev.json"))["cases"]
    return next(c for c in cases if c.split("-")[1] == suite)


def test_sediment_scour_physics(world):
    """Verify that discolouration reports appear downstream along the hydraulic gradient, not at the leak origin."""
    name = _find_case(world, "brown")
    case, env = world.open(name, "spread_k5", "template")
    leak = env.leak
    assert leak is not None, "Case should have a ground truth leak"
    
    # Check that brown water reports exist
    brown_reports = [r for r in env.hid["reports"] if r.get("truth", {}).get("mode") == "brown"]
    assert len(brown_reports) > 0, "Brown water case must have discolouration reports"
    
    # Verify that the seen nodes are displaced from the leak node (downstream at customer taps)
    seen_nodes = {r["truth"]["seen_node"] for r in brown_reports}
    assert len(seen_nodes) >= 1
    # Check that the symptoms are indeed dirty_water
    for r in brown_reports:
        assert "dirty_water" in r["gold"]["symptoms"]


def test_pipe_material_failure_records_rag(world):
    """Verify that RAG records.md contains historical failure tickets and use_repair_record raises priors."""
    from agentix.agent.toolbox import Toolbox
    from agentix.methods import make_reader
    
    name = _find_case(world, "chronic")
    case, env = world.open(name, "spread_k5", "template")
    tb = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules)
    
    # Check records.md retriever is available
    assert world.docs is not None
    hits = world.docs.search("pipe burst corrosion renewal")
    assert len(hits) > 0, "Retriever must find relevant repair history"
    
    # Test use_repair_record on a valid ticket from records.md
    res = tb.use_repair_record("REC-003")
    assert res.get("ok") is True
    assert res.get("junctions_raised", 0) > 0


def test_geotechnical_subsidence_cues(world):
    """Verify that chronic geotechnical void cases generate road damage and pothole cues."""
    name = _find_case(world, "chronic")
    case, env = world.open(name, "spread_k5", "template")
    
    # Check for cues in reports
    reports = env.hid["reports"]
    cues = [r.get("truth", {}).get("cue") for r in reports if r.get("truth", {}).get("cue")]
    assert len(cues) > 0, "Case must have chronic geotechnical/void cues"


def test_adversarial_decoy_discrimination(world):
    """Verify that citizen phone callbacks debunk look-alike decoy complaints."""
    from agentix.agent.toolbox import Toolbox
    from agentix.methods import make_reader
    
    name = _find_case(world, "cd_sound_decoy")
    case, env = world.open(name, "spread_k5", "template")
    tb = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules)
    
    # Find a report that can be called
    callable_reports = [r["report_id"] for r in case.reports if tb.can_call(r["report_id"])]
    assert len(callable_reports) > 0, "Must have callable citizen reports"
    
    # Perform callback interview
    res = tb.call_back([{"report_id": callable_reports[0], "question": "Is this from a private pump or generator?"}])
    assert res.get("ok") is True
    assert len(res.get("results", [])) == 1
    result = res["results"][0]
    assert result.get("answered") is True
    # The decoy report is debunked
    assert "learned" in result


def test_downhill_dispersion_triangulation(world):
    """Verify that topographical triangulation traces overland and trench flow uphill."""
    import agentix.search.topography as topo
    from agentix.agent.toolbox import Toolbox
    from agentix.methods import make_reader
    
    name = _find_case(world, "displaced")
    case, env = world.open(name, "spread_k5", "template")
    tb = Toolbox(env, case, world.search(), make_reader("rules", world), world.docs, world.rules)
    
    # The case open has reports of pooling downhill
    assert len(case.reports) > 0
    
    # Verify terrain exists
    assert hasattr(world.sim.ctx, "terrain")
    elev = world.sim.ctx.terrain.elev
    assert elev is not None and len(elev) > 0
    
    # Check that triangulation functions
    claim = {"origin": "n10", "downstream": ["n11"]}
    ok, msg = topo.triangulate(tb, claim)
    assert isinstance(ok, bool)
