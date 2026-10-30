"""Tests for modular baselines, ablations, and versioned evaluation logging."""
import json
import os
import shutil
import tempfile
import pytest

from aquamind.baselines.registry import BASELINE_METADATA, get_baseline_registry, is_llm_baseline
from aquamind.ablations.registry import ABLATION_METADATA, get_ablation_registry, is_llm_ablation
from aquamind.methods import registry, get_method_info
from aquamind.eval.logger import EvaluationLogger

BENCH = "data/benchmark/agentix_master" if os.path.exists("data/benchmark/agentix_master/dev.json") else "data/benchmark/agentix_final"


def test_baseline_metadata_completeness():
    """Verify all proposal baselines exist with metadata."""
    expected = [
        "sensitivity_match", "report_hotspot", "react_agent",
        "plain_llm", "fixed_rules", "classical_direct_inversion",
        "classical_bruteforce", "lila_style", "crew_practice"
    ]
    for name in expected:
        assert name in BASELINE_METADATA, f"Missing metadata for baseline {name}"
        meta = BASELINE_METADATA[name]
        assert "category" in meta
        assert "needs_llm" in meta
        assert "proposal_section" in meta


def test_ablation_metadata_completeness():
    """Verify the 6 proposal ablations exist with metadata."""
    expected = [
        "ablation_no_reports", "ablation_no_temp_sensor", "ablation_no_rag",
        "ablation_no_gate", "ablation_one_shot", "ablation_no_ai"
    ]
    for name in expected:
        assert name in ABLATION_METADATA, f"Missing metadata for ablation {name}"
        meta = ABLATION_METADATA[name]
        assert "category" in meta
        assert "needs_llm" in meta
        assert "proposal_section" in meta


def test_non_llm_baselines_instantiation():
    """Verify non-LLM baselines instantiate without an LLM."""
    baselines = get_baseline_registry(llm_factory=None)
    assert "classical_direct_inversion" in baselines
    assert "fixed_rules" in baselines
    assert "crew_practice" in baselines
    assert not is_llm_baseline("classical_direct_inversion")
    assert not is_llm_baseline("fixed_rules")
    assert is_llm_baseline("react_agent")


def test_non_llm_ablations_instantiation():
    """Verify non-LLM ablations instantiate without an LLM."""
    ablations = get_ablation_registry(llm_factory=None)
    assert "ablation_no_ai" in ablations
    assert "agentix_code" in ablations
    assert not is_llm_ablation("ablation_no_ai")
    assert is_llm_ablation("ablation_no_reports")


def test_individual_ablation_modules():
    """Verify each ablation can be imported directly from its dedicated file."""
    from aquamind.ablations.no_reports import build_no_reports
    from aquamind.ablations.no_temp_sensor import build_no_temp_sensor
    from aquamind.ablations.no_rag import build_no_rag
    from aquamind.ablations.no_gate import build_no_gate
    from aquamind.ablations.one_shot import build_one_shot, OneShotBrain
    from aquamind.ablations.no_ai import build_no_ai

    # Non-LLM builder
    m_no_ai = build_no_ai()
    assert "ablation_no_ai" in m_no_ai
    assert "agentix_code" in m_no_ai

    # Dummy LLM factory for checking instantiation structure
    dummy_llm = lambda: None
    m_reports = build_no_reports(dummy_llm)
    assert "ablation_no_reports" in m_reports
    assert "agentix_crew_no_reports" in m_reports

    m_temp = build_no_temp_sensor(dummy_llm)
    assert "ablation_no_temp_sensor" in m_temp

    m_rag = build_no_rag(dummy_llm)
    assert "ablation_no_rag" in m_rag

    m_gate = build_no_gate(dummy_llm)
    assert "ablation_no_gate" in m_gate

    m_oneshot = build_one_shot(dummy_llm)
    assert "ablation_one_shot" in m_oneshot
    assert OneShotBrain is not None



def test_multi_api_key_loading(monkeypatch, tmp_path):
    """Verify LLM client loads multiple API keys and sets up per-key rate pacing. Self-contained: fake keys, so it passes on any PC with or without a .env."""
    from aquamind.llm.client import LLM
    monkeypatch.chdir(tmp_path)                                 # no .env here: the keys below are the only ones
    for k in ("NVIDIA_API_KEY", "NVIDIA_API_KEY_1", "NVIDIA_API_KEY_2", "NVIDIA_API_KEY_3"):
        monkeypatch.delenv(k, raising=False)
    for i in (1, 2, 3):
        monkeypatch.setenv(f"NVIDIA_API_KEY_{i}", f"fake-key-{i}")
    llm = LLM(cache_dir=str(tmp_path / "llm"))
    assert len(llm.keys) >= 3, f"Expected at least 3 API keys, found {len(llm.keys)}"
    assert len(llm._key_tags) == len(llm.keys)
    # Check that each key has its own distinct rate file
    rate_files = [llm._rate_file(i) for i in range(len(llm.keys))]
    assert len(set(rate_files)) == len(llm.keys), "Each key must have its own separate rate limit file"



def test_evaluation_logger_subfolders_and_versioning():
    """Verify EvaluationLogger writes to proper category subfolder and auto-increments version (v1, v2, ...)."""
    temp_dir = tempfile.mkdtemp(prefix="agentix_test_logs_")
    try:
        # Run 1: classical_direct_inversion should go to baselines/bruteforce/v1_bruteforce_k5.log
        logger1 = EvaluationLogger(method_name="classical_direct_inversion", layout="spread_k5", base_log_dir=temp_dir)
        assert logger1.version == "v1"
        assert "baselines/bruteforce" in str(logger1.target_dir)

        logger1.log_case({
            "method": "classical_direct_inversion", "case": "dev-chronic-001", "suite": "chronic",
            "trigger": "scada_alarm", "final_guess": "n540", "layout": "spread_k5", "k": 5,
            "outcome": "found", "guess_error_m": 42.5, "q_lps": 3.5, "n_digs": 1,
            "checks_used": 2, "unsafe": 0, "llm_calls": 0, "tokens": 0, "wall_s": 0.4
        })
        summary1 = logger1.finalize()
        assert logger1.overview_file.exists() and logger1.metrics_file.exists()
        assert logger1.jsonl_file.exists()
        assert logger1.summary_file.exists()
        assert summary1["version"] == "v1"
        assert summary1["metrics"]["total_cases"] == 1
        assert summary1["metrics"]["found_within_100m_pct"] == 100.0

        # Run 2: same method and layout without a session: a new version v2, v1 untouched
        logger2 = EvaluationLogger(method_name="classical_direct_inversion", layout="spread_k5", base_log_dir=temp_dir)
        assert logger2.version == "v2"
        logger2.log_case({
            "method": "classical_direct_inversion", "case": "dev-false_alarm-002", "suite": "false_alarm",
            "trigger": "report", "final_guess": None, "layout": "spread_k5", "k": 5,
            "outcome": "hand_over", "guess_error_m": None, "q_lps": 0.0, "n_digs": 0,
            "checks_used": 1, "unsafe": 0, "llm_calls": 0, "tokens": 0, "wall_s": 0.2
        })
        summary2 = logger2.finalize()
        assert summary2["version"] == "v2"

        # Check master_index.json has both entries
        index_file = os.path.join(temp_dir, "master_index.json")
        assert os.path.exists(index_file)
        index = json.load(open(index_file))
        assert len(index) == 2
        assert index[0]["version"] == "v1"
        assert index[1]["version"] == "v2"

        assert logger1.txt_file.exists()
        log_text = logger1.overview_file.read_text()
        assert "CASE-BY-CASE" in log_text
        assert "dev-chronic-001" in log_text
        assert "chronic*" in log_text
        assert "YES" in log_text
        txt_text = logger1.txt_file.read_text()
        assert "dev-chronic-001" in txt_text
        assert "Actual Leak:" in txt_text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not os.path.exists(f"{BENCH}/dev.json"), reason="benchmark not assembled")
def test_fast_non_llm_execution():
    """Integration test: execute classical_direct_inversion on a real case."""
    from aquamind.eval.runner import World, run_case
    w = World(BENCH)
    methods = registry(None, "rules")
    method = methods["classical_direct_inversion"]

    cases = json.load(open(f"{BENCH}/dev.json"))["cases"]
    test_case = cases[0]

    res = run_case(w, method, test_case, "spread_k5")
    assert res is not None
    assert "outcome" in res
    assert "method" in res
    assert res["method"] in ("classical_direct_inversion", "classical_bruteforce")
    assert "guess_error_m" in res or res.get("outcome") == "no_leak"


ONE_LOOK_ONLY = {"sensitivity_match", "report_hotspot", "lila_style"}      # RQ1 baselines run by scripts/run_one_look.py (rq1.one_look), not as full cases


def test_registry_has_every_documented_method():
    """Regression: a refactor once left agentix_crew and the crew-role ablations out of registry() (inside an `except: pass`)."""
    m = registry(None, "rules")
    must = {"agentix_crew", "agentix_crew_no_override", "agentix_code", "react_agent", "plain_llm", "fixed_rules", "crew_practice",
            "classical_bruteforce"}
    for name, meta in ABLATION_METADATA.items():
        must |= {name, meta["alias"]}
    must |= {n for n in BASELINE_METADATA if n not in ONE_LOOK_ONLY}
    missing = sorted(must - set(m))
    assert not missing, f"not in registry(): {missing}"
    for name, meth in m.items():
        assert hasattr(meth, "run_case") and meth.name == name, name


def test_no_reports_ablation_hides_complaint_text_from_agents():
    m = registry(None, "rules")["ablation_no_reports"]
    assert m.reader_kind == "none" and m.search_kw.get("use_messages") is False


def test_no_temp_sensor_ablation_really_removes_a_tool():
    m = registry(None, "rules")
    assert m["ablation_no_temp_sensor"].tb_kw.get("allow_step") is False
    assert "allow_step" not in m["agentix_crew"].tb_kw


def test_one_shot_can_dig_and_is_not_the_gated_loop():
    m = registry(None, "rules")["ablation_one_shot"]
    assert m.tb_kw.get("use_gate") is False
    assert type(m.brain_factory(None)).__name__ == "OneShotBrain"
