"""Logs: one folder per (method, layout), five files; resume = append, fresh = new version, exact case numbers, one table row per case."""
import json

from aquamind.eval.logger import EvaluationLogger

ORDER = [f"dev-x-{i:03d}" for i in range(3)] + [f"test-x-{i:03d}" for i in range(4)]        # 7 cases: dev 1-3, test 4-7
S = "bench | agentix_crew | reader=local"


def row(case, layout="spread_k5", found=True, trace=None):
    r = {"method": "agentix_crew", "case": case, "layout": layout, "suite": "chronic", "trigger": "citizen_message", "q_lps": 2.0, "has_leak": True, "found": found,
         "outcome": "found" if found else "hand_over", "guess_error_m": 20.0 if found else 300.0, "n_digs": 1, "wrong_digs": 0, "checks_used": 2, "llm_calls": 3, "tokens": 10,
         "wall_s": 5.0, "budget": 5, "regime": "", "unsafe": 0, "violations": 0, "final_guess": "n12"}
    if trace is not None:
        r["trace"] = trace
    return r


def new(tmp_path, layout="spread_k5", **kw):
    return EvaluationLogger("agentix_crew", layout, str(tmp_path), session=S, case_order=ORDER, **kw)


def test_folder_per_layout_and_five_files(tmp_path):
    lg = new(tmp_path)
    lg.log_case(row("test-x-000"))
    d = tmp_path / "agentix" / "crew" / "k5"
    assert lg.target_dir == d
    assert sorted(p.name for p in d.iterdir()) == ["v1_k5_agentix_crew.jsonl", "v1_k5_agentix_crew.txt", "v1_k5_agentix_crew_metrics.json",
                                                   "v1_k5_agentix_crew_metrics.txt", "v1_k5_agentix_crew_overview.txt"]       # no trace yet: no trace file
    assert new(tmp_path, "spread_k3", resume=False).target_dir == tmp_path / "agentix" / "crew" / "k3"          # another layout, another folder


def test_each_file_holds_one_kind_of_information(tmp_path):
    lg = new(tmp_path)
    lg.log_case(row("test-x-000", trace=[{"tool": "field_check", "args": {"option": "F1"}, "result": {"ok": True}, "p_success": 0.4},
                                         {"turn": 0, "role": "forensic", "thought": "hypothesis H1"}]))
    cards, table, metrics = lg.txt_file.read_text(), lg.overview_file.read_text(), lg.metrics_file.read_text()
    assert "--- [k5 | CASE #004/7] test-x-000" in cards and "BENCHMARK" not in cards and "SUMMARY BENCHMARK METRICS" not in cards
    assert "test-x-000" in table and "SUMMARY BENCHMARK METRICS" not in table and "Actual Leak" not in table
    assert "SUMMARY BENCHMARK METRICS (spread_k5)" in metrics and "test-x-000" not in metrics
    ev = [json.loads(l) for l in lg.trace_file.read_text().splitlines()]
    assert [e["kind"] for e in ev] == ["tool_call", "llm_turn"] and ev[0]["case_no"] == 4 and ev[1]["role"] == "forensic" and ev[1]["step"] == 2
    lean = json.loads(lg.jsonl_file.read_text().splitlines()[0])
    assert "trace" not in lean and lean["trace_events"] == 2 and lean["pred_spot"] == "n12"          # the trace lives only in its own file


def test_one_table_row_per_case_numbered_by_benchmark_position(tmp_path):
    lg = new(tmp_path)
    for c in ("test-x-003", "dev-x-000", "test-x-000"):                                    # finished out of order
        lg.log_case(row(c))
    rows = [l for l in lg.overview_file.read_text().splitlines() if l[:3].strip().isdigit()]
    assert [l.split()[0] for l in rows] == ["1", "4", "7"] and len(rows) == 3


def test_resume_appends_to_the_same_files_and_never_duplicates(tmp_path):
    a = new(tmp_path)
    a.log_case(row("test-x-000"))
    a.log_case(row("dev-x-001"))
    b = new(tmp_path)                                                                  # the process died and was started again
    assert b.version == "v1" and b.resumed and len(b.rows) == 2
    assert b.backfill([row("test-x-000"), row("test-x-001")]) == 1                    # only the missed case is added
    b.log_case(row("test-x-001"))                                                      # a repeated case is ignored
    assert [r["case"] for r in b.rows].count("test-x-001") == 1 and len(b.jsonl_file.read_text().splitlines()) == 3
    assert len(list(tmp_path.glob("**/v*_k5_agentix_crew.txt"))) == 1


def test_a_resumed_card_keeps_its_key_action_although_the_trace_is_gone(tmp_path):
    a = new(tmp_path)
    a.log_case(row("test-x-000", trace=[{"tool": "propose_dig", "args": {}, "result": {"ok": True, "spot": "n7", "leak_found": True}}]))
    b = new(tmp_path)
    assert "dig at n7 (LEAK CONFIRMED)" in b.txt_file.read_text()


def test_a_fresh_run_starts_the_next_version(tmp_path):
    a = new(tmp_path)
    a.log_case(row("test-x-000"))
    a.finalize()
    b = new(tmp_path, resume=False)
    assert b.version == "v2" and not b.resumed and b.rows == []
    assert new(tmp_path).version == "v2"                                               # a later resume continues the NEWEST version
    assert len(list(tmp_path.glob("**/v1_k5_agentix_crew.txt"))) == 1                 # v1 untouched


def test_metrics_and_one_index_entry_per_log(tmp_path):
    lg = new(tmp_path)
    lg.log_case(row("test-x-000"))
    lg.log_case(row("test-x-001", found=False))
    s = lg.finalize()
    assert s["metrics"]["total_cases"] == 2 and s["metrics"]["found_rate_pct"] == 50.0
    lg2 = new(tmp_path)
    lg2.log_case(row("test-x-002"))
    lg2.finalize()
    idx = json.loads((tmp_path / "master_index.json").read_text())
    assert len(idx) == 1 and idx[0]["total_cases"] == 3 and idx[0]["trace_file"].endswith("_trace.jsonl")


def test_a_different_evaluation_does_not_mix_into_the_log(tmp_path):
    new(tmp_path).log_case(row("test-x-000"))
    other = EvaluationLogger("agentix_crew", "spread_k5", str(tmp_path), session="other bench", case_order=ORDER)
    assert other.version == "v2" and other.rows == []


def test_summary_by_case_type_and_failed_cases_are_below_the_metrics(tmp_path):
    lg = new(tmp_path)
    lg.log_case(row("test-x-000"))                                                     # success
    lg.log_case({**row("test-x-001", found=False), "n_digs": 0})                       # leak never dug
    lg.log_case({**row("test-x-002", found=False), "n_digs": 2, "wrong_digs": 2})      # dug, missed
    lg.log_case({**row("test-x-003"), "suite": "trap", "has_leak": False, "found": False, "outcome": "hand_over", "n_digs": 0, "guess_error_m": None})   # trap, no leak, closed: success
    lg.log_case({**row("dev-x-000"), "suite": "false_alarm", "has_leak": False, "found": False, "outcome": "hand_over", "n_digs": 1, "guess_error_m": None})  # false dig
    assert "SUMMARY BY CASE TYPE" not in lg.overview_file.read_text()                 # the table file holds only the table
    t = lg.metrics_file.read_text()
    assert t.index("SUMMARY BENCHMARK METRICS") < t.index("SUMMARY BY CASE TYPE") < t.index("FAILED CASES (3 of 5)")
    assert "ALL CASES" in t and "never dug (handed over / closed)" in t and "dug but missed the leak" in t and "crew sent, no leak (false dig)" in t
    m = lg._compute_summary({}, 0.0)["metrics"]
    assert m["leak_cases"] == 3 and m["no_leak_cases"] == 2                            # the trap case WITHOUT a leak is not a leak case
    assert m["false_digs_on_no_leak"] == "1/2"


def test_several_methods_and_layouts_in_one_run_never_mix(tmp_path):
    from aquamind.eval.logger import LoggerSet
    ls = LoggerSet(["fixed_rules", "react_agent"], ["spread_k5", "spread_k3"], str(tmp_path), "bench", "reader=local | text=realistic | regime=- | checks=5", case_order=ORDER)
    for m in ("fixed_rules", "react_agent"):
        for lay in ("spread_k5", "spread_k3"):
            ls.log_case({**row("test-x-000", lay), "method": m})
    ls.log_case({**row("test-x-001", "spread_k5", found=False), "method": "react_agent"})
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.glob("*/*/*") if p.is_dir()) == [
        "baselines/fixed_rules/k3", "baselines/fixed_rules/k5", "baselines/react_agent/k3", "baselines/react_agent/k5"]
    assert len(ls.get("react_agent", "spread_k5").rows) == 2 and len(ls.get("fixed_rules", "spread_k5").rows) == 1
    m = ls.get("react_agent", "spread_k5")._compute_summary({}, 0.0)["metrics"]
    assert m["total_cases"] == 2 and m["found_rate_pct"] == 50.0                       # only react_agent / k5 cases are in its metrics
    assert ls.backfill([{**row("test-x-002", "spread_k3"), "method": "fixed_rules"}]) == 1
    assert len(ls.get("fixed_rules", "spread_k3").rows) == 2


def test_two_crew_variants_have_separate_folders_and_version_numbers(tmp_path):
    a = EvaluationLogger("agentix_crew", "spread_k5", str(tmp_path), session="s1", case_order=ORDER)
    b = EvaluationLogger("agentix_crew_no_override", "spread_k5", str(tmp_path), session="s2", case_order=ORDER)
    assert a.target_dir != b.target_dir and a.version == "v1" and b.version == "v1"
    a.log_case(row("test-x-000"))
    assert EvaluationLogger("agentix_crew", "spread_k5", str(tmp_path), session="s3", case_order=ORDER).version == "v2"
