"""The evaluation metrics (agentix/eval/metrics.py) on small hand-checkable rows."""
import math

from agentix.eval import metrics as M


def row(case, found, wrong=0, digs=None, err=None, **kw):
    r = {"method": "m", "case": case, "layout": "L", "has_leak": True, "found": found, "wrong_digs": wrong, "n_digs": (digs if digs is not None else wrong + int(found)),
         "guess_error_m": err, "checks_used": 2, "hours_open": 10.0, "water_lost_m3": 100.0, "unsafe": 0, "violations": 0, "tool_calls": 4, "tool_errors": 1}
    r.update(kw)
    return r


REF = {("a", "L"): {"complaint_oracle_err_m": 40.0, "complaint_hotspot_err_m": 60.0},       # a complaint leads to the leak
       ("b", "L"): {"complaint_oracle_err_m": 400.0, "complaint_hotspot_err_m": 500.0},      # no complaint does
       ("c", "L"): {"complaint_oracle_err_m": None, "complaint_hotspot_err_m": None}}        # nobody complained


def test_first_dig_precision_and_dry_holes():
    rows = [row("a", True, 0, err=20.0), row("b", True, 1, err=70.0), row("c", False, 2, err=300.0)]
    s = M.summarize(rows, REF)
    assert s["found"][0] == 2 / 3
    assert s["first_dig"][0] == 1 / 3                       # only case a found it with no dry hole
    assert abs(s["dig_precision"] - 2 / 5) < 1e-9           # 2 leaks found by 5 digs
    assert abs(s["dry_per_leak_case"] - 1.0) < 1e-9


def test_error_curve_counts_a_missing_guess_as_a_miss():
    rows = [row("a", True, err=20.0), row("b", True, err=70.0), row("c", False, err=None)]
    s = M.summarize(rows, REF)
    assert s["within"][25][0] == 1 / 3 and s["within"][100][0] == 2 / 3 and s["within"][200][0] == 2 / 3
    assert s["err_median"] == 45.0


def test_strata_separate_cases_a_complaint_can_and_cannot_lead_to():
    rows = [row("a", True, err=20.0), row("b", True, err=70.0), row("c", False, err=300.0)]
    st = M.summarize(rows, REF)["strata"]
    assert st["A"]["n"] == 1 and st["A"]["found"][0] == 1.0          # a complaint is within 100 m
    assert st["B"]["n"] == 1 and st["B"]["found"][0] == 1.0          # complaints placed but all far: b was still found
    assert st["C"]["n"] == 1 and st["C"]["found"][0] == 0.0          # nobody complained: c was not


def test_win_vs_hotspot_uses_distance_to_the_leak():
    rows = [row("a", True, err=20.0),      # 20 < 60 win
            row("b", True, err=900.0),     # 900 > 500 loss
            row("c", False, err=1.0)]      # no complaint at all: not comparable
    w, l, p = M.summarize(rows, REF)["win_vs_hotspot"]
    assert (w, l) == (1, 1) and p == 1.0


def test_cost_uses_the_configured_prices():
    r = row("a", True, 1, err=10.0)                         # 2 digs, 2 checks, 100 m3
    assert abs(M.crew_cost(r) - (2 * 1.5 + 2 * 0.25)) < 1e-9
    assert abs(M.cost_cu(r) - (2 * 1.5 + 2 * 0.25 + 100 * 0.002)) < 1e-9


def test_giving_up_early_cannot_look_cheap():
    """A method that never digs and closes at once has a low cost per case but an undefined / huge cost per leak FOUND, and no 'hours until found'."""
    quit_ = [row("a", False, 0, digs=0, err=None, checks_used=0, hours_open=1.0, water_lost_m3=5.0) for _ in range(3)]
    s = M.summarize(quit_, REF)
    assert s["crew_cost"][0] == 0.0 and math.isnan(s["crew_cost_per_found"]) and math.isnan(s["hours_to_find"])
    worker = [row("a", True, 0, err=5.0, checks_used=4, hours_open=30.0, water_lost_m3=60.0), row("b", False, 2, err=400.0, checks_used=5, hours_open=70.0, water_lost_m3=300.0)]
    w = M.summarize(worker, REF)
    assert w["hours_to_find"] == 30.0 and w["water_to_find_m3"] == 60.0       # only the found leak counts for time and water
    assert abs(w["crew_cost_per_found"] - (1.5 * 3 + 0.25 * 9)) < 1e-9        # all spending (3 digs, 9 checks) divided by the 1 leak found


def test_calibration_from_the_trace():
    tr = [{"tool": "propose_dig", "result": {"ok": True, "p_success": 0.9, "leak_found": True}},
          {"tool": "propose_dig", "result": {"ok": True, "p_success": 0.9, "leak_found": False}},
          {"tool": "field_check", "result": {"ok": True}}]
    s = M.summarize([row("a", True, err=1.0, trace=tr)], REF)["calibration"]
    assert s["n_digs_logged"] == 2 and abs(s["brier"] - ((0.1 ** 2) + (0.9 ** 2)) / 2) < 1e-9
    assert s["hi_conf_n"] == 2 and s["hi_conf_success"] == 0.5


def test_a_method_without_a_trace_or_a_guess_does_not_break_the_summary():
    r = row("a", False, 0, err=None)
    r.pop("trace", None)
    s = M.summarize([r, {**r, "case": "n", "has_leak": False, "n_digs": 1}], REF)
    assert s["n_noleak"] == 1 and s["crew_sent_no_leak"] == (1, 1)
    assert s["calibration"]["n_digs_logged"] == 0 and math.isnan(s["err_median"])


def test_reference_strategies():
    rows = [row("a", True), row("b", True), row("c", True)]
    ref = {**REF, ("a", "L"): {**REF[("a", "L")], "chance_hit": 0.02}, ("b", "L"): {**REF[("b", "L")], "chance_hit": 0.01}, ("c", "L"): {"chance_hit": 0.03}}
    r = M.reference_rows(rows, ref)
    hot, oracle, rnd = r.values()
    assert hot == [True, False, False] and oracle == [True, False, False] and abs(sum(rnd) / 3 - 0.02) < 1e-9
