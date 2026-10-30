"""Evaluation logging: one folder per (method, layout), one file per kind of information.

  logs/evaluation/<category>/<method>/<layout>/                      e.g. agentix/crew/k5/
      v1_k5_agentix_crew.txt            all case cards, one block per case, numbered by benchmark position (#063/210)
      v1_k5_agentix_crew_overview.txt   the benchmark overview table: ONE ROW per case
      v1_k5_agentix_crew_metrics.txt    summary metrics, then the summary BY CASE TYPE and the list of FAILED cases (+ _metrics.json)
      v1_k5_agentix_crew_trace.jsonl    the investigation trace: ONE LINE per event (kind "llm_turn" = an agent's LLM call, "tool_call" = a tool)
      v1_k5_agentix_crew.jsonl          one lean result row per case (no trace); the source a resumed run reloads

A restart of the same evaluation appends to the newest version; a fresh full run starts the next one (v2, v3, ...). Cases finished earlier, in any
results file, are merged in once (backfill). All text files are rewritten in order after every case, so they always read top to bottom.
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def short_layout(layout: str) -> str:
    """spread_k5 -> k5, random_k10_s2 -> rk10s2 (used in file names and card headers)."""
    layout = layout.strip()
    m = re.fullmatch(r"spread_k(\d+)", layout)
    if m:
        return f"k{m.group(1)}"
    m = re.fullmatch(r"random_k(\d+)_s(\d+)", layout)
    return f"rk{m.group(1)}s{m.group(2)}" if m else re.sub(r"\W+", "", layout)


def layout_label(layouts: str) -> str:
    """'spread_k5,spread_k3,spread_k10' -> 'k5_k3_k10': the planned sweep, in the file name from the start so one log follows the whole evaluation."""
    return "_".join(short_layout(x) for x in str(layouts).split(",") if x.strip())


def _determine_category_and_folder(method_name: str, layout: str = "spread_k5") -> tuple[str, str]:
    """Map method name and layout to (category, subfolder).
    Modularizes by category, then method, then layout (e.g., agentix/crew/k5, baselines/fixed_rules/k5).
    """
    m = method_name.lower()
    lay_tag = short_layout(layout)
    if m in ("agentix_crew", "agentix"):
        return "agentix", f"agentix/crew/{lay_tag}"
    if m == "agentix_crew_no_override":
        return "agentix", f"agentix/crew_no_override/{lay_tag}"
    if "ablation" in m or m in ("agentix_code", "agentix_crew_no_reports", "agentix_crew_no_temp_sensor",
                                "agentix_crew_no_rag", "agentix_crew_no_gate", "agentix_crew_one_shot"):
        clean_name = m.replace("agentix_crew_", "").replace("ablation_", "")
        return "ablations", f"ablations/{clean_name}/{lay_tag}"
    # Baselines
    clean_base = m.replace("classical_direct_inversion", "bruteforce").replace("classical_bruteforce", "bruteforce")
    return "baselines", f"baselines/{clean_base}/{lay_tag}"


def _extract_key_action(r: dict[str, Any]) -> str:
    """A concise 1-line summary of the physical and operational actions of a case (kept in the lean row as `key_action`)."""
    if r.get("key_action"):
        return r["key_action"]
    trace = r.get("trace") or []
    actions = []
    for step in trace:
        tool = step.get("tool")
        res = step.get("result") or {}
        if tool == "field_check":
            kind = res.get("kind", "check")
            clearly_lit = res.get("clearly_lit")
            if clearly_lit:
                actions.append(f"{kind} lit {','.join(clearly_lit)}")
            elif res.get("heard"):
                actions.append("acoustic listening confirmed noise")
        elif tool == "propose_dig":
            spot = res.get("spot", "")
            found = res.get("leak_found")
            res_str = "LEAK CONFIRMED" if found else "dry hole"
            actions.append(f"dig at {spot} ({res_str})")
        elif tool == "close":
            verdict = step.get("args", {}).get("verdict", "closed")
            actions.append(f"closed as {verdict}")

    if actions:
        return " -> ".join(actions[:4])
    outcome = str(r.get("outcome", "-"))
    err = r.get("guess_error_m")
    if err is not None:
        return f"Investigation decided {outcome} (distance error: {err:.1f}m)"
    return f"Completed investigation with outcome '{outcome}'"


def _has_leak(r: dict[str, Any]) -> bool:
    """Truth flag written by the harness (a trap or false-alarm case may have no leak); older rows fall back to the leak flow."""
    return bool(r["has_leak"]) if "has_leak" in r else r.get("q_lps", 0.0) > 0


def _fail_reason(r: dict[str, Any]) -> str | None:
    """None = success. Leak case: a dig found it. No-leak case: closed without a dig."""
    if _has_leak(r):
        if r.get("found") or r.get("outcome") == "found":
            return None
        return "never dug (handed over / closed)" if r.get("n_digs", 0) == 0 else "dug but missed the leak"
    return None if r.get("n_digs", 0) == 0 else "crew sent, no leak (false dig)"


def _pred_spot(r: dict[str, Any]) -> str | None:
    """The spot the system would dig: its final guess, else the spot of its dig proposal in the trace."""
    if r.get("pred_spot"):
        return r["pred_spot"]
    if r.get("final_guess"):
        return str(r["final_guess"])
    for step in reversed(r.get("trace") or []):
        if isinstance(step, dict) and step.get("tool") == "propose_dig" and (step.get("result") or {}).get("spot"):
            return step["result"]["spot"]
    return None


def _format_case_card(case_num: int, r: dict[str, Any], total: int | None = None, layout: str | None = None) -> str:
    """Format limited, highly important information for a single case in human-readable plain text."""
    case_id = str(r.get("case", "-"))
    suite = str(r.get("suite", "-"))
    is_chronic = (suite == "chronic")
    suite_display = f"{suite} (silent underground leak)" if is_chronic else suite
    trigger = str(r.get("trigger", "-"))

    q_lps = r.get("q_lps", 0.0)
    has_leak = r.get("has_leak", q_lps > 0)
    surfaced = r.get("surfaced", False)
    if has_leak:
        leak_type = "Surface Burst" if surfaced else "Underground / Silent Leak"
        leak_str = f"YES ({q_lps:.2f} L/s, {leak_type})"
    else:
        leak_str = "NO (Clean False Alarm / Decoy)"

    pred_str = str(_pred_spot(r) or "-")
    err = r.get("guess_error_m")
    if err is not None:
        within_tag = "[SUCCESS <= 100m]" if err <= 100.0 else "[MISS > 100m]"
        err_str = f"{err:.1f} m  {within_tag}"
    else:
        err_str = "N/A (No dig proposed)"

    outcome = str(r.get("outcome", "-")).upper()
    digs = r.get("n_digs", 0)
    wrong_digs = r.get("wrong_digs", 0)
    checks = r.get("checks_used", 0)
    budget = r.get("budget", 5)

    llm_calls = r.get("llm_calls", 0)
    tokens = r.get("tokens", 0)
    wall_s = r.get("wall_s", 0.0)

    unsafe = r.get("unsafe", 0)
    violations = r.get("violations", 0)
    audit_str = "SAFE (0 violations)" if (unsafe == 0 and violations == 0) else f"WARNING: {unsafe} unsafe, {violations} violations"

    key_action = _extract_key_action(r)

    lines = [
        f"--- [{short_layout(layout) + ' | ' if layout else ''}CASE #{case_num:03d}{f'/{total}' if total else ''}] {case_id} [{suite_display}] ---",
        f"  Trigger:         {trigger}",
        f"  Actual Leak:     {leak_str}",
        f"  Predicted Spot:  {pred_str}",
        f"  Distance Error:  {err_str}",
        f"  Outcome:         {outcome} (Digs: {digs}, Wrong: {wrong_digs}, Field Checks: {checks}/{budget})",
        f"  AI Resources:    LLM Calls: {llm_calls} | Tokens: {tokens:,} | Wall Time: {wall_s:.1f}s",
        f"  Safety Audit:    {audit_str}",
        f"  Key Action:      {key_action}",
        "-" * 85,
    ]
    return "\n".join(lines)


class EvaluationLogger:
    """See the module docstring. `session` identifies one evaluation (bench, methods, reader, regime, budget): a process with the same session resumes."""

    def __init__(self, method_name: str, layout: str = "spread_k5",
                 base_log_dir: str = "logs", run_title: str | None = None,
                 version: str | None = None, session: str | None = None,
                 case_order: list[str] | None = None, resume: bool = True):
        self.session = session
        self.case_order = {n: i for i, n in enumerate(case_order or [], 1)}
        self.total = len(self.case_order) or None
        self.base_dir = Path(base_log_dir)
        self.method_name = method_name
        self.layout = layout
        self.layout_order = [x.strip() for x in str(layout).split(",") if x.strip()]
        self.label = layout_label(layout)

        if "," not in method_name:
            self.category, subpath = _determine_category_and_folder(method_name, layout=self.label)
            self.target_dir = self.base_dir / subpath
            tag = method_name.replace("classical_direct_inversion", "bruteforce").replace("classical_bruteforce", "bruteforce")
        else:
            self.category = "comparative"
            self.target_dir = self.base_dir / f"comparative/{self.label}"
            tag = "+".join(m.replace("ablation_", "").replace("agentix_crew_", "") for m in method_name.split(","))[:70]
        self.file_prefix = f"{self.label}_{tag}"                 # <layout>_<method>

        self.target_dir.mkdir(parents=True, exist_ok=True)
        found = self._find_resumable() if (version is None and resume and session) else None
        self.version = version or found or self._get_next_version()
        self.resumed = found is not None

        stem = f"{self.version}_{self.file_prefix}"              # v1_k5_agentix_crew
        self.txt_file = self.target_dir / f"{stem}.txt"                    # case cards
        self.overview_file = self.target_dir / f"{stem}_overview.txt"      # one row per case
        self.metrics_file = self.target_dir / f"{stem}_metrics.txt"        # summary metrics
        self.summary_file = self.target_dir / f"{stem}_metrics.json"
        self.trace_file = self.target_dir / f"{stem}_trace.jsonl"          # one line per LLM turn / tool call
        self.jsonl_file = self.target_dir / f"{stem}.jsonl"                # lean result rows (resume source)

        self.run_title = run_title or f"{self.method_name} on {self.layout}"
        self.start_time = datetime.datetime.now()
        self.first_started = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self._keys: set[tuple] = set()

        if self.resumed:
            self._load_previous()
        if self.resumed or not self.txt_file.exists():
            self._write_reports()

    # ------------------------------------------------------------------ files and versions
    @staticmethod
    def _key(r: dict[str, Any]) -> tuple:
        return (r.get("method"), r.get("case"), r.get("layout"), r.get("budget", 5), r.get("regime", ""))

    def _versions(self) -> dict[int, Path]:
        out = {}
        for f in self.target_dir.glob(f"v*_{self.file_prefix}.txt"):
            m = re.fullmatch(rf"v(\d+)_{re.escape(self.file_prefix)}\.txt", f.name)
            if m:
                out[int(m.group(1))] = f
        return out

    def _get_next_version(self) -> str:
        pat = re.compile(rf"v(\d+)_{re.escape(self.file_prefix)}(?:_overview|_metrics|_trace)?\.(?:txt|jsonl|json)")
        nums = [int(m.group(1)) for f in self.target_dir.iterdir() if (m := pat.fullmatch(f.name))]
        return f"v{max(nums, default=0) + 1}"

    def _find_resumable(self) -> str | None:
        hits = []
        for n, f in self._versions().items():
            try:
                if f"Session: {self.session}" in f.read_text(encoding="utf-8")[:1500]:
                    hits.append(n)
            except OSError:
                continue
        return f"v{max(hits)}" if hits else None

    def _load_previous(self) -> None:
        txt = self.txt_file.read_text(encoding="utf-8") if self.txt_file.exists() else ""
        m = re.search(r"First started: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", txt)
        if m:
            self.first_started = m.group(1)
        if self.jsonl_file.exists():
            for line in self.jsonl_file.read_text(encoding="utf-8").splitlines():
                try:
                    self._remember(json.loads(line))
                except ValueError:
                    continue                 # a half-written last line is dropped, never fatal

    def _remember(self, r: dict[str, Any]) -> bool:
        if self._key(r) in self._keys:
            return False
        self._keys.add(self._key(r))
        self.rows.append(r)
        lay = r.get("layout")
        if lay and lay not in self.layout_order:
            self.layout_order.append(lay)
        return True

    # ------------------------------------------------------------------ ordering and numbers
    def _card_no(self, r: dict[str, Any], fallback: int) -> int:
        return self.case_order.get(str(r.get("case")), fallback)

    def _sorted_rows(self) -> list[dict[str, Any]]:
        li = {l: i for i, l in enumerate(self.layout_order)}
        return sorted(self.rows, key=lambda r: (li.get(r.get("layout"), 99), self._card_no(r, 10**6)))

    # ------------------------------------------------------------------ the five files
    def _header(self, what: str) -> str:
        lines = ["=" * 100, f"AQUAMIND EVALUATION: {self.method_name.upper()} ({self.version}) | {what}",
                 f"Category: {self.category.upper()} | Layout: {self.label} | First started: {self.first_started}"
                 + (f" | Last resumed: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}" if self.resumed else ""),
                 f"Title: {self.run_title}"]
        if self.session:
            lines.append(f"Session: {self.session}")
        if self.total:
            lines.append(f"Case numbers: position in the benchmark of {self.total} cases (the same on every resume)")
        return "\n".join(lines + ["=" * 100, ""]) + "\n"

    def _write_reports(self) -> dict[str, Any]:
        rows = self._sorted_rows()
        with open(self.txt_file, "w", encoding="utf-8") as f:                     # 1. case cards
            f.write(self._header("CASES") + "CASE-BY-CASE DETAILED LOG:\n" + "-" * 100 + "\n")
            for i, r in enumerate(rows, 1):
                f.write(_format_case_card(self._card_no(r, i), r, self.total, r.get("layout")) + "\n")
        summary = self._compute_summary(self.metadata, (datetime.datetime.now() - self.start_time).total_seconds())
        with open(self.overview_file, "w", encoding="utf-8") as f:               # 2. one row per case
            f.write(self._header("BENCHMARK CASE-BY-CASE OVERVIEW TABLE") + "\n".join(self._overview(rows)) + "\n* 'chronic*' = silent underground hidden leak, no surface water.\n")
        with open(self.metrics_file, "w", encoding="utf-8") as f:                # 3. metrics
            f.write(self._header("SUMMARY METRICS") + self._metrics_text(summary) + "\n".join(self._suite_summary(rows)) + "\n")
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary

    @staticmethod
    def _trace_events(row: dict[str, Any], no: int) -> list[dict[str, Any]]:
        """The trace of one case as flat events: an agent's LLM turn has a `role`, a tool call has a `tool`."""
        out = []
        for i, t in enumerate(row.get("trace") or [], 1):
            if isinstance(t, dict):
                out.append({"case_no": no, "case": row.get("case"), "layout": row.get("layout"), "step": i,
                            "kind": "llm_turn" if "role" in t else "tool_call", **t})
        return out

    def _add(self, row: dict[str, Any]) -> bool:
        if self._key(row) in self._keys:
            return False
        lean = {k: v for k, v in row.items() if k != "trace"}                  # the trace goes to its own file
        lean["key_action"], lean["pred_spot"] = _extract_key_action(row), _pred_spot(row)
        lean["trace_events"] = len(row.get("trace") or [])
        self._remember(lean)
        with open(self.jsonl_file, "a", encoding="utf-8") as f:                # 4. lean result row
            f.write(json.dumps(lean, default=str) + "\n")
        ev = self._trace_events(row, self._card_no(row, 0))
        if ev:
            with open(self.trace_file, "a", encoding="utf-8") as f:            # 5. trace, one line per event
                f.writelines(json.dumps(e, default=str) + "\n" for e in ev)
        return True

    # ------------------------------------------------------------------ logging
    def log_case(self, row: dict[str, Any]) -> None:
        if self._add(row):
            self._write_reports()

    def backfill(self, rows: list[dict[str, Any]]) -> int:
        """Add finished cases this log does not hold yet (earlier processes, other results files). Returns how many were added."""
        n = sum(self._add(r) for r in rows)
        if n or not self.txt_file.exists():
            self._write_reports()
        return n

    def finalize(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.metadata = metadata or {}
        summary = self._write_reports()
        self._update_master_index(summary)
        return summary

    # ------------------------------------------------------------------ summary and table
    @staticmethod
    def _metrics(rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
        total_cases = len(rows)
        leak_cases = [r for r in rows if _has_leak(r)]
        no_leak_cases = [r for r in rows if not _has_leak(r)]
        errors = [r["guess_error_m"] for r in leak_cases if r.get("guess_error_m") is not None]
        median_error = float(np.median(errors)) if errors else None
        within_100m = sum(1 for e in errors if e <= 100.0)
        acc_100m = (within_100m / len(leak_cases)) * 100.0 if leak_cases else 0.0
        found_count = sum(1 for r in leak_cases if r.get("outcome") == "found")
        found_rate = (found_count / len(leak_cases)) * 100.0 if leak_cases else 0.0
        crews_on_no_leak = sum(1 for r in no_leak_cases if r.get("n_digs", 0) > 0)
        total_wall_s = sum(r.get("wall_s", 0.0) for r in rows)
        return {
            "total_cases": total_cases, "leak_cases": len(leak_cases), "no_leak_cases": len(no_leak_cases),
            "found_within_100m_count": within_100m, "found_within_100m_pct": round(acc_100m, 1),
            "median_distance_error_m": round(median_error, 1) if median_error is not None else None,
            "found_rate_pct": round(found_rate, 1), "false_digs_on_no_leak": f"{crews_on_no_leak}/{len(no_leak_cases)}",
            "unsafe_actions_count": sum(r.get("unsafe", 0) for r in rows),
            "total_llm_api_calls": sum(r.get("llm_calls", 0) for r in rows), "total_tokens": sum(r.get("tokens", 0) for r in rows),
            "avg_wall_time_per_case_s": round(total_wall_s / max(total_cases, 1), 1),
        }

    def _compute_summary(self, metadata: dict[str, Any], duration_s: float) -> dict[str, Any]:
        by_layout = {l: self._metrics([r for r in self.rows if r.get("layout") == l], 0.0) for l in self.layout_order
                     if any(r.get("layout") == l for r in self.rows)}
        return {"version": self.version, "title": self.run_title, "category": self.category, "method": self.method_name, "layout": self.label,
                "timestamp": self.first_started, "duration_seconds": round(duration_s, 1), "metadata": metadata,
                "metrics": self._metrics(self.rows, duration_s), "by_layout": by_layout}

    def _overview(self, rows: list[dict[str, Any]]) -> list[str]:
        """One concise row per case."""
        lines = [f"{'#':<4} {'Case ID':<24} {'Suite':<15} {'Trigger':<14} {'Predicted':<10} {'Dist Err':<10} {'<=100m':<7} {'Outcome':<10} {'Digs':<5} {'Checks':<7} {'LLM':<5} {'Time':<7}",
                 "-" * 125]
        for idx, r in enumerate(rows, 1):
            suite = str(r.get("suite", ""))[:14]
            suite = "chronic*" if suite == "chronic" else suite
            pred = _pred_spot(r)
            mn = re.search(r"\b(n\d+)\b", str(pred)) if pred else None
            pred_short = mn.group(1) if mn else (str(pred)[:9] if pred else "-")
            err = r.get("guess_error_m")
            within = "YES" if (err is not None and err <= 100.0) else ("NO" if err is not None else "-")
            lines.append(f"{self._card_no(r, idx):<4} {str(r.get('case', ''))[:23]:<24} {suite:<15} {str(r.get('trigger', ''))[:13]:<14} {pred_short:<10} "
                         f"{(f'{err:.1f} m' if err is not None else '-'):<10} {within:<7} {str(r.get('outcome', '-'))[:9]:<10} {str(r.get('n_digs', 0)):<5} "
                         f"{str(r.get('checks_used', 0)):<7} {str(r.get('llm_calls', 0)):<5} {r.get('wall_s', 0.0):.0f}s")
        return lines

    def _suite_summary(self, rows: list[dict[str, Any]]) -> list[str]:
        """Below the metrics: results by case type (suite), then every failed case with its reason."""
        if not rows:
            return []
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(str(r.get("suite", "?")), []).append(r)
        w = 125
        out = ["", "=" * w, "SUMMARY BY CASE TYPE (suite).  Success = a dig found the leak (leak case) / the case was closed without a dig (no-leak case).", "-" * w,
               f"{'Suite':<16} {'Cases':<6} {'Success':<8} {'Failed':<7} {'Success %':<10} {'Leak never dug':<15} {'Dug, missed':<12} {'False dig':<10} {'Guess<=100m':<12} {'Median err':<10}", "-" * w]

        def line(name, rs):
            reasons = [_fail_reason(r) for r in rs]
            ok = sum(x is None for x in reasons)
            leak = [r for r in rs if _has_leak(r)]
            e = [r["guess_error_m"] for r in leak if r.get("guess_error_m") is not None]
            within = f"{sum(x <= 100 for x in e)}/{len(leak)}" if leak else "-"
            return (f"{name:<16} {len(rs):<6} {ok:<8} {len(rs) - ok:<7} {100 * ok / len(rs):<10.1f} {reasons.count('never dug (handed over / closed)'):<15} "
                    f"{reasons.count('dug but missed the leak'):<12} {reasons.count('crew sent, no leak (false dig)'):<10} {within:<12} {(f'{np.median(e):.0f} m' if e else '-'):<10}")
        for name in sorted(by):
            out.append(line("chronic*" if name == "chronic" else name, by[name]))
        out += ["-" * w, line("ALL CASES", rows), "=" * w]
        failed = [(self._card_no(r, 0), r, _fail_reason(r)) for r in rows if _fail_reason(r)]
        out += ["", f"FAILED CASES ({len(failed)} of {len(rows)}):", "-" * w, f"{'#':<4} {'Case ID':<24} {'Suite':<15} {'Reason':<36} {'Guess error':<12} {'Digs':<5} {'Checks':<6}", "-" * w]
        for no, r, why in sorted(failed, key=lambda x: x[0]):
            err = r.get("guess_error_m")
            out.append(f"{no:<4} {str(r.get('case', ''))[:23]:<24} {str(r.get('suite', ''))[:14]:<15} {why:<36} {(f'{err:.0f} m' if err is not None else '-'):<12} "
                       f"{r.get('n_digs', 0):<5} {r.get('checks_used', 0):<6}")
        return out + ["-" * w]

    def _metrics_text(self, summary: dict[str, Any]) -> str:
        lines = []
        for lay, m in summary["by_layout"].items():
            lines += [f"SUMMARY BENCHMARK METRICS ({lay}):", "-" * 75,
                      f"  Total Cases Logged:          {m['total_cases']} (leak {m['leak_cases']}, no leak {m['no_leak_cases']})",
                      f"  Accuracy (Found <= 100m):    {m['found_within_100m_count']}/{m['leak_cases']} ({m['found_within_100m_pct']}%)",
                      f"  Correct digs (leak found):   {m['found_rate_pct']}%",
                      f"  Median Distance Error:       {m['median_distance_error_m']} m" if m["median_distance_error_m"] is not None else "  Median Distance Error:       N/A",
                      f"  False Digs (on No-Leak):     {m['false_digs_on_no_leak']}",
                      f"  Unsafe / Unapproved Digs:    {m['unsafe_actions_count']}",
                      f"  Total LLM API Calls:         {m['total_llm_api_calls']}",
                      f"  Total Tokens Consumed:       {m['total_tokens']:,}",
                      f"  Average Time / Case:         {m['avg_wall_time_per_case_s']}s", ""]
        return "\n".join(lines or ["no cases logged yet", ""]) + "\n"

    def _update_master_index(self, summary: dict[str, Any]) -> None:
        index_file = self.base_dir / "master_index.json"
        index: list[dict[str, Any]] = []
        if index_file.exists():
            try:
                index = json.loads(index_file.read_text(encoding="utf-8"))
            except Exception:
                index = []
        rel = lambda p: str(p.relative_to(self.base_dir.parent))
        entry = {"version": summary["version"], "category": summary["category"], "method": summary["method"], "layout": summary["layout"],
                 "timestamp": summary["timestamp"], "txt_file": rel(self.txt_file), "overview_file": rel(self.overview_file),
                 "metrics_file": rel(self.metrics_file), "trace_file": rel(self.trace_file),
                 "total_cases": summary["metrics"]["total_cases"], "within_100m_pct": summary["metrics"]["found_within_100m_pct"],
                 "median_error_m": summary["metrics"]["median_distance_error_m"], "total_llm_calls": summary["metrics"]["total_llm_api_calls"]}
        index = [e for e in index if e.get("txt_file") != entry["txt_file"]]        # one entry per log, updated on every resume
        index.append(entry)
        index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")


class LoggerSet:
    """One EvaluationLogger per (method, layout), rows routed automatically: a run of several methods and/or layouts still gives each its own folder
    and its own five files (cases, overview, metrics, trace, rows), so tables and metrics are never mixed across methods or layouts."""

    def __init__(self, methods: list[str], layouts: list[str], base_log_dir: str, bench: str, session_tail: str,
                 run_title: str | None = None, case_order: list[str] | None = None, resume: bool = True):
        self.cfg = dict(base_log_dir=base_log_dir, run_title=run_title, case_order=case_order, resume=resume)
        self.bench, self.tail = bench, session_tail
        self.loggers: dict[tuple[str, str], EvaluationLogger] = {}
        for m in methods:
            for lay in layouts:
                self.get(m, lay)

    def get(self, method: str, layout: str) -> EvaluationLogger:
        key = (method, layout)
        if key not in self.loggers:
            self.loggers[key] = EvaluationLogger(method, layout, self.cfg["base_log_dir"], self.cfg["run_title"], session=f"{self.bench} | {method} | {self.tail}",
                                                 case_order=self.cfg["case_order"], resume=self.cfg["resume"])
        return self.loggers[key]

    def log_case(self, row: dict[str, Any]) -> None:
        self.get(row["method"], row["layout"]).log_case(row)

    def backfill(self, rows: list[dict[str, Any]]) -> int:
        by: dict[tuple[str, str], list] = {}
        for r in rows:
            by.setdefault((r["method"], r["layout"]), []).append(r)
        return sum(self.get(m, lay).backfill(rs) for (m, lay), rs in by.items())

    def finalize(self, metadata: dict[str, Any] | None = None) -> dict:
        return {k: lg.finalize(metadata) for k, lg in self.loggers.items()}

    def describe(self) -> list[str]:
        return [f"[{lg.version}] {m} / {lay}: {'RESUMING' if lg.resumed else 'new log'} ({len(lg.rows)} cases) -> {lg.target_dir}/{lg.txt_file.stem}*"
                for (m, lay), lg in self.loggers.items()]


def generate_txt_from_jsonl(jsonl_path: str | Path, base_log_dir: str = "logs/evaluation", bench: str | None = None) -> list[Path]:
    """Build the five log files for any results .jsonl: one log per layout in it. With `bench`, cases are numbered by benchmark position."""
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {p}")
    order = None
    if bench:
        from .splits import cases_for
        order = []
        for sp in ("dev", "val", "test"):
            try:
                order += cases_for(bench, sp)
            except FileNotFoundError:
                pass
    out = []
    for lay in dict.fromkeys(r.get("layout", "spread_k5") for r in rows):
        lg = EvaluationLogger(method_name=rows[0].get("method", p.stem.split("_")[0]), layout=lay, base_log_dir=base_log_dir, case_order=order)
        lg.backfill([r for r in rows if r.get("layout", "spread_k5") == lay])
        lg.finalize(metadata={"source_jsonl": str(p)})
        out.append(lg.txt_file)
    return out
