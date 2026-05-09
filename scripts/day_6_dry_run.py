"""Day 6 Layer 4 full-pipeline dry-run.

Per docs/testing_strategy.md Layer 4: validates the full Day 7 → Day 8
orchestration end-to-end against real APIs at small scale before any
production-scale execution.

Scope:
  prompts: sum-015, sum-020 (both clear all caching thresholds; both have
           enough length to exercise compression meaningfully)
  models:  Sonnet 4.6, Haiku 4.5, GPT-5.4 (dated), GPT-5.4-mini (dated)
  levers:  baseline, caching, output_cap, batch, compression
  expected: ~32 calls, ~£0.50 cost

Uses a fresh tmp DB so skip-if-exists from prior step-1/2/3 smoke runs
doesn't suppress real API calls — Layer 4's job is to prove the full
pipeline works end-to-end, not to re-use prior data.

Five verifications run after the dry-run completes:
  1. Row counts (~44 results + 4 batch_jobs)
  2. Cost accounting (cost_so_far_gbp == sum(cost_usd)/GBP_USD_RATE ± £0.001)
  3. Engagement assertions (caching write/read engaged where above threshold;
     compression rows have compressed < original input tokens)
  4. No errors (no error rows, no NULL critical fields, no phase-log error events)
  5. Phase log readable (parses as JSONL; expected phase events present)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from runners._base import _read_cost_so_far_gbp, start_run  # noqa: E402
from runners.budget import GBP_USD_RATE  # noqa: E402
from runners.orchestrator import Orchestrator  # noqa: E402
from runners.schema import load_prompts  # noqa: E402

CAP_GBP = 5.0
TEST_MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
]
TARGET_PROMPT_IDS = ["sum-015", "sum-020"]

DRYRUN_DB_PATH = REPO_ROOT / "data" / "dryrun_results.db"
DRYRUN_PHASE_LOG = REPO_ROOT / "data" / "phase_log_dryrun.jsonl"
SCHEMA_SQL_PATH = REPO_ROOT / "data" / "schema.sql"


def _init_fresh_db(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    schema = SCHEMA_SQL_PATH.read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    _init_fresh_db(DRYRUN_DB_PATH)
    if DRYRUN_PHASE_LOG.exists():
        DRYRUN_PHASE_LOG.unlink()

    summary_prompts = load_prompts(REPO_ROOT / "prompts" / "summarisation.json")
    targets = [p for p in summary_prompts if p.prompt_id in TARGET_PROMPT_IDS]
    assert len(targets) == 2, f"expected 2 prompts, got {len(targets)}"

    rid = start_run(cost_cap_gbp=CAP_GBP, db_path=DRYRUN_DB_PATH)
    _print_section("Day 6 Layer 4 dry-run")
    print(f"db_path:       {DRYRUN_DB_PATH}")
    print(f"phase_log:     {DRYRUN_PHASE_LOG}")
    print(f"run_id:        {rid}")
    print(f"cap_gbp:       £{CAP_GBP}")
    print(f"prompts:       {[p.prompt_id for p in targets]}")
    print(f"models:        {TEST_MODELS}")

    orch = Orchestrator(
        run_id=rid, cap_gbp=CAP_GBP,
        db_path=DRYRUN_DB_PATH, phase_log_path=DRYRUN_PHASE_LOG,
    )

    # ---- Day 7 ----
    _print_section("Day 7 phases (sync)")
    t0 = time.perf_counter()
    day7 = orch.run_day_7(targets, TEST_MODELS)
    day7_wall = time.perf_counter() - t0
    print(f"Day 7 wall: {day7_wall:.1f}s")
    print(json.dumps({k: v for k, v in day7.items() if k != "batch_ids"}, indent=2))
    print(f"  batch_ids: {day7['batch_ids']}")

    # ---- Day 8 (waits for batches) ----
    _print_section("Day 8 phases (waits for batch completion)")
    t1 = time.perf_counter()
    # poll_interval_s=30 for the dry-run; quicker feedback than the production 60s
    day8 = orch.run_day_8()
    day8_wall = time.perf_counter() - t1
    print(f"Day 8 wall: {day8_wall:.1f}s  (includes batch poll wait)")
    print(json.dumps(day8, indent=2, default=str))

    # ---- Verifications ----
    _print_section("Five Verifications")
    issues: list[str] = []
    verify_row_counts(rid, issues)
    verify_cost_accounting(rid, issues)
    verify_engagement_assertions(rid, issues)
    verify_no_errors(rid, issues)
    verify_phase_log(rid, issues)

    _print_section("Result")
    print(f"Day 7 wall:      {day7_wall:.1f}s")
    print(f"Day 8 wall:      {day8_wall:.1f}s  (batch wait dominates)")
    print(f"Total wall:      {day7_wall + day8_wall:.1f}s")
    if issues:
        print(f"\nVERIFICATION FAILURES ({len(issues)}):")
        for i in issues:
            print(f"  - {i}")
        sys.exit(2)
    print("\nALL FIVE VERIFICATIONS PASSED ✓")


# ---------------------------------------------------------------------------
# Verifications
# ---------------------------------------------------------------------------

def verify_row_counts(rid: str, issues: list[str]) -> None:
    print("\n[V1] Row counts")
    with sqlite3.connect(DRYRUN_DB_PATH) as conn:
        n_results = conn.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ?", (rid,),
        ).fetchone()[0]
        n_batch_jobs = conn.execute(
            "SELECT COUNT(*) FROM batch_jobs WHERE run_id = ?", (rid,),
        ).fetchone()[0]
        per_lever = conn.execute(
            "SELECT optimisation_lever, COUNT(*) FROM results WHERE run_id = ? "
            "GROUP BY optimisation_lever ORDER BY optimisation_lever", (rid,),
        ).fetchall()
    print(f"    results:    {n_results} rows")
    for lever, n in per_lever:
        print(f"      {lever}: {n}")
    print(f"    batch_jobs: {n_batch_jobs} rows  (expected 4: one per model)")
    if n_batch_jobs != 4:
        issues.append(f"V1 batch_jobs row count: expected 4, got {n_batch_jobs}")
    if n_results < 30:
        issues.append(f"V1 results row count surprisingly low: {n_results}")


def verify_cost_accounting(rid: str, issues: list[str]) -> None:
    print("\n[V2] Cost accounting")
    with sqlite3.connect(DRYRUN_DB_PATH) as conn:
        sum_cost_usd = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM results WHERE run_id = ?", (rid,),
        ).fetchone()[0]
        cost_so_far_gbp = _read_cost_so_far_gbp(conn, rid)
    sum_cost_gbp_calc = sum_cost_usd / GBP_USD_RATE
    delta = abs(cost_so_far_gbp - sum_cost_gbp_calc)
    print(f"    cost_so_far_gbp:               £{cost_so_far_gbp:.6f}")
    print(f"    sum(cost_usd) / GBP_USD_RATE:  £{sum_cost_gbp_calc:.6f}")
    print(f"    delta:                         £{delta:.6f}  (target < £0.001)")
    if delta > 0.001:
        issues.append(f"V2 cost_so_far_gbp mismatches sum-of-cost_usd by £{delta:.6f}")


def verify_engagement_assertions(rid: str, issues: list[str]) -> None:
    print("\n[V3] Engagement assertions")
    with sqlite3.connect(DRYRUN_DB_PATH) as conn:
        # caching write rows: cache_creation_tokens > 0 (Anthropic) OR cached_tokens > 0 (OpenAI)
        # caching read rows:  cached_tokens > 0
        # compression rows:   compressed_input_tokens < original_input_tokens (in optimisation_config)
        cache_rows = conn.execute(
            "SELECT prompt_id, model, optimisation_config, cached_tokens, "
            "       cache_creation_tokens, provider "
            "FROM results WHERE run_id = ? AND optimisation_lever = 'caching'", (rid,),
        ).fetchall()
        comp_rows = conn.execute(
            "SELECT prompt_id, model, optimisation_config, input_tokens "
            "FROM results WHERE run_id = ? AND optimisation_lever = 'compression'", (rid,),
        ).fetchall()

    n_write_engaged = 0
    n_read_engaged = 0
    n_compression_engaged = 0
    n_compression_total = len(comp_rows)
    for prompt_id, model, cfg_json, cached, creation, provider in cache_rows:
        cfg = json.loads(cfg_json) if cfg_json else {}
        phase = cfg.get("cache_phase")
        if phase == "write":
            engaged = (provider == "anthropic" and creation > 0) or (provider == "openai")
            if engaged:
                n_write_engaged += 1
            else:
                issues.append(f"V3 caching WRITE not engaged: {prompt_id}/{model}")
        elif phase == "read":
            if cached > 0:
                n_read_engaged += 1
            else:
                issues.append(f"V3 caching READ not engaged: {prompt_id}/{model}")
    for prompt_id, model, cfg_json, billed_input in comp_rows:
        cfg = json.loads(cfg_json) if cfg_json else {}
        original = cfg.get("original_input_tokens")
        compressed = cfg.get("compressed_input_tokens")
        if original is None or compressed is None:
            issues.append(
                f"V3 compression {prompt_id}/{model}: missing original/compressed counts in optimisation_config"
            )
            continue
        if compressed >= original:
            issues.append(
                f"V3 compression {prompt_id}/{model}: no reduction "
                f"(compressed={compressed} >= original={original})"
            )
        else:
            n_compression_engaged += 1
    print(f"    caching writes engaged: {n_write_engaged} of {sum(1 for r in cache_rows if json.loads(r[2] or '{}').get('cache_phase') == 'write')}")
    print(f"    caching reads  engaged: {n_read_engaged} of {sum(1 for r in cache_rows if json.loads(r[2] or '{}').get('cache_phase') == 'read')}")
    print(f"    compression    engaged: {n_compression_engaged} of {n_compression_total}")


def verify_no_errors(rid: str, issues: list[str]) -> None:
    print("\n[V4] No errors")
    with sqlite3.connect(DRYRUN_DB_PATH) as conn:
        n_errored = conn.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ? AND error IS NOT NULL", (rid,),
        ).fetchone()[0]
        n_null = conn.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ? AND ("
            "result_id IS NULL OR run_id IS NULL OR model_version IS NULL OR "
            "response_text IS NULL OR cost_usd IS NULL)", (rid,),
        ).fetchone()[0]
    print(f"    errored rows:                {n_errored}")
    print(f"    NULL on critical fields:     {n_null}")
    if n_errored > 0:
        issues.append(f"V4 found {n_errored} errored rows in results")
    if n_null > 0:
        issues.append(f"V4 found {n_null} rows with NULL on critical fields")


def verify_phase_log(rid: str, issues: list[str]) -> None:
    print("\n[V5] Phase log readable")
    if not DRYRUN_PHASE_LOG.exists():
        issues.append("V5 phase log file does not exist")
        return
    raw = DRYRUN_PHASE_LOG.read_text().splitlines()
    entries: list[dict] = []
    for line_no, line in enumerate(raw, 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            issues.append(f"V5 line {line_no} is not valid JSON: {e}")
            return
    n_total = len(entries)
    n_error_events = sum(1 for e in entries if e["event"] == "error")
    expected_phases = {"day_7", "batch_submit", "baseline", "caching", "output_cap",
                       "day_8", "batch_retrieve", "compression_decide", "compression_run"}
    seen_phases = {e["phase"] for e in entries}
    missing = expected_phases - seen_phases
    print(f"    total events:     {n_total}")
    print(f"    error events:     {n_error_events}")
    print(f"    phases observed:  {sorted(seen_phases)}")
    if missing:
        issues.append(f"V5 missing phases in log: {sorted(missing)}")
    if n_error_events > 0:
        issues.append(f"V5 found {n_error_events} error events in phase log")


if __name__ == "__main__":
    main()
