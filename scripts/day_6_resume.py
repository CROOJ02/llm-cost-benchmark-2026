"""Resume the in-flight dry-run Day 8 against the existing dryrun DB + run_id.

The first run crashed during OpenAI batch poll on a transient network error;
the orchestrator now catches those, but we lost the in-process state. This
script picks up from the existing dryrun DB:

  - Reuses the existing run_id (no start_run — that would create a new one)
  - Re-runs Day 8 (retrieve_batches is idempotent on terminal-state batches,
    so already-retrieved or timed_out batches are filtered out; only the
    in-flight subset is polled until each finishes or hits per_batch_timeout)
  - decide_compression_tier + run_compression follow as normal
  - Same five verifications as the original dry-run

Re-running this script is safe: skip-if-exists in the results table prevents
duplicate inserts; the in-flight batch_jobs filter prevents duplicate batch
retrievals.
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

from runners._base import _read_cost_so_far_gbp  # noqa: E402
from runners.budget import GBP_USD_RATE  # noqa: E402
from runners.orchestrator import Orchestrator  # noqa: E402

CAP_GBP = 5.0
DRYRUN_DB_PATH = REPO_ROOT / "data" / "dryrun_results.db"
DRYRUN_PHASE_LOG = REPO_ROOT / "data" / "phase_log_dryrun.jsonl"

# Re-import the verifier helpers from the original dry-run script.
sys.path.insert(0, str(REPO_ROOT))
from scripts.day_6_dry_run import (  # noqa: E402
    verify_cost_accounting,
    verify_engagement_assertions,
    verify_no_errors,
    verify_phase_log,
    verify_row_counts,
)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def main() -> None:
    if not DRYRUN_DB_PATH.exists():
        raise SystemExit(f"no dry-run DB at {DRYRUN_DB_PATH}; run scripts.day_6_dry_run first")

    with sqlite3.connect(DRYRUN_DB_PATH) as conn:
        rid = conn.execute("SELECT run_id FROM runs LIMIT 1").fetchone()[0]
        in_flight = conn.execute(
            "SELECT model, status FROM batch_jobs WHERE status IN ('submitted', 'in_progress')",
        ).fetchall()

    _print_section("Day 6 Layer 4 dry-run — RESUME")
    print(f"db_path:           {DRYRUN_DB_PATH}", flush=True)
    print(f"phase_log:         {DRYRUN_PHASE_LOG}", flush=True)
    print(f"run_id (resumed):  {rid}", flush=True)
    print(f"in-flight batches: {in_flight}", flush=True)

    orch = Orchestrator(
        run_id=rid, cap_gbp=CAP_GBP,
        db_path=DRYRUN_DB_PATH, phase_log_path=DRYRUN_PHASE_LOG,
    )

    _print_section("Day 8 phases (resumed)")
    t1 = time.perf_counter()
    day8 = orch.run_day_8()
    day8_wall = time.perf_counter() - t1
    print(f"Day 8 wall: {day8_wall:.1f}s  (includes batch poll wait)", flush=True)
    print(json.dumps(day8, indent=2, default=str), flush=True)

    _print_section("Five Verifications")
    issues: list[str] = []
    verify_row_counts(rid, issues)
    verify_cost_accounting(rid, issues)
    verify_engagement_assertions(rid, issues)
    verify_no_errors(rid, issues)
    verify_phase_log(rid, issues)

    _print_section("Result")
    if issues:
        print(f"\nVERIFICATION FAILURES ({len(issues)}):", flush=True)
        for i in issues:
            print(f"  - {i}", flush=True)
        sys.exit(2)
    print("\nALL FIVE VERIFICATIONS PASSED ✓", flush=True)


if __name__ == "__main__":
    main()
