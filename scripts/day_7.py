"""Day 7 production sweep.

Runs the orchestrator's Day 7 phases against the full 102-prompt set on all
4 test models. Either mints a fresh run_id (default) or resumes an existing
run_id passed via --resume.

Phase order (per orchestrator.run_day_7):
  1. submit_batches  — submits 4 batches (one per model) FIRST so the 24h
                       provider-side SLA clock starts as early as possible.
                       Resume-safe: lever_batch.submit_batch's skip-if-exists
                       on (run_id, provider, model, lever) means already-
                       submitted batches are skipped on relaunch.
  2. run_baseline    — sync calls, all 102 prompts × 4 models
  3. run_caching     — 3-call sequence on sum-015..020 × 4 models
  4. run_output_cap  — sync calls with max_completion_tokens=200, all 102 × 4

Phase log streams to data/phase_log.jsonl (the orchestrator's default).

This script does NOT trigger Day 8. After Day 7 completes, review state and
invoke Day 8 separately.

Usage:
  poetry run python -m scripts.day_7                         # fresh run_id
  poetry run python -m scripts.day_7 --resume <run_id>       # resume existing
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from runners._base import _read_cost_so_far_gbp, start_run  # noqa: E402
from runners.orchestrator import (  # noqa: E402
    PHASE_LOG_PATH,
    TEST_MODELS,
    Orchestrator,
    load_all_prompts,
)

CAP_GBP = 300.0


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 7 production sweep")
    p.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume an existing run_id (skip-if-exists handles already-done work). "
             "Default: mint a fresh production run_id with cost_cap_gbp=300.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    prompts = load_all_prompts()
    assert len(prompts) == 102, f"expected 102 prompts, got {len(prompts)}"

    if args.resume:
        run_id = args.resume
        _print_section(f"Day 7 production sweep — RESUMING {run_id}")
    else:
        run_id = start_run(cost_cap_gbp=CAP_GBP)
        _print_section(f"Day 7 production sweep — FRESH run_id {run_id}")

    print(f"run_id:         {run_id}", flush=True)
    print(f"cap_gbp:        £{CAP_GBP}", flush=True)
    print(f"prompts:        {len(prompts)} ({sorted({p.task_category for p in prompts})})", flush=True)
    print(f"models:         {TEST_MODELS}", flush=True)
    print(f"phase_log:      {PHASE_LOG_PATH}", flush=True)

    from runners._base import DB_PATH
    with sqlite3.connect(DB_PATH) as conn:
        cost_before = _read_cost_so_far_gbp(conn, run_id)
    print(f"cost_so_far_gbp before: £{cost_before:.6f}", flush=True)

    orch = Orchestrator(run_id=run_id, cap_gbp=CAP_GBP)

    _print_section("Day 7 phases starting (batch_submit FIRST)")
    t0 = time.perf_counter()
    summary = orch.run_day_7(prompts, TEST_MODELS)
    wall_s = time.perf_counter() - t0

    _print_section("Day 7 complete")
    print(f"wall:                   {wall_s:.1f}s ({wall_s/60:.1f} min)", flush=True)
    print(f"summary:", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "batch_ids"}, indent=2), flush=True)
    print(f"  batch_ids: {summary['batch_ids']}", flush=True)

    with sqlite3.connect(DB_PATH) as conn:
        cost_after = _read_cost_so_far_gbp(conn, run_id)
    print(f"cost_so_far_gbp after:  £{cost_after:.6f}", flush=True)
    print(f"  delta this Day 7:     £{cost_after - cost_before:.6f}", flush=True)

    print(f"\nDAY 7 DONE. Pausing for review before Day 8 kickoff.", flush=True)


if __name__ == "__main__":
    main()
