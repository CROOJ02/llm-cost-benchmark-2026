"""Day 8 production sweep.

Runs the orchestrator's Day 8 phases against an existing Day 7 run_id:
  1. retrieve_batches — pull all 4 in-flight batches into the results table
                        (per-batch 30-min timeout; transient APIConnectionError
                        catch in the poll loop). Anthropic batch SLA is 24h
                        from submission; some batches may be tight on time.
  2. decide_compression_tier — read runs.cost_so_far_gbp, pick tier per the
                               §9 Day 8 ladder (proportional to cap).
  3. run_compression — runtime LLMLingua-2 on the chosen prompt subset
                       (`full` tier = all 102 prompts × 4 models = 408 calls).

run_id is REQUIRED via --resume — Day 8 must operate on the same run_id as
Day 7 to retrieve its batches and account into the same cost_so_far.

Phase log streams to data/phase_log.jsonl (the orchestrator's default).

Usage:
  poetry run python -m scripts.day_8 --resume <run_id>
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

from runners._base import _read_cost_so_far_gbp  # noqa: E402
from runners.orchestrator import (  # noqa: E402
    PHASE_LOG_PATH,
    Orchestrator,
)

CAP_GBP = 300.0


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 8 production sweep")
    p.add_argument(
        "--resume", required=True, metavar="RUN_ID",
        help="The Day 7 run_id to resume into. Required.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.resume

    _print_section(f"Day 8 production sweep — RESUMING {run_id}")
    print(f"run_id:         {run_id}", flush=True)
    print(f"cap_gbp:        £{CAP_GBP}", flush=True)
    print(f"phase_log:      {PHASE_LOG_PATH}", flush=True)

    from runners._base import DB_PATH
    with sqlite3.connect(DB_PATH) as conn:
        cost_before = _read_cost_so_far_gbp(conn, run_id)
        in_flight = conn.execute(
            "SELECT model, status, batch_id, submitted_at FROM batch_jobs "
            "WHERE run_id = ? ORDER BY submitted_at",
            (run_id,),
        ).fetchall()
    print(f"cost_so_far_gbp before: £{cost_before:.6f}", flush=True)
    print(f"batch_jobs at start:", flush=True)
    for model, status, bid, sub_at in in_flight:
        print(f"  {model:30s}  {status:12s}  {bid}  (submitted {sub_at})", flush=True)

    orch = Orchestrator(run_id=run_id, cap_gbp=CAP_GBP)

    _print_section("Day 8 phases starting (retrieve → tier → compression)")
    t0 = time.perf_counter()
    summary = orch.run_day_8()
    wall_s = time.perf_counter() - t0

    _print_section("Day 8 complete")
    print(f"wall:                   {wall_s:.1f}s ({wall_s/60:.1f} min)", flush=True)
    print(f"summary:", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)

    with sqlite3.connect(DB_PATH) as conn:
        cost_after = _read_cost_so_far_gbp(conn, run_id)
    print(f"cost_so_far_gbp after:  £{cost_after:.6f}", flush=True)
    print(f"  delta this Day 8:     £{cost_after - cost_before:.6f}", flush=True)

    print(f"\nDAY 8 DONE. Pausing for review.", flush=True)


if __name__ == "__main__":
    main()
