"""Day 9 backfill — fires fresh sync calls for the 33 (model, lever, prompt_id)
combinations missing from the production run-d1a9c980.

Pre-fix (Day 9 morning), these 33 rows were silently blocked by Day 6 dry-run
rows colliding on `(prompt_id, model, lever, config_hash, run_attempt)` because
the skip-if-exists query was run-id-agnostic. The skip-if-exists fix landed
earlier today (see methodology doc § "Skip-if-exists semantics"), so re-firing
under the same run_id now correctly inserts fresh rows under run-d1a9c980.

Usage:
  poetry run python -m scripts.day_9_backfill
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

from runners import _base, lever_compression, lever_output_cap, run_openai  # noqa: E402
from runners._base import DB_PATH, _read_cost_so_far_gbp  # noqa: E402
from runners.orchestrator import _adapter_for_model, load_all_prompts  # noqa: E402

RUN_ID = "run-d1a9c980-aee3-42df-9152-c5c6ea604532"
CAP_GBP = 300.0


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def main() -> None:
    prompts_by_id = {p.prompt_id: p for p in load_all_prompts()}

    # Recompute the missing set from the live DB (don't hardcode — proves fix
    # holds at execution time).
    from runners.orchestrator import TEST_MODELS
    LEVERS = ["baseline", "batch", "compression", "output_cap"]
    expected = {(m, lever, pid) for m in TEST_MODELS for lever in LEVERS for pid in prompts_by_id}
    with sqlite3.connect(DB_PATH) as conn:
        actual = set(conn.execute(
            "SELECT model, optimisation_lever, prompt_id FROM results WHERE run_id = ?",
            (RUN_ID,),
        ).fetchall())
    missing = sorted(expected - actual)

    _print_section(f"Day 9 backfill — {len(missing)} missing rows under {RUN_ID}")
    cost_before = 0.0
    with sqlite3.connect(DB_PATH) as conn:
        cost_before = _read_cost_so_far_gbp(conn, RUN_ID)
    print(f"cost_so_far_gbp before: £{cost_before:.6f}", flush=True)
    print(f"missing breakdown:", flush=True)
    from collections import Counter
    by_lever = Counter(lev for _, lev, _ in missing)
    for lev, n in sorted(by_lever.items()):
        print(f"  {lev:<12} {n}", flush=True)

    if not missing:
        print("Nothing to backfill.", flush=True)
        return

    _print_section("Firing calls")
    n_inserted = 0
    n_skipped = 0
    n_error = 0
    t0 = time.perf_counter()

    for i, (model, lever, pid) in enumerate(missing, start=1):
        prompt = prompts_by_id[pid]
        adapter = _adapter_for_model(model)
        print(f"[{i:>2}/{len(missing)}] {model:<32} {lever:<12} {pid}", end=" ", flush=True)
        try:
            if lever == "baseline":
                optimisation_config = run_openai.annotate_optimisation_config_for_reasoning_effort(None, model)
                row = _base.run_one(
                    adapter, prompt, model, lever="baseline",
                    optimisation_config=optimisation_config,
                    run_id=RUN_ID, cap_gbp=CAP_GBP,
                    completed=i - 1, planned=len(missing),
                )
            elif lever == "output_cap":
                row = lever_output_cap.run_output_cap_for_prompt(
                    adapter, prompt, model,
                    run_id=RUN_ID, cap_gbp=CAP_GBP,
                    completed=i - 1, planned=len(missing),
                )
            elif lever == "compression":
                row = lever_compression.run_compression_for_prompt(
                    adapter, prompt, model,
                    run_id=RUN_ID, cap_gbp=CAP_GBP,
                    completed=i - 1, planned=len(missing),
                )
            else:
                raise ValueError(f"unsupported backfill lever {lever!r}")
        except Exception as e:
            n_error += 1
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
            continue

        if row.get("skipped"):
            n_skipped += 1
            print("skipped (already present)", flush=True)
        elif row.get("error"):
            n_error += 1
            print(f"errored: {row.get('error')[:80]}", flush=True)
        else:
            n_inserted += 1
            print(f"ok  (out_tok={row.get('output_tokens')}, ${row.get('cost_usd', 0):.5f})", flush=True)

    wall = time.perf_counter() - t0

    _print_section("Backfill complete")
    print(f"wall:        {wall:.1f}s ({wall/60:.1f} min)", flush=True)
    print(f"inserted:    {n_inserted}", flush=True)
    print(f"skipped:     {n_skipped}", flush=True)
    print(f"errored:     {n_error}", flush=True)

    with sqlite3.connect(DB_PATH) as conn:
        cost_after = _read_cost_so_far_gbp(conn, RUN_ID)
        n_total = conn.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ?", (RUN_ID,),
        ).fetchone()[0]
    print(f"cost_so_far_gbp after:  £{cost_after:.6f}  (delta £{cost_after - cost_before:+.6f})", flush=True)
    print(f"results.run_id rows:    {n_total}  (target 1632)", flush=True)


if __name__ == "__main__":
    main()
