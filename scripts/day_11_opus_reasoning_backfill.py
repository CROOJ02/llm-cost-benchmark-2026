"""Day 11 Opus reasoning backfill — fills judge_a_reasoning on the 28 NEW
disagreement rows that emerged from the (Opus vs GPT-5.5) pairing.

Context: the Day 11 GPT-5.5 sweep recomputed judge_disagreement_flag from
(Opus, GPT-5.5) instead of (Opus, Mistral). 28 of the 80 new-disagreement
rows are in (prompt, lever) batches that weren't part of the original Day 11
Mistral-based reasoning re-fire, so judge_a_reasoning is NULL on those rows.

This script fires Opus only on the affected batches (21 unique batches) and
writes reasoning for every row whose judge_a_reasoning is currently NULL,
preserving any non-NULL existing reasoning via a strict WHERE clause guard.
Cost ~£0.50, wall ~3-5 min.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

from runners._base import DB_PATH  # noqa: E402
from runners.budget import estimate_cost_usd, usd_to_gbp  # noqa: E402
from runners.orchestrator import TEST_MODELS, load_all_prompts  # noqa: E402
from scoring.judge import (  # noqa: E402
    OPUS_MODEL, assemble_judge_call, _call_opus,
    _parse_judge_response,
)

RUN_ID = "run-d1a9c980-aee3-42df-9152-c5c6ea604532"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_judge_a_reasoning_if_null(
    conn: sqlite3.Connection, *, run_id: str, prompt_id: str, model: str,
    lever: str, reasoning: str, ts: str,
) -> int:
    """UPDATE one row's judge_a_reasoning if it's currently NULL.

    Returns rowcount (0 if no row matched OR existing reasoning preserved).

    Methodology lesson (Day 11): the original inline UPDATE in this script
    swapped two parameters in the binding tuple — `(model, pid)` instead of
    `(pid, model)` — so the WHERE clause was `prompt_id='<model name>' AND
    model='<prompt id>'`, which matched 0 rows. The script reported
    'updated=0/4' for every batch, looked like everything was already
    populated, and £0.78 of Opus calls landed in the void. Factoring the
    UPDATE into a keyword-only helper makes the parameter mapping explicit
    and unit-testable. See tests/test_runner_invariants.py for the
    regression test.
    """
    cur = conn.execute(
        """UPDATE results
              SET judge_a_reasoning = ?,
                  score_recomputed_at = ?
            WHERE run_id = ? AND prompt_id = ? AND model = ? AND optimisation_lever = ?
              AND judge_a_reasoning IS NULL""",
        (reasoning, ts, run_id, prompt_id, model, lever),
    )
    return cur.rowcount


def main() -> None:
    prompts_by_id = {p.prompt_id: p for p in load_all_prompts()}

    # Find affected (prompt, lever) batches
    with sqlite3.connect(DB_PATH) as conn:
        batches = conn.execute("""
            SELECT DISTINCT prompt_id, optimisation_lever
            FROM results
            WHERE run_id = ? AND judge_disagreement_flag = 1
              AND judge_a_reasoning IS NULL
            ORDER BY prompt_id, optimisation_lever
        """, (RUN_ID,)).fetchall()

    print(f"Opus reasoning backfill — {len(batches)} unique (prompt, lever) batches", flush=True)

    # Pre-load all responses per batch (need full 4-model batch to fire judge)
    batch_responses: dict[tuple[str, str], dict[str, str]] = {}
    with sqlite3.connect(DB_PATH) as conn:
        for pid, lever in batches:
            rows = conn.execute(
                "SELECT model, response_text FROM results "
                "WHERE run_id=? AND prompt_id=? AND optimisation_lever=? AND error IS NULL",
                (RUN_ID, pid, lever),
            ).fetchall()
            batch_responses[(pid, lever)] = dict(rows)
            if not all(m in batch_responses[(pid, lever)] for m in TEST_MODELS):
                print(f"  WARN: {pid} {lever} missing some model responses; will skip those positions")

    opus_client = anthropic.Anthropic()
    total_cost_usd = 0.0
    n_updated = 0
    n_skipped_existing = 0
    n_call_errors = 0
    ts = _now_iso()

    t0 = time.perf_counter()
    for i, (pid, lever) in enumerate(batches, start=1):
        prompt = prompts_by_id.get(pid)
        if prompt is None or prompt.scoring.tier_2_judge is None:
            print(f"  [{i:>2}/{len(batches)}] {pid} {lever}  SKIP (no tier_2_judge criteria)")
            continue
        responses = batch_responses[(pid, lever)]
        if len(responses) != 4:
            print(f"  [{i:>2}/{len(batches)}] {pid} {lever}  SKIP (only {len(responses)} model responses)")
            continue

        call = assemble_judge_call(prompt, responses, lever=lever)
        t_call = time.perf_counter()
        try:
            text, in_tok, out_tok, lat_ms = _call_opus(opus_client, call.user_message)
        except Exception as e:
            n_call_errors += 1
            print(f"  [{i:>2}/{len(batches)}] {pid} {lever}  EXCEPTION: {type(e).__name__}: {str(e)[:80]}")
            continue

        scores, reasoning, parse_err = _parse_judge_response(text)
        if reasoning is None:
            n_call_errors += 1
            print(f"  [{i:>2}/{len(batches)}] {pid} {lever}  PARSE_ERR: {parse_err}")
            continue

        # Per-position reasoning is keyed by label A/B/C/D in the JSON; map back
        # to model via call.position_to_model
        call_cost = estimate_cost_usd(OPUS_MODEL, input_tokens=in_tok, output_tokens=out_tok,
                                       cached_tokens=0, cache_creation_tokens=0)
        total_cost_usd += call_cost

        # UPDATE each row in this batch where judge_a_reasoning IS NULL.
        # Uses the keyword-only helper update_judge_a_reasoning_if_null so the
        # parameter mapping is explicit (see helper docstring for the
        # methodology lesson — earlier inline version had a positional-binding
        # bug that wrote nothing for £0.78).
        per_batch_updates = 0
        with sqlite3.connect(DB_PATH) as conn:
            for label, model in call.position_to_model.items():
                rsn_for_model = reasoning.get(label)
                if rsn_for_model is None:
                    continue
                rowcount = update_judge_a_reasoning_if_null(
                    conn,
                    run_id=RUN_ID, prompt_id=pid, model=model, lever=lever,
                    reasoning=rsn_for_model, ts=ts,
                )
                if rowcount > 0:
                    per_batch_updates += rowcount
                    n_updated += rowcount
                else:
                    n_skipped_existing += 1
            conn.commit()
        wall_call = time.perf_counter() - t_call
        print(f"  [{i:>2}/{len(batches)}] {pid:10} {lever:12} wall={wall_call:.1f}s  "
              f"updated={per_batch_updates}/4  cost=${call_cost:.4f}", flush=True)

    total_wall = time.perf_counter() - t0
    print()
    print(f"=== Summary ===")
    print(f"  rows updated:        {n_updated}")
    print(f"  rows skipped (existing reasoning preserved): {n_skipped_existing}")
    print(f"  call errors:         {n_call_errors}")
    print(f"  total wall:          {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print(f"  total Opus cost:     ${total_cost_usd:.4f} (~£{usd_to_gbp(total_cost_usd):.4f})")

    # Bump cost_so_far_gbp
    if total_cost_usd > 0:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp + ? WHERE run_id = ?",
                (usd_to_gbp(total_cost_usd), RUN_ID),
            )
            cost_after = conn.execute(
                "SELECT cost_so_far_gbp FROM runs WHERE run_id = ?", (RUN_ID,),
            ).fetchone()[0]
        print(f"  cost_so_far_gbp after: £{cost_after:.6f}")


if __name__ == "__main__":
    main()
