"""Day 9 Tier-1 scoring — computes scores in memory and surfaces a summary.

Default mode is DRY (no DB writes). Pass --write to persist scores.

Usage:
  poetry run python -m scripts.day_9_dryrun --run-id <run_id>
  poetry run python -m scripts.day_9_dryrun --run-id <run_id> --write
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners._base import DB_PATH  # noqa: E402
from runners.orchestrator import load_all_prompts  # noqa: E402
from scoring.tier_1 import score_row  # noqa: E402


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 9 Tier-1 scoring (dry by default)")
    p.add_argument("--run-id", required=True, metavar="RUN_ID")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N rows (debug).")
    p.add_argument("--show-fails", type=int, default=10, help="Show first N fail_format/fail_schema/fail_content samples.")
    p.add_argument("--write", action="store_true",
                   help="Persist scores to the results table. Single transaction; idempotent.")
    return p.parse_args()


def _persist_scores(conn: sqlite3.Connection, scored: list[tuple[str, dict]]) -> int:
    """Single-transaction UPDATE of all scored rows. Idempotent — re-running
    against rows that already have scores simply rewrites them with the same
    values (assuming the same scorer code). Returns the number of rows written.
    """
    ts = datetime.now(timezone.utc).isoformat()
    n = 0
    conn.execute("BEGIN")
    try:
        for result_id, payload in scored:
            conn.execute(
                """UPDATE results
                       SET tier_1_status = ?,
                           output_format_valid = ?,
                           truncated_due_to_cap = ?,
                           response_parsed = ?,
                           normalisation_steps_applied = ?,
                           rubric_score = ?,
                           score_recomputed_at = ?
                     WHERE result_id = ?""",
                (
                    payload["tier_1_status"],
                    payload["output_format_valid"],
                    payload["truncated_due_to_cap"],
                    payload["response_parsed"],
                    payload["normalisation_steps_applied"],
                    payload["rubric_score"],
                    ts,
                    result_id,
                ),
            )
            n += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return n


def main() -> None:
    args = _parse_args()
    run_id = args.run_id

    prompts_by_id = {p.prompt_id: p for p in load_all_prompts()}

    sql = (
        "SELECT result_id, prompt_id, task_category, complexity, model, provider, "
        "optimisation_lever, optimisation_config, output_tokens, response_text, error "
        "FROM results WHERE run_id = ? ORDER BY task_category, prompt_id, model, optimisation_lever"
    )
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(sql, (run_id,)))

    _print_section(f"Day 9 Tier-1 dry run — {run_id}")
    print(f"rows scored:  {len(rows)}", flush=True)
    if not rows:
        print("No rows for this run_id; nothing to score.", flush=True)
        return

    status_counts: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_model_lever: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    norm_step_counts: Counter[str] = Counter()
    norm_steps_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    fail_samples: dict[str, list[dict]] = defaultdict(list)

    skipped_no_prompt = 0
    scored_for_write: list[tuple[str, dict]] = []
    for r in rows:
        prompt = prompts_by_id.get(r["prompt_id"])
        if prompt is None:
            skipped_no_prompt += 1
            continue
        sr = score_row(dict(r), prompt)
        scored_for_write.append((r["result_id"], sr.to_db_dict()))
        status_counts[sr.tier_1_status] += 1
        by_category[r["task_category"]][sr.tier_1_status] += 1
        by_model_lever[(r["model"], r["optimisation_lever"])][sr.tier_1_status] += 1
        for step in sr.normalisation_steps_applied:
            norm_step_counts[step] += 1
            norm_steps_by_model[r["model"]][step] += 1
        if sr.tier_1_status in {"fail_format", "fail_schema", "fail_content", "truncated"}:
            if len(fail_samples[sr.tier_1_status]) < args.show_fails:
                fail_samples[sr.tier_1_status].append({
                    "result_id": r["result_id"],
                    "prompt_id": r["prompt_id"],
                    "model": r["model"],
                    "lever": r["optimisation_lever"],
                    "output_tokens": r["output_tokens"],
                    "detail": sr.detail,
                    "response_head": (r["response_text"] or "")[:160],
                })

    _print_section("Overall status counts")
    total = sum(status_counts.values())
    for status, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / total if total else 0.0
        print(f"  {status:30s}  {n:5d}  ({pct:5.1f}%)", flush=True)
    if skipped_no_prompt:
        print(f"  (skipped: {skipped_no_prompt} rows had no matching prompt)", flush=True)

    _print_section("Status by task_category")
    for cat in sorted(by_category):
        c = by_category[cat]
        total_cat = sum(c.values())
        print(f"  {cat:18s}  total={total_cat}", flush=True)
        for status in ["pass", "fail_format", "fail_schema", "fail_content", "truncated", "compression_unavailable", "error", "not_applicable"]:
            n = c.get(status, 0)
            if n:
                pct = 100.0 * n / total_cat
                print(f"      {status:25s}  {n:4d}  ({pct:5.1f}%)", flush=True)

    _print_section("Pass rate by (model, lever) — applicable rows only")
    print(f"  {'model':32s}  {'lever':12s}  {'pass':>5}  {'app':>5}  pass_rate", flush=True)
    for (model, lever), c in sorted(by_model_lever.items()):
        applicable = sum(c.values()) - c.get("not_applicable", 0) - c.get("compression_unavailable", 0) - c.get("error", 0)
        passes = c.get("pass", 0)
        rate = 100.0 * passes / applicable if applicable else float("nan")
        print(f"  {model:32s}  {lever:12s}  {passes:5d}  {applicable:5d}  {rate:6.1f}%", flush=True)

    _print_section("Normalisation steps applied (counts across all rows)")
    for step, n in sorted(norm_step_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {step:25s}  {n:5d}", flush=True)

    _print_section("Normalisation steps by model (refinement 2 audit)")
    for model in sorted(norm_steps_by_model):
        steps = norm_steps_by_model[model]
        breakdown = ", ".join(f"{s}={n}" for s, n in sorted(steps.items(), key=lambda kv: -kv[1]))
        print(f"  {model:32s}  {breakdown}", flush=True)

    _print_section("Sample failures (first N per bucket)")
    for status in ["fail_format", "fail_schema", "fail_content", "truncated"]:
        samples = fail_samples.get(status, [])
        if not samples:
            continue
        print(f"\n--- {status} ({len(samples)} sampled) ---", flush=True)
        for s in samples:
            print(f"  {s['prompt_id']:10s} {s['model']:32s} {s['lever']:12s} out_tok={s['output_tokens']}", flush=True)
            print(f"    detail: {json.dumps(s['detail'], default=str)[:200]}", flush=True)
            print(f"    head:   {s['response_head']!r}", flush=True)

    if args.write:
        _print_section("Persisting scores to DB")
        with sqlite3.connect(DB_PATH) as conn:
            n_written = _persist_scores(conn, scored_for_write)
            n_with_status = conn.execute(
                "SELECT COUNT(*) FROM results WHERE run_id = ? AND tier_1_status IS NOT NULL",
                (run_id,),
            ).fetchone()[0]
            n_total = conn.execute(
                "SELECT COUNT(*) FROM results WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
        print(f"rows updated:               {n_written}", flush=True)
        print(f"non-NULL tier_1_status:     {n_with_status} / {n_total}", flush=True)
        if n_with_status != n_total:
            print(f"WARNING: {n_total - n_with_status} rows lack tier_1_status post-write.", flush=True)
        else:
            print("OK — all rows in run have a non-NULL tier_1_status.", flush=True)
    else:
        _print_section("Dry run complete — NO DB writes performed")
        print(f"Run again with --write to persist tier_1_status / rubric_score / "
              f"normalisation_steps_applied / truncated_due_to_cap to results table.", flush=True)


if __name__ == "__main__":
    main()
