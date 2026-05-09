"""Day 10 Tier-2 judge scoring DRY RUN.

Fires 4 prompts (1 per applicable category) × 4 levers × 2 judges = 32 calls.
Total cost ~£0.50, total wall <2 min. Does NOT write back to the DB.

Surfaces:
  - did all 32 calls parse cleanly?
  - score range sanity (all in [0.0, 1.0])
  - did disagreement detection fire on at least one row? what's the rate?
  - position-to-model audit log per call
  - per-judge cost + latency

Usage:
  poetry run python -m scripts.day_10_dryrun --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from mistralai import Mistral

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

from runners._base import DB_PATH  # noqa: E402
from runners.budget import usd_to_gbp  # noqa: E402
from runners.orchestrator import TEST_MODELS, load_all_prompts  # noqa: E402
from scoring.disagreement import DISAGREEMENT_THRESHOLD, JudgePair, canonical_score, is_disagreement  # noqa: E402
from scoring.judge import score_one_batch  # noqa: E402

DRY_PROMPTS = ["cs-001", "rag-001", "rea-001", "sum-001"]
DRY_LEVERS = ["baseline", "batch", "compression", "output_cap"]


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 10 dual-judge dry run (32 calls)")
    p.add_argument("--run-id", required=True, metavar="RUN_ID",
                   help="Source run_id whose response_text rows feed the judge calls.")
    return p.parse_args()


def _load_responses_for(
    conn: sqlite3.Connection, run_id: str, prompt_id: str, lever: str,
) -> dict[str, str] | None:
    """Returns {model: response_text} for the (prompt, lever) under run_id, or
    None if any test model is missing."""
    rows = conn.execute(
        """SELECT model, response_text FROM results
           WHERE run_id = ? AND prompt_id = ? AND optimisation_lever = ?
             AND error IS NULL""",
        (run_id, prompt_id, lever),
    ).fetchall()
    by_model = dict(rows)
    if not all(m in by_model for m in TEST_MODELS):
        return None
    return {m: by_model[m] for m in TEST_MODELS}


def main() -> None:
    args = _parse_args()
    run_id = args.run_id

    prompts_by_id = {p.prompt_id: p for p in load_all_prompts()}
    targets = [(pid, lev) for pid in DRY_PROMPTS for lev in DRY_LEVERS]

    _print_section(f"Day 10 dual-judge DRY RUN — source run {run_id}")
    print(f"prompts:  {DRY_PROMPTS}", flush=True)
    print(f"levers:   {DRY_LEVERS}", flush=True)
    print(f"calls:    {len(targets)} (prompt,lever) batches × 2 judges = "
          f"{len(targets) * 2} judge API calls; {len(targets) * 4 * 2} row scores",
          flush=True)

    opus_client = anthropic.Anthropic()
    mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    rows_collected: list[dict] = []
    pairs_for_disagreement: list[JudgePair] = []
    parse_errors_by_judge: Counter[str] = Counter()
    cost_by_judge: dict[str, float] = {"opus": 0.0, "mistral": 0.0}
    latency_by_judge: dict[str, list[int]] = {"opus": [], "mistral": []}
    position_log: list[dict] = []

    _print_section("Firing")
    with sqlite3.connect(DB_PATH) as conn:
        for i, (pid, lever) in enumerate(targets, start=1):
            prompt = prompts_by_id.get(pid)
            if prompt is None:
                print(f"[{i:>2}/{len(targets)}] {pid:8s} {lever:12s}  SKIP (no prompt JSON)", flush=True)
                continue
            if prompt.scoring.tier_2_judge is None:
                print(f"[{i:>2}/{len(targets)}] {pid:8s} {lever:12s}  SKIP (no tier_2 criteria)", flush=True)
                continue
            responses = _load_responses_for(conn, run_id, pid, lever)
            if responses is None:
                print(f"[{i:>2}/{len(targets)}] {pid:8s} {lever:12s}  SKIP (missing response rows)", flush=True)
                continue

            print(f"[{i:>2}/{len(targets)}] {pid:8s} {lever:12s}  ", end="", flush=True)
            call, scored = score_one_batch(
                prompt, responses, lever,
                opus_client=opus_client, mistral_client=mistral_client,
            )
            position_log.append({
                "prompt_id": pid, "lever": lever,
                "position_to_model": call.position_to_model,
                "seed": call.seed,
            })
            for s in scored:
                rows_collected.append({
                    "prompt_id": s.prompt_id, "model": s.model, "lever": s.lever,
                    "judge": s.judge, "score": s.score,
                    "position": s.position_label, "judge_error": s.judge_error,
                    "reasoning": s.reasoning,
                })
                if s.judge_error:
                    parse_errors_by_judge[s.judge] += 1
                cost_by_judge[s.judge] += s.cost_usd
                latency_by_judge[s.judge].append(s.latency_ms)

            opus_scores = {r["model"]: r["score"] for r in rows_collected[-8:] if r["judge"] == "opus"}
            mistral_scores = {r["model"]: r["score"] for r in rows_collected[-8:] if r["judge"] == "mistral"}
            for model in TEST_MODELS:
                pairs_for_disagreement.append(JudgePair(
                    prompt_id=pid, model=model, lever=lever,
                    judge_a_score=opus_scores.get(model),
                    judge_b_score=mistral_scores.get(model),
                ))
            print(
                f"opus={list(opus_scores.values())} "
                f"mistral={list(mistral_scores.values())}",
                flush=True,
            )

    _print_section("Per-judge summary")
    for j in ("opus", "mistral"):
        lats = latency_by_judge[j]
        n_called = len(lats)
        n_err = parse_errors_by_judge[j]
        cost_gbp = usd_to_gbp(cost_by_judge[j])
        if lats:
            print(f"  {j:8s}  calls={n_called}  parse_errors={n_err}  "
                  f"latency p50={sorted(lats)[len(lats)//2]}ms  "
                  f"sum=${cost_by_judge[j]:.4f} (~£{cost_gbp:.4f})", flush=True)
        else:
            print(f"  {j:8s}  no calls fired", flush=True)

    _print_section("Score range sanity")
    all_scores = [r["score"] for r in rows_collected if r["score"] is not None]
    if all_scores:
        in_range = sum(1 for s in all_scores if 0.0 <= s <= 1.0)
        print(f"  scores in range [0.0, 1.0]: {in_range}/{len(all_scores)} "
              f"({100.0*in_range/len(all_scores):.0f}%)", flush=True)
        print(f"  min={min(all_scores):.3f}  max={max(all_scores):.3f}  "
              f"mean={sum(all_scores)/len(all_scores):.3f}", flush=True)
    else:
        print("  no scores collected", flush=True)

    _print_section("Disagreement check")
    n_pairs = len(pairs_for_disagreement)
    n_disagree = sum(1 for p in pairs_for_disagreement if is_disagreement(p.judge_a_score, p.judge_b_score))
    print(f"  pairs evaluated: {n_pairs}", flush=True)
    print(f"  disagreements (|Δ| > {DISAGREEMENT_THRESHOLD}): {n_disagree} "
          f"({100.0*n_disagree/n_pairs:.0f}% of pairs)" if n_pairs else "  no pairs", flush=True)
    for p in pairs_for_disagreement:
        if is_disagreement(p.judge_a_score, p.judge_b_score):
            delta = abs(p.judge_a_score - p.judge_b_score)
            canon = canonical_score(p.judge_a_score, p.judge_b_score)
            print(f"    {p.prompt_id:8s} {p.lever:12s} {p.model:32s} "
                  f"opus={p.judge_a_score:.2f} mistral={p.judge_b_score:.2f} "
                  f"Δ={delta:.2f} canonical={canon:.3f}", flush=True)

    _print_section("Position audit (per call)")
    for p in position_log:
        order = " ".join(f"{lab}={p['position_to_model'][lab]}"
                         for lab in ("A", "B", "C", "D"))
        print(f"  {p['prompt_id']:8s} {p['lever']:12s}  {order}", flush=True)

    # Quick aggregate: did any model get pinned to one slot across the 16 calls?
    slot_counts: dict[str, Counter[str]] = {m: Counter() for m in TEST_MODELS}
    for p in position_log:
        for lab, m in p["position_to_model"].items():
            slot_counts[m][lab] += 1
    print("\n  Slot occupancy (across 16 calls):", flush=True)
    print(f"  {'model':32s}  {'A':>3}  {'B':>3}  {'C':>3}  {'D':>3}", flush=True)
    for m in TEST_MODELS:
        c = slot_counts[m]
        print(f"  {m:32s}  {c['A']:>3}  {c['B']:>3}  {c['C']:>3}  {c['D']:>3}", flush=True)

    _print_section("Dry run complete — NO DB writes performed")
    print(f"32 judge API calls fired. Ready for production sweep on sign-off.", flush=True)


if __name__ == "__main__":
    main()
