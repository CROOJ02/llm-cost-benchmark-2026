"""Day 10 production Tier-2 dual-judge sweep.

Iterates 80 tier-2-applicable prompts (cs + rag + rea + sum) × 4 levers = 320
(prompt, lever) batches. Each batch is judged by Opus 4.6 + mistral-large-2512
in parallel. Each judge sees 4 anonymised model responses (A/B/C/D, randomised
per call). Total: 640 judge API calls → 2,560 row scores → 1,280 row-level
canonical scores after pairing.

Default mode is DRY (no DB writes). Pass --write to persist scores.

On --write the script:
  - UPDATEs results rows in run_id with judge_a_score, judge_b_score,
    judge_disagreement_flag, score_recomputed_at
  - bumps runs.cost_so_far_gbp by the judge cost
  - emits scoring/disagreements.csv for Day 11 human arbitration
  - logs position-to-model mapping per call to data/judge_position_log.jsonl
    (so the Day 12 position-bias audit can reconstruct without re-firing)

Usage:
  poetry run python -m scripts.day_10 --run-id <run_id>
  poetry run python -m scripts.day_10 --run-id <run_id> --write
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
from scoring.disagreement import (  # noqa: E402
    DISAGREEMENT_THRESHOLD, JudgePair, canonical_score,
    emit_disagreement_csv, is_disagreement,
)
from scoring.judge import score_one_batch  # noqa: E402

TIER2_CATEGORIES = ("customer_support", "rag_qa", "reasoning", "summarisation")
LEVERS = ("baseline", "batch", "compression", "output_cap")
PROVIDER_FAMILY = {
    "claude-sonnet-4-6": "Anthropic",
    "claude-haiku-4-5":  "Anthropic",
    "gpt-5.4-2026-03-05":      "OpenAI",
    "gpt-5.4-mini-2026-03-17": "OpenAI",
}
JUDGE_POS_LOG = REPO_ROOT / "data" / "judge_position_log.jsonl"
DISAGREEMENTS_CSV = REPO_ROOT / "scoring" / "disagreements.csv"
DEFAULT_JUDGE_CONCURRENCY = 4  # default; --concurrency overrides


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 10 dual-judge sweep")
    p.add_argument("--run-id", required=True, metavar="RUN_ID")
    p.add_argument("--write", action="store_true",
                   help="Persist judge scores + disagreement flags + cost. Default is dry.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only fire the first N (prompt, lever) batches. Debug.")
    p.add_argument("--missing-only", action="store_true",
                   help="Fire only (prompt, lever) batches where any tier-2 row is missing a "
                        "judge_a_score or judge_b_score. Used for recovery after partial runs. "
                        "score_one_batch dispatches both judges; the partial-results refactor "
                        "in scoring/judge.py persists each judge's success independently, so "
                        "re-running across already-scored sides is harmless (skip-if-already-set).")
    p.add_argument("--reasoning-only", action="store_true",
                   help="Re-fire ONLY the (prompt, lever) batches involved in existing judge "
                        "disagreements (judge_disagreement_flag=1) to populate the "
                        "judge_a_reasoning + judge_b_reasoning columns. Does NOT overwrite "
                        "existing scores or disagreement flags — those are preserved from "
                        "Day 10 (immutable post-scoring) to avoid disagreement-flag flapping "
                        "from fractional score drift at temperature=0. Reasoning is the only "
                        "new data written. See methodology doc § 'Day 11 reasoning re-fire'.")
    p.add_argument("--realign-scores", action="store_true",
                   help="Re-fire the (prompt, lever) batches involved in existing judge "
                        "disagreements AND OVERWRITE both score and reasoning with the new "
                        "API call's output, then recompute the disagreement flag. Use to "
                        "repair score-reasoning desyncs from a prior --reasoning-only pass "
                        "where temperature=0 didn't yield reproducible scores. The "
                        "arbitration target set is re-derived from the new aligned scores.")
    p.add_argument("--judges-only", default=None, metavar="NAMES",
                   help="Comma-separated list of judges to fire (e.g. 'gpt55' or 'gpt55,gemini'). "
                        "Other judges are skipped. Persistence updates ONLY the columns for the "
                        "fired judges (judge_b_* for gpt55, judge_c_* for gemini); judge_a_* "
                        "(Opus) is preserved from earlier sweeps. judge_disagreement_flag is "
                        "recomputed from (DB judge_a_score, fresh judge_b_score). Use for the "
                        "Day 11 panel-revision GPT-5.5-only sweep that keeps Opus stable.")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Number of batches in flight. Default 4. Day 10 recovery uses 2 "
                        "to stay under Mistral's TPM ceiling.")
    return p.parse_args()


def _load_responses_for(
    conn: sqlite3.Connection, run_id: str, prompt_id: str, lever: str,
) -> dict[str, str] | None:
    rows = conn.execute(
        """SELECT model, response_text FROM results
           WHERE run_id = ? AND prompt_id = ? AND optimisation_lever = ?
             AND error IS NULL
             AND (tier_1_status IS NULL OR tier_1_status != 'compression_unavailable')""",
        (run_id, prompt_id, lever),
    ).fetchall()
    by_model = dict(rows)
    if not all(m in by_model for m in TEST_MODELS):
        return None
    return {m: by_model[m] for m in TEST_MODELS}


def _result_id_lookup(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str, str], str]:
    """Map (prompt_id, model, lever) → result_id for the run, used by --write
    to UPDATE the right rows."""
    out: dict[tuple[str, str, str], str] = {}
    for pid, model, lever, rid in conn.execute(
        "SELECT prompt_id, model, optimisation_lever, result_id FROM results WHERE run_id = ?",
        (run_id,),
    ):
        out[(pid, model, lever)] = rid
    return out


def _reasoning_only_targets(
    conn: sqlite3.Connection, run_id: str, all_targets: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """(prompt_id, lever) batches that have ANY model row flagged
    judge_disagreement_flag=1 under run_id. Re-firing the whole batch is
    necessary because score_one_batch always scores all 4 model responses
    together — there's no per-row judge call to selectively re-fire."""
    rows = conn.execute(
        "SELECT DISTINCT prompt_id, optimisation_lever FROM results "
        "WHERE run_id = ? AND judge_disagreement_flag = 1",
        (run_id,),
    ).fetchall()
    flagged = set(rows)
    return [(pid, lever) for pid, lever in all_targets if (pid, lever) in flagged]


def _missing_only_targets(
    conn: sqlite3.Connection, run_id: str, all_targets: list[tuple[str, str]],
    judge_names: tuple[str, ...] = ("opus", "mistral"),
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], tuple[bool, bool]]]:
    """Filter `all_targets` to (prompt, lever) batches that need re-firing.

    A batch needs re-firing if ANY of its 4 model rows has a NULL judge_a_score
    OR judge_b_score. Returns (filtered_targets, skip_map) where skip_map[(pid,
    lever)] = (skip_opus, skip_mistral) tells the caller which judge sides to
    skip on a per-batch basis (because they're already filled in).
    """
    sql = (
        "SELECT prompt_id, optimisation_lever, model, judge_a_score, judge_b_score "
        "FROM results WHERE run_id = ? "
        "  AND task_category IN ('customer_support','rag_qa','reasoning','summarisation')"
    )
    by_batch: dict[tuple[str, str], dict[str, tuple]] = defaultdict(dict)
    for pid, lever, model, ja, jb in conn.execute(sql, (run_id,)):
        by_batch[(pid, lever)][model] = (ja, jb)

    filtered: list[tuple[str, str]] = []
    skip_map: dict[tuple[str, str], tuple[bool, bool]] = {}
    for pid, lever in all_targets:
        rows = by_batch.get((pid, lever), {})
        if len(rows) < len(TEST_MODELS):
            continue  # not all 4 models present; let the response-load step skip
        any_opus_null = any(ja is None for ja, _ in rows.values())
        any_mistral_null = any(jb is None for _, jb in rows.values())
        if not any_opus_null and not any_mistral_null:
            continue
        filtered.append((pid, lever))
        # Skip a judge side if EVERY row in this batch already has that side.
        skip_map[(pid, lever)] = (not any_opus_null, not any_mistral_null)
    return filtered, skip_map


def main() -> None:
    args = _parse_args()
    run_id = args.run_id
    concurrency = args.concurrency

    prompts_by_id = {p.prompt_id: p for p in load_all_prompts()}
    targets: list[tuple[str, str]] = []
    for prompt in prompts_by_id.values():
        if prompt.task_category not in TIER2_CATEGORIES:
            continue
        if prompt.scoring.tier_2_judge is None:
            continue
        for lever in LEVERS:
            targets.append((prompt.prompt_id, lever))
    targets.sort()
    if args.limit:
        targets = targets[: args.limit]

    exclusive_flags = sum([args.missing_only, args.reasoning_only, args.realign_scores, bool(args.judges_only)])
    if exclusive_flags > 1:
        sys.exit("ERROR: --missing-only / --reasoning-only / --realign-scores / --judges-only are mutually exclusive.")

    judges_only_set: set[str] | None = None
    if args.judges_only:
        judges_only_set = {j.strip() for j in args.judges_only.split(",") if j.strip()}
        unknown = judges_only_set - {"opus", "mistral", "gpt55", "gemini"}
        if unknown:
            sys.exit(f"ERROR: --judges-only contains unknown judges: {unknown}")
        # GPT-5.5-only sweep persists to judge_b_*; Gemini-only would persist
        # to judge_c_*. Mixed sweeps not supported by the persistence path.
        unsupported_combos = (
            ("opus" in judges_only_set and len(judges_only_set) > 1),
        )
        if any(unsupported_combos):
            sys.exit(f"ERROR: --judges-only with 'opus' is not supported; Opus stays "
                     f"in judge_a_* via existing data, this flag re-fires Opus which "
                     f"would overwrite known-good Day 10 scores.")

    skip_judge_per_batch: dict[tuple[str, str], tuple[bool, bool]] = {}
    if args.missing_only:
        with sqlite3.connect(DB_PATH) as conn:
            targets, skip_judge_per_batch = _missing_only_targets(conn, run_id, targets)
    elif args.reasoning_only or args.realign_scores:
        # Both modes target the same set: batches with judge_disagreement_flag=1.
        # The persistence behavior differs (narrow vs full overwrite).
        with sqlite3.connect(DB_PATH) as conn:
            targets = _reasoning_only_targets(conn, run_id, targets)

    mode_label = "WRITE" if args.write else "DRY"
    if args.missing_only:
        mode_label += "  [MISSING-ONLY recovery]"
    elif args.reasoning_only:
        mode_label += "  [REASONING-ONLY re-fire]"
    elif args.realign_scores:
        mode_label += "  [REALIGN-SCORES re-fire — overwrites scores+reasoning+flag]"
    _print_section(f"Day 10 dual-judge sweep — run {run_id}")
    print(f"mode:           {mode_label}", flush=True)
    print(f"batches:        {len(targets)}  ((prompt, lever) tuples)", flush=True)
    print(f"judge calls:    {len(targets) * 2}  (some sides may skip per skip_map)", flush=True)
    print(f"row scores:     {len(targets) * 4 * 2}", flush=True)
    print(f"concurrency:    {concurrency} batches in flight", flush=True)
    print(f"position log:   {JUDGE_POS_LOG}", flush=True)
    print(f"disagreements:  {DISAGREEMENTS_CSV}", flush=True)

    # Lazy-init: only create clients for judges that will actually fire.
    # GPT-5.5 + Gemini added Day 11 for the panel-revision sweep.
    opus_client = anthropic.Anthropic() if (judges_only_set is None or "opus" in judges_only_set) else None
    mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"]) if (judges_only_set is None or "mistral" in judges_only_set) else None
    gpt55_client = None
    gemini_client = None
    if judges_only_set and "gpt55" in judges_only_set:
        import openai as _openai
        gpt55_client = _openai.OpenAI(max_retries=0)
    if judges_only_set and "gemini" in judges_only_set:
        from google import genai as _genai
        gemini_client = _genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        result_id_by_key = _result_id_lookup(conn, run_id) if args.write else {}

        # Pre-load all (prompt, lever) responses up-front to make the workers stateless.
        batch_responses: dict[tuple[str, str], dict[str, str]] = {}
        skipped_no_response: list[tuple[str, str]] = []
        for pid, lever in targets:
            r = _load_responses_for(conn, run_id, pid, lever)
            if r is None:
                skipped_no_response.append((pid, lever))
            else:
                batch_responses[(pid, lever)] = r

    if skipped_no_response:
        print(f"\nWARNING: {len(skipped_no_response)} batches skipped — incomplete responses:",
              flush=True)
        for pid, lever in skipped_no_response[:10]:
            print(f"  {pid:8s} {lever:12s}", flush=True)

    fireable = [(pid, lever) for pid, lever in targets if (pid, lever) in batch_responses]
    print(f"\nfireable batches: {len(fireable)}", flush=True)
    if not fireable:
        print("Nothing to fire. Exiting.", flush=True)
        return

    rows_collected: list[dict] = []
    # defaultdicts so any judge name (opus/mistral/gpt55/gemini) can be added
    # without pre-declaration. Day 11 panel revision added gpt55 and gemini;
    # original hardcoded {"opus", "mistral"} keys would KeyError on first batch.
    cost_by_judge: dict[str, float] = defaultdict(float)
    latency_by_judge: dict[str, list[int]] = defaultdict(list)
    parse_errors_by_judge: Counter[str] = Counter()
    position_log: list[dict] = []

    def _fire_one(pid: str, lever: str) -> tuple[str, str, dict, list]:
        prompt = prompts_by_id[pid]
        responses = batch_responses[(pid, lever)]
        if judges_only_set is not None:
            # Judge-subset sweep — fire only the judges in the set, exactly as
            # specified. Skip-map from --missing-only doesn't apply here.
            judges = tuple(sorted(judges_only_set))
        else:
            skip_opus, skip_mistral = skip_judge_per_batch.get((pid, lever), (False, False))
            judges = tuple(j for j, skip in (("opus", skip_opus), ("mistral", skip_mistral)) if not skip)
        call, scored = score_one_batch(
            prompt, responses, lever,
            judge_names=judges,
            opus_client=opus_client, mistral_client=mistral_client,
            gpt55_client=gpt55_client, gemini_client=gemini_client,
        )
        return pid, lever, {
            "prompt_id": pid, "lever": lever, "seed": call.seed,
            "position_to_model": call.position_to_model,
        }, scored

    _print_section(f"Firing {len(fireable)} batches")
    t0 = time.perf_counter()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_fire_one, pid, lever): (pid, lever) for pid, lever in fireable}
        for fut in concurrent.futures.as_completed(futures):
            pid_orig, lever_orig = futures[fut]
            try:
                pid, lever, pos_entry, scored = fut.result()
            except Exception as e:
                completed += 1
                print(f"[{completed:>3}/{len(fireable)}] {pid_orig:10s} {lever_orig:12s}  EXCEPTION {type(e).__name__}: {e}",
                      flush=True)
                continue
            position_log.append(pos_entry)
            for s in scored:
                rows_collected.append({
                    "prompt_id": s.prompt_id, "model": s.model, "lever": s.lever,
                    "judge": s.judge, "score": s.score, "position": s.position_label,
                    "judge_error": s.judge_error, "reasoning": s.reasoning,
                    "cost_usd": s.cost_usd, "latency_ms": s.latency_ms,
                })
                if s.judge_error:
                    parse_errors_by_judge[s.judge] += 1
                cost_by_judge[s.judge] += s.cost_usd
                latency_by_judge[s.judge].append(s.latency_ms)
            completed += 1
            if completed % 20 == 0 or completed == len(fireable):
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(fireable) - completed) / rate if rate > 0 else 0
                print(f"[{completed:>3}/{len(fireable)}] elapsed={elapsed:.0f}s "
                      f"rate={rate:.2f} batches/s  eta~{eta:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    _print_section("Sweep complete — aggregating")
    # Pair (judge_a=opus, judge_b=<active judge B>) scores per (prompt, model, lever).
    # Reasoning is carried in a parallel dict so JudgePair (used downstream
    # for disagreement detection) stays score-only — Day 11 added reasoning
    # capture without changing the disagreement contract. The "judge B" slot
    # is GPT-5.5 post-Day-11 panel revision; legacy "mistral" preserved for
    # any backward-compat sweeps.
    by_row: dict[tuple[str, str, str], dict[str, float | None]] = defaultdict(
        lambda: {"opus": None, "mistral": None, "gpt55": None, "gemini": None}
    )
    by_row_reasoning: dict[tuple[str, str, str], dict[str, str | None]] = defaultdict(
        lambda: {"opus": None, "mistral": None, "gpt55": None, "gemini": None}
    )
    by_row_meta: dict[tuple[str, str, str], dict] = {}
    for r in rows_collected:
        key = (r["prompt_id"], r["model"], r["lever"])
        by_row[key][r["judge"]] = r["score"]
        by_row_reasoning[key][r["judge"]] = r["reasoning"]
        by_row_meta.setdefault(key, {"position": r["position"], "judge_errors": []})
        if r["judge_error"]:
            by_row_meta[key]["judge_errors"].append((r["judge"], r["judge_error"]))

    # Determine which judge name maps to "judge B" slot for this sweep.
    # Default mistral preserves legacy 2-judge behavior. Day 11 panel revision
    # uses gpt55. Gemini support reserved for v2.
    if judges_only_set and "gpt55" in judges_only_set:
        slot_b_judge = "gpt55"
    elif judges_only_set and "gemini" in judges_only_set:
        slot_b_judge = "gemini"
    else:
        slot_b_judge = "mistral"

    # When --judges-only is set, judge_a (Opus) is preserved from the DB
    # rather than re-fired. Pre-fetch existing Opus scores so pairs (used by
    # disagreement detection, calibration metric, and CSV emission) reflect
    # the stable DB-archived Opus + the fresh Judge B from this sweep.
    db_opus_scores: dict[tuple[str, str, str], float | None] = {}
    db_opus_reasoning: dict[tuple[str, str, str], str | None] = {}
    if judges_only_set is not None:
        with sqlite3.connect(DB_PATH) as _conn:
            for key in by_row.keys():
                pid, model, lev = key
                row = _conn.execute(
                    "SELECT judge_a_score, judge_a_reasoning FROM results "
                    "WHERE run_id=? AND prompt_id=? AND model=? AND optimisation_lever=?",
                    (run_id, pid, model, lev),
                ).fetchone()
                db_opus_scores[key] = row[0] if row else None
                db_opus_reasoning[key] = row[1] if row else None

    pairs: list[JudgePair] = []
    pair_reasoning: dict[tuple[str, str, str], tuple[str | None, str | None]] = {}
    for (pid, model, lever), scores in by_row.items():
        prompt = prompts_by_id[pid]
        responses = batch_responses.get((pid, lever), {})
        # judge_a_score: prefer fresh Opus from this run if Opus fired; else
        # use DB Opus (--judges-only path). For default 2-judge sweeps, Opus
        # fires fresh and scores["opus"] is the new score.
        if judges_only_set is not None:
            ja = db_opus_scores.get((pid, model, lever))
            ja_rsn = db_opus_reasoning.get((pid, model, lever))
        else:
            ja = scores["opus"]
            ja_rsn = by_row_reasoning[(pid, model, lever)]["opus"]
        jb = scores[slot_b_judge]
        jb_rsn = by_row_reasoning[(pid, model, lever)][slot_b_judge]
        pairs.append(JudgePair(
            prompt_id=pid, model=model, lever=lever,
            judge_a_score=ja, judge_b_score=jb,
            response_text=responses.get(model, ""),
            tier_2_criteria=prompt.scoring.tier_2_judge.criteria,
        ))
        pair_reasoning[(pid, model, lever)] = (ja_rsn, jb_rsn)

    n_disagree = sum(1 for p in pairs if is_disagreement(p.judge_a_score, p.judge_b_score))

    # Cross-judge calibration metric: Opus vs <slot B judge> mean per provider family
    family_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for p in pairs:
        family = PROVIDER_FAMILY.get(p.model, "Other")
        if p.judge_a_score is not None:
            family_scores[(family, "opus")].append(p.judge_a_score)
        if p.judge_b_score is not None:
            family_scores[(family, slot_b_judge)].append(p.judge_b_score)

    # Position-by-model audit (model × slot)
    slot_counts: dict[str, Counter[str]] = {m: Counter() for m in TEST_MODELS}
    slot_score_sum: dict[tuple[str, str], list[float]] = defaultdict(list)
    pos_by_call: dict[tuple[str, str], dict[str, str]] = {}
    for entry in position_log:
        pos_by_call[(entry["prompt_id"], entry["lever"])] = entry["position_to_model"]
        for label, model in entry["position_to_model"].items():
            slot_counts[model][label] += 1
    for r in rows_collected:
        if r["score"] is None:
            continue
        slot_score_sum[(r["model"], r["position"])].append(r["score"])

    _print_section("Per-judge summary")
    # Iterate over judges that actually fired (any of opus/mistral/gpt55/gemini)
    # rather than the hardcoded legacy 2-judge set.
    for j in sorted(latency_by_judge.keys()):
        lats = latency_by_judge[j]
        n_called = len(lats)
        n_err = parse_errors_by_judge[j]
        cost_gbp = usd_to_gbp(cost_by_judge[j])
        if lats:
            slats = sorted(lats)
            p50 = slats[len(slats)//2]
            p95 = slats[int(len(slats)*0.95)]
            print(f"  {j:8s}  scored_rows={n_called}  parse_errors={n_err}  "
                  f"latency p50={p50}ms p95={p95}ms  "
                  f"sum=${cost_by_judge[j]:.4f} (~£{cost_gbp:.4f})", flush=True)

    _print_section("Score distribution")
    all_scores = [r["score"] for r in rows_collected if r["score"] is not None]
    if all_scores:
        in_range = sum(1 for s in all_scores if 0.0 <= s <= 1.0)
        print(f"  scores in range [0.0, 1.0]: {in_range}/{len(all_scores)} ({100.0*in_range/len(all_scores):.1f}%)", flush=True)
        print(f"  min={min(all_scores):.3f}  max={max(all_scores):.3f}  mean={sum(all_scores)/len(all_scores):.3f}", flush=True)
        # Distribution buckets
        buckets = Counter()
        for s in all_scores:
            if s == 1.0: buckets["1.00"] += 1
            elif s >= 0.8: buckets["0.80–0.99"] += 1
            elif s >= 0.6: buckets["0.60–0.79"] += 1
            elif s >= 0.4: buckets["0.40–0.59"] += 1
            elif s >= 0.2: buckets["0.20–0.39"] += 1
            else: buckets["0.00–0.19"] += 1
        for k in ("1.00", "0.80–0.99", "0.60–0.79", "0.40–0.59", "0.20–0.39", "0.00–0.19"):
            n = buckets[k]
            print(f"    {k:12s}  {n:5d}  ({100.0*n/len(all_scores):5.1f}%)", flush=True)

    _print_section("Disagreement summary")
    n_pairs = len(pairs)
    n_complete = sum(1 for p in pairs if p.judge_a_score is not None and p.judge_b_score is not None)
    print(f"  pairs total:           {n_pairs}", flush=True)
    print(f"  pairs with both scored: {n_complete}", flush=True)
    print(f"  disagreements:         {n_disagree}  ({100.0*n_disagree/n_complete:.1f}% of complete pairs)", flush=True)

    _print_section("Cross-judge calibration metric (per provider family)")
    print(f"  {'family':12s}  {'judge':8s}  {'mean':>6}  {'n':>5}", flush=True)
    family_means: dict[tuple[str, str], float] = {}
    for (fam, judge), vs in sorted(family_scores.items()):
        mean = sum(vs) / len(vs)
        family_means[(fam, judge)] = mean
        print(f"  {fam:12s}  {judge:8s}  {mean:6.3f}  {len(vs):>5}", flush=True)
    print(f"\n  {'family':12s}  {'opus mean':>10}  {slot_b_judge + ' mean':>13}  {'Δ (residual self-bias)':>25}",
          flush=True)
    for fam in ("Anthropic", "OpenAI"):
        o = family_means.get((fam, "opus"))
        m = family_means.get((fam, slot_b_judge))
        if o is not None and m is not None:
            d = o - m
            flag = "  ⚠ |Δ|>0.05" if abs(d) > 0.05 else "  ✓"
            print(f"  {fam:12s}  {o:>10.3f}  {m:>13.3f}  {d:>+25.3f}{flag}", flush=True)

    _print_section("Position-by-model occupancy (n=320 calls)")
    print(f"  {'model':32s}  {'A':>5}  {'B':>5}  {'C':>5}  {'D':>5}  expected~{len(fireable)/4:.0f}", flush=True)
    for m in TEST_MODELS:
        c = slot_counts[m]
        print(f"  {m:32s}  {c['A']:>5}  {c['B']:>5}  {c['C']:>5}  {c['D']:>5}", flush=True)

    _print_section("Position-by-model mean score (Day 12 audit deliverable)")
    print(f"  {'model':32s}  {'A':>6}  {'B':>6}  {'C':>6}  {'D':>6}  {'row mean':>8}", flush=True)
    findings: list[str] = []
    for m in TEST_MODELS:
        slot_means = {}
        for label in ("A", "B", "C", "D"):
            vs = slot_score_sum[(m, label)]
            slot_means[label] = sum(vs)/len(vs) if vs else float("nan")
        valid = [v for v in slot_means.values() if v == v]  # drop NaN
        row_mean = sum(valid)/len(valid) if valid else float("nan")
        cells = "  ".join(f"{slot_means[l]:6.3f}" for l in ("A", "B", "C", "D"))
        print(f"  {m:32s}  {cells}  {row_mean:8.3f}", flush=True)
        for label in ("A", "B", "C", "D"):
            if slot_means[label] == slot_means[label] and abs(slot_means[label] - row_mean) > 0.03:
                findings.append(f"{m} slot {label}: {slot_means[label]:.3f} vs row mean {row_mean:.3f} "
                                f"(Δ={slot_means[label]-row_mean:+.3f})")
    if findings:
        print(f"\n  ⚠ position-bias findings (|Δ|>0.03 from row mean):", flush=True)
        for f in findings:
            print(f"    {f}", flush=True)
    else:
        print(f"\n  ✓ no cell deviates from row mean by >0.03", flush=True)

    if not args.write:
        _print_section("DRY RUN — no DB writes performed")
        print(f"Run again with --write to persist scores + emit disagreements CSV.", flush=True)
        return

    _print_section("Persisting to DB + emitting artefacts")
    ts = _now_iso()
    n_updated = 0
    n_judge_err_rows = 0
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN")
        try:
            for p in pairs:
                rid = result_id_by_key.get((p.prompt_id, p.model, p.lever))
                if rid is None:
                    continue
                new_rsn_a, new_rsn_b = pair_reasoning.get(
                    (p.prompt_id, p.model, p.lever), (None, None)
                )
                if args.reasoning_only:
                    # Narrow update: ONLY reasoning columns. Original scores +
                    # disagreement flag are preserved as immutable post-Day-10
                    # state. COALESCE preserves any reasoning that's already
                    # been written (idempotent re-run safety).
                    conn.execute(
                        """UPDATE results
                              SET judge_a_reasoning = COALESCE(?, judge_a_reasoning),
                                  judge_b_reasoning = COALESCE(?, judge_b_reasoning),
                                  score_recomputed_at = ?
                            WHERE result_id = ?""",
                        (new_rsn_a, new_rsn_b, ts, rid),
                    )
                    n_updated += 1
                    continue

                if judges_only_set is not None:
                    # Judge-subset sweep (Day 11 panel revision GPT-5.5 path).
                    # Update ONLY the columns for the fired judge:
                    #   - judge_b_score  ← fresh fired judge's score (was NULL)
                    #   - judge_b_reasoning ← fresh fired judge's reasoning
                    #   - judge_disagreement_flag ← recomputed from
                    #     (DB judge_a_score, fresh judge_b_score)
                    #   - score_recomputed_at ← now
                    # Explicitly NOT touched:
                    #   - judge_a_score, judge_a_reasoning (Opus, frozen Day 10)
                    #   - judge_b_mistral_score, judge_b_mistral_reasoning (archive)
                    #   - judge_c_score, judge_c_reasoning, judge_c_name (Gemini, NULL)
                    # p.judge_a_score is the DB-archived Opus (loaded above into
                    # pairs); p.judge_b_score is the fresh Judge-B call. Both
                    # are passed to is_disagreement to compute the new flag.
                    db_opus_for_row = p.judge_a_score
                    fresh_b_score = p.judge_b_score
                    disagree = 1 if is_disagreement(db_opus_for_row, fresh_b_score) else 0
                    if fresh_b_score is None:
                        n_judge_err_rows += 1
                    conn.execute(
                        """UPDATE results
                              SET judge_b_score = ?,
                                  judge_b_reasoning = ?,
                                  judge_disagreement_flag = ?,
                                  score_recomputed_at = ?
                            WHERE result_id = ?""",
                        (fresh_b_score, new_rsn_b, disagree, ts, rid),
                    )
                    n_updated += 1
                    continue

                # Default + --missing-only path: write scores AND reasoning.
                # Fetch existing values so a NULL on either side doesn't
                # clobber previously-persisted data.
                existing = conn.execute(
                    "SELECT judge_a_score, judge_b_score, judge_a_reasoning, judge_b_reasoning "
                    "FROM results WHERE result_id = ?",
                    (rid,),
                ).fetchone()
                ex_a, ex_b, ex_rsn_a, ex_rsn_b = (existing or (None, None, None, None))
                final_a = p.judge_a_score if p.judge_a_score is not None else ex_a
                final_b = p.judge_b_score if p.judge_b_score is not None else ex_b
                final_rsn_a = new_rsn_a if new_rsn_a is not None else ex_rsn_a
                final_rsn_b = new_rsn_b if new_rsn_b is not None else ex_rsn_b
                disagree = 1 if is_disagreement(final_a, final_b) else 0
                if final_a is None or final_b is None:
                    n_judge_err_rows += 1
                conn.execute(
                    """UPDATE results
                          SET judge_a_score = ?,
                              judge_b_score = ?,
                              judge_a_reasoning = ?,
                              judge_b_reasoning = ?,
                              judge_disagreement_flag = ?,
                              score_recomputed_at = ?
                        WHERE result_id = ?""",
                    (final_a, final_b, final_rsn_a, final_rsn_b, disagree, ts, rid),
                )
                n_updated += 1
            # Bump cost_so_far_gbp
            judge_cost_gbp = sum(usd_to_gbp(v) for v in cost_by_judge.values())
            conn.execute(
                "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp + ? WHERE run_id = ?",
                (judge_cost_gbp, run_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        cost_after = conn.execute(
            "SELECT cost_so_far_gbp FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()[0]

    # Emit disagreements CSV
    n_csv = emit_disagreement_csv(pairs, DISAGREEMENTS_CSV)

    # Append position log to JSONL
    JUDGE_POS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with JUDGE_POS_LOG.open("a") as f:
        for entry in position_log:
            f.write(json.dumps({**entry, "run_id": run_id, "ts": ts}) + "\n")

    print(f"  rows updated:                  {n_updated}", flush=True)
    print(f"  rows with one+ judge_error:    {n_judge_err_rows}", flush=True)
    print(f"  cost_so_far_gbp after:         £{cost_after:.6f}  (delta £+{usd_to_gbp(sum(cost_by_judge.values())):.4f})", flush=True)
    print(f"  disagreements CSV:             {DISAGREEMENTS_CSV}  ({n_csv} rows for Day 11)", flush=True)
    print(f"  position log appended to:      {JUDGE_POS_LOG}  ({len(position_log)} new entries)", flush=True)
    print(f"\n  wall:                          {wall:.1f}s ({wall/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
