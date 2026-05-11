"""Day 12 step 1 — Lever × model cost-quality matrix.

Read-only against data/results.db. Computes per-(model, lever):
  - n_rows
  - mean canonical_score
  - mean cost_usd
  - tier_1_pass_rate_all        (pass / n_rows; counts not_applicable as non-pass)
  - tier_1_pass_rate_applicable (pass / (pass + fail_format + fail_content + truncated))
  - delta canonical_score vs same-model baseline
  - cost ratio vs same-model baseline (fraction; 1.0 = same cost, 0.5 = half cost)
  - delta tier_1_pass_rate_all vs same-model baseline
  - delta tier_1_pass_rate_applicable vs same-model baseline

Tier-2 population only: filter on `canonical_score IS NOT NULL`. The 496
Tier-1-only rows have no judge scoring and are excluded.

Output: stdout 16-row matrix + analysis/out/lever_matrix.csv.
"""

from __future__ import annotations

import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.db"
OUT_DIR = ROOT / "analysis" / "out"
OUT_CSV = OUT_DIR / "lever_matrix.csv"

MATRIX_QUERY = """
SELECT
    model,
    optimisation_lever AS lever,
    COUNT(*)                                                            AS n_rows,
    AVG(canonical_score)                                                AS mean_canonical,
    AVG(cost_usd)                                                       AS mean_cost_usd,
    SUM(CASE WHEN tier_1_status = 'pass' THEN 1 ELSE 0 END)             AS pass_count,
    SUM(CASE WHEN tier_1_status = 'fail_format' THEN 1 ELSE 0 END)      AS fail_format_count,
    SUM(CASE WHEN tier_1_status = 'fail_content' THEN 1 ELSE 0 END)     AS fail_content_count,
    SUM(CASE WHEN tier_1_status = 'truncated' THEN 1 ELSE 0 END)        AS truncated_count,
    SUM(CASE WHEN tier_1_status = 'not_applicable' THEN 1 ELSE 0 END)   AS not_applicable_count
FROM results
WHERE canonical_score IS NOT NULL
GROUP BY model, optimisation_lever
ORDER BY model, optimisation_lever
"""

STATUS_QUERY = """
SELECT model, optimisation_lever AS lever, tier_1_status, COUNT(*) AS n
FROM results
WHERE canonical_score IS NOT NULL
GROUP BY model, optimisation_lever, tier_1_status
"""

SPOTCHECK_MODEL = "claude-sonnet-4-6"
SPOTCHECK_LEVER = "compression"
SPOTCHECK_QUERY = """
SELECT prompt_id, task_category, canonical_score, cost_usd, tier_1_status,
       judge_a_score, judge_b_score, judge_disagreement_flag,
       SUBSTR(response_text, 1, 150) AS response_snippet
FROM results
WHERE canonical_score IS NOT NULL
  AND model = ?
  AND optimisation_lever = ?
  AND tier_1_status = ?
ORDER BY prompt_id
LIMIT ?
"""

IDENTITY_QUERY = """
SELECT prompt_id, task_category, tier_1_status
FROM results
WHERE canonical_score IS NOT NULL
  AND model = ?
  AND optimisation_lever = ?
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Surface the SQL so the user can verify the source columns ---
    print("[query: matrix aggregation]")
    for line in MATRIX_QUERY.strip().splitlines():
        print(f"   {line}")

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cells = [dict(r) for r in conn.execute(MATRIX_QUERY)]

    # Baseline lookup per model for delta computation.
    by_model_baseline = {r["model"]: r for r in cells if r["lever"] == "baseline"}
    missing = {r["model"] for r in cells} - set(by_model_baseline)
    if missing:
        raise SystemExit(f"[error] no baseline row for models: {missing}")

    def pass_rates(row: dict) -> tuple[float, float]:
        all_denom = row["n_rows"]
        applicable_denom = (
            row["pass_count"]
            + row["fail_format_count"]
            + row["fail_content_count"]
            + row["truncated_count"]
        )
        rate_all = row["pass_count"] / all_denom if all_denom else 0.0
        rate_applicable = (
            row["pass_count"] / applicable_denom if applicable_denom else 0.0
        )
        return rate_all, rate_applicable

    matrix_rows: list[dict] = []
    for r in cells:
        base = by_model_baseline[r["model"]]
        rate_all, rate_applicable = pass_rates(r)
        base_rate_all, base_rate_applicable = pass_rates(base)
        delta_canonical = r["mean_canonical"] - base["mean_canonical"]
        cost_ratio = (
            r["mean_cost_usd"] / base["mean_cost_usd"] if base["mean_cost_usd"] else 0.0
        )
        matrix_rows.append(
            {
                "model": r["model"],
                "lever": r["lever"],
                "n_rows": r["n_rows"],
                "mean_canonical_score": round(r["mean_canonical"], 4),
                "mean_cost_usd": round(r["mean_cost_usd"], 6),
                "tier_1_pass_rate_all": round(rate_all, 4),
                "tier_1_pass_rate_applicable": round(rate_applicable, 4),
                "delta_canonical_vs_baseline": round(delta_canonical, 4),
                "cost_ratio_vs_baseline": round(cost_ratio, 4),
                "delta_t1_pass_rate_all_vs_baseline": round(rate_all - base_rate_all, 4),
                "delta_t1_pass_rate_applicable_vs_baseline": round(
                    rate_applicable - base_rate_applicable, 4
                ),
            }
        )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)

    # --- Cell counts: confirm 80 per cell or surface deviations ---
    cell_count_summary = Counter(r["n_rows"] for r in matrix_rows)
    print("\n[cell counts] (model, lever) cell n_rows distribution")
    for n, count in sorted(cell_count_summary.items()):
        print(f"   n={n}: {count} cells")
    if set(cell_count_summary) == {80}:
        print("   → all 16 cells have exactly 80 rows (4 models × 4 levers × 80 prompts)")
    else:
        print("   → non-uniform; per-cell counts surfaced in the matrix below")

    # --- 16-row matrix ---
    print(
        f"\n[matrix] 4 models × 4 levers (n=1,280 tier-2 rows; "
        f"baseline rows show trivial Δ/ratio for completeness)"
    )
    header = (
        f"{'model':<28} {'lever':<12} {'n':>4}  "
        f"{'canon':>6}  {'cost($)':>10}  "
        f"{'t1_all':>7}  {'t1_appl':>7}  "
        f"{'Δcanon':>7}  {'cost_x':>7}  {'Δt1_all':>7}  {'Δt1_app':>7}"
    )
    print(header)
    print("-" * len(header))
    last_model = None
    for r in matrix_rows:
        if last_model and r["model"] != last_model:
            print()  # separator between model blocks
        last_model = r["model"]
        marker = "  ←base" if r["lever"] == "baseline" else ""
        print(
            f"{r['model']:<28} {r['lever']:<12} {r['n_rows']:>4}  "
            f"{r['mean_canonical_score']:>6.3f}  {r['mean_cost_usd']:>10.6f}  "
            f"{r['tier_1_pass_rate_all']:>7.3f}  "
            f"{r['tier_1_pass_rate_applicable']:>7.3f}  "
            f"{r['delta_canonical_vs_baseline']:>+7.3f}  "
            f"{r['cost_ratio_vs_baseline']:>7.3f}  "
            f"{r['delta_t1_pass_rate_all_vs_baseline']:>+7.3f}  "
            f"{r['delta_t1_pass_rate_applicable_vs_baseline']:>+7.3f}"
            f"{marker}"
        )

    # --- tier_1_status distribution per cell (helps reader interpret pass_rate
    # denominator: 'not_applicable' rows count as non-pass in the pass_rate as
    # defined above) ---
    status_dist: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in conn.execute(STATUS_QUERY):
        status_dist[(row["model"], row["lever"])][row["tier_1_status"]] += row["n"]
    all_statuses = sorted({s for d in status_dist.values() for s in d.keys()})
    print(f"\n[tier_1_status] distribution per (model, lever) — denominator context for t1_pass")
    print(f"   statuses present: {all_statuses}")
    print(f"   {'cell':<42}  " + "  ".join(f"{s[:14]:>14}" for s in all_statuses))
    for r in matrix_rows:
        key = (r["model"], r["lever"])
        cell_label = f"{r['model']} / {r['lever']}"
        counts = "  ".join(f"{status_dist[key].get(s, 0):>14d}" for s in all_statuses)
        print(f"   {cell_label:<42}  {counts}")

    # --- Spot-check: stratified sample (2 fail_content + 2 pass + 1 fail_format) ---
    print(
        f"\n[spot-check] stratified sample from ({SPOTCHECK_MODEL}, {SPOTCHECK_LEVER}): "
        f"2 fail_content + 2 pass + 1 fail_format"
    )
    print(
        f"   {'prompt_id':<10} {'category':<18} {'tier_1':<14} "
        f"{'canon':>6}  {'response_snippet (≤150 chars)'}"
    )
    sample_spec = [("fail_content", 2), ("pass", 2), ("fail_format", 1)]
    for status, k in sample_spec:
        rows = list(
            conn.execute(SPOTCHECK_QUERY, (SPOTCHECK_MODEL, SPOTCHECK_LEVER, status, k))
        )
        if not rows:
            print(f"   (no rows with tier_1_status={status} in this cell)")
            continue
        for row in rows:
            snippet = (row["response_snippet"] or "").replace("\n", "⏎")
            print(
                f"   {row['prompt_id']:<10} {row['task_category']:<18} "
                f"{row['tier_1_status']:<14} "
                f"{row['canonical_score']:>6.3f}  {snippet!r}"
            )

    # --- Batch identity check across all 4 models ---
    print("\n[identity] baseline vs batch — are the pass sets identical per model?")
    identity_models = [
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini-2026-03-17",
    ]
    for model in identity_models:
        base_rows = list(conn.execute(IDENTITY_QUERY, (model, "baseline")))
        batch_rows = list(conn.execute(IDENTITY_QUERY, (model, "batch")))
        base_by_prompt = {(r["prompt_id"], r["task_category"]): r["tier_1_status"] for r in base_rows}
        batch_by_prompt = {(r["prompt_id"], r["task_category"]): r["tier_1_status"] for r in batch_rows}
        base_pass = {k for k, s in base_by_prompt.items() if s == "pass"}
        batch_pass = {k for k, s in batch_by_prompt.items() if s == "pass"}
        inter = base_pass & batch_pass
        only_base = sorted(base_pass - batch_pass)
        only_batch = sorted(batch_pass - base_pass)
        print(f"\n   {model}")
        print(
            f"      baseline pass: {len(base_pass)}    "
            f"batch pass: {len(batch_pass)}    intersection: {len(inter)}"
        )
        print(f"      pass in baseline but fail in batch: {len(only_base)}")
        for p in only_base[:3]:
            batch_status = batch_by_prompt.get(p, "?")
            print(f"         {p[0]} ({p[1]}) → batch_status={batch_status}")
        print(f"      pass in batch but fail in baseline: {len(only_batch)}")
        for p in only_batch[:3]:
            base_status = base_by_prompt.get(p, "?")
            print(f"         {p[0]} ({p[1]}) → baseline_status={base_status}")
        if base_pass == batch_pass:
            print("      → identical pass sets (bit-for-bit pass/fail-equivalent)")
        else:
            divergence = len(only_base) + len(only_batch)
            print(f"      → pass sets diverge ({divergence} per-prompt differences)")

    conn.close()

    # --- One-line summary for at-a-glance ---
    cheapest = min(matrix_rows, key=lambda r: r["mean_cost_usd"])
    highest_q = max(matrix_rows, key=lambda r: r["mean_canonical_score"])
    print(
        f"\n[summary] 1,280 tier-2 rows across 4 models × 4 levers; "
        f"cheapest cell = {cheapest['model']}/{cheapest['lever']} "
        f"@ ${cheapest['mean_cost_usd']:.6f} (canon={cheapest['mean_canonical_score']:.3f}); "
        f"highest-quality cell = {highest_q['model']}/{highest_q['lever']} "
        f"@ canon={highest_q['mean_canonical_score']:.3f} "
        f"(${highest_q['mean_cost_usd']:.6f}). "
        f"Wrote {OUT_CSV.relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
