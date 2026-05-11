"""Day 12 step 3 — Per-category breakdown + sub-analyses.

Read-only against data/results.db. Surfaces:

  1. Main category × lever table (4 task categories × 4 levers = 16 cells; not 5 —
     the 'extraction' category is Tier-1-only with no judge scoring and is
     excluded by canonical_score IS NOT NULL).
  2. Per-category lever ranking — does the global ranking hold per category?
  3. Compression-where-it-doesn't-fail check.
  4. Batch-quality-drop hot spots.
  5. Per-category Pareto frontier (16 (model, lever) points per category).
  6. Reasoning drill-down: full 4×4 (model × lever) canon + tier-1 matrices.
  7. Per-category cost sensitivity (cost spread within the near-best quality band).

Output: stdout summary + analysis/out/category_breakdown.csv.
"""

from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.db"
OUT_DIR = ROOT / "analysis" / "out"
OUT_CSV = OUT_DIR / "category_breakdown.csv"

CATEGORY_LEVER_QUERY = """
SELECT
    task_category,
    optimisation_lever AS lever,
    COUNT(*)                                                            AS n_rows,
    AVG(canonical_score)                                                AS mean_canonical,
    AVG(cost_usd)                                                       AS mean_cost_usd,
    SUM(CASE WHEN tier_1_status = 'pass' THEN 1 ELSE 0 END)             AS pass_count,
    SUM(CASE WHEN tier_1_status IN ('pass','fail_format','fail_content','truncated') THEN 1 ELSE 0 END) AS applicable_count
FROM results
WHERE canonical_score IS NOT NULL
GROUP BY task_category, optimisation_lever
ORDER BY task_category, optimisation_lever
"""

CATEGORY_MODEL_LEVER_QUERY = """
SELECT
    task_category,
    model,
    optimisation_lever AS lever,
    COUNT(*)             AS n_rows,
    AVG(canonical_score) AS mean_canonical,
    AVG(cost_usd)        AS mean_cost_usd,
    AVG(CASE WHEN tier_1_status = 'pass' THEN 1.0 ELSE 0.0 END) AS tier_1_pass_rate_all
FROM results
WHERE canonical_score IS NOT NULL
GROUP BY task_category, model, optimisation_lever
ORDER BY task_category, model, optimisation_lever
"""

LEVERS = ["baseline", "batch", "compression", "output_cap"]
MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
]


def is_dominator(p: dict, q: dict) -> bool:
    return (
        p["mean_canonical"] >= q["mean_canonical"]
        and p["mean_cost_usd"] <= q["mean_cost_usd"]
        and (
            p["mean_canonical"] > q["mean_canonical"]
            or p["mean_cost_usd"] < q["mean_cost_usd"]
        )
    )


def pareto_optimal(points: list[dict]) -> list[dict]:
    return [p for p in points if not any(is_dominator(o, p) for o in points if o is not p)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    cells = [dict(r) for r in conn.execute(CATEGORY_LEVER_QUERY)]
    by_cat: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in cells:
        # None means "no applicable rows" (e.g., summarisation, where Tier-1
        # doesn't define a deterministic check for free-text outputs).
        # Distinguish from 0.0, which would mean "all applicable rows failed".
        r["tier_1_pass_rate_applicable"] = (
            r["pass_count"] / r["applicable_count"]
            if r["applicable_count"] > 0
            else None
        )
        by_cat[r["task_category"]][r["lever"]] = r

    categories = sorted(by_cat.keys())
    print(f"[scope] {len(categories)} Tier-2 task categories: {categories}")
    print(
        "   note: 'extraction' is the 5th PRD category but is Tier-1-only "
        "(canonical_score IS NULL on all extraction rows). 4×4=16 cells, not 5×4=20."
    )

    # --- 1. Main category × lever table ---
    csv_rows = []
    print(
        f"\n[main] category × lever — 4 categories × 4 levers, "
        f"each cell aggregates 80 rows (20 prompts × 4 models)"
    )
    header = (
        f"{'category':<18} {'lever':<12} {'n':>4}  "
        f"{'canon':>6}  {'cost($)':>10}  {'t1_appl':>8}  {'Δcanon':>7}"
    )
    print(header)
    print("-" * len(header))
    last_cat = None
    for cat in categories:
        base = by_cat[cat]["baseline"]
        for lever in LEVERS:
            r = by_cat[cat][lever]
            delta_canon = r["mean_canonical"] - base["mean_canonical"]
            if last_cat and cat != last_cat:
                print()
            last_cat = cat
            marker = "  ←base" if lever == "baseline" else ""
            t1_display = (
                f"{r['tier_1_pass_rate_applicable']:>8.3f}"
                if r["tier_1_pass_rate_applicable"] is not None
                else f"{'n/a':>8}"
            )
            print(
                f"{cat:<18} {lever:<12} {r['n_rows']:>4}  "
                f"{r['mean_canonical']:>6.3f}  {r['mean_cost_usd']:>10.6f}  "
                f"{t1_display}  "
                f"{delta_canon:>+7.3f}{marker}"
            )
            csv_rows.append(
                {
                    "task_category": cat,
                    "lever": lever,
                    "n_rows": r["n_rows"],
                    "mean_canonical_score": round(r["mean_canonical"], 4),
                    "mean_cost_usd": round(r["mean_cost_usd"], 6),
                    "tier_1_pass_rate_applicable": (
                        round(r["tier_1_pass_rate_applicable"], 4)
                        if r["tier_1_pass_rate_applicable"] is not None
                        else "n/a"
                    ),
                    "delta_canonical_vs_category_baseline": round(delta_canon, 4),
                }
            )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # --- 2. Per-category lever ranking ---
    # Global ranking from step 1 is: baseline > batch > output_cap > compression
    # (by canonical_score, averaged across all models).
    GLOBAL_RANK = ("baseline", "batch", "output_cap", "compression")
    print(f"\n[lever-rank] global lever quality ranking: {' > '.join(GLOBAL_RANK)}")
    print(f"   per-category ranking (sorted by mean_canonical desc):")
    for cat in categories:
        ranked = sorted(LEVERS, key=lambda L: -by_cat[cat][L]["mean_canonical"])
        deltas = [
            f"{L}({by_cat[cat][L]['mean_canonical']:.3f})" for L in ranked
        ]
        flip = "" if tuple(ranked) == GLOBAL_RANK else "  ← FLIP vs global"
        print(f"   {cat:<18} {' > '.join(deltas)}{flip}")

    # --- 3. Compression-where-it-doesn't-destroy-quality ---
    print(f"\n[compression] cells where compression is competitive (Δcanon > -0.05 vs baseline):")
    found_compression_safe = False
    for cat in categories:
        base = by_cat[cat]["baseline"]
        comp = by_cat[cat]["compression"]
        delta = comp["mean_canonical"] - base["mean_canonical"]
        if delta > -0.05:
            print(
                f"   {cat:<18} Δcanon={delta:+.3f}  (compression={comp['mean_canonical']:.3f}, "
                f"baseline={base['mean_canonical']:.3f})  "
                f"cost saving = {(1 - comp['mean_cost_usd']/base['mean_cost_usd']):.0%}"
            )
            found_compression_safe = True
    if not found_compression_safe:
        print("   (none) — compression destroys quality in every category at the aggregate level.")
    print("\n   compression Δcanon by category (for reference):")
    for cat in categories:
        base = by_cat[cat]["baseline"]
        comp = by_cat[cat]["compression"]
        delta = comp["mean_canonical"] - base["mean_canonical"]
        print(f"   {cat:<18} Δcanon={delta:+.3f}")

    # --- 4. Batch quality drop hot spots ---
    print(
        f"\n[batch] cells where batch quality drop is larger than the global ~0–2% pattern "
        f"(Δcanon < -0.02):"
    )
    found_batch_drop = False
    for cat in categories:
        base = by_cat[cat]["baseline"]
        bat = by_cat[cat]["batch"]
        delta = bat["mean_canonical"] - base["mean_canonical"]
        if delta < -0.02:
            print(
                f"   {cat:<18} Δcanon={delta:+.3f}  "
                f"(batch={bat['mean_canonical']:.3f}, baseline={base['mean_canonical']:.3f})"
            )
            found_batch_drop = True
    if not found_batch_drop:
        print("   (none) — batch within ±0.02 in all 4 categories.")
    print("\n   batch Δcanon by category (for reference):")
    for cat in categories:
        base = by_cat[cat]["baseline"]
        bat = by_cat[cat]["batch"]
        delta = bat["mean_canonical"] - base["mean_canonical"]
        print(f"   {cat:<18} Δcanon={delta:+.3f}")

    # --- 5. Widest spread between best and worst lever ---
    print("\n[spread] best-lever vs worst-lever canon gap per category:")
    spreads = []
    for cat in categories:
        canons = [by_cat[cat][L]["mean_canonical"] for L in LEVERS]
        spread = max(canons) - min(canons)
        spreads.append((cat, spread, max(canons), min(canons)))
    for cat, spread, mx, mn in sorted(spreads, key=lambda x: -x[1]):
        print(f"   {cat:<18} spread={spread:.3f}  (best={mx:.3f}, worst={mn:.3f})")

    # --- 6. Per-category Pareto frontier ---
    cell_rows = [dict(r) for r in conn.execute(CATEGORY_MODEL_LEVER_QUERY)]
    by_cat_points: dict[str, list[dict]] = defaultdict(list)
    for r in cell_rows:
        by_cat_points[r["task_category"]].append(r)

    print("\n[per-cat pareto] (model, lever) Pareto frontier within each category:")
    for cat in categories:
        pts = by_cat_points[cat]
        pareto = pareto_optimal(pts)
        models_on_frontier = {p["model"] for p in pareto}
        is_anthropic_present = any("claude" in m for m in models_on_frontier)
        is_openai_present = any("gpt" in m for m in models_on_frontier)
        provider_summary = []
        if is_anthropic_present:
            provider_summary.append("Anthropic")
        if is_openai_present:
            provider_summary.append("OpenAI")
        print(
            f"\n   {cat}: {len(pareto)} of {len(pts)} cells Pareto-optimal — "
            f"providers on frontier: {', '.join(provider_summary)}"
        )
        for p in sorted(pareto, key=lambda p: p["mean_canonical"]):
            print(
                f"      {p['model']:<28} {p['lever']:<12} "
                f"canon={p['mean_canonical']:.3f}  cost=${p['mean_cost_usd']:.6f}"
            )

    # --- 7. Reasoning drill-down ---
    print("\n[reasoning] 4×4 (model × lever) drill-down — canonical_score:")
    reasoning_pts = {(p["model"], p["lever"]): p for p in by_cat_points["reasoning"]}
    print(f"   {'model':<28} " + "  ".join(f"{L:>11}" for L in LEVERS))
    for m in MODELS:
        cells_str = "  ".join(
            f"{reasoning_pts[(m, L)]['mean_canonical']:>11.3f}" for L in LEVERS
        )
        print(f"   {m:<28} {cells_str}")

    print("\n[reasoning] 4×4 (model × lever) drill-down — tier_1_pass_rate (all denominator):")
    print(f"   {'model':<28} " + "  ".join(f"{L:>11}" for L in LEVERS))
    for m in MODELS:
        cells_str = "  ".join(
            f"{reasoning_pts[(m, L)]['tier_1_pass_rate_all']:>11.3f}" for L in LEVERS
        )
        print(f"   {m:<28} {cells_str}")

    # Compare GPT-5.4 dominance across categories
    print("\n[gpt-5.4 dominance] per-category: is gpt-5.4 baseline the highest-canon cell?")
    for cat in categories:
        best = max(by_cat_points[cat], key=lambda p: p["mean_canonical"])
        gpt54_baseline = next(
            p for p in by_cat_points[cat]
            if p["model"] == "gpt-5.4-2026-03-05" and p["lever"] == "baseline"
        )
        is_top = best is gpt54_baseline
        marker = "✓" if is_top else " "
        print(
            f"   {marker} {cat:<18} best = {best['model']}/{best['lever']} "
            f"(canon={best['mean_canonical']:.3f}); "
            f"gpt-5.4/baseline canon={gpt54_baseline['mean_canonical']:.3f}"
        )

    # --- 8. Per-category cost sensitivity ---
    print(
        "\n[cost-sensitivity] within each category, cost spread among cells at ≥90% of "
        "category-best canon (high spread = many cheap alternatives exist):"
    )
    for cat in categories:
        pts = by_cat_points[cat]
        best_canon = max(p["mean_canonical"] for p in pts)
        threshold = 0.9 * best_canon
        near_best = [p for p in pts if p["mean_canonical"] >= threshold]
        if not near_best:
            continue
        min_cost = min(p["mean_cost_usd"] for p in near_best)
        max_cost = max(p["mean_cost_usd"] for p in near_best)
        ratio = max_cost / min_cost if min_cost > 0 else float("inf")
        cheapest = min(near_best, key=lambda p: p["mean_cost_usd"])
        most_exp = max(near_best, key=lambda p: p["mean_cost_usd"])
        print(
            f"   {cat:<18} {len(near_best):>2} near-best cells; "
            f"cost ratio {ratio:.1f}× "
            f"(cheapest: {cheapest['model']}/{cheapest['lever']} @ ${min_cost:.6f}; "
            f"most exp: {most_exp['model']}/{most_exp['lever']} @ ${max_cost:.6f})"
        )

    conn.close()
    print(f"\n[done] wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
