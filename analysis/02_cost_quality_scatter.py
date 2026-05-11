"""Day 12 step 2 — Cost-quality scatter data + Pareto frontier.

Read-only against data/results.db. Surfaces the (cost, quality) pairs that drive
the eventual scatter plot, identifies the Pareto frontier, and surfaces two
headline scalars: highest-quality cell and cheapest-cell-at-≥90%-of-best-quality.

Pareto definition used: point P dominates point Q iff
    canon(P) ≥ canon(Q) AND cost(P) ≤ cost(Q)
    AND (canon(P) > canon(Q) OR cost(P) < cost(Q))
A point is Pareto-optimal iff no other point dominates it.

Output: stdout summary + analysis/out/cost_quality_scatter.csv.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.db"
OUT_DIR = ROOT / "analysis" / "out"
OUT_CSV = OUT_DIR / "cost_quality_scatter.csv"

QUERY = """
SELECT
    model,
    optimisation_lever AS lever,
    COUNT(*)             AS n_rows,
    AVG(canonical_score) AS mean_canonical,
    AVG(cost_usd)        AS mean_cost_usd
FROM results
WHERE canonical_score IS NOT NULL
GROUP BY model, optimisation_lever
ORDER BY mean_cost_usd
"""

QUALITY_THRESHOLD = 0.90  # fraction of best quality for the "cheapest near-best" scalar


def is_dominator(p: dict, q: dict) -> bool:
    """p dominates q: at least as good on both, strictly better on at least one."""
    return (
        p["mean_canonical"] >= q["mean_canonical"]
        and p["mean_cost_usd"] <= q["mean_cost_usd"]
        and (
            p["mean_canonical"] > q["mean_canonical"]
            or p["mean_cost_usd"] < q["mean_cost_usd"]
        )
    )


def best_dominator(q: dict, points: list[dict]) -> dict | None:
    """Return the dominator with the highest canon (the 'strongest upgrade')."""
    dominators = [p for p in points if p is not q and is_dominator(p, q)]
    if not dominators:
        return None
    return max(dominators, key=lambda p: (p["mean_canonical"], -p["mean_cost_usd"]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    points = [dict(r) for r in conn.execute(QUERY)]
    conn.close()

    # Identify Pareto-optimal points
    for p in points:
        p["pareto_optimal"] = not any(is_dominator(o, p) for o in points if o is not p)

    pareto = [p for p in points if p["pareto_optimal"]]
    dominated = [p for p in points if not p["pareto_optimal"]]

    # --- CSV ---
    csv_rows = []
    for p in points:
        dom = best_dominator(p, points) if not p["pareto_optimal"] else None
        csv_rows.append(
            {
                "model": p["model"],
                "lever": p["lever"],
                "n_rows": p["n_rows"],
                "mean_canonical_score": round(p["mean_canonical"], 4),
                "mean_cost_usd": round(p["mean_cost_usd"], 6),
                "pareto_optimal": int(p["pareto_optimal"]),
                "dominated_by_model": dom["model"] if dom else "",
                "dominated_by_lever": dom["lever"] if dom else "",
            }
        )
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # --- 16-row table with Pareto flag ---
    print("[scatter] 16 (model, lever) points sorted by cost ascending")
    header = (
        f"{'model':<28} {'lever':<12} {'n':>4}  "
        f"{'canon':>6}  {'cost($)':>10}  {'pareto':>7}"
    )
    print(header)
    print("-" * len(header))
    for p in points:
        flag = "  ★" if p["pareto_optimal"] else ""
        print(
            f"{p['model']:<28} {p['lever']:<12} {p['n_rows']:>4}  "
            f"{p['mean_canonical']:>6.3f}  {p['mean_cost_usd']:>10.6f}  "
            f"{'YES' if p['pareto_optimal'] else 'no':>7}{flag}"
        )

    # --- Pareto frontier (ordered by canon ascending; cost also ascending by construction) ---
    print(f"\n[pareto frontier] {len(pareto)} of 16 points")
    print(
        f"   {'model':<28} {'lever':<12} {'canon':>6}  {'cost($)':>10}  {'canon/best':>10}  {'cost/best_q':>11}"
    )
    best_q = max(p["mean_canonical"] for p in points)
    best_q_cost = max(p["mean_cost_usd"] for p in pareto)  # cost of the highest-canon Pareto point
    best_q_point = max(points, key=lambda p: p["mean_canonical"])
    for p in sorted(pareto, key=lambda p: p["mean_canonical"]):
        canon_pct = p["mean_canonical"] / best_q
        cost_vs_top = p["mean_cost_usd"] / best_q_point["mean_cost_usd"]
        print(
            f"   {p['model']:<28} {p['lever']:<12} "
            f"{p['mean_canonical']:>6.3f}  {p['mean_cost_usd']:>10.6f}  "
            f"{canon_pct:>10.1%}  {cost_vs_top:>11.1%}"
        )

    # --- Dominated points: who dominates them ---
    print(f"\n[dominated points] {len(dominated)} of 16 points")
    print("   Shown with the strongest dominator (highest-canon point that beats them on both axes).")
    for p in sorted(dominated, key=lambda p: -p["mean_canonical"]):
        dom = best_dominator(p, points)
        assert dom is not None  # by construction
        delta_canon = dom["mean_canonical"] - p["mean_canonical"]
        cost_saving = 1.0 - dom["mean_cost_usd"] / p["mean_cost_usd"]
        print(
            f"   {p['model']:<28} {p['lever']:<12} "
            f"(canon={p['mean_canonical']:.3f}, ${p['mean_cost_usd']:.6f})  "
            f"← dominated by {dom['model']}/{dom['lever']} "
            f"(canon={dom['mean_canonical']:.3f} [+{delta_canon:.3f}], "
            f"${dom['mean_cost_usd']:.6f} [{cost_saving:.0%} cheaper])"
        )

    # --- Two headline scalars ---
    threshold = QUALITY_THRESHOLD * best_q
    near_best = [p for p in points if p["mean_canonical"] >= threshold]
    cheapest_near_best = min(near_best, key=lambda p: p["mean_cost_usd"])
    print("\n[headlines]")
    print(
        f"   highest quality at any cost: "
        f"{best_q_point['model']} / {best_q_point['lever']} — "
        f"canon={best_q_point['mean_canonical']:.3f}, "
        f"cost=${best_q_point['mean_cost_usd']:.6f}"
    )
    print(
        f"   cheapest at ≥{QUALITY_THRESHOLD:.0%} of best quality "
        f"(canon ≥ {threshold:.3f}): "
        f"{cheapest_near_best['model']} / {cheapest_near_best['lever']} — "
        f"canon={cheapest_near_best['mean_canonical']:.3f} "
        f"({cheapest_near_best['mean_canonical']/best_q:.1%} of best), "
        f"cost=${cheapest_near_best['mean_cost_usd']:.6f} "
        f"({cheapest_near_best['mean_cost_usd']/best_q_point['mean_cost_usd']:.1%} of best-quality cost)"
    )

    print(f"\n[summary] {len(pareto)} Pareto-optimal cells, {len(dominated)} dominated; "
          f"wrote {OUT_CSV.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
