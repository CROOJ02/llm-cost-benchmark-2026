"""Day 13 chart (d) — Cost-sensitivity strip plot, per task category.

Reads analysis/out/category_model_lever.csv. Renders 4 horizontal strips (one
per task category) stacked vertically. Each strip plots its 16 (model, lever)
cells as dots — provider colour, lever shape, both consistent with chart (a).

The shaded band on each strip marks the near-best region (canonical_score
≥ 0.9 × category-best). The narrative: in CS, RAG, and reasoning the near-best
band is wide on the cost axis (cheap cells reach near-best quality); in
summarisation the band is narrow and clustered to the right (you pay for
near-best quality on this task).

Output: analysis/out/charts/cost_sensitivity.{png,svg}.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "analysis" / "out" / "category_model_lever.csv"
OUT_DIR = ROOT / "analysis" / "out" / "charts"

CATEGORIES = ["customer_support", "rag_qa", "reasoning", "summarisation"]
CATEGORY_LABEL = {
    "customer_support": "Customer support",
    "rag_qa": "RAG QA",
    "reasoning": "Reasoning",
    "summarisation": "Summarisation",
}

# Same palette / shape mapping as chart (a) for cross-chart consistency.
COLOR_ANTHROPIC = "#E69F00"
COLOR_OPENAI = "#0072B2"
LEVER_MARKER = {
    "baseline": "o",
    "batch": "s",
    "compression": "v",
    "output_cap": "D",
}
LEVER_LABEL = {
    "baseline": "Baseline (sync)",
    "batch": "Batch",
    "compression": "Compression",
    "output_cap": "Output cap",
}

NEAR_BEST_FRAC = 0.9
NEAR_BEST_FILL = "#A8D8A8"  # soft green band fill (alpha applied separately)
NEAR_BEST_ALPHA = 0.22


def provider_of(model: str) -> str:
    return "Anthropic" if model.startswith("claude") else "OpenAI"


def friendly_model(m: str) -> str:
    import re
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", m)


def load_points() -> list[dict]:
    out: list[dict] = []
    with CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            out.append({
                "category": r["task_category"],
                "model": r["model"],
                "lever": r["lever"],
                "canon": float(r["mean_canonical_score"]),
                "cost": float(r["mean_cost_usd"]),
            })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = load_points()

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, sharey=True)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        by_cat[p["category"]].append(p)

    for ax, cat in zip(axes, CATEGORIES):
        cat_points = by_cat[cat]
        best_canon = max(p["canon"] for p in cat_points)
        threshold = NEAR_BEST_FRAC * best_canon

        # Shade near-best band (full strip width, y from threshold to top).
        ax.axhspan(
            threshold, 1.05,
            facecolor=NEAR_BEST_FILL, alpha=NEAR_BEST_ALPHA, zorder=0,
        )

        # Plot 16 dots
        for p in cat_points:
            color = COLOR_ANTHROPIC if provider_of(p["model"]) == "Anthropic" else COLOR_OPENAI
            ax.scatter(
                p["cost"], p["canon"],
                s=110, c=color, marker=LEVER_MARKER[p["lever"]],
                edgecolor="black", linewidth=0.6, alpha=0.92, zorder=3,
            )

        # Annotate cheapest near-best cell. Position text immediately adjacent
        # to the cell — short arrow only. xytext is in DATA coords with
        # multiplicative x-offset (log axis) and additive y-offset below the cell.
        # Multiplier 1.5 ≈ 0.18 decades ≈ ~9% of strip width.
        near_best = [p for p in cat_points if p["canon"] >= threshold]
        cheapest = min(near_best, key=lambda p: p["cost"])
        label_text = (
            f"Cheapest near-best:\n{friendly_model(cheapest['model'])} "
            f"{cheapest['lever']} (${cheapest['cost']:.6f}, canon {cheapest['canon']:.3f})"
        )
        ax.annotate(
            label_text,
            xy=(cheapest["cost"], cheapest["canon"]),
            xytext=(cheapest["cost"] * 1.5, cheapest["canon"] - 0.085),
            fontsize=9,
            ha="left",
            va="top",
            arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.88),
        )

        # Strip label in upper-left of axes
        n_near = len(near_best)
        cost_ratio = (
            max(p["cost"] for p in near_best) / min(p["cost"] for p in near_best)
        )
        ax.text(
            0.012, 0.93,
            f"{CATEGORY_LABEL[cat]}  —  {n_near} near-best cells, {cost_ratio:.1f}× cost spread",
            transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="top",
        )

        # Near-best threshold annotation on the right edge
        ax.axhline(threshold, linestyle=":", linewidth=0.7, color="darkgreen",
                   alpha=0.5, zorder=1)

        ax.grid(True, alpha=0.22, zorder=0)
        ax.set_axisbelow(True)

    # X axis: shared log scale covering all data ($0.000170 to $0.011302)
    axes[-1].set_xscale("log")
    axes[-1].set_xlim(0.00015, 0.013)
    # Explicit ticks at meaningful cost points
    tick_values = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]

    def _fmt_dollar(v: float, _pos) -> str:
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return f"${s}"

    axes[-1].xaxis.set_major_locator(FixedLocator(tick_values))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_fmt_dollar))
    axes[-1].xaxis.set_minor_locator(FixedLocator([]))
    axes[-1].set_xlabel("Cost per task (USD, log scale)", fontsize=10.5)

    # Y axis: shared 0.35-1.05 to capture the lowest cell (RAG compression
    # ~0.396) and leave headroom above 1.0 for the band shading.
    axes[0].set_ylim(0.35, 1.05)
    for ax in axes:
        ax.set_ylabel("Canon", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    # Suptitle
    fig.suptitle(
        "Cost-sensitivity by task category — how much cost spread exists at near-best quality?",
        fontsize=12.5, y=0.985,
    )

    # Figure-level legend at the bottom
    provider_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ANTHROPIC,
               markeredgecolor="black", markersize=10, label="Anthropic"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OPENAI,
               markeredgecolor="black", markersize=10, label="OpenAI"),
    ]
    lever_handles = [
        Line2D([0], [0], marker=LEVER_MARKER[L], color="w",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=10, label=LEVER_LABEL[L])
        for L in ("baseline", "batch", "compression", "output_cap")
    ]
    band_handle = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=NEAR_BEST_FILL, markeredgecolor="darkgreen",
               markersize=12, label="Near-best band (≥90% of category best)"),
    ]
    fig.legend(
        handles=provider_handles + lever_handles + band_handle,
        loc="lower center",
        ncol=7,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
    )

    fig.text(
        0.012, 0.005,
        ("Near-best band (shaded) = ≥90% of best canon in that category. Customer support, RAG, "
         "and reasoning each have ~10 near-best cells with 9–14× cost spread. Summarisation has "
         "only 4 near-best cells in a 2.2× cost band."),
        fontsize=8.5, ha="left", style="italic", color="dimgray",
    )

    fig.subplots_adjust(top=0.94, bottom=0.115, hspace=0.18, left=0.07, right=0.97)

    png = OUT_DIR / "cost_sensitivity.png"
    svg = OUT_DIR / "cost_sensitivity.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)

    print(f"[wrote] {png.relative_to(ROOT)} ({png.stat().st_size:,} bytes)")
    print(f"[wrote] {svg.relative_to(ROOT)} ({svg.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
