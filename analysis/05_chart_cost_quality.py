"""Day 13 chart (a) — Cost-vs-quality scatter, headline figure.

Reads analysis/out/cost_quality_scatter.csv (not the DB — charts regenerate
from CSVs only, so the scatter survives data-collection changes via 02_*.py).
Renders 16 (model, lever) points coloured by provider, shaped by lever, with
the Pareto frontier overlaid as a dashed connecting line.

Output: analysis/out/charts/cost_quality_scatter.{png,svg}.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "analysis" / "out" / "cost_quality_scatter.csv"
OUT_DIR = ROOT / "analysis" / "out" / "charts"

# Wong colour-blind safe palette (Nature Methods 2011): distinguishable under
# protanopia/deuteranopia/tritanopia. Orange + blue is the safest pair.
COLOR_ANTHROPIC = "#E69F00"  # warm orange
COLOR_OPENAI = "#0072B2"  # deep blue

LEVER_MARKER = {
    "baseline": "o",  # circle
    "batch": "s",  # square
    "compression": "v",  # triangle-down
    "output_cap": "D",  # diamond
}
LEVER_LABEL = {
    "baseline": "Baseline (sync)",
    "batch": "Batch",
    "compression": "Compression",
    "output_cap": "Output cap",
}


def provider_of(model: str) -> str:
    if model.startswith("claude"):
        return "Anthropic"
    if model.startswith("gpt"):
        return "OpenAI"
    raise ValueError(f"unknown provider for model {model!r}")


def friendly_model(m: str) -> str:
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", m)


def load_rows() -> list[dict]:
    with CSV.open(newline="") as f:
        return [
            {
                **r,
                "mean_canonical_score": float(r["mean_canonical_score"]),
                "mean_cost_usd": float(r["mean_cost_usd"]),
                "pareto_optimal": int(r["pareto_optimal"]),
            }
            for r in csv.DictReader(f)
        ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Pareto frontier first so points draw on top.
    pareto_sorted = sorted(
        (r for r in rows if r["pareto_optimal"] == 1),
        key=lambda r: r["mean_cost_usd"],
    )
    ax.plot(
        [p["mean_cost_usd"] for p in pareto_sorted],
        [p["mean_canonical_score"] for p in pareto_sorted],
        linestyle="--",
        color="black",
        linewidth=1.2,
        alpha=0.55,
        zorder=2,
    )

    # 16 data points
    for r in rows:
        provider = provider_of(r["model"])
        color = COLOR_ANTHROPIC if provider == "Anthropic" else COLOR_OPENAI
        ax.scatter(
            r["mean_cost_usd"],
            r["mean_canonical_score"],
            s=140,
            c=color,
            marker=LEVER_MARKER[r["lever"]],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            alpha=0.95,
        )

    # Annotations for the two headline-scalar cells.
    cheap94 = next(
        r for r in rows
        if r["model"] == "gpt-5.4-mini-2026-03-17" and r["lever"] == "batch"
    )
    frontier = next(
        r for r in rows
        if r["model"] == "gpt-5.4-2026-03-05" and r["lever"] == "baseline"
    )
    # cheap94: text south-east of the marker, short arrow up-left to the point.
    # Empty space exists in the band y in [0.78, 0.83] at x in [0.0007, 0.0009]
    # — between gpt-5.4-mini output_cap (0.000922, 0.857) and gpt-5.4-mini
    # compression (0.000929, 0.685). Arrow path stays clear of all markers.
    ax.annotate(
        "94% of frontier quality\n@ 13% of frontier cost",
        xy=(cheap94["mean_cost_usd"], cheap94["mean_canonical_score"]),
        xytext=(0.00072, 0.78),
        fontsize=9.5,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.8),
    )
    # Frontier label: nudged inward from the right edge so it sits cleanly
    # within the plot area (was at x=0.0048, y=0.985 — too close to border).
    ax.annotate(
        "Frontier quality",
        xy=(frontier["mean_cost_usd"], frontier["mean_canonical_score"]),
        xytext=(0.0040, 0.975),
        fontsize=9.5,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.8),
    )

    # Axes
    ax.set_xscale("log")
    ax.set_xlim(0.0005, 0.006)
    ax.set_ylim(0.55, 1.0)  # was 0.60 — captures haiku compression at 0.587
    ax.set_xlabel("Cost per task (USD, log scale)", fontsize=10.5)
    ax.set_ylabel(
        "Canonical score (Tier-2, mean across 80 prompts per cell)", fontsize=10.5
    )

    # Explicit major ticks. Default log-scale ticking only shows one label
    # because the data spans <1 decade ($0.0005-$0.006).
    from matplotlib.ticker import FixedLocator

    def _fmt_dollar(v: float, _pos) -> str:
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return f"${s}"

    ax.xaxis.set_major_locator(FixedLocator([0.0005, 0.001, 0.002, 0.005]))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_dollar))
    ax.xaxis.set_minor_locator(FixedLocator([]))  # suppress minor ticks
    ax.tick_params(axis="both", labelsize=9.5)

    ax.set_title(
        "Cost vs Quality across 4 models × 4 levers (Tier-2 canonical scores, 1,280 rows)",
        fontsize=12,
        pad=10,
    )

    # Legend: provider colour + lever shape + Pareto line.
    provider_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ANTHROPIC,
               markeredgecolor="black", markersize=10, label="Anthropic"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OPENAI,
               markeredgecolor="black", markersize=10, label="OpenAI"),
    ]
    lever_handles = [
        Line2D([0], [0], marker=LEVER_MARKER[L], color="w", markerfacecolor="lightgray",
               markeredgecolor="black", markersize=10, label=LEVER_LABEL[L])
        for L in ("baseline", "batch", "compression", "output_cap")
    ]
    pareto_handle = [
        Line2D([0], [0], linestyle="--", color="black", alpha=0.55,
               label="Pareto frontier"),
    ]
    ax.legend(
        handles=provider_handles + lever_handles + pareto_handle,
        loc="lower left",
        framealpha=0.92,
        fontsize=9,
        title="Provider / Lever",
        title_fontsize=9.5,
    )

    ax.grid(True, alpha=0.25, zorder=0)

    # Footer note (single line; wraps in writeup if needed).
    fig.text(
        0.012, 0.005,
        ("Pareto frontier in dashed line. Anthropic Sonnet/baseline scored 0.993 on RAG "
         "specifically (not shown on aggregate plot — see per-category chart)."),
        fontsize=8.5,
        ha="left",
        style="italic",
        color="dimgray",
    )

    fig.tight_layout(rect=(0, 0.035, 1, 1))

    png = OUT_DIR / "cost_quality_scatter.png"
    svg = OUT_DIR / "cost_quality_scatter.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)

    print(f"[wrote] {png.relative_to(ROOT)} ({png.stat().st_size:,} bytes)")
    print(f"[wrote] {svg.relative_to(ROOT)} ({svg.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
