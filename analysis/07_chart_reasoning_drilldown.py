"""Day 13 chart (c) — Reasoning task drill-down grouped bar chart.

Reads analysis/out/category_model_lever.csv, filters to task_category=reasoning,
renders 16 bars (4 model groups × 4 lever bars per group) showing canonical_score.

Design choice: colour BY LEVER (not by provider). Provider-colour + lever-pattern
was the alternative; lever-colour produces cleaner visual separation when bars
are grouped by model. The Anthropic-vs-OpenAI story is still visible because
the compression bars are uniformly shorter in the Anthropic groups (the
gap-widening narrative the footer describes).

No secondary y-axis. Tier-1 pass rate was a candidate overlay but would
duplicate the canon story (both metrics drop on the same cells). Deferred to
a small standalone Tier-1 chart if needed in the writeup.

Output: analysis/out/charts/reasoning_drilldown.{png,svg}.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "analysis" / "out" / "category_model_lever.csv"
OUT_DIR = ROOT / "analysis" / "out" / "charts"

MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
]
LEVERS = ["baseline", "batch", "compression", "output_cap"]

MODEL_LABEL = {
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
    "gpt-5.4-2026-03-05": "gpt-5.4",
    "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
}
LEVER_LABEL = {
    "baseline": "Baseline (sync)",
    "batch": "Batch",
    "compression": "Compression",
    "output_cap": "Output cap",
}

# Lever colour mapping. Semantic intent: baseline = neutral reference,
# batch = cool/favored, compression = warning red, output_cap = warm moderate.
# Chosen for max within-group contrast (4 distinct hues) and intuitive read.
LEVER_COLOR = {
    "baseline": "#4A5568",     # slate-700, neutral reference
    "batch": "#0072B2",        # Wong blue
    "compression": "#D55E00",  # Wong vermillion (warning)
    "output_cap": "#E69F00",   # Wong orange (moderate)
}


def load_reasoning() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["task_category"] != "reasoning":
                continue
            out[(r["model"], r["lever"])] = float(r["mean_canonical_score"])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_reasoning()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    n_models = len(MODELS)
    n_levers = len(LEVERS)
    bar_width = 0.20
    group_centers = np.arange(n_models, dtype=float)

    for j, lever in enumerate(LEVERS):
        offsets = group_centers + (j - (n_levers - 1) / 2) * bar_width
        values = [data[(m, lever)] for m in MODELS]
        ax.bar(
            offsets,
            values,
            bar_width,
            label=LEVER_LABEL[lever],
            color=LEVER_COLOR[lever],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    # Axes
    ax.set_ylim(0.55, 1.00)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=10.5)
    ax.set_ylabel("Canonical score (mean across 20 reasoning prompts per cell)",
                  fontsize=10.5)
    ax.set_xlabel("Model", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    # Title
    ax.set_title(
        "Reasoning task: canonical score by model × lever (n=20 prompts per cell)",
        fontsize=12,
        pad=10,
    )

    # Legend in lower-right — the gpt-5.4-mini group is rightmost and its
    # shortest bar (compression at 0.761) leaves the y<0.72 band empty in
    # that x region. Same "legend lives where data isn't" pattern as chart (a).
    ax.legend(
        loc="lower right",
        framealpha=0.92,
        fontsize=9.5,
        title="Lever",
        title_fontsize=9.5,
        ncol=2,
    )

    # Annotations for the extremes.
    # Highest: gpt-5.4 / baseline (model idx=2, lever idx=0)
    highest_x = group_centers[2] + (0 - (n_levers - 1) / 2) * bar_width
    highest_y = data[("gpt-5.4-2026-03-05", "baseline")]
    ax.annotate(
        f"Highest reasoning cell:\ngpt-5.4 baseline ({highest_y:.3f})",
        xy=(highest_x, highest_y),
        xytext=(2.55, 0.985),
        fontsize=9.5,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.8),
    )

    # Lowest: haiku / compression (model idx=1, lever idx=2)
    lowest_x = group_centers[1] + (2 - (n_levers - 1) / 2) * bar_width
    lowest_y = data[("claude-haiku-4-5", "compression")]
    ax.annotate(
        f"Lowest reasoning cell:\nhaiku compression ({lowest_y:.3f})",
        xy=(lowest_x, lowest_y),
        xytext=(1.35, 0.69),
        fontsize=9.5,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.8),
    )

    # Footer
    fig.text(
        0.012,
        0.005,
        (
            "Anthropic models (sonnet, haiku) show wider lever-induced degradation than OpenAI on "
            "this category. Sonnet baseline trails GPT-5.4 baseline by 0.027; under compression, "
            "gap widens to 0.093."
        ),
        fontsize=8.5,
        ha="left",
        style="italic",
        color="dimgray",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    png = OUT_DIR / "reasoning_drilldown.png"
    svg = OUT_DIR / "reasoning_drilldown.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)

    print(f"[wrote] {png.relative_to(ROOT)} ({png.stat().st_size:,} bytes)")
    print(f"[wrote] {svg.relative_to(ROOT)} ({svg.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
