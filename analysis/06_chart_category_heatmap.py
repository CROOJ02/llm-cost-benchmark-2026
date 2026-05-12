"""Day 13 chart (b) — Per-category model × lever heatmap.

Reads analysis/out/category_model_lever.csv (emitted by 03_category_breakdown.py
alongside category_breakdown.csv). Renders a 2×2 grid of subplots, one per
Tier-2 task category, each showing canonical_score for the 16 (model, lever)
cells in that category. Cell colour: diverging RdYlGn_r colormap; cell text:
canonical_score to 3 decimals.

Output: analysis/out/charts/category_heatmap.{png,svg}.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "analysis" / "out" / "category_model_lever.csv"
OUT_DIR = ROOT / "analysis" / "out" / "charts"

# Inherit conventions from chart (a):
#   - Friendly model names in displayed prose
#   - Same colour-blind awareness (RdYlGn_r tested OK under deuteranopia)
MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5",
          "gpt-5.4-2026-03-05", "gpt-5.4-mini-2026-03-17"]
LEVERS = ["baseline", "batch", "compression", "output_cap"]
CATEGORIES = ["customer_support", "rag_qa", "reasoning", "summarisation"]

MODEL_LABEL = {
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
    "gpt-5.4-2026-03-05": "gpt-5.4",
    "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
}
LEVER_LABEL = {
    "baseline": "baseline",
    "batch": "batch",
    "compression": "compression",
    "output_cap": "output_cap",
}
CATEGORY_LABEL = {
    "customer_support": "Customer support",
    "rag_qa": "RAG QA",
    "reasoning": "Reasoning",
    "summarisation": "Summarisation",
}

# Diverging colormap clipped to a data-driven range for better contrast.
# Actual data span is roughly 0.49–0.99, so a vmin of 0.45 maps red to the
# worst observed cell and green to the best. The text labels carry the exact
# value — colour exists to drive at-a-glance pattern recognition.
VMIN = 0.45
VMAX = 1.0
CMAP = "RdYlGn"  # mpl name — we want red=low, green=high (default direction)


def load_cells() -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    with CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            out[(r["task_category"], r["model"], r["lever"])] = float(
                r["mean_canonical_score"]
            )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, cat in enumerate(CATEGORIES):
        ax = axes[idx]
        matrix = [
            [cells[(cat, m, L)] for L in LEVERS]
            for m in MODELS
        ]
        im = ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")

        # Cell text labels
        for i, _m in enumerate(MODELS):
            for j, _L in enumerate(LEVERS):
                v = matrix[i][j]
                # Pick text colour for contrast against the cell colour.
                # Cells near the colormap midpoint (yellow) get black; very
                # red or very green cells get white for contrast.
                norm = (v - VMIN) / (VMAX - VMIN)
                txt_color = "white" if (norm < 0.2 or norm > 0.8) else "black"
                ax.text(
                    j, i, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="medium",
                )

        ax.set_xticks(range(len(LEVERS)))
        ax.set_xticklabels([LEVER_LABEL[L] for L in LEVERS], fontsize=9)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9)
        ax.set_title(CATEGORY_LABEL[cat], fontsize=11, pad=8)
        ax.tick_params(top=False, bottom=True, left=True, right=False)

    # Shared colourbar to the right of the 2×2 grid.
    fig.subplots_adjust(right=0.88, hspace=0.32, wspace=0.20, top=0.90, bottom=0.10)
    cbar_ax = fig.add_axes((0.91, 0.10, 0.022, 0.80))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Canonical score (mean across 20 prompts per cell)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Canonical score by model × lever, per task category "
        "(Tier-2, 20 rows per cell)",
        fontsize=12.5, y=0.965,
    )

    fig.text(
        0.012, 0.012,
        ("Cell text is mean canonical_score to 3 decimals (n=20 prompts per cell). "
         "Colour scale: red=low, green=high, clipped to [0.45, 1.0] for contrast. "
         "Friendly model names; full IDs in CSV."),
        fontsize=8.5, ha="left", style="italic", color="dimgray",
    )

    png = OUT_DIR / "category_heatmap.png"
    svg = OUT_DIR / "category_heatmap.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)

    print(f"[wrote] {png.relative_to(ROOT)} ({png.stat().st_size:,} bytes)")
    print(f"[wrote] {svg.relative_to(ROOT)} ({svg.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
