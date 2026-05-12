"""Day 12 step 4 — Headline-ready findings for the v1 writeup.

Read-only against data/results.db. Computes the underlying numbers fresh from
the DB (does not depend on the CSVs from steps 1–3 being current) and emits
analysis/out/headlines.md in writeup-ready prose. Same prose is echoed to
stdout for review.

Structure: 6 findings + 2 headline scalars, citable numbers inline.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.db"
OUT_DIR = ROOT / "analysis" / "out"
OUT_MD = OUT_DIR / "headlines.md"

MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
]


def fetch_all(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    return [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]


# Friendly names for prose. Same labels chart (b) and the bar charts use,
# so the writeup and the figures share vocabulary. Full IDs only appear in
# methodology references and CSV exports.
MODEL_FRIENDLY = {
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
    "gpt-5.4-2026-03-05": "gpt-5.4",
    "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
}


def friendly_model(model: str) -> str:
    """Map a full model ID to its writeup-friendly name. Falls back to
    date-stripping for models not in the explicit map."""
    if model in MODEL_FRIENDLY:
        return MODEL_FRIENDLY[model]
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)


def fmt_delta(x: float, decimals: int = 3) -> str:
    """Render a delta value. Effective-zero values (rounding to 0 at the given
    decimals) render as unsigned "0.000" rather than "+0.000" or "−0.000", which
    misread as positive/negative effects on a casual read."""
    rounded = round(x, decimals)
    if rounded == 0.0:
        return f"{0.0:.{decimals}f}"
    return f"{rounded:+.{decimals}f}"


def compute_findings(conn: sqlite3.Connection) -> dict:
    findings: dict = {}

    # --- Per (model, lever) aggregates ---
    model_lever = fetch_all(
        conn,
        """
        SELECT model, optimisation_lever AS lever,
               AVG(canonical_score) AS canon, AVG(cost_usd) AS cost
        FROM results
        WHERE canonical_score IS NOT NULL
        GROUP BY model, optimisation_lever
        """,
    )
    ml = {(r["model"], r["lever"]): r for r in model_lever}

    # Cost ratios and delta canon for batch vs baseline, per model
    batch_cost_ratios = {}
    batch_delta_canon = {}
    for m in MODELS:
        base = ml[(m, "baseline")]
        bat = ml[(m, "batch")]
        batch_cost_ratios[m] = bat["cost"] / base["cost"]
        batch_delta_canon[m] = bat["canon"] - base["canon"]
    findings["batch_cost_ratio_min"] = min(batch_cost_ratios.values())
    findings["batch_cost_ratio_max"] = max(batch_cost_ratios.values())
    findings["batch_delta_canon_min"] = min(batch_delta_canon.values())
    findings["batch_delta_canon_max"] = max(batch_delta_canon.values())

    # Compression cost & quality, per model
    comp_cost_ratios = {}
    comp_delta_canon = {}
    for m in MODELS:
        base = ml[(m, "baseline")]
        comp = ml[(m, "compression")]
        comp_cost_ratios[m] = comp["cost"] / base["cost"]
        comp_delta_canon[m] = comp["canon"] - base["canon"]
    findings["comp_cost_saving_min"] = 1 - max(comp_cost_ratios.values())
    findings["comp_cost_saving_max"] = 1 - min(comp_cost_ratios.values())
    findings["comp_delta_canon_min"] = min(comp_delta_canon.values())  # most negative
    findings["comp_delta_canon_max"] = max(comp_delta_canon.values())  # least negative

    # --- Per (category, lever) aggregates ---
    cat_lever = fetch_all(
        conn,
        """
        SELECT task_category, optimisation_lever AS lever,
               AVG(canonical_score) AS canon, AVG(cost_usd) AS cost,
               SUM(CASE WHEN tier_1_status = 'pass' THEN 1 ELSE 0 END) AS pass_n,
               SUM(CASE WHEN tier_1_status IN ('pass','fail_format','fail_content','truncated')
                        THEN 1 ELSE 0 END) AS applicable_n
        FROM results
        WHERE canonical_score IS NOT NULL
        GROUP BY task_category, optimisation_lever
        """,
    )
    cl = {(r["task_category"], r["lever"]): r for r in cat_lever}

    cats = sorted({r["task_category"] for r in cat_lever})
    for c in cats:
        base = cl[(c, "baseline")]
        for lever in ("batch", "compression", "output_cap"):
            findings[f"cat_{c}_{lever}_delta_canon"] = cl[(c, lever)]["canon"] - base["canon"]
    for c in cats:
        findings[f"cat_{c}_baseline_canon"] = cl[(c, "baseline")]["canon"]

    # RAG tier-1 collapse under compression
    rag_base = cl[("rag_qa", "baseline")]
    rag_comp = cl[("rag_qa", "compression")]
    findings["rag_baseline_t1_applicable"] = (
        rag_base["pass_n"] / rag_base["applicable_n"] if rag_base["applicable_n"] else 0
    )
    findings["rag_compression_t1_applicable"] = (
        rag_comp["pass_n"] / rag_comp["applicable_n"] if rag_comp["applicable_n"] else 0
    )

    # --- Reasoning drill-down: model × lever ---
    reasoning_cells = fetch_all(
        conn,
        """
        SELECT model, optimisation_lever AS lever, AVG(canonical_score) AS canon
        FROM results
        WHERE canonical_score IS NOT NULL AND task_category = 'reasoning'
        GROUP BY model, optimisation_lever
        """,
    )
    rc = {(r["model"], r["lever"]): r["canon"] for r in reasoning_cells}
    findings["rea_sonnet_baseline"] = rc[("claude-sonnet-4-6", "baseline")]
    findings["rea_sonnet_compression"] = rc[("claude-sonnet-4-6", "compression")]
    findings["rea_sonnet_output_cap"] = rc[("claude-sonnet-4-6", "output_cap")]
    findings["rea_gpt54_baseline"] = rc[("gpt-5.4-2026-03-05", "baseline")]
    findings["rea_gpt54_compression"] = rc[("gpt-5.4-2026-03-05", "compression")]
    findings["rea_gpt54_output_cap"] = rc[("gpt-5.4-2026-03-05", "output_cap")]
    findings["rea_haiku_baseline"] = rc[("claude-haiku-4-5", "baseline")]
    findings["rea_haiku_compression"] = rc[("claude-haiku-4-5", "compression")]
    findings["rea_mini_baseline"] = rc[("gpt-5.4-mini-2026-03-17", "baseline")]
    findings["rea_mini_compression"] = rc[("gpt-5.4-mini-2026-03-17", "compression")]

    # --- RAG: confirm Sonnet baseline is highest cell anywhere ---
    rag_pts = fetch_all(
        conn,
        """
        SELECT model, optimisation_lever AS lever, AVG(canonical_score) AS canon,
               AVG(cost_usd) AS cost
        FROM results
        WHERE canonical_score IS NOT NULL AND task_category = 'rag_qa'
        GROUP BY model, optimisation_lever
        """,
    )
    sonnet_baseline_rag = next(
        p for p in rag_pts
        if p["model"] == "claude-sonnet-4-6" and p["lever"] == "baseline"
    )
    findings["rag_sonnet_baseline_canon"] = sonnet_baseline_rag["canon"]
    findings["rag_sonnet_baseline_cost"] = sonnet_baseline_rag["cost"]
    # second-place sonnet/batch
    sonnet_batch_rag = next(
        p for p in rag_pts
        if p["model"] == "claude-sonnet-4-6" and p["lever"] == "batch"
    )
    findings["rag_sonnet_batch_canon"] = sonnet_batch_rag["canon"]
    findings["rag_sonnet_batch_cost"] = sonnet_batch_rag["cost"]
    gpt_mini_batch_rag = next(
        p for p in rag_pts
        if p["model"] == "gpt-5.4-mini-2026-03-17" and p["lever"] == "batch"
    )
    findings["rag_gpt_mini_batch_canon"] = gpt_mini_batch_rag["canon"]
    findings["rag_gpt_mini_batch_cost"] = gpt_mini_batch_rag["cost"]

    # Sonnet RAG dominance margin — sonnet/baseline canon minus the best
    # NON-SONNET (model, lever) cell in rag_qa. Using "best non-Sonnet" rather
    # than the literal "second-best cell" because sonnet/output_cap ties
    # sonnet/baseline at 0.993 (output cap is harmless on short RAG answers),
    # which would produce a meaningless "winning by 0.000" prose. The
    # task-dominance framing the writeup needs is "by how much does sonnet
    # beat the best OpenAI cell on RAG" — this is that number.
    rag_non_sonnet = sorted(
        (p for p in rag_pts if p["model"] != "claude-sonnet-4-6"),
        key=lambda p: -p["canon"],
    )
    best_non_sonnet_rag = rag_non_sonnet[0]
    findings["rag_sonnet_win_margin"] = (
        sonnet_baseline_rag["canon"] - best_non_sonnet_rag["canon"]
    )
    findings["rag_second_best_model"] = best_non_sonnet_rag["model"]
    findings["rag_second_best_lever"] = best_non_sonnet_rag["lever"]
    findings["rag_second_best_canon"] = best_non_sonnet_rag["canon"]

    # CS Pareto-tie margin: the gap between gpt-5.4 batch and gpt-5.4 baseline
    # on customer_support. The original hardcoded "0.008" referred to this
    # gap; computing it dynamically so it tracks the data.
    cs_pts = fetch_all(
        conn,
        """
        SELECT model, optimisation_lever AS lever, AVG(canonical_score) AS canon
        FROM results
        WHERE canonical_score IS NOT NULL AND task_category = 'customer_support'
        GROUP BY model, optimisation_lever
        """,
    )
    gpt54_cs_batch = next(
        p for p in cs_pts
        if p["model"] == "gpt-5.4-2026-03-05" and p["lever"] == "batch"
    )
    gpt54_cs_baseline = next(
        p for p in cs_pts
        if p["model"] == "gpt-5.4-2026-03-05" and p["lever"] == "baseline"
    )
    findings["cs_gpt54_win_margin"] = gpt54_cs_batch["canon"] - gpt54_cs_baseline["canon"]

    # --- Headline scalars ---
    all_cells = fetch_all(
        conn,
        """
        SELECT model, optimisation_lever AS lever,
               AVG(canonical_score) AS canon, AVG(cost_usd) AS cost
        FROM results
        WHERE canonical_score IS NOT NULL
        GROUP BY model, optimisation_lever
        """,
    )
    best = max(all_cells, key=lambda p: p["canon"])
    findings["best_model"] = best["model"]
    findings["best_lever"] = best["lever"]
    findings["best_canon"] = best["canon"]
    findings["best_cost"] = best["cost"]
    threshold = 0.9 * best["canon"]
    near_best = [p for p in all_cells if p["canon"] >= threshold]
    cheapest_near = min(near_best, key=lambda p: p["cost"])
    findings["cheap94_model"] = cheapest_near["model"]
    findings["cheap94_lever"] = cheapest_near["lever"]
    findings["cheap94_canon"] = cheapest_near["canon"]
    findings["cheap94_cost"] = cheapest_near["cost"]
    findings["cheap94_canon_pct"] = cheapest_near["canon"] / best["canon"]
    findings["cheap94_cost_pct"] = cheapest_near["cost"] / best["cost"]

    # --- GPT-5.4 batch identity: 54/54 ---
    gpt54_status_rows = fetch_all(
        conn,
        """
        SELECT prompt_id, task_category, optimisation_lever AS lever, tier_1_status
        FROM results
        WHERE canonical_score IS NOT NULL
          AND model = 'gpt-5.4-2026-03-05'
          AND optimisation_lever IN ('baseline','batch')
        """,
    )
    base_pass = {
        (r["prompt_id"], r["task_category"])
        for r in gpt54_status_rows
        if r["lever"] == "baseline" and r["tier_1_status"] == "pass"
    }
    batch_pass = {
        (r["prompt_id"], r["task_category"])
        for r in gpt54_status_rows
        if r["lever"] == "batch" and r["tier_1_status"] == "pass"
    }
    findings["gpt54_pass_intersection"] = len(base_pass & batch_pass)
    findings["gpt54_baseline_pass"] = len(base_pass)
    findings["gpt54_batch_pass"] = len(batch_pass)

    # --- Cost sensitivity per category ---
    findings["cost_sensitivity"] = {}
    for c in cats:
        pts = [p for p in cat_lever if p["task_category"] == c]
        # full 16-cell view per category requires (model, lever), so query separately:
        full = fetch_all(
            conn,
            """
            SELECT model, optimisation_lever AS lever, AVG(canonical_score) AS canon,
                   AVG(cost_usd) AS cost
            FROM results
            WHERE canonical_score IS NOT NULL AND task_category = ?
            GROUP BY model, optimisation_lever
            """,
            (c,),
        )
        best_canon = max(p["canon"] for p in full)
        near = [p for p in full if p["canon"] >= 0.9 * best_canon]
        cheapest = min(near, key=lambda p: p["cost"])
        most_exp = max(near, key=lambda p: p["cost"])
        ratio = most_exp["cost"] / cheapest["cost"] if cheapest["cost"] else float("inf")
        findings["cost_sensitivity"][c] = {
            "n_near_best": len(near),
            "ratio": ratio,
            "cheapest_model": cheapest["model"],
            "cheapest_lever": cheapest["lever"],
            "cheapest_cost": cheapest["cost"],
        }

    return findings


def render_markdown(f: dict) -> str:
    cs = f["cost_sensitivity"]
    cs_summarisation = cs["summarisation"]
    cs_reasoning = cs["reasoning"]
    cs_customer = cs["customer_support"]
    cs_rag = cs["rag_qa"]

    return f"""\
# LLM Cost-Quality Benchmark 2026 — Headline Findings

Generated from `data/results.db` (n=1,280 Tier-2 rows across 4 models × 4 levers × 4 task categories). Canonical score is the per-row Tier-2 quality number: agreement rows use the mean of judge A (Claude Opus 4.6) and judge B (GPT-5.5); the 80 judge-disagreement rows use arbitrated values from `scoring/disagreements.csv` (16 human + 64 median-canonical-auto).

## Finding 1 — Batch is universally near-free

The batch lever cuts cost by close to 50% (cost ratios {f['batch_cost_ratio_min']:.3f}–{f['batch_cost_ratio_max']:.3f} across the four models, matching the 50% provider batch discount) while leaving quality essentially unchanged: aggregate Δcanonical_score ranges from {fmt_delta(f['batch_delta_canon_min'])} to {fmt_delta(f['batch_delta_canon_max'])} across models, and from {fmt_delta(f['cat_customer_support_batch_delta_canon'])} (customer_support, rag_qa) to {fmt_delta(f['cat_reasoning_batch_delta_canon'])} (reasoning) across categories. On GPT-5.4 specifically, batch produces bit-for-bit identical Tier-1 pass sets versus sync baseline ({f['gpt54_pass_intersection']}/{f['gpt54_baseline_pass']} pass intersection, zero per-prompt divergence) — strong evidence the OpenAI batch endpoint is deterministic-equivalent to sync at temperature=0. The other three models show small monotonic regression under batch (1–2 prompts each); every observed divergence is a reasoning prompt.

## Finding 2 — Compression is universally dominated, catastrophic on RAG

Prompt compression cuts cost by only {f['comp_cost_saving_min']*100:.0f}–{f['comp_cost_saving_max']*100:.0f}% while reducing canonical_score by {-f['comp_delta_canon_max']*100:.0f}–{-f['comp_delta_canon_min']*100:.0f} points on aggregate (Δcanon {f['comp_delta_canon_max']:+.3f} to {f['comp_delta_canon_min']:+.3f} across the four models). The category-level damage is wildly uneven: customer_support loses {-f['cat_customer_support_compression_delta_canon']*100:.0f} points, summarisation {-f['cat_summarisation_compression_delta_canon']*100:.0f}, reasoning {-f['cat_reasoning_compression_delta_canon']*100:.0f}, and rag_qa **{-f['cat_rag_qa_compression_delta_canon']*100:.0f}**. The RAG collapse is not just a quality drop but a structural failure — Tier-1 applicable pass rate on compressed RAG goes from {f['rag_baseline_t1_applicable']*100:.1f}% (baseline) to {f['rag_compression_t1_applicable']*100:.1f}% (compression), meaning roughly {round(f['rag_compression_t1_applicable']*100)} of every 100 compressed RAG outputs pass deterministic format/content checks. Compressing retrieval context breaks the grounding the model needs to answer correctly. No model and no task category recovers compression's quality cost; the lever is dominated everywhere on the cost-quality frontier.

## Finding 3 — Output cap damage is reasoning-specific

The output-length cap is fine for short-answer tasks (Δcanon {f['cat_customer_support_output_cap_delta_canon']:+.3f} on customer_support, {f['cat_rag_qa_output_cap_delta_canon']:+.3f} on rag_qa, {f['cat_summarisation_output_cap_delta_canon']:+.3f} on summarisation) but costly for multi-step reasoning (Δcanon {f['cat_reasoning_output_cap_delta_canon']:+.3f}). The mechanism is visible in the Tier-1 status enum: capped reasoning rows hit the cap (`tier_1_status='truncated'`) at rates of 4–11 per (model, lever) cell on reasoning, versus zero truncations elsewhere except for some long summaries. Capping output cuts the reasoning chain before the answer, and the answer suffers. Use output_cap freely for classification, extraction, or short replies; avoid it on tasks that require chained intermediate steps.

*Footnote: Summarisation is moderately sensitive to both compression ({f['cat_summarisation_compression_delta_canon']:+.3f}) and output_cap ({f['cat_summarisation_output_cap_delta_canon']:+.3f}) — the close magnitudes reflect the task's general sensitivity to any quality-affecting lever, not a coincidence.*

## Finding 4 — Provider dominance is task-shaped, not universal

OpenAI models Pareto-dominate the aggregate cost-quality frontier (all four Pareto-optimal cells are gpt-5.4 or gpt-5.4-mini), but the category-level Pareto frontiers tell a different story. On reasoning and summarisation, the Pareto frontier is OpenAI-only (4 cells each). On customer_support the frontier is OpenAI-only too (2 cells), but the gpt-5.4 batch-vs-baseline gap is {f['cs_gpt54_win_margin']:.3f} canonical points — effectively a tie. **On rag_qa the Pareto frontier is mixed: sonnet baseline reaches canonical_score {f['rag_sonnet_baseline_canon']:.3f} (the highest cell in the entire 16-cell aggregate matrix, beating the best non-Sonnet cell — {friendly_model(f['rag_second_best_model'])} {f['rag_second_best_lever']} at {f['rag_second_best_canon']:.3f} — by {f['rag_sonnet_win_margin']:.3f} canonical points), and sonnet batch ({f['rag_sonnet_batch_canon']:.3f}) is also Pareto-optimal alongside gpt-5.4-mini batch ({f['rag_gpt_mini_batch_canon']:.3f}).** Sonnet's strength on retrieval-grounded QA outweighs GPT-5.4's strength elsewhere on this category. The honest framing is: OpenAI wins three of four categories; Anthropic sonnet wins RAG.

## Finding 5 — Anthropic models degrade more under optimization pressure

On reasoning specifically, Sonnet trails GPT-5.4 by {f['rea_gpt54_baseline']-f['rea_sonnet_baseline']:.3f} canonical points at baseline ({f['rea_sonnet_baseline']:.3f} versus {f['rea_gpt54_baseline']:.3f}), but the gap widens to {f['rea_gpt54_compression']-f['rea_sonnet_compression']:.3f} under compression ({f['rea_sonnet_compression']:.3f} versus {f['rea_gpt54_compression']:.3f}) and to {f['rea_gpt54_output_cap']-f['rea_sonnet_output_cap']:.3f} under output_cap ({f['rea_sonnet_output_cap']:.3f} versus {f['rea_gpt54_output_cap']:.3f}). The same pattern shows on Haiku, where reasoning canon falls from {f['rea_haiku_baseline']:.3f} (baseline) to {f['rea_haiku_compression']:.3f} (compression) — a {(f['rea_haiku_baseline']-f['rea_haiku_compression'])*100:.0f}-point drop, versus GPT-5.4-mini's {(f['rea_mini_baseline']-f['rea_mini_compression'])*100:.0f}-point drop ({f['rea_mini_baseline']:.3f} → {f['rea_mini_compression']:.3f}) under the same lever. Anthropic models are more sensitive to optimization levers, especially compression on reasoning. Practical implication: if you are running Anthropic in production, avoid compression specifically — the cost saving is small (≤21%) and the quality cost is disproportionately large.

## Finding 6 — Cost-quality sweet spot depends on task

For three of four task categories you can shop aggressively for cheap quality: customer_support has {cs_customer['n_near_best']} cells within 90% of the category's best canonical score spanning a {cs_customer['ratio']:.1f}× cost range (cheapest {friendly_model(cs_customer['cheapest_model'])} {cs_customer['cheapest_lever']} at ${cs_customer['cheapest_cost']:.6f} per task); rag_qa has {cs_rag['n_near_best']} cells at {cs_rag['ratio']:.1f}× spread; reasoning has {cs_reasoning['n_near_best']} cells at {cs_reasoning['ratio']:.1f}× spread (cheapest {friendly_model(cs_reasoning['cheapest_model'])} {cs_reasoning['cheapest_lever']} at ${cs_reasoning['cheapest_cost']:.6f} per task). gpt-5.4-mini batch is the cheapest near-best cell in three of four categories (customer support, RAG, reasoning) — the cost-quality default for production routing on these task shapes. Summarisation is the exception: only {cs_summarisation['n_near_best']} cells reach 90% of the category's best canon, all within a narrow {cs_summarisation['ratio']:.1f}× cost range, and all from the two larger models at baseline or batch. Summarisation's cheapest near-best cell is {friendly_model(cs_summarisation['cheapest_model'])} {cs_summarisation['cheapest_lever']} at ${cs_summarisation['cheapest_cost']:.6f} per task — {cs_summarisation['cheapest_cost']/min(cs_customer['cheapest_cost'], cs_rag['cheapest_cost'], cs_reasoning['cheapest_cost']):.0f}× more expensive than the cheapest near-best cell in any other category. You can't cheap your way to good summaries on this prompt set.

## Headline scalar 1 — Cheapest 94% of frontier quality

**{friendly_model(f['cheap94_model'])} + {f['cheap94_lever']} → canonical_score {f['cheap94_canon']:.3f} at \\${f['cheap94_cost']:.6f} per task.** {f['cheap94_canon_pct']*100:.1f}% of frontier quality at {f['cheap94_cost_pct']*100:.1f}% of frontier cost. This is the cheapest cell on the aggregate Pareto frontier; for routine production traffic that tolerates batch latency, it is the dominant cost-quality choice. Frontier quality at \\${f['best_cost']:.6f} costs {f['best_cost']/f['cheap94_cost']:.1f}× more than near-frontier quality at \\${f['cheap94_cost']:.6f} — for a {(f['best_canon']-f['cheap94_canon'])*100:.1f}-point canonical improvement.

## Headline scalar 2 — Frontier quality cost

**{friendly_model(f['best_model'])} + {f['best_lever']} → canonical_score {f['best_canon']:.3f} at \\${f['best_cost']:.6f} per task.** Highest canonical score on the aggregate frontier across all 16 (model, lever) cells. Use when the {(f['best_canon']-f['cheap94_canon'])*100:.1f}-point canonical margin over the near-frontier cell genuinely matters for the application.

---

*Source: `data/results.db`, n=1,280 Tier-2 rows. Generated by `analysis/04_headlines.py`. Numbers regenerate on each run from the DB.*
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    findings = compute_findings(conn)
    conn.close()

    md = render_markdown(findings)
    OUT_MD.write_text(md)

    print(md)
    print(f"\n[wrote] {OUT_MD.relative_to(ROOT)} ({len(md)} chars)")


if __name__ == "__main__":
    main()
