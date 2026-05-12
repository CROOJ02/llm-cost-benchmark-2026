# LLM Cost-Optimisation Benchmark 2026 — v1

An empirical benchmark of cost-optimisation levers for production LLM workloads. 1,280 Tier-2-scored responses across four frontier models, four operational levers, and four task categories. Every claim in the writeup is traceable to a SQL query against `data/results.db`.

**Status:** v1 complete (May 2026). Methodology documented, results published, 4 charts rendered. The full writeup is at [docs/writeup/v1.md](docs/writeup/v1.md).

---

## Headline findings

Six findings from the writeup, one line each. Click through for the per-finding prose and supporting data.

1. **Batch is operationally equivalent to sync** (and 50% cheaper) — on gpt-5.4 specifically, batch produces a bit-for-bit identical Tier-1 pass set vs sync at temperature=0. [→](docs/writeup/v1.md#finding-1--batch-is-operationally-equivalent-to-sync-and-50-cheaper)
2. **Compression is universally dominated, catastrophic on RAG** — 18–21% cost saving for 18–22 canonical points lost on aggregate; on RAG specifically, Tier-1 pass rate collapses 80% → 4%. [→](docs/writeup/v1.md#finding-2--compression-is-universally-dominated-catastrophic-on-rag)
3. **Output cap damage is reasoning-specific** — fine on short-answer tasks, costs −0.13 canonical on multi-step reasoning. [→](docs/writeup/v1.md#finding-3--output-cap-damage-is-reasoning-specific)
4. **Provider dominance is task-shaped, not universal** — OpenAI Pareto-dominates the aggregate frontier, but Sonnet wins RAG with canon 0.993 (highest cell anywhere). [→](docs/writeup/v1.md#finding-4--provider-dominance-is-task-shaped-not-universal)
5. **Anthropic models degrade more under optimisation pressure** — the Sonnet-vs-GPT-5.4 reasoning gap widens from 0.027 (baseline) to 0.093 (compression). [→](docs/writeup/v1.md#finding-5--anthropic-models-degrade-more-under-optimisation-pressure)
6. **The cost-quality sweet spot depends on the task** — three categories tolerate 9–14× cost spread; summarisation only 2.2×. [→](docs/writeup/v1.md#finding-6--the-cost-quality-sweet-spot-depends-on-the-task)

**Headline scalar:** `gpt-5.4-mini + batch` delivers 94.2% of frontier quality at 13.2% of frontier cost ($0.000557 per task vs $0.004220).

---

## What's measured

**Models (4):** Claude Sonnet 4.6, Claude Haiku 4.5, GPT-5.4, GPT-5.4-mini — pinned to dated snapshots (`claude-sonnet-4-6`, `claude-haiku-4-5`, `gpt-5.4-2026-03-05`, `gpt-5.4-mini-2026-03-17`).

**Cost-optimisation levers (4):**
- `baseline` — synchronous API call, no optimisation.
- `batch` — provider batch API (50% list-price discount on both providers).
- `output_cap` — explicit `max_tokens` cap to bound output cost.
- `compression` — runtime LLM-Lingua-2 prompt compression (rate=0.5).

**Tasks (102 prompts × 5 categories):**
- `customer_support` (20 prompts) — classification + reply, structured JSON output.
- `rag_qa` (20 prompts) — retrieval-grounded QA with citation requirements.
- `extraction` (22 prompts) — structured-field extraction. Tier-1-only.
- `summarisation` (20 prompts) — long-document → executive summary.
- `reasoning` (20 prompts) — multi-step quantitative reasoning.

**Scope:** 4 models × 4 levers × 80 Tier-2 prompts = **1,280 Tier-2-scored responses**. Extraction's 22 Tier-1-only rows × 4 models × 4 levers add a further 352 rows for format-rigour analysis (Tier-2 not applicable).

---

## Quality scoring

Two independent tiers, so cost and quality can be evaluated jointly.

**Tier 1 — deterministic.** Per-row checks derived from each prompt's `expected` block (JSON validity, schema, content checks). Lever-aware status enum distinguishes design-induced failure (`truncated`, `compression_unavailable`) from genuine model failure. See [scoring/tier_1.py](scoring/tier_1.py).

**Tier 2 — dual-judge panel.** Two LLM judges score each response against a per-prompt rubric:
- **Judge A:** Claude Opus 4.6 (`claude-opus-4-6`)
- **Judge B:** GPT-5.5 (`gpt-5.5`)

The v1 panel went through a mid-benchmark revision: the original Judge B was Mistral Large 2512, replaced after Day 10 audit surfaced systematic failure modes (non-determinism at temperature=0, score-reasoning desync, and hallucinated completeness on truncated responses). GPT-5.5 was adopted after passing a targeted 12-prompt validation against the known Mistral failure modes. **Mistral data is archived verbatim** in `judge_b_mistral_*` columns of the results table for transparency and v2 cross-judge analysis. The full panel-revision narrative is in [docs/methodology/prompt_design_decisions.md](docs/methodology/prompt_design_decisions.md#day-11-panel-revision) and is also summarised in the writeup.

**Disagreement resolution.** Of the 80 judge disagreements (|Δ| ≥ 0.3 between judges), 16 cases were arbitrated by a human operator; the remaining 64 were resolved by median-canonical-auto (midpoint between the two judges). Both methods produce a single `canonical_score` per row; the `arbitration_method` column tags how each row was resolved.

---

## Reading the results

Public artefacts in this repo:

- **The writeup:** [docs/writeup/v1.md](docs/writeup/v1.md) — 3,700-word document with methodology, six findings, four figures, limitations.
- **Methodology decisions:** [docs/methodology/prompt_design_decisions.md](docs/methodology/prompt_design_decisions.md) — full audit trail including the Day 11 panel revision.
- **Headlines (regenerable):** [analysis/out/headlines.md](analysis/out/headlines.md) — writeup-spine prose with citable numbers, regenerated from the DB on each run of `analysis/04_headlines.py`.
- **Figures (PNG + SVG):**
  - [analysis/out/charts/cost_quality_scatter.png](analysis/out/charts/cost_quality_scatter.png) — 16-point cost-vs-quality scatter with Pareto frontier.
  - [analysis/out/charts/category_heatmap.png](analysis/out/charts/category_heatmap.png) — 4×4 per-category model × lever heatmaps.
  - [analysis/out/charts/reasoning_drilldown.png](analysis/out/charts/reasoning_drilldown.png) — reasoning bar chart, 16 cells.
  - [analysis/out/charts/cost_sensitivity.png](analysis/out/charts/cost_sensitivity.png) — 4-strip cost-sensitivity plot.
- **All prompt JSONs:** [prompts/](prompts/) — 102 prompts as authored.
- **Source data:** `data/results.db` — SQLite database with 1,776 rows. Schema in [data/schema.sql](data/schema.sql).

---

## Reproducing locally

Requirements: Python 3.11+, Poetry, SQLite. Provider API keys only needed if you want to re-run the data collection; the analysis layer is read-only against the committed `data/results.db`.

```bash
# Clone and install
git clone https://github.com/CROOJ02/llm-cost-benchmark-2026.git
cd llm-cost-benchmark-2026
poetry install --with dev   # includes matplotlib for chart regeneration

# Verify the DB is intact (1,280 Tier-2 rows expected)
sqlite3 data/results.db "SELECT COUNT(*) FROM results WHERE canonical_score IS NOT NULL;"

# Run the analysis pipeline against the committed DB (read-only)
poetry run python analysis/01_lever_matrix.py         # 4×4 matrix + identity check
poetry run python analysis/02_cost_quality_scatter.py # Pareto frontier + headline scalars
poetry run python analysis/03_category_breakdown.py   # per-category drill-down
poetry run python analysis/04_headlines.py            # regenerates analysis/out/headlines.md

# Regenerate the 4 charts (writes PNG + SVG to analysis/out/charts/)
poetry run python analysis/05_chart_cost_quality.py
poetry run python analysis/06_chart_category_heatmap.py
poetry run python analysis/07_chart_reasoning_drilldown.py
poetry run python analysis/08_chart_cost_sensitivity.py

# Run the test suite (162 items)
poetry run pytest
```

To re-collect data from scratch (full sweep), copy `.env.example` to `.env`, populate the provider API keys (Anthropic, OpenAI, Mistral, Gemini optional), and consult the per-day scripts in [scripts/](scripts/). The full sweep cost £35.49 of a £300 cap.

---

## Methodology highlights

- **Hard cost cap** of £300 across all measurement runs; current spend tracked in `runs.cost_so_far_gbp`.
- **Skip-if-exists is run_id-scoped:** each run produces independent measurements (see methodology doc).
- **Bias safeguards:** Tier-1 criteria sourced from prompt JSONs, never from inspecting responses; provider-style normalisation pipeline applied uniformly; per-row audit of which normalisation steps fired.
- **Reproducibility:** every result row carries `model_version`, `temperature`, `optimisation_config`, `config_hash`, and `run_attempt`.
- **Same-family judge bias** is a documented v1 limitation (Opus shares family with Sonnet/Haiku; GPT-5.5 with GPT-5.4 family). The Day 11 panel revision eliminated a Mistral-driven cross-provider calibration offset; residual same-family bias is acknowledged in [the writeup's Limitations section](docs/writeup/v1.md#limitations).

---

## Project context

This benchmark is published as part of the InferOps research line — an AI inference efficiency platform currently in pre-build validation. The benchmark methodology is the engine; if you are running LLM workloads in production and want to apply these findings to your actual workload, early-adopter information is at [inferops.org](https://inferops.org) (and in the [writeup's conclusion](docs/writeup/v1.md#conclusion--early-adopter-offer)).

The benchmark itself is research, not product. Code, data, prompts, and writeup are all public so others can reproduce, critique, or extend.

---

## License

MIT. See [LICENSE](LICENSE).
