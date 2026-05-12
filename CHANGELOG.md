# Changelog

## v1.0 — 12 May 2026

Initial public release. Empirical benchmark of cost-optimisation levers for production LLM workloads.

### What's in v1.0

- **102 prompts** across five task categories (customer support, RAG QA, extraction, summarisation, reasoning) — 80 Tier-2-scored prompts plus 22 Tier-1-only extraction prompts.
- **Four production frontier models** at fixed snapshots: `claude-sonnet-4-6`, `claude-haiku-4-5`, `gpt-5.4-2026-03-05`, `gpt-5.4-mini-2026-03-17`.
- **Four cost-optimisation levers**: baseline (sync), batch, output_cap (max-tokens cap), compression (LLM-Lingua-2 prompt compression).
- **1,280 Tier-2-scored responses** (4 models × 4 levers × 80 prompts) plus 352 Tier-1-only rows for extraction format-rigour analysis.
- **Two-tier scoring**:
  - **Tier 1** — deterministic per-row checks derived from each prompt's `expected` block (JSON validity, schema, content).
  - **Tier 2** — dual-judge LLM panel: Claude Opus 4.6 (Judge A) + GPT-5.5 (Judge B). Mid-benchmark panel revision documented (original Judge B was Mistral Large 2512; replaced after Day 10 audit surfaced systematic failure modes including hallucinated completeness on truncated responses).
- **Hybrid disagreement arbitration**: 16 cases with |Δ| > 0.3 received human arbitration; 64 cases with |Δ| ≤ 0.3 resolved by median-canonical-auto.
- **v1 writeup** in `docs/writeup/v1.md` (3,700 words, 8 sections, references 4 figures).
- **Four published charts** in `analysis/out/charts/` (PNG + SVG): cost-quality scatter with Pareto frontier, per-category model × lever heatmap, reasoning drill-down bars, cost-sensitivity strip plot.
- **Full methodology audit trail** in `docs/methodology/prompt_design_decisions.md`, including all design decisions, bias safeguards, the Day 11 panel revision, and the hybrid arbitration approach.
- **Source data** committed at `data/results.db` (4 MB SQLite) so every claim in the writeup is traceable to a SQL query. Schema in `data/schema.sql`.

### Headline findings

- Batch is operationally equivalent to sync (and 50% cheaper). On gpt-5.4, batch produces a bit-for-bit identical Tier-1 pass set vs sync at temperature=0.
- Compression is universally dominated and catastrophic on RAG (Tier-1 pass rate 80% → 4%).
- Output cap damage is reasoning-specific (−0.13 canonical on reasoning vs ≤ −0.07 elsewhere).
- Provider dominance is task-shaped: OpenAI Pareto-dominates the aggregate frontier; Sonnet wins RAG with canon 0.993 (highest cell anywhere).
- Anthropic models degrade more under optimisation pressure than OpenAI on reasoning specifically.
- `gpt-5.4-mini + batch` delivers 94.2% of frontier quality at 13.2% of frontier cost — the actionable v1 recommendation for cost-sensitive routing.

### Known v1 limitations

- Single-shot scoring per (prompt, model, lever, judge); no statistical replication. v2 will run n=5 with bootstrap CIs.
- Same-family judge bias (Opus + GPT-5.5 share families with test models). v2 will explore family-disjoint judges (Grok 4, fine-tuned specialised evaluators).
- 102 prompts is small relative to real production workloads.
- May 2026 pricing/model snapshot — re-measurement recommended quarterly.

See `docs/writeup/v1.md#limitations` for the full discussion.

### Cost

Total benchmark spend: £35.49 of a £300 hard cap.

---

*v2 scope is informed by Phase 0.5 discovery findings and community feedback. Pull requests welcomed.*
