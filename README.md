# LLM Cost Benchmark 2026 — v1.0

An empirical benchmark of cost-optimisation levers for production LLM workloads. Measures cost, latency, and quality (deterministic + judge-scored) across four cost levers and four production models, against 102 prompts in five task categories.

This is **v1** — a focused first release intended to enable rigorous methodology over broad coverage. See [Scope and limitations](docs/methodology/prompt_design_decisions.md#scope-and-limitations-v1) for what's deliberately out of scope, and the [CHANGELOG](CHANGELOG.md) for what's planned next.

## What it measures

**Models (4):** Anthropic Claude Sonnet 4.6, Claude Haiku 4.5; OpenAI GPT-5.4, GPT-5.4-mini.

**Cost-optimisation levers (4 + 1):**
- `baseline` — straight sync API call, no optimisation
- `batch` — provider batch API (50% list-price discount on both providers)
- `output_cap` — `max_tokens=200` to bound output cost
- `compression` — runtime LLMLingua-2 prompt compression (`rate=0.5`)
- `caching` (subset) — prompt-caching engagement on the 5 longest summarisation prompts

**Tasks (102 prompts × 5 categories):** customer-support classification + reply (20), RAG QA with citations (20), structured extraction (22), summarisation (20), multi-step reasoning (20).

**Quality scoring (two tiers):**
- **Tier 1 — deterministic.** Per-row checks derived from each prompt's `expected` block. Lever-aware status buckets distinguish design-induced failure (truncation, compression unavailability) from genuine model failure. See [scoring/tier_1.py](scoring/tier_1.py).
- **Tier 2 — dual judge.** Opus 4.6 + Mistral Large 2512 score against per-prompt criteria; human arbitration on disagreement. (Day 10 work, in progress.)

## Reading the results

Public artefacts (when v1 ships):
- All prompt JSONs in [`prompts/`](prompts/)
- All captured model outputs and scores in `data/results.db` (SQLite; schema in [`data/schema.sql`](data/schema.sql))
- Methodology decisions and audit trail in [`docs/methodology/prompt_design_decisions.md`](docs/methodology/prompt_design_decisions.md)
- Findings writeup linked here on release

## Methodology highlights

- **Hard cost cap:** £300 across all measurement runs. Current spend tracked in `runs.cost_so_far_gbp`.
- **Skip-if-exists is run_id-scoped:** each run produces independent measurements (see methodology doc).
- **Bias safeguards:** Tier-1 criteria sourced from prompt JSONs not responses; provider-style normalisation pipeline applied uniformly; per-row audit of which normalisation steps fired. See doc for details.
- **Reproducibility:** every result row carries `model_version`, `temperature`, `optimisation_config`, `config_hash`, and `run_attempt`.

## Project context

A research artefact accompanying [InferOps](https://inferops.org), an AI inference efficiency platform currently in pre-build validation (Phase 0.5). Findings inform — and are informed by — discovery conversations with production teams optimising LLM spend.

## License

MIT.
