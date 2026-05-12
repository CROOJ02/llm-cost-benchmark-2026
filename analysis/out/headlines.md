# LLM Cost-Quality Benchmark 2026 — Headline Findings

Generated from `data/results.db` (n=1,280 Tier-2 rows across 4 models × 4 levers × 4 task categories). Canonical score is the per-row Tier-2 quality number: agreement rows use the mean of judge A (Claude Opus 4.6) and judge B (GPT-5.5); the 80 judge-disagreement rows use arbitrated values from `scoring/disagreements.csv` (16 human + 64 median-canonical-auto).

## Finding 1 — Batch is universally near-free

The batch lever cuts cost by close to 50% (cost ratios 0.476–0.501 across the four models, matching the 50% provider batch discount) while leaving quality essentially unchanged: aggregate Δcanonical_score ranges from -0.017 to +0.001 across models, and from 0.000 (customer_support, rag_qa) to -0.015 (reasoning) across categories. On GPT-5.4 specifically, batch produces bit-for-bit identical Tier-1 pass sets versus sync baseline (54/54 pass intersection, zero per-prompt divergence) — strong evidence the OpenAI batch endpoint is deterministic-equivalent to sync at temperature=0. The other three models show small monotonic regression under batch (1–2 prompts each); every observed divergence is a reasoning prompt.

## Finding 2 — Compression is universally dominated, catastrophic on RAG

Prompt compression cuts cost by only 18–21% while reducing canonical_score by 18–22 points on aggregate (Δcanon -0.180 to -0.224 across the four models). The category-level damage is wildly uneven: customer_support loses 5 points, summarisation 7, reasoning 21, and rag_qa **45**. The RAG collapse is not just a quality drop but a structural failure — Tier-1 applicable pass rate on compressed RAG goes from 80.0% (baseline) to 3.8% (compression), meaning roughly 4 of every 100 compressed RAG outputs pass deterministic format/content checks. Compressing retrieval context breaks the grounding the model needs to answer correctly. No model and no task category recovers compression's quality cost; the lever is dominated everywhere on the cost-quality frontier.

## Finding 3 — Output cap damage is reasoning-specific

The output-length cap is fine for short-answer tasks (Δcanon -0.004 on customer_support, -0.003 on rag_qa, -0.067 on summarisation) but costly for multi-step reasoning (Δcanon -0.132). The mechanism is visible in the Tier-1 status enum: capped reasoning rows hit the cap (`tier_1_status='truncated'`) at rates of 4–11 per (model, lever) cell on reasoning, versus zero truncations elsewhere except for some long summaries. Capping output cuts the reasoning chain before the answer, and the answer suffers. Use output_cap freely for classification, extraction, or short replies; avoid it on tasks that require chained intermediate steps.

*Footnote: Summarisation is moderately sensitive to both compression (-0.069) and output_cap (-0.067) — the close magnitudes reflect the task's general sensitivity to any quality-affecting lever, not a coincidence.*

## Finding 4 — Provider dominance is task-shaped, not universal

OpenAI models Pareto-dominate the aggregate cost-quality frontier (all four Pareto-optimal cells are gpt-5.4 or gpt-5.4-mini), but the category-level Pareto frontiers tell a different story. On reasoning and summarisation, the Pareto frontier is OpenAI-only (4 cells each). On customer_support the frontier is OpenAI-only too (2 cells), but the gpt-5.4 batch-vs-baseline gap is 0.007 canonical points — effectively a tie. **On rag_qa the Pareto frontier is mixed: sonnet baseline reaches canonical_score 0.993 (the highest cell in the entire 16-cell aggregate matrix, beating the best non-Sonnet cell — gpt-5.4-mini batch at 0.985 — by 0.008 canonical points), and sonnet batch (0.990) is also Pareto-optimal alongside gpt-5.4-mini batch (0.985).** Sonnet's strength on retrieval-grounded QA outweighs GPT-5.4's strength elsewhere on this category. The honest framing is: OpenAI wins three of four categories; Anthropic Sonnet wins RAG.

## Finding 5 — Anthropic models degrade more under optimisation pressure

On reasoning specifically, Sonnet trails GPT-5.4 by 0.027 canonical points at baseline (0.943 versus 0.970), but the gap widens to 0.093 under compression (0.708 versus 0.801) and to 0.105 under output_cap (0.796 versus 0.901). The same pattern shows on Haiku, where reasoning canon falls from 0.909 (baseline) to 0.631 (compression) — a 28-point drop, versus GPT-5.4-mini's 17-point drop (0.932 → 0.761) under the same lever. Anthropic models are more sensitive to optimisation levers, especially compression on reasoning. Practical implication: if you are running Anthropic in production, avoid compression specifically — the cost saving is small (≤21%) and the quality cost is disproportionately large.

## Finding 6 — Cost-quality sweet spot depends on task

For three of four task categories you can shop aggressively for cheap quality: customer_support has 9 cells within 90% of the category's best canonical score spanning a 10.0× cost range (cheapest gpt-5.4-mini batch at $0.000170 per task); rag_qa has 9 cells at 8.9× spread; reasoning has 10 cells at 13.9× spread (cheapest gpt-5.4-mini batch at $0.000431 per task). gpt-5.4-mini batch is the cheapest near-best cell in three of four categories (customer support, RAG, reasoning) — the cost-quality default for production routing on these task shapes. Summarisation is the exception: only 4 cells reach 90% of the category's best canon, all within a narrow 2.2× cost range, and all from the two larger models at baseline or batch. Summarisation's cheapest near-best cell is gpt-5.4 batch at $0.005045 per task — 30× more expensive than the cheapest near-best cell in any other category. You can't cheap your way to good summaries on this prompt set.

## Headline scalar 1 — Cheapest 94% of frontier quality

**gpt-5.4-mini + batch → canonical_score 0.881 at \$0.000557 per task.** 94.2% of frontier quality at 13.2% of frontier cost. This is the cheapest cell on the aggregate Pareto frontier; for routine production traffic that tolerates batch latency, it is the dominant cost-quality choice. Frontier quality at \$0.004220 costs 7.6× more than near-frontier quality at \$0.000557 — for a 5.4-point canonical improvement.

## Headline scalar 2 — Frontier quality cost

**gpt-5.4 + baseline → canonical_score 0.935 at \$0.004220 per task.** Highest canonical score on the aggregate frontier across all 16 (model, lever) cells. Use when the 5.4-point canonical margin over the near-frontier cell genuinely matters for the application.

---

*Source: `data/results.db`, n=1,280 Tier-2 rows. Generated by `analysis/04_headlines.py`. Numbers regenerate on each run from the DB.*
