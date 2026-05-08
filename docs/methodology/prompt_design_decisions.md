# Prompt Design Decisions

This document records prompt-design choices that diverge from the PRD's literal specification, so the writeup methodology section can cite a single source.

## Length variance in easy-tier RAG and summarisation prompts

The PRD specifies context lengths of 500–1500 words for RAG (§3 Cat 2) and 1500–3000 words for summarisation (§3 Cat 4). The committed prompt files run shorter than the PRD floor at the easy end of each category — RAG easies at roughly 400–480 words, summarisation easies at roughly 890–1180 words — and below the PRD upper bound across the rest of both categories. The design choice is deliberate: complexity is varied through the difficulty of the underlying task — citation precision in RAG, faithful coverage of main points in summarisation — not through the brute size of the input document. A short context with a directly stated answer is a cleaner test of "easy citation" than a longer context where the answer sentence is buried, and padding either category's easies with neutral background content would dilute what the easies are designed to test rather than improve it. Spot-checks on representative prompts (RAG rag-002 at the easy end with a single-sentence citation; summarisation sum-002 at 1177 words covering an article with multiple statistics and a nuanced thesis; summarisation sum-007 at 889 words covering a self-contained customer-comms email with a hallucination risk) confirmed the affected prompts hold up methodologically at their actual lengths. The writeup limitations section should note: "easy-tier inputs run shorter than the PRD spec range to keep the complexity gradient on task difficulty rather than input size."

## Mistral rate-limit alias vs versioned-model finding

During Day 1 setup, rate-limit verification on Mistral was performed against the `mistral-large-latest` alias, which exposed an alias-level limit (~15 RPM) rather than the underlying versioned model's actual capacity. Direct verification on Day 4 against the console limits page revealed `mistral-large-2411` at ~100 RPM and `mistral-large-2512` at ~360 RPM. Methodology lesson: when verifying provider rate limits, query against the specific model version that will be used in production runs, not the alias.

## Anthropic prompt-caching minimum lengths and Haiku 4.5 caveat

Anthropic's `cache_control` ephemeral block is silently ignored on prompts shorter than a model-specific minimum. Per the prompt-caching docs (verified 2026-05-04):

- Claude Sonnet 4.6: **2048 tokens** minimum
- Claude Haiku 4.5: **4096 tokens** minimum
- Claude Opus 4.6 / 4.7: 4096 tokens minimum
- Older Sonnet/Opus generations (Sonnet 4.5, Opus 4 / 4.1, Sonnet 4, Sonnet 3.7): 1024 tokens

Cache pricing on the 5-minute default TTL: cache writes are **1.25×** the base input rate, cache reads are **0.1×** (a 90% discount). Break-even is computed from the observed write/read multipliers in the Day 12 analysis rather than asserted here — break-even is an output of the empirical measurement, not an input assumption.

These thresholds matter for the Day 5 caching lever's coverage. The summarisation category was selected because its prompts (~2,680–3,300 input tokens) clear Sonnet 4.6's 2048-token floor on every prompt. They do NOT clear Haiku 4.5's 4096-token floor on any prompt, which means **the caching lever is empirically a no-op on Haiku 4.5 across our entire test set**. Customer support, RAG, extraction, and reasoning prompts are below 2048 tokens, so caching is a no-op for those categories on every Anthropic model in the test set.

Methodology consequence: Day 5's caching lever measurements on Anthropic are scoped to Sonnet 4.6 × summarisation. The Haiku 4.5 result is reported as "caching unavailable at our prompt sizes" rather than as a measurement, and the writeup limitations section will note the prompt-size threshold as the cause.

### OpenAI prompt-caching specs (for cross-provider comparison)

Per OpenAI's prompt-caching guide (verified 2026-05-04 against `developers.openai.com/api/docs/guides/prompt-caching`): caching activates automatically on prompts containing **1024 tokens or more** with no opt-in or API parameter required. Cache reads can reduce input token cost by **up to 90%** and latency by up to 80%. Cached prefixes "generally remain active for 5 to 10 minutes of inactivity, up to a maximum of one hour" (vs Anthropic's explicit 5-minute default TTL on the ephemeral block). Cache hits are reported in `usage.prompt_tokens_details.cached_tokens` on the chat-completion response. All five summarisation prompts in the test set (~2,680–3,300 input tokens) clear OpenAI's 1024-token threshold on every model, so caching engages across the entire summarisation set on the OpenAI side — wider coverage than the Anthropic side, which is constrained by Sonnet 4.6's 2048-token floor and is unavailable on Haiku 4.5 entirely.

### Caching test design (Day 5)

For each in-scope (model, prompt) pair, the runner records three calls: one baseline call (no `cache_control`), one cache-write call (with `cache_control`, on a cold cache), and one cache-read call (with `cache_control`, within the 5-minute TTL of the write). Three calls × five prompts × N models gives the empirical multipliers that the Day 12 analysis script then projects to amortised cost at any N reads, rather than pegging the result to one observed N. This is the cheaper path to a more general output (~£0.13 per model vs ~£0.26 for an N=5 amortisation observation), and the empirical write/read multipliers double as a sanity check on Anthropic's stated 1.25× / 0.1× pricing — divergence from those numbers is itself a finding worth surfacing. We use the default 5-minute TTL for cache writes (`cache_control: {"type": "ephemeral"}`, no `ttl` override). The 1-hour TTL option costs 2.0× rather than 1.25× for writes; testing the 1-hour TTL would change the break-even point but not the cache-read economics, which is the load-bearing measurement here.

### Caching measures both cost AND latency

The caching lever is reported as a pair of multipliers per (model, prompt): a **cost multiplier** (cache_read_cost / baseline_cost) and a **latency multiplier** (cache_read_latency / baseline_latency). Provider docs claim cache reads are dramatically faster as well as cheaper, but the latency claim is independent of the cost claim and worth measuring directly. The Day 12 analysis script must compute both ratios and the writeup must report both — production decisions about caching are sensitive to latency just as much as to cost (e.g. an interactive copilot may value sub-500ms latency over the 90% read discount). `latency_ms` is already captured per-call in the results schema; no additional instrumentation needed.

### Cross-provider test-set consistency

When the OpenAI runner lands and the caching lever is exercised on OpenAI models, it uses **the same five prompts (sum-001..005)** as the Anthropic Sonnet 4.6 measurement, not a different set. This isolates the "caching-mechanism difference between providers" variable cleanly: any divergence in the cost or latency multipliers between Sonnet 4.6 and GPT-4o is attributable to the providers' caching implementations (Anthropic's explicit `cache_control` opt-in vs OpenAI's automatic prompt caching), not to differences in the input prompts being cached. The two measurements are directly comparable because both providers' thresholds are cleared by all five prompts (see the per-provider specs above).
