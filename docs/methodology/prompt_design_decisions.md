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

These thresholds determine which (model, prompt) pairs the caching lever can measure. Per a `count_tokens` survey of all 20 summarisation prompts (verified 2026-05-04 against Anthropic's tokenizer for Sonnet 4.6), the corpus splits into three bands: easies (sum-001..007) at **1,186–1,789 tokens**, mediums (sum-008..014) at **2,875–3,323 tokens**, and hards (sum-015..020) at **3,158–3,838 tokens**. The actual range across all 20 summarisation prompts is therefore approximately **1,186–3,838 input tokens**. An earlier draft of this doc claimed "~2,680–3,300 input tokens" — that range applied to mediums and hards only; easies are below 2048 and would not engage caching on any Anthropic model in the test set.

Of the 20 summarisation prompts, **13 clear Sonnet 4.6's 2048-token floor (the mediums and the hards) and 0 clear Haiku 4.5's 4096-token floor**. The longest prompt in the corpus (sum-020 at 3,838 tokens) is still 258 tokens short of Haiku's threshold. Customer support, RAG, extraction, and reasoning prompts (all below 2048 tokens) are out of caching scope for every Anthropic model in the test set.

**Methodology consequence:** Day 5's caching lever measurements on Anthropic are scoped to **Sonnet 4.6 × {sum-015, sum-016, sum-017, sum-018, sum-020}** — the 5 longest hards, 3,348–3,838 tokens (see "Prompt subset selection" subsection below for rationale). Haiku 4.5 is reported as "caching unavailable at our prompt sizes" — a property of the prompt corpus relative to Haiku's higher threshold, not a property of the model. Where caching engages on a (model, prompt) pair we measure cost and latency multipliers; where it doesn't engage we record the unavailability with the threshold check that triggered it. The writeup limitations section notes the prompt-size threshold as the cause.

### OpenAI prompt-caching specs (for cross-provider comparison)

Per OpenAI's prompt-caching guide (verified 2026-05-04 against `developers.openai.com/api/docs/guides/prompt-caching`): caching activates automatically on prompts containing **1024 tokens or more** with no opt-in or API parameter required. Cache reads can reduce input token cost by **up to 90%** and latency by up to 80%. Cached prefixes "generally remain active for 5 to 10 minutes of inactivity, up to a maximum of one hour" (vs Anthropic's explicit 5-minute default TTL on the ephemeral block). Cache hits are reported in `usage.prompt_tokens_details.cached_tokens` on the chat-completion response. All five hards in our caching test set (sum-015..020 at 3,158–3,838 tokens) clear OpenAI's 1024-token threshold by a wide margin, so caching engages on every OpenAI (model, prompt) pair in the test — wider engagement than the Anthropic side, which is constrained by Sonnet 4.6's 2048-token floor and is unavailable on Haiku 4.5 across our entire corpus.

**OpenAI caches in 1024-token chunks (empirical observation, Day 5).** For sum-015 on GPT-4o (input 3,046 tokens after our `prompt_tokens − cached_tokens` normalisation, total prompt content 3,046 tokens), OpenAI cached exactly 2 × 1024 = 2,048 tokens on the cache_read call, leaving the remaining 998 tokens uncached and charged at the full input rate. Other prompts in the test set show similar chunked behaviour (cached_tokens values of 2,176 / 3,072 / 3,200 / 3,328 across the five prompts × two models). Production prompts whose total length isn't a clean multiple of 1024 will see this partial-caching effect; its impact on the observed cache_read multiplier is captured in the scaling formula in the *Scope and bias considerations* subsection below.

### OpenAI cache-warming asymmetry — write multiplier structurally unobservable

The 3-call test design (baseline / cache-write / cache-read) maps cleanly onto Anthropic's explicit `cache_control` opt-in: a baseline call with no `cache_control` does NOT touch cache; a write call with `cache_control` and a cold cache writes to it; a subsequent read call with `cache_control` hits the cache. The three calls produce three distinct measurements: baseline cost, cache-write cost (with the 1.25× write premium on the cached portion), and cache-read cost (with the 0.1× read discount).

OpenAI's automatic caching does not match this shape. Because the cache writes opportunistically on every API call regardless of opt-in, the "baseline" call (no special config) inadvertently warms the cache. By the time the second call ("cache-write" in our terminology) executes, the cache is already populated — making the labelled write call effectively a cache read. The Day 12 analysis surfaces this empirically by comparing the OpenAI write and read multipliers (which should be nearly identical, both reflecting cache hits) against the Anthropic write and read multipliers (which should differ materially, write at ~1.25× and read at ~0.1×).

**Anthropic write multiplier is observable; OpenAI's is not.** The OpenAI "write" column in the Day 12 analysis should be interpreted as a second observation of the cache-read multiplier rather than as an independent write measurement. The economically relevant production figure on OpenAI is the cache-read multiplier (cost of a cached call once the cache is warm); the OpenAI baseline call captures the cache-miss cost. Anthropic captures all three (uncached, write, read) cleanly — the asymmetry is a property of the providers' caching implementations, not the test design.

### Caching test design (Day 5)

For each in-scope (model, prompt) pair, the runner records three calls: one baseline call (no `cache_control`), one cache-write call (with `cache_control`, on a cold cache), and one cache-read call (with `cache_control`, within the 5-minute TTL of the write). Three calls × five prompts × applicable models gives the empirical multipliers that the Day 12 analysis script then projects to amortised cost at any N reads, rather than pegging the result to one observed N. The empirical write/read multipliers double as a sanity check on Anthropic's stated 1.25× / 0.1× pricing — divergence from those numbers is itself a finding worth surfacing. We use the default 5-minute TTL for cache writes (`cache_control: {"type": "ephemeral"}`, no `ttl` override). The 1-hour TTL option costs 2.0× rather than 1.25× for writes; testing the 1-hour TTL would change the break-even point but not the cache-read economics, which is the load-bearing measurement here.

**`cache_control` placement.** The runner places `cache_control: {"type": "ephemeral"}` on the user-message text content block (see `runners/run_anthropic.py::call_anthropic`). Per Anthropic's cache-breakpoint semantics, this caches everything from the start of the prompt up to and including the breakpoint — i.e. the system prompt + the user-message body (the deterministic prefix). The output (assistant response) is never cached. This placement is the load-bearing assumption for the cost-multiplier interpretation: the cached portion is the input/prompt content, the discount applies to that portion only, and the small uncached overhead Anthropic surfaces on cache_read calls (3 input_tokens for sum-015, similar for others) is structural Anthropic overhead beyond our cached content, not a runner artefact.

### Prompt subset selection

The five prompts used by the caching lever are **sum-015, sum-016, sum-017, sum-018, sum-020** — the 5 longest hards, at 3,348 / 3,422 / 3,609 / 3,471 / 3,838 input tokens respectively (sum-019 at 3,158 tokens is the smallest hard and was dropped to keep the set at 5). Selection rationale:

- **All 5 clear Sonnet 4.6's 2048-token floor by 1,300+ tokens**, providing comfortable margin against any tokenizer drift between the count_tokens survey and the live API call. Caching is observable on Sonnet 4.6 across the full 5-prompt sample.
- **All 5 sit 250–700 tokens below Haiku 4.5's 4096-token floor**, cleanly demonstrating Haiku unavailability as a structural property of the corpus rather than a borderline case. No prompt in the entire 20-prompt summarisation set clears Haiku's threshold.
- **All 5 clear OpenAI's 1024-token threshold by 2,300+ tokens**, so OpenAI auto-caching engages on every (model, prompt) pair. The Anthropic and OpenAI measurements use the same 5 prompts to isolate the caching-mechanism-difference variable.
- **Homogeneous selection (all hards) maximises the cached-portion signal at our narrow available range.** Mixing complexity tiers would have introduced cross-tier variance in the small N=5 sample without a corresponding methodological gain.

### Caching measures both cost AND latency

The caching lever is reported as a pair of multipliers per (model, prompt): a **cost multiplier** (cache_read_cost / baseline_cost) and a **latency multiplier** (cache_read_latency / baseline_latency). Provider docs claim cache reads are dramatically faster as well as cheaper, but the latency claim is independent of the cost claim and worth measuring directly. The Day 12 analysis script must compute both ratios and the writeup must report both — production decisions about caching are sensitive to latency just as much as to cost (e.g. an interactive copilot may value sub-500ms latency over the 90% read discount). `latency_ms` is already captured per-call in the results schema; no additional instrumentation needed.

Both multipliers are reported as a **range (min / median / max) across the N=5 prompt sample**, not solely as an average. With N=5 a mean alone hides heterogeneity that a reader extrapolating to production deserves to see; the range surfaces whether caching benefit is consistent across prompts of similar size or varies materially within the sample.

### Cross-provider test-set consistency

The caching lever uses **the same five prompts (sum-015, 016, 017, 018, 020)** on Anthropic and OpenAI test models, not a different set per provider. This isolates the "caching-mechanism difference between providers" variable cleanly: any divergence in the cost or latency multipliers between Sonnet 4.6 and GPT-4o is attributable to the providers' caching implementations (Anthropic's explicit `cache_control` opt-in vs OpenAI's automatic prompt caching, with the write-multiplier asymmetry noted above), not to differences in the input prompts being cached. The two measurements are directly comparable because both providers' thresholds are cleared by all five prompts (see the per-provider specs above).

### Scope and bias considerations

The caching lever measures cost and latency multipliers at conditions where caching engages. Provider minimum token thresholds (1024 for OpenAI; 2048 for Sonnet 4.6; 4096 for Haiku 4.5 and Opus) determine which (model, prompt) pairs produce a measurement vs which are reported as "caching unavailable at our prompt sizes." The "unavailable" designation reflects our test set selection, not a property of the underlying model — production workloads with longer prompts may see caching benefits we did not measure. Where caching engages, observed multipliers are reported as a range across the N=5 prompt sample, not solely as an average, to surface heterogeneity. The OpenAI write multiplier is structurally unobservable in our design (automatic caching writes on the baseline call); the OpenAI "write" measurement reported in our results reflects the cost of a cached call when the cache was warmed by a prior call, which is the economically relevant number for production reasoning.

Caching multipliers were measured at **3,348–3,838 input tokens specifically** (the sum-015..020 subset), not "summarisation prompts" generically. Production workloads with substantially different cached prefix sizes — particularly near or below provider thresholds — may see different multipliers. Multipliers near the threshold are likely to be more variable (smaller cached portion relative to overall request); multipliers well above the threshold (e.g. 10k-token prompts common in long-context RAG) may show more pronounced cache-read savings as the cached portion dominates the cost arithmetic.

**Applying our findings to a production workload — the explicit scaling formula.** The observed cache_read cost multiplier `M` decomposes as:

```
M ≈ (input_share × cached_input_discount) + (output_share × 1.0)
```

where `input_share = baseline_input_cost / total_baseline_cost` and `output_share = baseline_output_cost / total_baseline_cost`. The output term is `× 1.0` because caching never discounts output tokens. A production user computes their own `input_share` and `output_share` from their workload's input/output token ratio and the model's input/output rates.

For **Sonnet 4.6** ($3/MTok input, $15/MTok output) applied to our test set's ~3,500 input + ~220 output tokens:
- input_share = (3,500 × $3) / (3,500 × $3 + 220 × $15) = $10,500 / $13,800 ≈ **0.78**
- output_share ≈ **0.22**
- Predicted M ≈ 0.78 × 0.1 + 0.22 × 1.0 = 0.078 + 0.22 = **0.30**
- Observed median: **0.326×** — within ~10% of the formula. Anthropic's published 0.1× discount is being applied correctly on the cached portion; the 0.30–0.33 multiplier reflects the structural reality that 22% of total cost is output, which caching can't reach.

For **GPT-4o** ($2.50/MTok input, $10/MTok output) applied to our test set's ~3,000 input + ~130 output tokens:
- input_share = (3,000 × $2.50) / (3,000 × $2.50 + 130 × $10) = $7,500 / $8,800 ≈ **0.85**
- output_share ≈ **0.15**
- Predicted M (assuming all input cached at 0.5×) ≈ 0.85 × 0.5 + 0.15 × 1.0 = 0.425 + 0.15 = **0.58**
- Observed median: **0.617×** — slightly above the formula's floor because OpenAI's 1024-token chunked caching left ~33% of input uncached on at least one prompt in our sample (sum-015: 998 of 3,046 tokens uncached, charged at full input rate). The chunked-caching effect raises the effective `cached_input_discount` from 0.5 toward 1.0 in proportion to the uncached tail. A production user whose prompt is a clean multiple of 1024 will see closer to the formula's 0.58 floor; a user whose prompt has a substantial uncached remainder will see closer to our 0.617 observation.

For **GPT-4o-mini** ($0.15/MTok input, $0.60/MTok output) applied to our test set's ~3,000 input + ~130 output tokens:
- input_share = (3,000 × $0.15) / (3,000 × $0.15 + 130 × $0.60) = $450 / $528 ≈ **0.85**
- output_share ≈ **0.15** (GPT-4o-mini and GPT-4o share the same 4:1 input-to-output rate ratio, so the input/output cost shares are identical despite the 16× absolute price difference)
- Predicted M ≈ 0.85 × 0.5 + 0.15 × 1.0 = **0.58**
- Observed median: **0.591×** — closest match to the formula across the three engaging models, again with chunked-caching pushing slightly above the floor.

A production user applying these findings should plug in their own `input_share`, `output_share`, and (for OpenAI) the cached fraction implied by their prompt length modulo 1024. Workloads with a high output share (e.g. long-form generation from short prompts) will see shallower cost savings than ours; workloads with very long prompts well above 1024-token multiples will see deeper savings on OpenAI as the chunked-caching tail becomes a smaller fraction of total input.

### Day 6 model-currency revision: GPT-4o → GPT-5.4 family, OpenAI re-measurement

#### Why the swap

The Day 5 OpenAI numbers above were measured against `gpt-4o` and `gpt-4o-mini`, the most recent OpenAI flagship and budget-tier models *at the time of Day 5 (4 May 2026)*. Five days later, on Day 6 (8 May 2026), this turned out to be one major OpenAI generation behind. OpenAI's GPT-5 family launched August 2025; GPT-5.1, 5.2, 5.3 followed across late 2025 and Q1 2026; GPT-5.4 (`gpt-5.4-2026-03-05`) and `gpt-5.4-mini-2026-03-17` are the current production lineup at the time of the Day 6 dry-run; GPT-5.5 launched 23 April 2026 but does not have a `-mini` companion (only `gpt-5.5-pro`), so the cleanest flagship/budget pair for our benchmark is GPT-5.4 + GPT-5.4-mini. GPT-4o was retired from the consumer ChatGPT product on 13 February 2026; it remains callable via the API but on lower-priority infrastructure — we observed this empirically as a 4h+ batch-API queue wait on `gpt-4o-mini` during the Day 6 Layer 4 dry-run, with `gpt-4o` and `gpt-4o-mini` synchronous calls still working in seconds. Switching to GPT-5.4 family makes the benchmark numbers reflect what a developer choosing OpenAI in May 2026 actually picks.

The Day 5 GPT-4o numbers above are preserved as historical record. The figures cited in the published writeup are the GPT-5.4 numbers below. Both Anthropic models (Sonnet 4.6, Haiku 4.5) are unchanged — Anthropic's lineup is current.

#### What changed in pricing

GPT-5.4 family pricing (verified 8 May 2026 against `developers.openai.com/api/docs/pricing`, corroborated against `benchlm.ai` and `devtk.ai`):

| Model         | Input $/MTok | Output $/MTok | Cache-read multiplier | Δ vs GPT-4o equivalent |
| ------------- | -----------: | ------------: | --------------------: | ---------------------- |
| `gpt-5.4`      | 2.50         | 15.00         | **0.10**              | output 50% more expensive; cache-read 5× deeper discount |
| `gpt-5.4-mini` | 0.75         | 4.50          | **0.10**              | input 5×, output 7.5× more expensive than GPT-4o-mini; cache-read 5× deeper discount |

The cache-read multiplier shift is the biggest analytic change: 0.5× → 0.1×, matching Anthropic's published rate. **This invalidates the Day 5 GPT-4o cache_read multipliers as production-relevant figures**; the GPT-5.4 re-measurement below is the figure to cite.

#### Re-measurement: caching multipliers on GPT-5.4 family (8 May 2026, sum-015..020)

Running the same caching test (5 longest summarisation prompts: sum-015, 016, 017, 018, 020 at 3,158–3,838 Anthropic-counted tokens — well above the 1024-token OpenAI caching threshold) against the new models:

| Model          | cost_write multiplier (min/median/max) | cost_read multiplier (min/median/max) | latency_read multiplier |
| -------------- | -------------------------------------- | ------------------------------------- | ----------------------- |
| `gpt-5.4`      | 0.428 / 0.511 / 0.952×                 | **0.411 / 0.475 / 0.494×**            | 0.816 / 0.942 / 0.999×  |
| `gpt-5.4-mini` | 0.390 / 0.423 / 0.456×                 | **0.377 / 0.425 / 0.482×**            | 0.816 / 0.978 / 1.144×  |

**Comparison vs Day 5 GPT-4o numbers** — improvement is real but smaller than the naïve 5× cache_read_mult shift would suggest:

- GPT-4o cost_read median **0.617×** → GPT-5.4 cost_read median **0.475×** (~**23% improvement** in the cost-saving ratio)
- GPT-4o-mini cost_read median **0.591×** → GPT-5.4-mini cost_read median **0.425×** (~**28% improvement**)

**Why the improvement is ~25–30% instead of 5×.** Re-applying the cost-dilution formula `M ≈ (input_share × cached_input_discount) + (output_share × 1.0)` with the new pricing:

For **GPT-5.4** ($2.50/MTok input, $15.00/MTok output) on ~3,000 input + ~225 output tokens:
- input_share = (3,000 × $2.50) / (3,000 × $2.50 + 225 × $15.00) = $7,500 / $10,875 ≈ **0.69**
- output_share ≈ **0.31** (vs 0.15 on GPT-4o — output dilution doubled because GPT-5.4 charges 50% more per output token)
- Predicted M (assuming all input cached at 0.1×) ≈ 0.69 × 0.10 + 0.31 × 1.0 = 0.069 + 0.31 = **0.38**
- Observed median: **0.475×** — above the formula's floor for the same chunked-caching reason as Day 5 (cached_tokens land in 1024-token blocks, leaving an uncached tail that's billed at full rate; sum-015 at 3,000+ tokens has ~1,000 token tail uncached, raising the effective cached_input_discount from 0.10 toward ~0.40 in proportion).

For **GPT-5.4-mini** ($0.75/MTok input, $4.50/MTok output) on the same prompts:
- input_share = (3,000 × $0.75) / (3,000 × $0.75 + 225 × $4.50) = $2,250 / $3,263 ≈ **0.69**
- output_share ≈ **0.31** (mini and flagship share the same 6:1 input-to-output rate ratio, so input/output cost shares are identical)
- Predicted M ≈ 0.69 × 0.10 + 0.31 × 1.0 = **0.38**
- Observed median: **0.425×** — closer to the formula floor than GPT-5.4. The smaller absolute prices may make 1024-token tail effects less impactful in dollar terms, but the percentage shape is the same.

**The headline finding for the writeup is the doubled output-share.** Going from GPT-4o to GPT-5.4 raised output cost share from ~0.15 to ~0.31 of total bill on our prompt sizes — output cost now dominates input cost more than it did, so even a perfect 0× cache_read multiplier on the input portion couldn't drop total cost below the output-share floor (~0.31 here). Day 12 production-recommendation framing should emphasise this: caching's headline cost-saving on GPT-5.x is bounded by output share much more tightly than on GPT-4o, despite the deeper underlying input discount.

**Cache-write multiplier note.** The 0.952× outlier for sum-018 on GPT-5.4 is consistent with the OpenAI cache-warming asymmetry documented above (auto-caching populates the cache on prior calls within the 5-10 min TTL window; the labelled "write" call may already see a partial cache hit). The GPT-5.4-mini write multipliers (0.39–0.46) cluster cleanly with its read multipliers — both are observations of the same underlying cached-call cost. The Anthropic cache-write multiplier remains the only directly observable write number.

#### Day 6 finding: OpenAI auto-caching is account-level and contaminates baseline measurements across sessions

OpenAI's automatic prompt caching is **not session-bound** — it lives at the API account level and persists for the documented 5–10 minutes of inactivity (up to one hour). Any prior call against the same prompt + model leaves cache state that contaminates subsequent "baseline" measurements: the next caller sees `cached_tokens > 0` on what is supposed to be a cold call, and gets billed at the cache-read rate for the cached portion.

This was observed empirically during the Day 6 Layer 4 dry-run. The dry-run's `run_baseline` phase on `gpt-5.4-2026-03-05` produced rows with `cached_tokens=2816` for both sum-015 and sum-020 — because the GPT-5.4 caching smoke test had run on the same prompts ~2 minutes prior. The dry-run's "sync baseline" cost numbers for gpt-5.4 were ~50% lower than they should have been; methodologically the baseline measurement was invalid for those rows.

**Day 7 risk.** The full benchmark's baseline phase MUST run on cold cache state. If smoke testing or any prior runs touched the prompts in the last ~10–15 minutes, baseline measurements for those prompts will be artificially cheap and methodologically invalid. Mitigation in code: `runners/orchestrator.run_baseline` checks each result row for `cached_tokens > 0` and emits a `phase='baseline', event='warning', payload.warning='baseline_cache_contamination'` event to the phase log. This does not prevent the contaminated measurement (caching is server-side and unstoppable from our side), but creates an auditable signal so Day 12 analysis can flag affected rows. Operational mitigation: schedule Day 7's baseline phase as the first call after a ≥15 min quiet window, OR use prompts that have never been touched by smoke runs in the same calendar day. The smoke testing phases (Day 5–6) by design touched only sum-015..020 plus cs-001..005; the production baseline will hit the full 102 prompts, of which ~92 will be cold by virtue of never having been run before in any context.

Anthropic's `cache_control: ephemeral` is opt-in: baseline calls without `cache_control` do NOT engage Anthropic's cache regardless of prior calls. This contamination risk is OpenAI-specific.

#### GPT-5.4 reasoning_effort and temperature interaction (empirical decomposition)

GPT-5.4 (reasoning model family) does not accept `temperature=0` when `reasoning_effort` is set explicitly to `'low'`; the API rejects with `400 BadRequest`. The benchmark therefore uses the API default `reasoning_effort='medium'` with `temperature=0` (matching all other models in the matrix). A controlled 2×2 experiment on cs-001..005 (5 prompts × 2 GPT-5.4 models) decomposed the variable contributions: `temperature 0→1` inflates cost by 52% on flagship and 93% on mini (primarily through 1.8–2.5× longer outputs from sampling diversity); `reasoning_effort medium→low` at `temperature=1` reduces cost by 6% on flagship and 24% on mini (the actual reasoning overhead). The cheapest available configuration is `medium + temp=0`, which is 43–46% cheaper than `low + temp=1`. Production teams setting `reasoning_effort='low'` explicitly for non-reasoning workloads should be aware that the API forces `temperature=1` in that mode, and the temperature side-effect costs more than the reasoning_effort saves.

The decomposition data lives in the `results` table tagged with `optimisation_config.experimental_comparison=true` (50 rows across two run_ids covering medium+temp=1, low+temp=1, plus yesterday's medium+temp=0 baseline). Day 12 main analysis filters those rows out to avoid mixing methodology-comparison data with the production benchmark.

A separate compounding finding from the same re-measurement: caching engagement on GPT-5.4 is flaky in some configurations (1 of 5 prompts on `gpt-5.4` showed cache-read returning `cached_tokens=0` despite cache-write succeeding). Day 7 production tolerates this via the `caching_unavailable` row pattern that mirrors `compression_unavailable` — a single-prompt cache miss no longer crashes the full benchmark sweep.

## Day 6+ orchestration: batch submit/retrieve split, compression timing, dynamic budget gate

Three architectural decisions about the Day 6+ runner orchestration, captured before the lever modules and orchestrator land.

### Batch API integration: separate submit and retrieve operations

The batch lever splits into two methods rather than one blocking call. `submit_batch` writes a row to a new `batch_jobs` table (`batch_id`, `run_id`, `provider`, `model`, `lever`, `status`, `submitted_at`, `retrieved_at`, `completed_at`, `prompt_ids` as a JSON array, `request_count`, `error`) and returns immediately. `retrieve_batch` polls the provider's batch-status endpoint, pulls completed results when ready, and writes them to the `results` table. Per PRD §9, Day 7 calls `submit_batch` alongside the synchronous baseline run; Day 8 starts by calling `retrieve_batch` to collect the results from Day 7's submissions.

The split is load-bearing for two reasons. First, batch processing on both providers takes between 1 and 24 hours of provider-side queueing time that the orchestrator script cannot productively wait on without blocking the rest of Day 7's work. Second, the `batch_jobs` table makes the in-flight-jobs state recoverable across script restarts: if the orchestrator crashes between submit and retrieve, the `batch_id` is persisted and retrieval resumes cleanly on any subsequent run. A single blocking call would lose state on failure and forfeit the batch discount on the next attempt — the original submission is still queued at the provider and would be billed regardless.

### Compression timing: runtime compression inside the lever, not preprocessing

The compression lever invokes LLMLingua-2 at call time, transforming the prompt body before the API request, rather than pre-computing compressed prompts and storing them as artefacts. LLMLingua-2 compression timing measured 2026-05-05 on Mac CPU against actual benchmark prompts: **cold path** (first compression after model load) **1.50s against sum-001** at 1,589 Anthropic input tokens (LLMLingua-2's BERT tokenizer counts 1,379); **warm path** (subsequent compression, model in memory) **1.59s against sum-020** at 3,838 Anthropic input tokens (LLMLingua-2 count 3,356). Compression ratios observed (compressed / original by LLMLingua-2's count): **50.5% on sum-001, 48.5% on sum-020** — both close to the requested `rate=0.5` target.

Compression ratios reported by LLMLingua-2 are computed in its own tokenizer's counts; the actual cost saving on the API side is determined by Anthropic's re-tokenization of the compressed string. The lever measures both: `original_input_tokens` and `compressed_input_tokens` in `optimisation_config` are recorded using Anthropic's `count_tokens` for both values, ensuring the compression ratio Day 12 analyses reflects what was actually billed. **This is a binding requirement on `lever_compression.py`: it must call Anthropic's `count_tokens` against the compressed string to get the billable token count, not record LLMLingua-2's claimed compressed count.** The `optimisation_config` column is `TEXT` in SQLite (functionally JSON via the json1 extension); typical compression configs serialise to ~80 bytes, well below any practical limit.

**Compression target vs billed reduction (Day 6 finding).** LLMLingua-2 with `rate=0.5` produced **47.7% reduction** in its own BERT-tokenizer counts on sum-015 (3,055 → 1,458 BERT tokens) but only **44.5% reduction** in Anthropic's `count_tokens` (3,348 → 1,858 Anthropic tokens). The ~3 percentage-point gap reflects that LLMLingua-2 optimises sequence selection for the tokenizer it was trained against; the resulting compressed string then re-tokenises differently in production tokenizers. Production developers enabling LLMLingua-2 expecting `rate=0.5` to mean 50% billed input reduction will observe a consistently lower (~45%) actual reduction. The Day 12 analysis reports both LLMLingua-2 BERT-counted ratios (for comparing against the compressor's stated behaviour) and Anthropic/OpenAI `count_tokens`-based ratios (for production cost reasoning). The cross-provider implication is one step further: the same compressed string re-tokenised by OpenAI's `tiktoken` (`o200k_base` on the GPT-4o family) will produce yet another count, so the ~45% Anthropic figure is not directly portable across providers either.

The 30-prompt compression run on Day 8 is bounded at ~30 × 1.6s ≈ **48s** of total CPU overhead. **Init cost: 8.6s per orchestrator process** (model load from local disk; one-time per session, paid once per Day 8 run, paid again on any restart). The 48s compression total is exclusive of this 8.6s init. Runtime compression keeps the lever's contract self-contained (input prompt → output result, with compression as one of the runner's internal levers) and avoids a separate preprocessing artefact that would need its own hash-keyed cache, refresh logic, and methodology footprint.

### Orchestrator semantics: dynamic phased plan with budget gate before compression

The orchestrator runs phases in sequence: `baseline` → `caching` → `output_cap` → `batch_submit` → (Day 7 ends; Day 8 starts) → `batch_retrieve` → `budget_check` → `compression_decide` → `compression_run`. The budget gate sits between `batch_retrieve` and `compression_run`, implementing the four-tier ladder from PRD §9 Day 8: full matrix if >£180 remaining under the £300 cap, 60-prompt stratified subset if >£120, 30-prompt subset if >£80, operator's call (15-prompt subset or skip) if £40–80, skip entirely if <£40. The `compression_decide` phase reads `runs.cost_so_far_gbp`, computes the chosen tier, and writes the decision (tier name + rationale + remaining headroom) to a phase log; `compression_run` then iterates the chosen subset. Phase ordering is dynamic in the sense that `compression_run`'s behaviour depends on the budget state at execution time rather than on a predetermined config — the orchestrator records the actual decision in the run log, so the Day 12 analysis and the writeup can cite which compression tier actually ran (and why) rather than guessing.

## Skip-if-exists semantics

The orchestrator's skip-if-exists logic was hardened during Day 6–9 in two passes. First, batch and sync result rows are now distinguished as separate levers (`lever='batch'` vs `lever='baseline'`) so the same prompt+model can produce both rows in a single run. Second, the skip-if-exists query was made `run_id`-aware so each run produces independent measurements rather than reusing rows from prior runs. The latter fix surfaced during Day 9 backfill when 33 expected baseline/output_cap/compression rows were absent from the production run because prior dry-run rows were silently blocking insertion via shared `(prompt_id, model, lever, config_hash, run_attempt)` keys. Both fixes ensure runs produce reproducible, isolated datasets.

The fix required pairing application-level scoping with a corresponding schema UNIQUE constraint update — the application logic determines when skip-if-exists fires, but the schema constraint determines what duplicate-prevention the DB enforces. Either alone is insufficient: app-only would produce IntegrityErrors at insert time; schema-only would allow duplicate rows to accumulate. Both layers must agree on what "duplicate" means.

## Tier-1 deterministic scoring (Day 9)

Tier-1 scoring derives criteria from each prompt's `tier_1_deterministic.expected` block — never from inspecting the model's response. A single normalisation pipeline (whitespace strip, markdown-fence strip, pre-JSON preamble strip, conditional smart-quote translation, single-level wrapper unwrap) is applied to every response regardless of provider/model; the steps that fired on each row are recorded in the new `normalisation_steps_applied` column so any provider-style asymmetry is visible in Day 12 analysis rather than hidden. Lever-aware pre-checks distinguish design-induced failure (output_cap truncation, `compression_unavailable` rows) from genuine model failure: a response at exactly the 200-token cap that parses cleanly is a genuine pass, not truncation; a response at the cap that fails to parse is `truncated_due_to_cap`, not `fail_format`. rag_qa answers split into a short-factoid bucket (≤3 words → expected substring of model answer) and a phrasal bucket (≥7 words → answer content deferred to Tier-2 judges, citations still scored deterministically); the corpus has zero answers in the 4–6 word gap, so the bucketing is unambiguous.

### Pre-Tier-2 directional findings from the Day 9 dry run

(1) **Compression as quality-destroyer.** Tier-1 deterministic checks show 35–50 percentage point pass rate loss across all four models when compression is applied to extraction-style prompts (LLMLingua-2 destroys field markers required for downstream parsing). The 15% cost savings come at material correctness cost; cost ratios in Day 12 must be quality-adjusted.

(2) **Batch as cost-optimisation-not-quality-loss.** Batch lever rows show pass rates within 1–2 percentage points of baseline rows across all four models, validating that the published 50% batch discount is real with quality preserved.

These are pre-Tier-2 directional signals; Day 10 judge scoring will refine them.
