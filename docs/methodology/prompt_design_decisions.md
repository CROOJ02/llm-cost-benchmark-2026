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

### OpenAI cache-warming asymmetry — write multiplier structurally unobservable

The 3-call test design (baseline / cache-write / cache-read) maps cleanly onto Anthropic's explicit `cache_control` opt-in: a baseline call with no `cache_control` does NOT touch cache; a write call with `cache_control` and a cold cache writes to it; a subsequent read call with `cache_control` hits the cache. The three calls produce three distinct measurements: baseline cost, cache-write cost (with the 1.25× write premium on the cached portion), and cache-read cost (with the 0.1× read discount).

OpenAI's automatic caching does not match this shape. Because the cache writes opportunistically on every API call regardless of opt-in, the "baseline" call (no special config) inadvertently warms the cache. By the time the second call ("cache-write" in our terminology) executes, the cache is already populated — making the labelled write call effectively a cache read. The Day 12 analysis surfaces this empirically by comparing the OpenAI write and read multipliers (which should be nearly identical, both reflecting cache hits) against the Anthropic write and read multipliers (which should differ materially, write at ~1.25× and read at ~0.1×).

**Anthropic write multiplier is observable; OpenAI's is not.** The OpenAI "write" column in the Day 12 analysis should be interpreted as a second observation of the cache-read multiplier rather than as an independent write measurement. The economically relevant production figure on OpenAI is the cache-read multiplier (cost of a cached call once the cache is warm); the OpenAI baseline call captures the cache-miss cost. Anthropic captures all three (uncached, write, read) cleanly — the asymmetry is a property of the providers' caching implementations, not the test design.

### Caching test design (Day 5)

For each in-scope (model, prompt) pair, the runner records three calls: one baseline call (no `cache_control`), one cache-write call (with `cache_control`, on a cold cache), and one cache-read call (with `cache_control`, within the 5-minute TTL of the write). Three calls × five prompts × applicable models gives the empirical multipliers that the Day 12 analysis script then projects to amortised cost at any N reads, rather than pegging the result to one observed N. The empirical write/read multipliers double as a sanity check on Anthropic's stated 1.25× / 0.1× pricing — divergence from those numbers is itself a finding worth surfacing. We use the default 5-minute TTL for cache writes (`cache_control: {"type": "ephemeral"}`, no `ttl` override). The 1-hour TTL option costs 2.0× rather than 1.25× for writes; testing the 1-hour TTL would change the break-even point but not the cache-read economics, which is the load-bearing measurement here.

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
