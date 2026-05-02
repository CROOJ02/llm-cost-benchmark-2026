# LLM Cost Benchmark 2026 — PRD

**Project:** Public empirical research artefact supporting InferOps positioning
**Version:** 1.0
**Date:** April 2026
**Status:** Pre-build
**Owner:** James Crooks
**Repository:** `/Users/jamescrooks/llm-cost-benchmark-2026/` (local)
**Publish target:** Wednesday 14 May 2026
**API budget cap:** £300 hard, £250 soft warning

---

## 1. Why This Exists

InferOps is currently in Phase 0.5 validation. Cold outreach to Heads of AI has produced zero replies on 50 emails. Deliverability is one suspected cause but a deeper problem is that we have no concrete artefact to anchor a credibility play in our outreach. We're a solo founder with a landing page and a thesis, asking Heads of AI for 20 minutes of their time.

This benchmark raises the credibility floor. We will run an empirical study answering one specific question:

_Across the AI workloads enterprises actually run in production in 2026, where does the boundary fall between tasks where cheaper models produce equivalent quality and tasks where they don't? And how do other cost-optimisation levers (caching, output capping, batch processing, prompt compression) affect that boundary?_

The artefact's job is not to be the InferOps product. It is a terminal piece of work — built, shipped, walked away from. It serves three goals:

1. **Generate a non-pitchy artefact** for cold email referencing ("I just shipped a 2026 benchmark on which AI tasks can safely run on cheaper models — curious whether the patterns match what you're seeing internally")
2. **Empirically test the central InferOps claim** that safe cost optimisation is possible in real AI workloads, with public evidence rather than client conversations we can't share
3. **Build credibility** on the measurement half of InferOps so discovery calls have somewhere to go

This is research, not product. Code, data, and writeup are all public. Methodology is published in full so others can reproduce or critique. The benchmark is downstream of InferOps positioning — it serves the cold outreach campaign and Phase 0.5 discovery calls.

---

## 2. Scope

### In Scope

- **5 task categories** grounded in 2026 production AI use cases
- **20 prompts per category** = 100 prompts total, varying complexity (easy/medium/hard)
- **4 models tested**: Claude Sonnet 4.6, Claude Haiku 4.5, GPT-4o, GPT-4o-mini
- **5 optimisation levers**: baseline, prompt caching, output capping, batch processing, prompt compression
- **Tiered scoring**: deterministic rubrics (Tier 1), dual-judge blind evaluation (Tier 2), human arbitration on disagreements (Tier 3)
- **Public writeup** of 2,500–3,500 words covering methodology, findings, cross-category patterns, and InferOps positioning paragraph
- **Reproducible artefact** — all prompts, all data, all rubrics, and all runners published

### Explicit Non-Goals

The non-goals matter more than the goals. Without explicit non-goals, scope creeps.

- **Not a product.** No web UI, no SaaS dashboard, no recommendations engine. Captured data is inspected via SQL queries and Python scripts.
- **Not the InferOps SDK.** The runners share conceptual lineage with InferOps's instrumentation but are not the production codebase. Code from this repo will not be lifted into InferOps. Any reusability is incidental.
- **Not Gemini, Mistral, or Cohere.** Four models, two providers. Coverage limitation acknowledged in writeup.
- **Not multi-modal.** Text in, text out. No vision, audio, or document understanding.
- **Not statistically rigorous.** 20 prompts per category supports directional findings, not peer-reviewed statistical claims. Writeup framing is "exploratory benchmark" not "definitive benchmark."
- **Not a routing-layer study.** Calls go directly to provider native SDKs. LiteLLM, LangChain, and other abstraction layers are addressed separately in InferOps's product PRD.
- **Not a quality-of-output study beyond cost-relevance.** We measure whether outputs meet a defined rubric; we don't analyse style, creativity, or qualitative differences beyond what the rubric captures.

If at any point we find ourselves implementing what looks like an InferOps SDK feature, we stop and rescope. The test: at the end of two weeks, do we have a published writeup or do we have unfinished tooling?

---

## 3. The Five Task Categories

Each category gets 20 prompts, 7 easy / 7 medium / 6 hard. All prompts use synthetic data only — no real names, real email addresses, real company names, or any data that could be misread as belonging to a real person. Use placeholders: `Customer A`, `customer@example.com`, `Acme Co`.

### Category 1 — Customer Support Classification + Reply

**Why this category:** customer support automation is the single largest production AI workload in 2026 (Klarna, Bank of America, Octopus Energy etc). High volume, high spend, often over-modelled.

**Input shape:** A customer email (50–300 words) describing an issue with a fictional product or service.

**Task:** (a) Classify the issue into one of {billing, technical, feature_request, complaint, other}. (b) Draft a 2-sentence acknowledgement reply.

**Output format:** JSON `{"category": "...", "reply": "..."}`

**Scoring tier:** Tier 1 deterministic (classification matches expected) + Tier 2 dual-judge (reply quality).

### Category 2 — RAG-based Q&A

**Why this category:** dominant pattern in B2B SaaS copilots. Retrieve, then answer with citation. Long contexts, often Sonnet by default.

**Input shape:** A retrieved context paragraph (500–1500 words) plus a specific question.

**Task:** Answer the question using only the context. Cite which sentence(s) support the answer.

**Output format:** JSON `{"answer": "...", "supporting_sentences": [1, 3]}`

**Scoring tier:** Tier 1 deterministic (correct answer + correct supporting sentence indices) + Tier 2 dual-judge (answer phrasing quality).

### Category 3 — Structured Data Extraction

**Why this category:** common in form processing, invoice handling, lead enrichment. High volume, classification-like outputs, frequently over-modelled.

**Input shape:** Unstructured text (50–400 words) containing extractable fields — invoice text, customer message, document.

**Task:** Extract specified fields as JSON.

**Output format:** JSON matching prompt-specified schema (e.g. `{"name": "...", "email": "...", "amount": ..., "date": "..."}`)

**Scoring tier:** Tier 1 deterministic (each field correct = 1 point, normalised).

### Category 4 — Document Summarisation

**Why this category:** universal across knowledge work — meeting notes, articles, threads, reports.

**Input shape:** A 1500–3000 word document (article, meeting transcript, report).

**Task:** Produce a 3-bullet summary covering the main points without hallucinating.

**Output format:** Plain text with 3 bulleted points.

**Scoring tier:** Tier 2 dual-judge only. Judges score on (a) coverage of main points, (b) accuracy / no hallucinations, (c) appropriate length.

### Category 5 — Multi-Step Reasoning

**Why this category:** agentic patterns where waste compounds. Most fragile category methodologically.

**Risk note:** This category is flagged as the highest-risk in the benchmark. If by end of Day 3 we cannot design 20 prompts that reliably differentiate model capability, the category is **dropped from v1** and replaced with additional prompts in Categories 1–4. The writeup explicitly notes the limitation.

**Input shape:** A scenario requiring 2–4 logical steps to reach a final answer (e.g. proration calculation, multi-condition eligibility check, simple workflow planning).

**Task:** Provide reasoning and final answer.

**Output format:** JSON `{"reasoning": "...", "final_answer": "..."}`

**Scoring tier:** Tier 1 deterministic (final answer correctness) + Tier 2 dual-judge (reasoning plausibility).

---

## 4. Models Tested

Four models, two providers. Captured `model_version` strings on every API call to track silent updates.

| Tier | Model             | Provider  | Why included                                        |
| ---- | ----------------- | --------- | --------------------------------------------------- |
| Top  | Claude Sonnet 4.6 | Anthropic | Industry-default top-tier for B2B SaaS AI           |
| Top  | GPT-4o            | OpenAI    | Most-used proprietary alternative to Sonnet         |
| Mid  | Claude Haiku 4.5  | Anthropic | Industry-default cheaper-tier comparison for Sonnet |
| Mid  | GPT-4o-mini       | OpenAI    | Industry-default cheaper-tier comparison for GPT-4o |

**Models explicitly excluded from v1:** Gemini Flash 2.5, Mistral Large, Cohere Command R+, open-source models (Llama, DeepSeek, Qwen). Writeup acknowledges this is a coverage limitation. Future rounds may extend.

**Temperature:** 0 on all calls. Reproducibility caveat: output token counts and response text vary ~5% even at temperature 0. Cost varies proportionally.

---

## 5. Optimisation Levers

| Lever                      | Sonnet 4.6                         | Haiku 4.5                          | GPT-4o                         | GPT-4o-mini                    |
| -------------------------- | ---------------------------------- | ---------------------------------- | ------------------------------ | ------------------------------ |
| Baseline (no optimisation) | ✓                                  | ✓                                  | ✓                              | ✓                              |
| Prompt caching             | ✓ (configured via `cache_control`) | ✓ (configured via `cache_control`) | ✓ (observed automatic caching) | ✓ (observed automatic caching) |
| Output cap                 | ✓ (`max_tokens=200`)               | ✓                                  | ✓                              | ✓                              |
| Batch processing           | ✓ (Anthropic batch API)            | ✓                                  | ✓ (OpenAI batch API)           | ✓                              |
| Prompt compression         | ✓ (LLMLingua-2 client-side)        | ✓                                  | ✓                              | ✓                              |

**Day 1 verification:** before committing to the full matrix, verify on Day 1 that LLMLingua-2 runs acceptably on James's Mac (CPU, no GPU). If it doesn't, the compression lever is dropped and noted as a documented limitation.

**Total runs:**

- Baseline: 100 prompts × 4 models = 400
- Caching: 100 × 4 = 400
- Output cap: 100 × 4 = 400
- Batch: 100 × 4 = 400
- Compression: 30-prompt stratified subset × 4 models = 120 (conditional on Day 8 budget gate — see §9 Day 8; or 0 if dropped from Day 1 verification)
- **Total: 1,720 model runs** (1,600 if compression skipped for budget or technical reasons)

**Plus dual-judge layer:** Tier 2 prompts × 4 models × 2 judges per lever, plus a smaller proportional set across the compression subset. Order of ~2,000 total judgement calls (~1,900 if compression skipped). Cheaper per call than the model runs.

**Compression as a subset, not the full matrix:** running compression on the full 100-prompt matrix would consume ~£30–60 of headroom that's better reserved for completion and the dual-judge layer. The 30-prompt stratified subset keeps the compression finding directional (one cost-quality datapoint per category) rather than full-coverage. The subset is stratified across the 5 categories so each contributes to the signal.

**Estimated total cost:** £150–280 across all model runs and judgement calls. Hard cap **£300**, soft warning **£250**.

---

## 6. Data Schema

### Prompt Files

Stored as JSON files, one per category: `prompts/customer_support.json`, `prompts/rag_qa.json`, `prompts/extraction.json`, `prompts/summarisation.json`, `prompts/reasoning.json`.

Each file contains an array of prompt objects:

```json
{
  "prompt_id": "cs-001",
  "task_category": "customer_support",
  "complexity": "medium",
  "input": {
    "system": "You classify customer emails and draft acknowledgement replies.",
    "user": "Hi, my last invoice charged me £150 but my plan is supposed to be £99..."
  },
  "scoring": {
    "tier_1_deterministic": {
      "expected": { "category": "billing" }
    },
    "tier_2_judge": {
      "criteria": "Reply acknowledges the issue and indicates investigation without committing to specific outcomes."
    }
  },
  "metadata": {
    "input_tokens_approx": 65,
    "notes": "Standard billing dispute, clear category"
  }
}
```

### SQLite Schema

Single database file: `data/results.db`.

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    cost_so_far_usd REAL DEFAULT 0,
    cost_cap_usd REAL NOT NULL,
    status TEXT NOT NULL  -- 'running' / 'completed' / 'aborted_cost' / 'aborted_error'
);

CREATE TABLE results (
    -- Identification
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    run_attempt INTEGER DEFAULT 1,

    -- What was tested
    prompt_id TEXT NOT NULL,
    task_category TEXT NOT NULL,
    complexity TEXT NOT NULL,

    -- Which model + how
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    optimisation_lever TEXT NOT NULL,
    optimisation_config TEXT,
    config_hash TEXT NOT NULL,

    -- Cost and performance
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER DEFAULT 0,
    latency_ms INTEGER NOT NULL,
    cost_usd REAL NOT NULL,

    -- Output
    response_text TEXT NOT NULL,
    response_parsed TEXT,
    output_format_valid INTEGER DEFAULT 1,

    -- Scoring
    rubric_score REAL,
    judge_a_score REAL,
    judge_b_score REAL,
    judge_disagreement_flag INTEGER DEFAULT 0,
    human_score REAL,
    final_score REAL,
    score_recomputed_at TEXT,

    -- Reproducibility
    model_version TEXT,
    temperature REAL DEFAULT 0,
    error TEXT,

    UNIQUE(prompt_id, model, optimisation_lever, config_hash, run_attempt)
);

CREATE INDEX idx_results_prompt ON results(prompt_id);
CREATE INDEX idx_results_model ON results(model);
CREATE INDEX idx_results_lever ON results(optimisation_lever);
CREATE INDEX idx_results_run ON results(run_id);
```

---

## 7. Scoring Methodology

### Tier 1 — Deterministic Rubrics

Categories: classification (Cat 1 partial), Q&A `supporting_sentences` only (Cat 2 partial — see RAG note below), structured extraction (Cat 3), reasoning final-answer correctness (Cat 5 partial).

Implementation: a Python function per category in `scoring/tier_1.py` that takes the parsed response and the expected output, returns a 0.0–1.0 score. Deterministic, repeatable.

**RAG Q&A scoring split (revision applied 2026-04-30 after Day 2 prompt review):**

- Tier 1 deterministic scoring for RAG covers ONLY the `supporting_sentences` integer array. The `answer` text field is scored entirely by the Tier 2 dual-judge layer against the prompt's `tier_2_judge.criteria`. The expected answer in `tier_1_deterministic.expected.answer` serves as the reference answer provided to judges, not as a verbatim string-match target.
- For `supporting_sentences`, the rubric accepts the minimal citation set that supports the answer, plus any superset that includes additional sentences explicitly establishing the question's subject or context. A model citing `[answer_sentence]` and a model citing `[subject_sentence, answer_sentence]` both score 1.0. A model citing unrelated sentences or missing the answer sentence scores 0.0.

### Tier 2 — Dual-Judge Blind Evaluation

Categories: customer support reply quality (Cat 1 partial), Q&A phrasing (Cat 2 partial), summarisation (Cat 4), reasoning plausibility (Cat 5 partial).

**Two judges, different model families:**

- Judge A: Claude Opus 4.6 (Anthropic family — same family as Sonnet/Haiku in test set, residual bias acknowledged)
- Judge B: Mistral Large (different family entirely — primary bias control)

Both judges are configured to score on the prompt-specific Tier 2 criteria.

**Blinding mechanism:**

- Each prompt has 4 model responses (one per model in test set)
- Responses are labelled Response A, Response B, Response C, Response D in randomised order _per prompt_
- The judge is shown all 4 responses with the prompt and rubric, and scores each on a 0.0–1.0 scale
- The judge does not know which model produced which response
- Position randomisation per prompt controls position bias across the dataset

**Scoring math:**

- Each response gets two judge scores (`judge_a_score`, `judge_b_score`)
- If both judges agree within 0.2 (i.e. |a - b| ≤ 0.2): score = median(a, b), no disagreement flag
- If judges disagree by more than 0.2: `judge_disagreement_flag = 1`, sent to Tier 3

**Partial-credit guidance (Day 10 judge prompt template):**

When the judge prompt template is built on Day 10, it must include explicit partial-credit instructions. Some prompts test multiple facts in a single answer (e.g. RAG questions like "X and Y?" where the model could be right on X but wrong on Y). Judges should:

- Score 1.0 when the response satisfies all criteria fully and accurately
- Score 0.5–0.7 when the response covers part of the criteria correctly but is missing one or more components
- Score 0.0–0.3 when the response is wrong on the central facts or misleading
- Use intermediate values where appropriate; do not collapse to only 0 or 1

### Tier 3 — Human Arbitration on Disagreements Only

James reads only the cases where the two judges disagreed substantively. Estimated 10–25 cases across the full benchmark, taking 30–60 minutes total.

Process:

1. Open `scoring/disagreements.csv` generated automatically from the database
2. For each row, read the prompt, read the response, score 0.0–1.0 with brief justification
3. Save and run `scoring/finalise_scores.py` to compute `final_score` for arbitrated rows

The final_score logic:

- Tier 1 prompts: `final_score = rubric_score`
- Tier 2 prompts (no disagreement): `final_score = median(judge_a_score, judge_b_score)`
- Tier 2 prompts (disagreement): `final_score = human_score`
- Mixed Tier 1+2 prompts: `final_score = 0.5 * rubric_score + 0.5 * tier_2_score`

Recomputation: if scoring logic changes, run `scoring/finalise_scores.py --recompute-all`. Updates `score_recomputed_at` for all affected rows.

---

## 8. Repository Structure

```
llm-cost-benchmark-2026/
├── README.md
├── PRD.md                          (this file)
├── LICENSE                         (MIT)
├── pyproject.toml                  (Poetry-managed deps)
├── .env.example                    (API key template — never commit real keys)
├── prompts/
│   ├── customer_support.json
│   ├── rag_qa.json
│   ├── extraction.json
│   ├── summarisation.json
│   └── reasoning.json
├── runners/
│   ├── __init__.py
│   ├── run_anthropic.py            (calls Anthropic SDK directly)
│   ├── run_openai.py               (calls OpenAI SDK directly)
│   ├── lever_baseline.py
│   ├── lever_caching.py            (sets cache_control on Anthropic / observes OpenAI auto)
│   ├── lever_output_cap.py         (sets max_tokens=200)
│   ├── lever_batch.py              (batch API integration)
│   ├── lever_compression.py        (LLMLingua-2 client-side, may be dropped)
│   ├── budget.py                   (cost cap tracking)
│   └── orchestrator.py             (top-level runner that orchestrates the matrix)
├── scoring/
│   ├── __init__.py
│   ├── tier_1.py                   (deterministic rubrics per category)
│   ├── judge.py                    (dual-judge blind evaluation)
│   ├── disagreements.py            (generates CSV for human arbitration)
│   ├── finalise_scores.py          (computes final_score from tiers)
│   └── prompts/
│       ├── judge_prompt_template.txt
│       └── judge_examples/
├── data/
│   ├── results.db                  (SQLite — committed for reproducibility)
│   └── disagreements.csv           (generated, human-edited)
├── analysis/
│   ├── __init__.py
│   ├── cross_model.py              (queries db, builds summary tables)
│   ├── cross_lever.py              (queries db, builds lever-impact tables)
│   ├── charts.py                   (generates writeup charts)
│   └── charts/                     (output SVG/PNG)
├── docs/
│   └── (other planning docs as needed)
├── tests/
│   ├── test_schema.py
│   ├── test_runners.py
│   └── test_scoring.py
└── WRITEUP.md                      (the public artefact)
```

---

## 9. Daily Plan

**Calendar:** Tuesday 30 April 2026 → Wednesday 14 May 2026 (15 calendar days, ~50–60 hours of focused work).

### Day 1 — Setup and verification (Tue 30 Apr)

- Install Python 3.11+ via Homebrew (Claude Code does this)
- Set up Poetry project with deps: `anthropic`, `openai`, `llmlingua`, `pydantic`, `python-dotenv`, `pytest`
- Create the SQLite database with schema from Section 6
- Verify Anthropic API tier (need tier 2+ for reasonable concurrency); same for OpenAI
- Verify Mistral Large API access (sign up if needed)
- Verify LLMLingua-2 runs on Mac CPU at acceptable speed (<30s per compression)
  - **If LLMLingua-2 fails:** drop compression lever, document as limitation, proceed
- Write the prompt JSON schema validator
- Write 3 customer support prompts end-to-end as test cases
- **Done when:** can validate a JSON prompt against schema and store a placeholder result row in SQLite

### Day 2 — Prompt writing (part 1) (Wed 1 May)

- Write all 20 customer support prompts with full Tier 1 / Tier 2 scoring metadata
- Write all 20 RAG Q&A prompts
- Write all 20 structured extraction prompts
- All prompts schema-valid, all using synthetic data only
- **Done when:** 60 prompts in version control

### Day 3 — Prompt writing (part 2) and reasoning go/no-go (Thu 2 May)

- Write all 20 summarisation prompts
- Attempt to write all 20 multi-step reasoning prompts
  - **If reasoning prompts cannot reliably differentiate models by end of day:** drop Category 5 from v1, write 5 additional prompts each for Categories 1–4 to compensate, document the drop in PRD
- All 100 prompts schema-valid
- **Done when:** 100 prompts (or 80 with documented Cat 5 drop) in version control

### Day 4 — Anthropic runner (Fri 3 May)

- Build `runners/run_anthropic.py` with:
  - Native `anthropic` SDK calls
  - Retry with exponential backoff on 429
  - Cost tracking against `runs` table
  - Cost-cap enforcement reflecting §10: hard cap **£300**, soft warning at **£250** (the runner reads these from a single source so a future revision changes one place)
  - Skip-if-exists logic checking `(prompt_id, model, lever, config_hash, run_attempt)`
  - Concurrency configurable via `INFEROPS_CONCURRENCY` env var (default 4)
- Test against 5 prompts on Sonnet 4.6 baseline
- **Done when:** 5 result rows in SQLite, each correctly scored on cost

### Day 5 — OpenAI runner + first lever (Sat 4 May)

- Build `runners/run_openai.py` mirroring Anthropic runner
- Build `runners/lever_caching.py`:
  - Anthropic: configures `cache_control` on system prompt
  - OpenAI: observes automatic caching via `cached_tokens` field
- Test both against 5 prompts each
- **Done when:** caching lever measurable on both providers

### Day 6 — Remaining levers (Sun 5 May)

- Build `runners/lever_output_cap.py` (sets `max_tokens=200`)
- Build `runners/lever_batch.py` (handles async batch APIs for both providers)
- Build `runners/lever_compression.py` (LLMLingua-2 client-side, if not dropped)
- Build `runners/orchestrator.py` to drive the full matrix
- **Done when:** all levers operational against test prompts

### Day 7 — Run baseline + submit batch jobs (Mon 6 May)

- Submit batch jobs for all 100 prompts × 4 models on baseline (batch turnaround ~24h)
- Run synchronous baseline for all 100 prompts × 4 models = 400 runs
- Watch cost tracker. Target: under £80 spent by end of day
- **Done when:** 400 baseline rows in SQLite, batch jobs submitted

### Day 8 — Run lever matrix (Tue 7 May)

- Collect batch job results from Day 7
- Run sync caching and output cap levers across full matrix (~800 additional model runs)
- **Budget check before compression** (revision applied with the £300 cap decision):
  - If **>£80 remaining** under the £300 cap: run compression on the 30-prompt stratified subset (120 model runs)
  - If **£40–£80 remaining**: operator's call — run a reduced subset (e.g. 15 prompts) or skip
  - If **<£40 remaining**: skip compression entirely; document as a budget-gated drop in the writeup limitations section
- Watch cost tracker. Target: under £180 total spent by end of day, leaving headroom for judge calls on Days 10–11
- **Done when:** ~1,720 rows in SQLite (or 1,600 if compression skipped for budget or technical reasons)

### Day 9 — Tier 1 scoring + KILL-SWITCH CHECKPOINT (Wed 8 May)

- Implement Tier 1 deterministic rubrics in `scoring/tier_1.py` per category
- Run rubric scoring across all applicable result rows
- **KILL-SWITCH:** review preliminary Tier 1 scores. If models score 95%+ across the board with no differentiation, the substitutability story collapses. If so:
  - Stop the planned work
  - Pivot the writeup to "We expected substitutability gradients. We didn't find them in our test set. Here's what we did find and what it means."
  - Compress remaining days into a shorter, honest writeup
- **If kill-switch not triggered:** proceed normally
- **Done when:** Tier 1 prompts have `rubric_score` populated OR pivot decision made

### Day 10 — Dual-judge scoring (Thu 9 May)

- Build `scoring/judge.py` with:
  - Blind evaluation (response order randomised per prompt)
  - Calls Claude Opus 4.6 as Judge A, Mistral Large as Judge B
  - Position randomisation within each prompt's response set
- Run judge passes across Tier 2 results
- Identify disagreements (judges differ by >0.2)
- **Done when:** Tier 2 prompts have both judge scores, disagreement flags set

### Day 11 — Human arbitration + finalise scores (Fri 10 May)

- Generate `data/disagreements.csv` from disagreement-flagged rows
- Read each disagreement, score 0.0–1.0
- Run `scoring/finalise_scores.py` to compute all `final_score` values
- **Done when:** every result row has `final_score` populated

### Day 12 — Analysis and charts (Sat 11 May)

- Build `analysis/cross_model.py` — summary tables of model × category performance
- Build `analysis/cross_lever.py` — lever impact analysis
- Generate writeup charts in `analysis/charts/`
- Identify the headline findings (target: at least 3 quantified findings)
- **Done when:** charts generated, top findings documented

### Day 13 — Writeup draft (Sun 12 May)

- Write `WRITEUP.md` end to end
- Sections: TL;DR, methodology, findings per category, cross-category patterns, lever impact analysis, limitations, InferOps positioning paragraph
- Include all charts and key tables
- **Done when:** complete first draft, possibly rough

### Day 14 — Polish, verify, ship-ready (Mon 13 May)

- Edit writeup for tone (technical, dry, no marketing language)
- Verify every claim against SQLite data — every number traceable to a query
- Read writeup aloud once to catch tone drift
- Reproducibility check: snapshot `data/results.db`, re-run 10 prompts from scratch, confirm match within tolerance
- **Done when:** writeup ready to publish

### Day 15 — Buffer / publish (Wed 14 May)

- Anything that overran lands here
- Cold-email hook line drafted (specific reference line for InferOps + LenzAI emails)
- Publish:
  - Push to GitHub (resolves the deferred GitHub auth task)
  - Post to LinkedIn
  - Submit to Hacker News (Tuesday morning UK time is optimal — adjust day if needed)
- **Done when:** writeup is live on GitHub and at least one external channel

---

## 10. Edge Cases and Operational Concerns

### Rate Limits

Anthropic and OpenAI tier-based rate limits checked on Day 1. Retry logic uses exponential backoff with jitter on 429 errors. Concurrency configurable via `INFEROPS_CONCURRENCY` env var (default 4).

### Cost Cap

Hard cap: **£300**. Soft warning at **£250**.

Before every API call, runner checks `cost_so_far_usd + estimated_cost > cost_cap_usd`. If exceeded:

- Runner aborts with clear stderr message: `"Cost cap of £300 reached. Completed N of M planned runs. Raise --cost-cap (e.g. --cost-cap=350) and re-run with --force-resume to continue. Skip-if-exists logic prevents redoing completed work."`
- `runs.status = 'aborted_cost'`
- Re-running with raised cap or `--force-resume` continues from where it stopped (skip-if-exists handles this naturally)

### Skip-if-Exists Logic

Before every API call, runner checks SQLite for matching `(prompt_id, model, optimisation_lever, config_hash, run_attempt)`. If found and `error IS NULL`, skip. `--force` flag creates new attempt (`run_attempt + 1`) instead of skipping.

### Determinism

All runs at `temperature=0`. All runs capture exact `model_version` from API response.

Reproducibility caveat in writeup: "Output token counts and response text vary by approximately 5% across runs even at temperature=0; cost varies proportionally. Structured outputs (JSON extraction) are mostly identical but not guaranteed bit-for-bit."

### Synthetic Data Discipline

All prompts use clearly synthetic placeholders. No real names, real email addresses, real company names, or any data that could be misread as belonging to a real person. Any prompt that fails this discipline must be rewritten before commit.

### Public Repo Hygiene

API keys never committed. `.env` is gitignored. `.env.example` is committed showing the variable names. Pre-commit hook checks for accidental key patterns (`sk-`, `sk-ant-`, etc.).

---

## 11. Risks and Unknowns

| Risk                                                                 | Likelihood | Mitigation                                                                          |
| -------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------- |
| LLMLingua-2 doesn't run cleanly on Mac CPU                           | Medium     | Day 1 verification; drop compression lever if it fails                              |
| Reasoning category (Cat 5) can't be designed to differentiate models | Medium     | Day 3 go/no-go; drop Cat 5 if unworkable                                            |
| Mistral Large API access requires approval delays                    | Low        | Day 1 verification; fallback to Cohere Command R+ as Judge B if delayed             |
| Provider rate limits throttle the run                                | Medium     | Day 1 tier verification; configurable concurrency                                   |
| Total runs blow past £300 cost cap                                   | Low        | Hard cap enforced at runner level; compression scoped to a 30-prompt subset to preserve headroom; Day 8 budget gate skips compression if <£40 remains |
| Findings are weak / models score equivalently across the board       | Medium     | Day 9 kill-switch with pivot writeup option                                         |
| Two-week scope slips into three                                      | Medium     | Day 15 buffer; if slips beyond Day 15, ship what we have honestly                   |
| Writeup tone drifts toward marketing language                        | High       | Tone discipline in PRD; read aloud check on Day 14                                  |
| Judge bias survives the dual-judge blinding                          | Medium     | Acknowledged in writeup limitations; human arbitration on disagreements as backstop |

---

## 12. Cold Outreach Hook

The benchmark must produce one specific output for use in cold email: a line that references the writeup naturally and invites reply.

**Draft target line (refined after writeup is done, not before):**

> "Just shipped a 2026 benchmark on which AI workloads can safely run on cheaper models — turns out [HEADLINE FINDING] is sharper than I expected. [link]. Curious whether these patterns match what you're seeing in your stack."

Where `[HEADLINE FINDING]` is filled in based on actual results. Do not pre-promise a specific finding before we have data.

This line goes into:

- InferOps cold email sequence
- LenzAI cold email sequence (as credibility signal in email 2 or 3)
- LinkedIn outreach DMs to Heads of AI
- Any podcast or conference outreach

**Note:** wiring this line into the actual InferOps/LenzAI templates is a separate post-ship task with its own checklist. It is not part of this PRD's definition of done.

---

## 13. Definition of Done

The benchmark ships when:

1. **Writeup is complete and defensible.** Covers methodology, findings per category, cross-category patterns, lever impact analysis, limitations, and InferOps positioning paragraph. Every claim is traceable to a SQL query against `data/results.db`. Read aloud once for tone.
2. **At least 3 quantified, distinct findings appear in the writeup.**
3. **All data is reproducible.** `prompts/*.json`, `data/results.db`, and the runners are public. Reproducibility check on Day 14 passes within tolerance.
4. **All result rows have `final_score` populated.**
5. **Public repo is shippable.** README updated from placeholder to public-facing. License in place. No API keys leaked.
6. **Cold-email hook line drafted** (templates wiring is separate post-ship task).

Numerical thresholds (≥2,000 rows etc) are guidance, not gates. If we hit Day 14 with 1,800 rows but a publishable writeup, we ship.

---

## 14. What This Is Not

A reminder for week 2 when the temptation will be highest:

- Not the InferOps SDK
- Not a tool other people will use
- Not a benchmark with statistical claims rigorous enough for peer review
- Not a sales asset
- Not a demonstration of every InferOps feature

This is a **terminal piece of writing with code attached**. If we find ourselves wanting to add features, refactor for reuse, or productise anything: stop. The artefact ships. We move on.

---

_End of PRD v1.0_

**Next steps after PRD review:**

1. Review this PRD, push back on anything that feels wrong
2. Iterate to v1.1 if needed
3. Commit `docs/PRD.md` as the second commit on the local repo
4. Open Claude Code in the repo, paste the PRD, ask it to set up the Poetry project per Day 1
5. Begin Day 1 work
