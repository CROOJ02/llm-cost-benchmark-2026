-- LLM Cost Benchmark 2026 — SQLite schema
-- Source of truth: docs/PRD.md §6.

-- WAL mode: enables concurrent reads while a write is in flight, which matters
-- once the runner uses ThreadPoolExecutor with INFEROPS_CONCURRENCY > 1.
-- Persists in the DB file once set; safe to keep at the top of schema.sql.
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    cost_so_far_gbp REAL DEFAULT 0,
    cost_cap_gbp REAL NOT NULL,
    status TEXT NOT NULL  -- 'running' / 'completed' / 'aborted_cost' / 'aborted_error'
);

CREATE TABLE IF NOT EXISTS results (
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
    -- input_tokens: prompt tokens NOT served from cache (uncached portion).
    --   For Anthropic this maps to resp.usage.input_tokens directly. For OpenAI
    --   this is prompt_tokens − cached_tokens since OpenAI's prompt_tokens
    --   includes cached.
    -- cached_tokens: tokens read from cache on this call (cache READ).
    -- cache_creation_tokens: tokens written to cache on this call (cache WRITE,
    --   Anthropic only — OpenAI does not separately bill or expose creation).
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    latency_ms INTEGER NOT NULL,
    cost_usd REAL NOT NULL,

    -- Output
    response_text TEXT NOT NULL,
    response_parsed TEXT,
    output_format_valid INTEGER DEFAULT 1,

    -- Scoring (3-judge panel as of Day 11 revision; see methodology doc
    -- § "Scope and rigor positioning"). Judge A = Opus 4.6 (Anthropic);
    -- Judge B = GPT-5.5 (OpenAI, replacing Mistral large 2512 after Day 11
    -- analysis revealed Mistral quality issues); Judge C = Gemini 3.1 Pro
    -- preview (Google, added as third judge). Mistral's archived Day 10–11
    -- scores + reasoning are preserved verbatim in judge_b_mistral_*
    -- columns for transparency and v2 cross-judge analysis; they are NOT
    -- used in canonical-score or disagreement-flag computation post-revision.
    rubric_score REAL,
    judge_a_score REAL,
    judge_b_score REAL,
    judge_c_score REAL,
    judge_disagreement_flag INTEGER DEFAULT 0,
    -- Tier-2 judge reasoning (Day 11). Each judge outputs one short sentence
    -- per response in the JSON 'reasoning' field; persisted so Day 11
    -- arbitration can show the judges' thinking alongside the scores. The
    -- original Day 10 UPDATE statement discarded these — gap closed Day 11
    -- via a --reasoning-only re-fire (later upgraded to --realign-scores).
    -- See methodology doc § "Day 11 reasoning re-fire methodology".
    judge_a_reasoning TEXT,
    judge_b_reasoning TEXT,
    judge_c_reasoning TEXT,
    -- judge_c_name records the model ID actually called for the Judge C slot
    -- (e.g. 'gemini-3.1-pro-preview'). Confirms the schema is aware of the
    -- new judge and lets v2 analysis distinguish judge revisions over time.
    judge_c_name TEXT,
    -- Mistral archive (preserved from Day 10–11 original 2-judge sweep).
    -- These columns are NOT updated post-Day-11-revision; their content is
    -- frozen. Day 12 analysis can re-derive Mistral-vs-others comparisons
    -- by reading these alongside the active judge_b/c columns.
    judge_b_mistral_score REAL,
    judge_b_mistral_reasoning TEXT,
    -- canonical_score (Day 12) is the single per-row Tier-2 quality number for
    -- downstream analysis. Population rule:
    --   - judge_disagreement_flag = 0 → canonical_score = mean(judge_a_score,
    --     judge_b_score) (judges agreed, take the simple average)
    --   - judge_disagreement_flag = 1 → canonical_score = human_score from
    --     scoring/disagreements.csv (16 human-arbitrated + 64 median-canonical-
    --     auto; both kinds carry a final canonical value in the CSV)
    -- This column is the analysis-facing one; the judge_*_score columns remain
    -- the underlying ingredients for audit and methodology surfaces.
    canonical_score REAL,
    human_score REAL,
    final_score REAL,
    score_recomputed_at TEXT,

    -- Tier-1 deterministic scoring (Day 9). tier_1_status is the per-row enum:
    --   'pass' | 'fail_format' | 'fail_schema' | 'fail_content' | 'truncated'
    --   | 'compression_unavailable' | 'error' | 'not_applicable'
    -- normalisation_steps_applied is a JSON array of step names (e.g.
    -- ["fence_strip", "preamble_strip", "unwrap_result"]) so Day 12 analysis can
    -- see whether any provider's outputs systematically depend on a given step.
    -- truncated_due_to_cap is a separate flag for output_cap rows where the
    -- response hit the cap AND failed to parse — distinguishes design-induced
    -- truncation from genuine format failure.
    tier_1_status TEXT,
    normalisation_steps_applied TEXT,
    truncated_due_to_cap INTEGER DEFAULT 0,

    -- Reproducibility
    model_version TEXT,
    temperature REAL DEFAULT 0,
    error TEXT,

    -- run_id is included so each run_id is an independent measurement: two
    -- rows that differ only in run_id are now permitted (where the prior
    -- schema would have rejected the second insert at the DB layer even
    -- after the application-level skip-if-exists correctly let it through).
    -- Pairs with the run_id-aware skip-if-exists query in _base.py — neither
    -- alone is sufficient. See "Skip-if-exists semantics" in
    -- docs/methodology/prompt_design_decisions.md.
    UNIQUE(prompt_id, model, optimisation_lever, config_hash, run_attempt, run_id)
);

CREATE INDEX IF NOT EXISTS idx_results_prompt ON results(prompt_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON results(model);
CREATE INDEX IF NOT EXISTS idx_results_lever ON results(optimisation_lever);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);

-- Batch jobs: one row per submitted batch. Submit (Day 7) writes pending; retrieve
-- (Day 8) updates status and pulls per-prompt results into the results table. The
-- split lets the orchestrator survive script restarts during the 1–24h provider-side
-- batch queue without losing batch_id state and forfeiting the batch discount on
-- re-submission. See methodology doc § "Day 6+ orchestration".
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    provider TEXT NOT NULL,           -- 'anthropic' / 'openai'
    model TEXT NOT NULL,
    lever TEXT NOT NULL,              -- 'batch' for the Day 7 batch sweep — distinct from
                                      -- 'baseline' (sync) so result rows from retrieve coexist
                                      -- with sync baseline rows on the same (prompt, model);
                                      -- PRD §5 lists batch as its own lever in the matrix
    status TEXT NOT NULL,             -- 'submitted' / 'in_progress' / 'completed' / 'failed' / 'expired' / 'timed_out' / 'cancelled'
    submitted_at TEXT NOT NULL,
    retrieved_at TEXT,
    completed_at TEXT,
    prompt_ids TEXT NOT NULL,         -- JSON array of prompt_ids included in this batch
    request_count INTEGER NOT NULL,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_run ON batch_jobs(run_id);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);
