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

    -- Scoring
    rubric_score REAL,
    judge_a_score REAL,
    judge_b_score REAL,
    judge_disagreement_flag INTEGER DEFAULT 0,
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
