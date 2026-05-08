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

    -- Reproducibility
    model_version TEXT,
    temperature REAL DEFAULT 0,
    error TEXT,

    UNIQUE(prompt_id, model, optimisation_lever, config_hash, run_attempt)
);

CREATE INDEX IF NOT EXISTS idx_results_prompt ON results(prompt_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON results(model);
CREATE INDEX IF NOT EXISTS idx_results_lever ON results(optimisation_lever);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
