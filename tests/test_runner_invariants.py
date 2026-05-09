"""Layer 1 unit tests for runner invariants.

Fourteen cases (4 original + 2 added Day 6 alongside the model-currency rework +
1 added Day 6 day-2 alongside the lever='batch' refactor + 1 added Day 6 day-2
alongside the OpenAI cache-contamination methodology finding + 1 added Day 6
day-2 alongside the compression-engagement softening + 1 added Day 6 day-2
alongside the caching-engagement softening + 2 added Day 7 alongside the
batch-submit transient-error retry + 2 added Day 7 alongside the sync-call
transient-error retry):
  1. test_cost_reservation_released_under_retry_exhaustion — when the API
     hits 429s 6 times and exhausts retries, the runner writes an error row
     and releases the reserved cost; runs.cost_so_far_gbp stays unchanged
     against pre-call value.
  2. test_reservation_released_on_non_success_exit_paths — parameterised
     across (a) non-rate-limit exception during API call, (b) accounting
     exception in post-call path, (c) DB insert exception. All three must
     leave runs.cost_so_far_gbp unchanged from pre-call value.
  3. test_skip_if_exists_distinguishes_lever_variants — a baseline result
     for prompt X must NOT cause a subsequent caching call for prompt X to
     be skipped (the skip composite key includes lever name AND config_hash).
  4. test_schema_migration_safety — adding the batch_jobs table did not break
     existing representative queries against runs and results.
  5. test_retrieve_batches_marks_stale_batch_timed_out — a batch_jobs row
     older than per_batch_timeout_s gets marked 'timed_out' and dropped from
     the in-flight set; retrieve_batches proceeds without it. Caps Day 7-8
     exposure to single-batch tail-latency outliers.
  6. test_poll_openai_batch_inserts_results_from_constructed_payload —
     synthetic OpenAI batch retrieve payload (constructed per OpenAI's
     documented batch output JSONL format) flows through _poll_openai_batch
     and lands as result rows. Closes the code-path gap that the Day 6
     dry-run didn't cover (real OpenAI batch retrieval blocked by slow queue).
  7. test_sync_baseline_and_batch_baseline_coexist — sync baseline and batch
     retrieval produce distinct result rows for the same (prompt_id, model).
     Without the lever='baseline' vs lever='batch' distinction the schema's
     UNIQUE constraint collapses them and Day 12 can't compute the batch
     lever's cost ratio (cost(batch)/cost(baseline) per (prompt, model)).
  8. test_run_baseline_warns_on_cache_contamination — when run_baseline
     receives a row with cached_tokens > 0, the orchestrator logs a
     phase='baseline', event='warning' to phase_log so Day 12 can flag
     methodologically-contaminated baseline measurements (per Day 6
     finding on OpenAI auto-caching account-level persistence).
  9. test_compression_unavailable_inserts_row — when LLMLingua-2 produces
     no Anthropic-counted reduction (compressed >= original), the lever
     inserts a `compression_status='unavailable'` row instead of raising.
     Mirrors caching's `caching_available=False` pattern at the lever-
     return level but persists as a row so Day 12 can distinguish
     "didn't try compression" from "tried, tokeniser asymmetry made it
     unavailable" via SQL JOIN.
 10. test_caching_unavailable_inserts_row — when cache_read returns
     cached_tokens=0 (empirically observed on GPT-5.4 sum-018), the lever
     inserts a `lever='caching_unavailable'` marker row instead of raising
     CachingEngagementError. The actual write/read API rows still exist
     under lever='caching'; the marker carries the observed write/read
     state + skip_reason so Day 12 can quantify caching reliability.
 11. test_batch_submit_retry_succeeds_on_transient_5xx — when the provider
     returns a 504 (Cloudflare gateway timeout) on the first batches.create
     call, _retry_batch_submit retries after the Cloudflare-provided
     retry_after delay and succeeds on attempt 2. Caught Day 7 attempt 1
     crash where this exact 504 took down the whole submit_batches loop.
 12. test_batch_submit_retry_exhaustion_raises — after 3 retries (4 attempts
     total) of persistent 5xx, _retry_batch_submit raises the last error so
     the orchestrator can decide what to do (currently bubbles up).
 13. test_sync_call_retry_succeeds_on_transient_5xx — call_openai_with_retry
     catches InternalServerError (504), waits Cloudflare's retry_after
     (mocked sleep), and succeeds on retry. Same coverage class as Test 11
     but for the sync-call path that runs all 800+ Day 7 sync calls.
 14. test_sync_call_retry_exhaustion_raises — 4 consecutive 504s exhausts the
     shared max_retries=3 budget; the original InternalServerError bubbles
     up so _base.run_one's reservation-release path can fire.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import httpx
import openai
import pytest

from runners import _base
from runners._base import _read_cost_so_far_gbp, run_one, start_run
from runners.run_anthropic import ANTHROPIC_ADAPTER, DEFAULT_MAX_RETRIES
from tests.conftest import make_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_429() -> anthropic.RateLimitError:
    fake_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    fake_response = httpx.Response(status_code=429, headers={}, request=fake_request)
    return anthropic.RateLimitError(
        message="429 rate limit",
        response=fake_response,
        body={"error": {"type": "rate_limit_error", "message": "rate_limit"}},
    )


def _fake_success_message(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    text_block = MagicMock(type="text", text="ok")
    msg = MagicMock(content=[text_block], model="claude-sonnet-4-6", stop_reason="end_turn")
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 0
    return msg


def _set_cost_so_far(db_path: Path, run_id: str, gbp: float) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET cost_so_far_gbp = ? WHERE run_id = ?", (gbp, run_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Test 1: cost reservation released under retry exhaustion
# ---------------------------------------------------------------------------

def test_cost_reservation_released_under_retry_exhaustion(monkeypatch, fresh_db):
    """Pre-call cost £10. Call hits 429 × (max_retries+1). Post-call cost still £10.
    The estimated reservation made before the API call is fully released; no
    phantom cost is left in the cap accumulator."""
    monkeypatch.setattr("runners.run_anthropic.time.sleep", lambda s: None)
    monkeypatch.setattr("runners.run_anthropic.random.uniform", lambda lo, hi: 0.0)
    monkeypatch.setattr("runners.run_anthropic.count_input_tokens", lambda *a, **kw: 1000)

    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    _set_cost_so_far(fresh_db, rid, 10.0)

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = [_fake_429() for _ in range(DEFAULT_MAX_RETRIES + 1)]

    result = run_one(
        ANTHROPIC_ADAPTER, make_prompt(), "claude-sonnet-4-6", lever="baseline",
        run_id=rid, cap_gbp=300.0, completed=0, planned=1,
        client=mock_client, db_path=fresh_db,
        max_retries=DEFAULT_MAX_RETRIES, base_delay=1.0,
    )

    assert result["error"] is not None and "RateLimitError" in result["error"]
    with sqlite3.connect(fresh_db) as conn:
        assert _read_cost_so_far_gbp(conn, rid) == pytest.approx(10.0), (
            "reservation leaked into cost_so_far_gbp after retry exhaustion"
        )
        n = conn.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ? AND error IS NOT NULL", (rid,),
        ).fetchone()[0]
        assert n == 1


# ---------------------------------------------------------------------------
# Test 2: reservation released on non-success exit paths
# ---------------------------------------------------------------------------

class _FakeAdapter:
    """Standalone adapter for fault-injection. Lets each test parameter pick
    where the failure happens (in the API call, in cost computation, in DB insert)."""
    name = "anthropic"
    rate_limit_error = anthropic.RateLimitError

    def __init__(self, *, fail_call: BaseException | None = None,
                 input_tokens: int = 100, output_tokens: int = 50):
        self._fail_call = fail_call
        self._input = input_tokens
        self._output = output_tokens

    def make_client(self):
        return MagicMock()

    def count_input_tokens(self, client, prompt, model):
        return 1000

    def call_with_retry(self, client, prompt, model, max_tokens, max_retries, base_delay,
                        *, optimisation_config=None):
        if self._fail_call is not None:
            raise self._fail_call
        return {
            "response_text": "ok",
            "input_tokens": self._input,
            "output_tokens": self._output,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "model_version": model,
            "latency_ms": 100,
            "stop_reason": "end_turn",
        }


def test_reservation_released_on_non_rate_limit_exception(fresh_db):
    """An adapter exception that is not a RateLimitError must release the
    reservation before propagating."""
    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    _set_cost_so_far(fresh_db, rid, 10.0)

    adapter = _FakeAdapter(fail_call=RuntimeError("transport error"))
    with pytest.raises(RuntimeError, match="transport error"):
        run_one(
            adapter, make_prompt(), "claude-sonnet-4-6", lever="baseline",
            run_id=rid, cap_gbp=300.0, completed=0, planned=1,
            db_path=fresh_db,
        )

    with sqlite3.connect(fresh_db) as conn:
        assert _read_cost_so_far_gbp(conn, rid) == pytest.approx(10.0), (
            "reservation leaked into cost_so_far_gbp after non-rate-limit API exception"
        )


def test_reservation_released_on_post_call_accounting_failure(monkeypatch, fresh_db):
    """If estimate_cost_usd raises AFTER a successful API call, the reservation
    must still be released."""
    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    _set_cost_so_far(fresh_db, rid, 10.0)

    monkeypatch.setattr(
        "runners._base.estimate_cost_usd",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("kaboom")),
    )

    adapter = _FakeAdapter()
    with pytest.raises(ValueError, match="kaboom"):
        run_one(
            adapter, make_prompt(), "claude-sonnet-4-6", lever="baseline",
            run_id=rid, cap_gbp=300.0, completed=0, planned=1,
            db_path=fresh_db,
        )

    with sqlite3.connect(fresh_db) as conn:
        assert _read_cost_so_far_gbp(conn, rid) == pytest.approx(10.0)


def test_reservation_released_on_db_insert_failure(monkeypatch, fresh_db):
    """If the result-row insert raises (e.g. constraint violation), the
    reservation must still be released."""
    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    _set_cost_so_far(fresh_db, rid, 10.0)

    def _raising_insert(conn, row):
        raise sqlite3.IntegrityError("forced insert failure")

    monkeypatch.setattr("runners._base._insert_row", _raising_insert)

    adapter = _FakeAdapter()
    with pytest.raises(sqlite3.IntegrityError, match="forced insert failure"):
        run_one(
            adapter, make_prompt(), "claude-sonnet-4-6", lever="baseline",
            run_id=rid, cap_gbp=300.0, completed=0, planned=1,
            db_path=fresh_db,
        )

    with sqlite3.connect(fresh_db) as conn:
        assert _read_cost_so_far_gbp(conn, rid) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Test 3: skip-if-exists distinguishes lever variants
# ---------------------------------------------------------------------------

def test_skip_if_exists_distinguishes_lever_variants(fresh_db):
    """A baseline row for prompt X must NOT cause a subsequent caching call for
    prompt X to be skipped — different lever names hash to different
    config_hashes, so skip-if-exists treats them as distinct."""
    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    prompt = make_prompt("test-001", task_category="summarisation")
    adapter = _FakeAdapter(input_tokens=100, output_tokens=50)

    # First call — baseline.
    r1 = run_one(
        adapter, prompt, "claude-sonnet-4-6", lever="baseline",
        run_id=rid, cap_gbp=300.0, completed=0, planned=1,
        db_path=fresh_db,
    )
    assert not r1.get("skipped")

    with sqlite3.connect(fresh_db) as conn:
        n_after_baseline = conn.execute(
            "SELECT COUNT(*) FROM results WHERE prompt_id='test-001'"
        ).fetchone()[0]
    assert n_after_baseline == 1

    # Second call — caching with config (different lever AND different config).
    r2 = run_one(
        adapter, prompt, "claude-sonnet-4-6", lever="caching",
        optimisation_config={"cache_phase": "read", "enable_cache": True},
        run_id=rid, cap_gbp=300.0, completed=1, planned=2,
        db_path=fresh_db,
    )
    assert not r2.get("skipped"), "caching call was incorrectly skipped because baseline existed"

    # Third call — caching with same config: should now skip.
    r3 = run_one(
        adapter, prompt, "claude-sonnet-4-6", lever="caching",
        optimisation_config={"cache_phase": "read", "enable_cache": True},
        run_id=rid, cap_gbp=300.0, completed=2, planned=3,
        db_path=fresh_db,
    )
    assert r3.get("skipped"), "duplicate (lever+config) call was not skipped"

    with sqlite3.connect(fresh_db) as conn:
        rows = conn.execute(
            "SELECT optimisation_lever, config_hash FROM results "
            "WHERE prompt_id='test-001' ORDER BY timestamp"
        ).fetchall()
    assert len(rows) == 2  # baseline + caching; the duplicate caching was skipped
    levers = {r[0] for r in rows}
    hashes = {r[1] for r in rows}
    assert levers == {"baseline", "caching"}
    assert len(hashes) == 2, "baseline and caching should have distinct config_hashes"


def test_skip_if_exists_is_run_id_scoped(fresh_db):
    """Regression for the Day 9 cross-run skip bug.

    Skip-if-exists must scope to a single run_id: a row written under run_id_A
    must NOT cause a fresh run_id_B from skipping the same (prompt, model,
    lever, config_hash, run_attempt) tuple. Each run_id is a fresh measurement.

    Inverse must still hold: a duplicate within the same run_id IS skipped, so
    intra-run idempotency for orchestrator restarts is preserved.

    Until 2026-05-09 the query at _existing_successful_row was run-id-agnostic,
    which caused 33 missing rows in production run-d1a9c980 because Day 6
    dry-run rows blocked the production baseline phase from firing. See
    methodology doc, Day 9 audit.
    """
    rid_a = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    rid_b = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    prompt = make_prompt("test-001", task_category="summarisation")
    adapter = _FakeAdapter(input_tokens=100, output_tokens=50)

    r1 = run_one(
        adapter, prompt, "claude-sonnet-4-6", lever="baseline",
        run_id=rid_a, cap_gbp=300.0, completed=0, planned=1, db_path=fresh_db,
    )
    assert not r1.get("skipped")

    r2 = run_one(
        adapter, prompt, "claude-sonnet-4-6", lever="baseline",
        run_id=rid_b, cap_gbp=300.0, completed=0, planned=1, db_path=fresh_db,
    )
    assert not r2.get("skipped"), (
        "fresh run_id was wrongly blocked by a row in a prior run_id; "
        "skip-if-exists must filter by run_id"
    )
    assert r2["run_id"] == rid_b
    assert r2["run_attempt"] == 1

    r3 = run_one(
        adapter, prompt, "claude-sonnet-4-6", lever="baseline",
        run_id=rid_a, cap_gbp=300.0, completed=1, planned=2, db_path=fresh_db,
    )
    assert r3.get("skipped"), (
        "duplicate within the same run_id must still be skipped — "
        "intra-run idempotency must not regress"
    )

    with sqlite3.connect(fresh_db) as conn:
        run_id_counts = dict(conn.execute(
            "SELECT run_id, COUNT(*) FROM results WHERE prompt_id='test-001' GROUP BY run_id"
        ).fetchall())
    assert run_id_counts == {rid_a: 1, rid_b: 1}, (
        f"each run_id should have exactly one row for the prompt; got {run_id_counts}"
    )


# ---------------------------------------------------------------------------
# Test 4: schema migration safety
# ---------------------------------------------------------------------------

def test_schema_migration_safety(fresh_db):
    """Representative queries against runs and results work after the
    batch_jobs migration. Catches the class of bug where a new table breaks
    implicit assumptions held by working queries (e.g. UNION shapes, FK refs)."""
    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    prompt = make_prompt()
    adapter = _FakeAdapter()
    run_one(
        adapter, prompt, "claude-sonnet-4-6", lever="baseline",
        run_id=rid, cap_gbp=300.0, completed=0, planned=1, db_path=fresh_db,
    )

    with sqlite3.connect(fresh_db) as conn:
        # Schema introspection — all three tables present.
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert tables == {"runs", "results", "batch_jobs"}

        # runs query — pull a typical column shape.
        run_row = conn.execute(
            "SELECT run_id, started_at, cost_so_far_gbp, cost_cap_gbp, status FROM runs "
            "WHERE run_id = ?", (rid,),
        ).fetchone()
        assert run_row is not None
        assert run_row[0] == rid and run_row[3] == 300.0 and run_row[4] == "running"

        # results query — JOINed shape that Day 12 analysis will use.
        result_rows = conn.execute(
            """SELECT r.prompt_id, r.optimisation_lever, r.input_tokens,
                      r.output_tokens, r.cost_usd, runs.cost_cap_gbp
               FROM results r JOIN runs ON r.run_id = runs.run_id
               WHERE r.run_id = ?""", (rid,),
        ).fetchall()
        assert len(result_rows) == 1
        assert result_rows[0][1] == "baseline" and result_rows[0][5] == 300.0

        # batch_jobs query — empty but queryable, expected columns selectable.
        batch_rows = conn.execute(
            "SELECT batch_id, run_id, provider, model, lever, status, "
            "submitted_at, prompt_ids, request_count FROM batch_jobs"
        ).fetchall()
        assert batch_rows == []

        # WAL pragma persisted (matters for ThreadPoolExecutor with concurrency > 1).
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode == "wal"


# ---------------------------------------------------------------------------
# Test 5: per-batch timeout in retrieve_batches
# ---------------------------------------------------------------------------

def test_retrieve_batches_marks_stale_batch_timed_out(monkeypatch, fresh_db, tmp_path):
    """A batch_jobs row whose age exceeds per_batch_timeout_s gets marked
    'timed_out' and dropped from the in-flight set; retrieve_batches proceeds
    without polling it. Caps Day 7-8 exposure to single-batch tail-latency
    outliers (e.g. the 4h+ gpt-4o-mini wait that surfaced this bug)."""
    from datetime import datetime, timedelta, timezone
    import json
    from runners.orchestrator import Orchestrator

    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    # Insert a batch_jobs row that's already 1 hour old (>> 60s timeout).
    stale_submitted_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    seed = {
        "batch_id": "batch_stale_test", "run_id": rid, "provider": "openai",
        "model": "gpt-5.4-mini-2026-03-17", "lever": "batch",
        "status": "in_progress", "submitted_at": stale_submitted_at,
        "retrieved_at": None, "completed_at": None,
        "prompt_ids": json.dumps(["sum-015"]), "request_count": 1, "error": None,
    }
    with sqlite3.connect(fresh_db) as conn:
        cols = ", ".join(seed.keys())
        ph = ", ".join(["?"] * len(seed))
        conn.execute(f"INSERT INTO batch_jobs ({cols}) VALUES ({ph})", list(seed.values()))
        conn.commit()

    # Asserting that the orchestrator never reaches the OpenAI poll: monkeypatch
    # _poll_openai_batch to raise if called.
    monkeypatch.setattr(
        Orchestrator, "_poll_openai_batch",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("poll should be skipped for timed_out batch")),
    )
    # Avoid loading the real prompts dir — empty dict is fine since we never poll.
    monkeypatch.setattr("runners.orchestrator.load_all_prompts", lambda: [])

    orch = Orchestrator(
        run_id=rid, cap_gbp=300.0, db_path=fresh_db,
        phase_log_path=tmp_path / "phase_log.jsonl",
    )
    retrieved = orch.retrieve_batches(per_batch_timeout_s=60)

    assert retrieved == []
    with sqlite3.connect(fresh_db) as conn:
        row = conn.execute(
            "SELECT status, error FROM batch_jobs WHERE batch_id = ?", ("batch_stale_test",),
        ).fetchone()
    assert row[0] == "timed_out"
    assert "per-batch timeout exceeded" in row[1]


# ---------------------------------------------------------------------------
# Test 6: synthetic _poll_openai_batch retrieve
# ---------------------------------------------------------------------------

def test_poll_openai_batch_inserts_results_from_constructed_payload(monkeypatch, fresh_db, tmp_path):
    """Construct an OpenAI batch retrieve payload (per OpenAI's documented
    batch output format) and run it through _poll_openai_batch. Verifies the
    code path inserts result rows correctly with the per-row 50% batch
    discount applied — the path the Day 6 dry-run couldn't exercise live
    because gpt-4o-mini batch queue was multi-hour."""
    import json
    from runners.orchestrator import Orchestrator

    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    test_prompt = make_prompt("sum-015", task_category="summarisation")

    seed = {
        "batch_id": "batch_synthetic_test", "run_id": rid, "provider": "openai",
        "model": "gpt-5.4-mini-2026-03-17", "lever": "batch",
        "status": "submitted", "submitted_at": _base._now_iso(),
        "retrieved_at": None, "completed_at": None,
        "prompt_ids": json.dumps(["sum-015"]), "request_count": 1, "error": None,
    }
    with sqlite3.connect(fresh_db) as conn:
        cols = ", ".join(seed.keys())
        ph = ", ".join(["?"] * len(seed))
        conn.execute(f"INSERT INTO batch_jobs ({cols}) VALUES ({ph})", list(seed.values()))
        conn.commit()

    # Synthetic OpenAI batch retrieve payload — exact JSONL format from
    # https://platform.openai.com/docs/guides/batch (output file shape).
    output_jsonl = json.dumps({
        "id": "batch_req_001",
        "custom_id": "sum-015",
        "response": {
            "status_code": 200,
            "request_id": "req_abc",
            "body": {
                "id": "chatcmpl_xyz",
                "object": "chat.completion",
                "created": 1762547951,
                "model": "gpt-5.4-mini-2026-03-17",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthetic test response"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1500,
                    "completion_tokens": 100,
                    "total_tokens": 1600,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        },
        "error": None,
    })

    fake_batch = MagicMock(status="completed", output_file_id="file-fake", errors=None)
    fake_file_content = MagicMock(text=output_jsonl)
    fake_client = MagicMock()
    fake_client.batches.retrieve.return_value = fake_batch
    fake_client.files.content.return_value = fake_file_content
    monkeypatch.setattr("runners.orchestrator.openai.OpenAI", lambda: fake_client)

    orch = Orchestrator(
        run_id=rid, cap_gbp=300.0, db_path=fresh_db,
        phase_log_path=tmp_path / "phase_log.jsonl",
    )
    n = orch._poll_openai_batch(seed, prompts_by_id={"sum-015": test_prompt})

    assert n == 1
    with sqlite3.connect(fresh_db) as conn:
        result = conn.execute(
            "SELECT optimisation_lever, input_tokens, output_tokens, "
            "       cached_tokens, response_text, model_version, cost_usd, latency_ms "
            "FROM results WHERE run_id = ?", (rid,),
        ).fetchone()
        batch_status = conn.execute(
            "SELECT status FROM batch_jobs WHERE batch_id = ?", ("batch_synthetic_test",),
        ).fetchone()[0]
    assert result[0] == "batch"   # lever propagates from batch_jobs.lever
    assert result[1] == 1500      # prompt_tokens - cached_tokens
    assert result[2] == 100
    assert result[3] == 0
    assert result[4] == "synthetic test response"
    assert result[5] == "gpt-5.4-mini-2026-03-17"
    assert result[7] == 0         # batch latency_ms is 0 (no per-request timing)
    # Cost: gpt-5.4-mini at $0.75/M input + $4.50/M output, ×0.5 batch discount.
    expected_cost_usd = (1500 * 0.75 + 100 * 4.50) / 1_000_000.0 * 0.5
    assert result[6] == pytest.approx(expected_cost_usd, rel=1e-9)
    assert batch_status == "completed"


# ---------------------------------------------------------------------------
# Test 7: sync baseline and batch baseline coexist as distinct rows
# ---------------------------------------------------------------------------

def test_sync_baseline_and_batch_baseline_coexist(fresh_db, tmp_path):
    """Sync baseline (lever='baseline') and batch retrieval (lever='batch')
    produce distinct result rows for the same (prompt_id, model). Without
    the lever distinction, the schema's UNIQUE(prompt_id, model, lever,
    config_hash, run_attempt) collapses them and only the first inserted
    survives — Day 12 then can't compute cost(batch)/cost(baseline) per
    (prompt, model). With the distinct lever values both rows coexist."""
    from runners.orchestrator import Orchestrator

    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    prompt = make_prompt("sum-015", task_category="summarisation")
    adapter = _FakeAdapter(input_tokens=3000, output_tokens=200)

    # 1. Insert sync baseline row (the way Day 7's run_baseline phase does).
    sync_row = run_one(
        adapter, prompt, "gpt-5.4-2026-03-05", lever="baseline",
        run_id=rid, cap_gbp=300.0, completed=0, planned=1, db_path=fresh_db,
    )
    assert not sync_row.get("skipped")
    assert sync_row["optimisation_lever"] == "baseline"

    # 2. Insert a batch result row for the same (prompt, model) via the
    #    orchestrator's _insert_batch_result_row helper with lever='batch'.
    orch = Orchestrator(
        run_id=rid, cap_gbp=300.0, db_path=fresh_db,
        phase_log_path=tmp_path / "phase_log.jsonl",
    )
    inserted = orch._insert_batch_result_row(
        prompt=prompt, model="gpt-5.4-2026-03-05",
        provider="openai", lever="batch",
        message_meta={
            "input_tokens": 3000, "output_tokens": 200,
            "cached_tokens": 0, "cache_creation_tokens": 0,
            "response_text": "batch result text",
            "model_version": "gpt-5.4-2026-03-05",
        },
    )
    assert inserted is True

    # 3. Both rows coexist with distinct lever + config_hash.
    with sqlite3.connect(fresh_db) as conn:
        rows = conn.execute(
            "SELECT optimisation_lever, config_hash, ROUND(cost_usd, 6) AS cost "
            "FROM results WHERE prompt_id='sum-015' AND model='gpt-5.4-2026-03-05' "
            "ORDER BY optimisation_lever"
        ).fetchall()
    assert len(rows) == 2, f"expected 2 distinct rows, got {len(rows)}: {rows}"
    levers = {r[0] for r in rows}
    hashes = {r[1] for r in rows}
    assert levers == {"baseline", "batch"}
    assert len(hashes) == 2, "baseline and batch should have distinct config_hashes"

    # Sanity: batch row should be cheaper (50% batch discount applied to gpt-5.4).
    by_lever = {r[0]: r[2] for r in rows}
    assert by_lever["batch"] < by_lever["baseline"], (
        f"batch cost {by_lever['batch']} should be < sync cost {by_lever['baseline']}"
    )


# ---------------------------------------------------------------------------
# Test 8: run_baseline emits a phase warning on cache contamination
# ---------------------------------------------------------------------------

def test_run_baseline_warns_on_cache_contamination(monkeypatch, fresh_db, tmp_path):
    """When run_baseline receives a result row with cached_tokens > 0, the
    orchestrator logs a phase='baseline', event='warning' event with
    payload.warning='baseline_cache_contamination'. The auditable signal
    lets Day 12 analysis flag methodologically-contaminated baseline rows
    (OpenAI auto-caching is account-level with a 5–10 min TTL, so prior
    runs against the same prompt+model can contaminate baseline calls)."""
    from runners.orchestrator import Orchestrator
    from tests.conftest import read_phase_log

    log_path = tmp_path / "phase_log.jsonl"
    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)

    # Mock _base.run_many to return one row with cached_tokens > 0 (simulates
    # OpenAI auto-cache hit on a baseline call) and one row with cached=0.
    contaminated_prompt = make_prompt("sum-015", task_category="summarisation")
    cold_prompt = make_prompt("sum-020", task_category="summarisation")

    def _fake_run_many(adapter, prompts, model, lever, *, run_id, cap_gbp, db_path, **kw):
        rows = []
        for p in prompts:
            cached = 2816 if p.prompt_id == "sum-015" else 0
            row = _base._new_row(
                prompt=p, model=model, provider=adapter.name, lever=lever,
                config_hash=_base._config_hash(lever, None), optimisation_config=None,
                run_id=run_id, run_attempt=1,
            )
            row.update({
                "input_tokens": 200 if p.prompt_id == "sum-015" else 3000,
                "output_tokens": 100, "cached_tokens": cached, "cost_usd": 0.001,
                "latency_ms": 100, "response_text": "ok", "model_version": model,
            })
            with sqlite3.connect(db_path) as conn:
                _base._insert_row(conn, row)
                conn.commit()
            rows.append(row)
        return rows

    monkeypatch.setattr(_base, "run_many", _fake_run_many)

    orch = Orchestrator(
        run_id=rid, cap_gbp=300.0, db_path=fresh_db, phase_log_path=log_path,
    )
    results = orch.run_baseline(
        [contaminated_prompt, cold_prompt], ["gpt-5.4-2026-03-05"],
    )
    assert len(results) == 2

    log = read_phase_log(log_path)
    warnings = [
        e for e in log
        if e["phase"] == "baseline" and e["event"] == "warning"
    ]
    assert len(warnings) == 1, (
        f"expected 1 cache-contamination warning, got {len(warnings)}: {warnings}"
    )
    payload = warnings[0]["payload"]
    assert payload["warning"] == "baseline_cache_contamination"
    assert payload["prompt_id"] == "sum-015"  # the contaminated one, not sum-020
    assert payload["cached_tokens"] == 2816
    assert payload["model"] == "gpt-5.4-2026-03-05"


# ---------------------------------------------------------------------------
# Test 9: compression_unavailable inserts a row instead of raising
# ---------------------------------------------------------------------------

def test_compression_unavailable_inserts_row(monkeypatch, fresh_db):
    """When LLMLingua-2 produces no Anthropic-counted reduction (compressed
    >= original — typical for short or structured prompts), the compression
    lever inserts a `compression_status='unavailable'` row rather than
    raising CompressionEngagementError. Day 12 can then distinguish
    'didn't try' from 'tried, tokeniser asymmetry made it unavailable'."""
    import json
    from runners import lever_compression

    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    prompt = make_prompt(
        "ext-008", task_category="extraction",
        system="Extract fields per the schema below.",
        user="Schema: {field_a: str, field_b: int}. Document: tiny.",  # short
    )
    adapter = _FakeAdapter()

    # Mock LLMLingua-2: return a "compressed" string that's actually LARGER
    # in Anthropic's counts than the original. (Short prompts often expand
    # under cross-tokeniser re-counting — the empirical Day 6 ext-008 case.)
    monkeypatch.setattr(
        lever_compression, "compress_user_text",
        lambda text, rate=0.5: {
            "compressed_text": text + " (compressed marker for test)",
            "compress_ms": 10,
            "llmlingua_origin_tokens": 100,
            "llmlingua_compressed_tokens": 50,  # BERT-counted reduction
            "llmlingua_rate_actual": 0.5,
        },
    )
    # Mock the Anthropic count_tokens calls to simulate the asymmetry:
    # compressed comes back longer than original in Anthropic's tokenizer.
    counts = iter([100, 110])  # original=100, compressed=110 → no reduction
    monkeypatch.setattr(
        lever_compression, "_count_tokens_direct",
        lambda *a, **kw: next(counts),
    )
    # Stub Anthropic client construction to avoid real auth.
    monkeypatch.setattr(
        lever_compression.anthropic, "Anthropic",
        lambda *a, **kw: MagicMock(),
    )

    result = lever_compression.run_compression_for_prompt(
        adapter, prompt, "claude-sonnet-4-6",
        run_id=rid, cap_gbp=300.0, completed=0, planned=1, db_path=fresh_db,
    )

    # Result reflects unavailability without raising
    assert result["compression_available"] is False
    assert "no Anthropic-counted reduction" in result["skip_reason"]
    assert result["original_input_tokens"] == 100
    assert result["compressed_input_tokens"] == 110

    # Row was persisted with the unavailable marker
    with sqlite3.connect(fresh_db) as conn:
        rows = conn.execute(
            "SELECT optimisation_lever, optimisation_config, input_tokens, "
            "       output_tokens, cost_usd, response_text, error "
            "FROM results WHERE prompt_id='ext-008'",
        ).fetchall()
    assert len(rows) == 1
    cfg = json.loads(rows[0][1])
    assert rows[0][0] == "compression"
    assert cfg["compression_status"] == "unavailable"
    assert cfg["original_input_tokens"] == 100
    assert cfg["compressed_input_tokens"] == 110
    assert cfg["compression_ratio_anthropic"] == 1.1
    assert rows[0][2] == 0          # no API call → no input_tokens billed
    assert rows[0][3] == 0          # no output_tokens
    assert rows[0][4] == 0.0        # zero cost
    assert rows[0][5] == ""         # empty response
    assert rows[0][6] is None       # no error — unavailability is not failure


# ---------------------------------------------------------------------------
# Test 10: caching_unavailable inserts a marker row instead of raising
# ---------------------------------------------------------------------------

def test_caching_unavailable_inserts_row(monkeypatch, fresh_db):
    """When cache_read returns cached_tokens=0 (empirically observed on
    GPT-5.4 sum-018 during Day 6 day-2 re-measurement), the caching lever
    inserts a `lever='caching_unavailable'` marker row instead of raising
    CachingEngagementError. The marker carries observed write+read cached
    counts and the skip_reason in optimisation_config so Day 12 can quantify
    caching reliability across (model, prompt) pairs."""
    import json as _json
    from runners import lever_caching, run_openai

    rid = start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    prompt = make_prompt("sum-015", task_category="summarisation")

    # Stand in for `_base.run_one` — return rows with the desired cached_tokens
    # state per lever / cache_phase to drive the soft-fail path:
    #   - baseline: normal row
    #   - caching write: cached=2816 (cache was warm; this is fine for OpenAI)
    #   - caching read:  cached=0 (the failure case we're testing)
    call_count = {"baseline": 0, "write": 0, "read": 0}
    def _fake_run_one(adapter, p, model, lever, *,
                      run_id, cap_gbp, completed, planned,
                      optimisation_config=None, force_new_attempt=False,
                      db_path, client=None, **kw):
        cfg = optimisation_config or {}
        if lever == "baseline":
            call_count["baseline"] += 1
            cached = 0
        elif lever == "caching" and cfg.get("cache_phase") == "write":
            call_count["write"] += 1
            cached = 2816  # write call shows warm cache from prior state
        elif lever == "caching" and cfg.get("cache_phase") == "read":
            call_count["read"] += 1
            cached = 0  # ← the failure: read returns no cached tokens
        else:
            raise AssertionError(f"unexpected lever={lever} cfg={cfg}")
        config_hash = _base._config_hash(lever, cfg)
        row = _base._new_row(
            prompt=p, model=model, provider=adapter.name, lever=lever,
            config_hash=config_hash, optimisation_config=cfg,
            run_id=run_id, run_attempt=1,
        )
        row.update({
            "input_tokens": 3000, "output_tokens": 100,
            "cached_tokens": cached, "cache_creation_tokens": 0,
            "cost_usd": 0.005, "latency_ms": 200,
            "response_text": "ok", "model_version": model,
        })
        with sqlite3.connect(db_path) as conn:
            _base._insert_row(conn, row)
            conn.commit()
        return {**row, "skipped": False}

    monkeypatch.setattr(_base, "run_one", _fake_run_one)
    # Avoid creating a real Anthropic/OpenAI client for the count_tokens path.
    monkeypatch.setattr(
        run_openai.OPENAI_ADAPTER, "count_input_tokens",
        lambda *a, **kw: 3000,  # well above 1024 OpenAI cache threshold
    )
    monkeypatch.setattr(
        run_openai.OPENAI_ADAPTER, "make_client",
        lambda: MagicMock(),
    )

    # Run the caching test on a single OpenAI prompt+model
    result = lever_caching.run_caching_for_prompt(
        run_openai.OPENAI_ADAPTER, prompt, "gpt-5.4-2026-03-05",
        run_id=rid, cap_gbp=300.0, completed=0, planned=1, db_path=fresh_db,
    )

    # Lever didn't raise; signals unavailability
    assert result["caching_available"] is False
    assert "cache-read did not hit cache" in result["skip_reason"]
    assert "caching_unavailable_row" in result
    assert call_count == {"baseline": 1, "write": 1, "read": 1}

    # DB has 4 rows: baseline + caching/write + caching/read + caching_unavailable
    with sqlite3.connect(fresh_db) as conn:
        rows = conn.execute(
            "SELECT optimisation_lever, optimisation_config, cached_tokens, cost_usd "
            "FROM results WHERE prompt_id='sum-015' "
            "ORDER BY optimisation_lever, optimisation_config",
        ).fetchall()
    levers = [r[0] for r in rows]
    assert sorted(levers) == ["baseline", "caching", "caching", "caching_unavailable"]

    # The unavailable marker row has the observed state + zero cost
    unavail = [r for r in rows if r[0] == "caching_unavailable"][0]
    cfg = _json.loads(unavail[1])
    assert cfg["caching_status"] == "unavailable"
    assert cfg["observed_write_cached_tokens"] == 2816
    assert cfg["observed_read_cached_tokens"] == 0
    assert "cache-read did not hit cache" in cfg["skip_reason"]
    assert unavail[2] == 0      # cached_tokens on marker row is 0 (no API call)
    assert unavail[3] == 0.0    # zero cost


# ---------------------------------------------------------------------------
# Tests 11+12: batch-submit transient-5xx retry
# ---------------------------------------------------------------------------

def _fake_openai_504(retry_after: int | None = 120) -> openai.InternalServerError:
    """Construct an OpenAI InternalServerError matching the actual Cloudflare
    504 body shape that crashed Day 7 attempt 1's batches.create call."""
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/batches")
    body: dict = {
        "type": "https://developers.cloudflare.com/.../error-504/",
        "title": "Error 504: Gateway time-out",
        "status": 504,
        "detail": "The origin web server did not respond to Cloudflare within the allowed time.",
        "error_code": 504,
        "cloudflare_error": True,
    }
    if retry_after is not None:
        body["retry_after"] = retry_after
    fake_response = httpx.Response(status_code=504, request=fake_request, json=body)
    return openai.InternalServerError(
        message="504 Gateway time-out",
        response=fake_response,
        body=body,
    )


def test_batch_submit_retry_succeeds_on_transient_5xx(monkeypatch):
    """First call raises Cloudflare 504 with retry_after=10; retry waits the
    indicated delay (mocked) and succeeds on attempt 2. Day 7 attempt 1 hit
    exactly this on gpt-5.4-mini batches.create — fix is verified end-to-end."""
    from runners import lever_batch

    sleep_calls: list[float] = []
    monkeypatch.setattr(lever_batch.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = {"n": 0}
    def _flaky_create():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _fake_openai_504(retry_after=10)
        return MagicMock(id="batch_success_001")

    result = lever_batch._retry_batch_submit(_flaky_create)

    assert call_count["n"] == 2
    assert result.id == "batch_success_001"
    assert sleep_calls == [10.0]  # honoured Cloudflare retry_after, not the 120s default


def test_batch_submit_retry_exhaustion_raises(monkeypatch):
    """4 consecutive 504s (1 initial + 3 retries) exhausts the wrapper's
    retry budget and raises the last error so the orchestrator can decide.
    Verifies max_retries=3 (matching submit_batch's contract)."""
    from runners import lever_batch

    sleep_calls: list[float] = []
    monkeypatch.setattr(lever_batch.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = {"n": 0}
    def _always_504():
        call_count["n"] += 1
        # Body without retry_after → wrapper falls back to default_delay_s (120)
        raise _fake_openai_504(retry_after=None)

    with pytest.raises(openai.InternalServerError):
        lever_batch._retry_batch_submit(_always_504, max_retries=3, default_delay_s=120.0)

    # 1 initial + 3 retries = 4 attempts; 3 sleeps between them
    assert call_count["n"] == 4
    assert sleep_calls == [120.0, 120.0, 120.0]


# ---------------------------------------------------------------------------
# Tests 13+14: sync-call transient-5xx retry (call_openai_with_retry)
# ---------------------------------------------------------------------------

def test_sync_call_retry_succeeds_on_transient_5xx(monkeypatch):
    """Sync-call path: chat.completions.create raises Cloudflare 504 once
    (with retry_after=15), then succeeds. call_openai_with_retry waits the
    indicated delay and returns successfully on retry. Verifies sync calls
    survive transient OpenAI 5xx — the failure class that crashed Day 7
    submit_batches and would equally crash Day 7's 800+ baseline calls."""
    from runners import run_openai
    from tests.conftest import make_prompt

    sleep_calls: list[float] = []
    monkeypatch.setattr(run_openai.time, "sleep", lambda s: sleep_calls.append(s))

    mock_client = MagicMock(spec=openai.OpenAI)
    success = MagicMock()
    success.choices = [MagicMock(message=MagicMock(content="ok"), finish_reason="stop")]
    success.usage = MagicMock(prompt_tokens=10, completion_tokens=2,
                              prompt_tokens_details=MagicMock(cached_tokens=0))
    success.model = "gpt-5.4-2026-03-05"
    mock_client.chat.completions.create.side_effect = [
        _fake_openai_504(retry_after=15),
        success,
    ]

    result = run_openai.call_openai_with_retry(
        mock_client, make_prompt(), "gpt-5.4-2026-03-05",
        max_tokens=16, max_retries=3, base_delay=1.0,
    )

    assert mock_client.chat.completions.create.call_count == 2
    assert sleep_calls == [15.0]   # honoured Cloudflare retry_after, not 120s default
    assert result["response_text"] == "ok"
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 2


def test_sync_call_retry_exhaustion_raises(monkeypatch):
    """4 consecutive 504s (1 initial + max_retries=3 retries) exhausts the
    shared retry budget; call_openai_with_retry raises the original
    InternalServerError so _base.run_one's outer try/except can release the
    reservation and propagate up to the orchestrator's per-model loop."""
    from runners import run_openai
    from tests.conftest import make_prompt

    sleep_calls: list[float] = []
    monkeypatch.setattr(run_openai.time, "sleep", lambda s: sleep_calls.append(s))

    mock_client = MagicMock(spec=openai.OpenAI)
    # Body without retry_after → wrapper falls back to DEFAULT_TRANSIENT_5XX_DELAY
    mock_client.chat.completions.create.side_effect = [
        _fake_openai_504(retry_after=None) for _ in range(4)
    ]

    with pytest.raises(openai.InternalServerError):
        run_openai.call_openai_with_retry(
            mock_client, make_prompt(), "gpt-5.4-2026-03-05",
            max_tokens=16, max_retries=3, base_delay=1.0,
        )

    assert mock_client.chat.completions.create.call_count == 4
    assert sleep_calls == [120.0, 120.0, 120.0]   # DEFAULT_TRANSIENT_5XX_DELAY × 3
