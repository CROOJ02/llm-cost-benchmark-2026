"""Layer 3 integration tests for runners/orchestrator.py.

Three tests per the testing strategy doc:
  1. test_run_day_7_phases_execute_in_order — mock all lever calls,
     verify phase log records start/complete events in the correct order
     (batch_submit FIRST, then baseline → caching → output_cap), final
     state has results in DB and one batch_jobs row per (provider, model).
  2. test_day_8_idempotent_re_entry — mock batch retrieval to insert results;
     run run_day_8 twice; verify the second call inserts no new rows and
     batch_jobs.status correctly tracks "already retrieved".
  3. test_decide_compression_tier_ladder — parameterised across the 5 ladder
     boundaries (£100/£150/£200/£230/£270 cost_so_far on a £300 cap).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from runners import _base, lever_batch, lever_caching, lever_compression, lever_output_cap, orchestrator
from runners.orchestrator import Orchestrator
from tests.conftest import make_prompt, read_phase_log


def _count(db_path: Path, table: str, where: str = "1=1", params: tuple = ()) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()[0])


# ---------------------------------------------------------------------------
# Test 1: phase transition state
# ---------------------------------------------------------------------------

def _fake_run_many(adapter, prompts, model, lever, *, run_id, cap_gbp, db_path, **kw):
    """Stand-in for _base.run_many: inserts one fake successful row per prompt."""
    rows = []
    cfg = kw.get("optimisation_config")
    for p in prompts:
        row = _base._new_row(
            prompt=p, model=model, provider=adapter.name, lever=lever,
            config_hash=_base._config_hash(lever, cfg), optimisation_config=cfg,
            run_id=run_id, run_attempt=1,
        )
        row.update({
            "input_tokens": 100, "output_tokens": 50,
            "cost_usd": 0.0001, "latency_ms": 123,
            "response_text": f"fake-{p.prompt_id}-{lever}",
            "model_version": model,
        })
        with sqlite3.connect(db_path) as conn:
            _base._insert_row(conn, row)
            conn.commit()
        rows.append(row)
    return rows


def test_run_day_7_phases_execute_in_order(monkeypatch, tmp_path, fresh_db, temp_phase_log):
    """All four Day 7 phases run in order, each emits start+complete events to
    phase_log, results land in DB, batch_jobs has one row per (provider, model)."""

    # --- mocks ---
    # baseline goes through _base.run_many; lever modules expose their own
    # test entrypoints we mock separately.
    monkeypatch.setattr(_base, "run_many", _fake_run_many)

    def _fake_caching_test(adapter, prompts, model, *, run_id, cap_gbp, db_path, **kw):
        """Mirror lever_caching.run_caching_test's row-insert contract,
        respecting skip-if-exists for the baseline cell (run_baseline already
        inserted it earlier in the phase sequence). Mirrors the real lever's
        reasoning_effort annotation on each cell so the config_hashes match
        the orchestrator's annotated baseline (for GPT-5.4 family)."""
        from runners import run_openai
        annotate = run_openai.annotate_optimisation_config_for_reasoning_effort
        out = []
        for p in prompts:
            cells = []
            for lever, cfg in [
                ("baseline", annotate(None, model)),
                ("caching",  annotate({"cache_phase": "write", "enable_cache": True}, model)),
                ("caching",  annotate({"cache_phase": "read",  "enable_cache": True}, model)),
            ]:
                config_hash = _base._config_hash(lever, cfg)
                with sqlite3.connect(db_path) as conn:
                    existing = _base._existing_successful_row(
                        conn, p.prompt_id, model, lever, config_hash, 1, run_id,
                    )
                if existing is not None:
                    cells.append(existing)
                    continue
                row = _base._new_row(
                    prompt=p, model=model, provider=adapter.name, lever=lever,
                    config_hash=config_hash, optimisation_config=cfg,
                    run_id=run_id, run_attempt=1,
                )
                row.update({"input_tokens": 100, "output_tokens": 50,
                            "cost_usd": 0.0001, "latency_ms": 123,
                            "response_text": f"fake-{p.prompt_id}-{lever}",
                            "model_version": model})
                with sqlite3.connect(db_path) as conn:
                    _base._insert_row(conn, row)
                    conn.commit()
                cells.append(row)
            out.append({
                "prompt_id": p.prompt_id, "model": model,
                "baseline": cells[0], "cache_write": cells[1], "cache_read": cells[2],
                "caching_available": True, "skip_reason": None,
            })
        return out

    def _fake_output_cap_test(adapter, prompts, model, *, run_id, cap_gbp, max_tokens, db_path, **kw):
        rows = []
        for p in prompts:
            cfg = {"max_tokens": max_tokens}
            row = _base._new_row(
                prompt=p, model=model, provider=adapter.name, lever="output_cap",
                config_hash=_base._config_hash("output_cap", cfg), optimisation_config=cfg,
                run_id=run_id, run_attempt=1,
            )
            row.update({"input_tokens": 100, "output_tokens": max_tokens,
                        "cost_usd": 0.0001, "latency_ms": 123,
                        "response_text": "fake", "model_version": model})
            with sqlite3.connect(db_path) as conn:
                _base._insert_row(conn, row)
                conn.commit()
            rows.append(row)
        return rows

    submit_calls: list[tuple[str, int]] = []
    def _fake_submit_batch(prompts, model, *, run_id, db_path, **kw):
        submit_calls.append((model, len(prompts)))
        batch_id = f"batch_{model}_{len(submit_calls)}"
        row = {
            "batch_id": batch_id, "run_id": run_id, "provider": "anthropic" if model.startswith("claude") else "openai",
            "model": model, "lever": "batch", "status": "submitted",
            "submitted_at": _base._now_iso(), "retrieved_at": None, "completed_at": None,
            "prompt_ids": json.dumps([p.prompt_id for p in prompts]),
            "request_count": len(prompts), "error": None,
        }
        with sqlite3.connect(db_path) as conn:
            cols = ", ".join(row.keys())
            ph = ", ".join(["?"] * len(row))
            conn.execute(f"INSERT INTO batch_jobs ({cols}) VALUES ({ph})", list(row.values()))
            conn.commit()
        return {**row, "skipped": False}

    monkeypatch.setattr(lever_caching, "run_caching_test", _fake_caching_test)
    monkeypatch.setattr(lever_output_cap, "run_output_cap_test", _fake_output_cap_test)
    monkeypatch.setattr(lever_batch, "submit_batch", _fake_submit_batch)

    # --- run ---
    rid = _base.start_run(cost_cap_gbp=300.0, db_path=fresh_db)
    orch = Orchestrator(run_id=rid, cap_gbp=300.0, db_path=fresh_db, phase_log_path=temp_phase_log)

    prompts = [
        make_prompt("sum-015", task_category="summarisation"),
        make_prompt("sum-016", task_category="summarisation"),
    ]
    models = ["claude-sonnet-4-6", "gpt-5.4-2026-03-05"]

    # CACHING_SUBSET_IDS includes sum-015 and sum-016 — they'll exercise caching.
    summary = orch.run_day_7(prompts, models)

    # --- assertions ---
    # Phase log captures the expected sequence.
    log = read_phase_log(temp_phase_log)
    starts = [(e["phase"], e["event"]) for e in log if e["event"] == "start"]
    expected_start_phases = [
        "day_7",
        "batch_submit", "batch_submit",       # one per model
        "baseline",     "baseline",
        "caching",      "caching",
        "output_cap",   "output_cap",
    ]
    assert [p for p, _ in starts] == expected_start_phases, (
        f"phase ordering wrong; got {[p for p, _ in starts]}"
    )

    # Each non-day_7 phase has matching start and complete.
    for phase in ("batch_submit", "baseline", "caching", "output_cap"):
        n_start = sum(1 for e in log if e["phase"] == phase and e["event"] == "start")
        n_done  = sum(1 for e in log if e["phase"] == phase and e["event"] == "complete")
        assert n_start == n_done == len(models), (
            f"phase {phase!r}: {n_start} starts vs {n_done} completes"
        )

    # batch_jobs: one row per model.
    assert _count(fresh_db, "batch_jobs", "run_id = ?", (rid,)) == len(models)

    # Per model: 2 baseline (from baseline phase) + 4 caching (write+read on each
    # of 2 prompts; baseline cell skip-if-exists) + 2 output_cap = 8 rows per model.
    # No baseline rows from SUBMIT (batch results land at retrieve, not submit).
    assert _count(fresh_db, "results", "run_id = ?", (rid,)) == 8 * len(models)

    # Summary returned mirrors phase_log totals.
    assert summary["n_baseline"] == 2 * len(models)
    assert len(summary["batch_ids"]) == len(models)


# ---------------------------------------------------------------------------
# Test 2: Day 8 idempotent re-entry
# ---------------------------------------------------------------------------

def test_day_8_idempotent_re_entry(monkeypatch, tmp_path, fresh_db, temp_phase_log):
    """Mock batch retrieval to insert results + mark completed. Call run_day_8
    twice and assert the second is a no-op (no duplicate rows; status stays
    'completed'; compression skipped via tier='skip')."""

    rid = _base.start_run(cost_cap_gbp=300.0, db_path=fresh_db)

    # Seed one submitted batch row.
    test_prompt = make_prompt("sum-015", task_category="summarisation")
    seed_batch = {
        "batch_id": "batch_test_001", "run_id": rid, "provider": "anthropic",
        "model": "claude-sonnet-4-6", "lever": "batch", "status": "submitted",
        "submitted_at": _base._now_iso(),
        "retrieved_at": None, "completed_at": None,
        "prompt_ids": json.dumps(["sum-015"]),
        "request_count": 1, "error": None,
    }
    with sqlite3.connect(fresh_db) as conn:
        cols = ", ".join(seed_batch.keys())
        ph = ", ".join(["?"] * len(seed_batch))
        conn.execute(f"INSERT INTO batch_jobs ({cols}) VALUES ({ph})", list(seed_batch.values()))
        conn.commit()

    poll_calls: list[str] = []

    def _fake_poll_anthropic(self, job, prompts_by_id):
        """Stand in for the real Anthropic batch poll — uses the orchestrator's
        own `_insert_batch_result_row` + `_update_batch_status` so the
        idempotency seam is exercised with real persistence helpers."""
        poll_calls.append(job["batch_id"])
        prompt = prompts_by_id["sum-015"]
        inserted = self._insert_batch_result_row(
            prompt=prompt, model=job["model"], provider="anthropic", lever=job["lever"],
            message_meta={
                "input_tokens": 1000, "output_tokens": 200,
                "cached_tokens": 0, "cache_creation_tokens": 0,
                "response_text": "hello",
                "model_version": "claude-sonnet-4-6",
            },
        )
        self._update_batch_status(
            job["batch_id"], "completed",
            retrieved_at=_base._now_iso(), completed_at=_base._now_iso(),
        )
        return 1 if inserted else 0

    monkeypatch.setattr(Orchestrator, "_poll_anthropic_batch", _fake_poll_anthropic)
    # Redirect prompt loading so we don't need the real prompt file on disk.
    monkeypatch.setattr(orchestrator, "load_all_prompts", lambda: [test_prompt])
    # Stub run_compression — this test isolates batch retrieval idempotency, not
    # compression. Without the stub, decide_compression_tier() with cost_so_far=0
    # picks tier='full' and runs the actual lever (loads LLMLingua-2 + hits the API).
    monkeypatch.setattr(Orchestrator, "run_compression", lambda self, decision: [])

    orch = Orchestrator(run_id=rid, cap_gbp=300.0, db_path=fresh_db, phase_log_path=temp_phase_log)

    # First Day-8 invocation — pulls 1 result, marks batch completed, skips compression.
    summary1 = orch.run_day_8()
    n_results_after_1 = _count(fresh_db, "results", "run_id = ?", (rid,))
    assert n_results_after_1 == 1
    assert poll_calls == ["batch_test_001"]
    with sqlite3.connect(fresh_db) as conn:
        status1 = conn.execute(
            "SELECT status FROM batch_jobs WHERE batch_id = ?", ("batch_test_001",)
        ).fetchone()[0]
    assert status1 == "completed"

    # Second invocation — must be a no-op for retrieval (in-flight set is empty).
    summary2 = orch.run_day_8()
    n_results_after_2 = _count(fresh_db, "results", "run_id = ?", (rid,))
    assert n_results_after_2 == n_results_after_1, "duplicate result rows inserted on re-entry"
    assert poll_calls == ["batch_test_001"], "polled the completed batch a second time"
    assert summary2["n_retrieved_batches"] == 0


# ---------------------------------------------------------------------------
# Test 3: budget gate decision logic (parameterised across the 5 boundaries)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cap_gbp,cost_so_far,expected_tier",
    [
        # Original PRD-spec values at £300 cap (backward compatibility check).
        # Percentages: 200/300=66.7% → full; 150/300=50% → 60-subset;
        # 100/300=33.3% → 30-subset; 70/300=23.3% → operator-call; 30/300=10% → skip.
        (300.0, 100.0, "full"),
        (300.0, 150.0, "60-subset"),
        (300.0, 200.0, "30-subset"),
        (300.0, 230.0, "operator-call"),
        (300.0, 270.0, "skip"),
        # Scale-invariance: same percentages at £5 cap (the dry-run's working cap).
        # Percentages: 4.00/5=80% → full; 2.50/5=50% → 60-subset;
        # 1.50/5=30% → 30-subset; 1.00/5=20% → operator-call; 0.50/5=10% → skip.
        (5.0,   1.00, "full"),
        (5.0,   2.50, "60-subset"),
        (5.0,   3.50, "30-subset"),
        (5.0,   4.00, "operator-call"),
        (5.0,   4.50, "skip"),
    ],
)
def test_decide_compression_tier_ladder(
    fresh_db, temp_phase_log, cap_gbp, cost_so_far, expected_tier,
):
    """§9 Day 8 ladder is proportional to cap, not absolute GBP. Verified at
    £300 (the PRD's original numbers) and at £5 (the dry-run's working cap):
    the same headroom percentage produces the same tier regardless of scale."""
    rid = _base.start_run(cost_cap_gbp=cap_gbp, db_path=fresh_db)
    with sqlite3.connect(fresh_db) as conn:
        conn.execute(
            "UPDATE runs SET cost_so_far_gbp = ? WHERE run_id = ?", (cost_so_far, rid),
        )
        conn.commit()

    orch = Orchestrator(
        run_id=rid, cap_gbp=cap_gbp,
        db_path=fresh_db, phase_log_path=temp_phase_log,
    )
    decision = orch.decide_compression_tier()

    headroom_pct = (cap_gbp - cost_so_far) / cap_gbp
    assert decision["tier"] == expected_tier, (
        f"cap=£{cap_gbp}, cost_so_far=£{cost_so_far}: "
        f"headroom_pct={headroom_pct:.1%} → expected {expected_tier!r}, got {decision['tier']!r}"
    )
    assert decision["cost_so_far_gbp"] == pytest.approx(cost_so_far)
    assert decision["headroom_gbp"] == pytest.approx(cap_gbp - cost_so_far)
    assert decision["cap_gbp"] == cap_gbp

    # Decision was logged.
    log = read_phase_log(temp_phase_log)
    decisions = [e for e in log if e["phase"] == "compression_decide" and e["event"] == "decision"]
    assert len(decisions) == 1
    assert decisions[0]["payload"]["tier"] == expected_tier
