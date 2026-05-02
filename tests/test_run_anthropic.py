"""Tests for runners/run_anthropic.py retry logic (Step D).

Exercises three paths:
- 429 → 429 → success: confirms 3 API attempts, 2 sleeps with exponential delays
- 429 × 6 (exhaustion): confirms error row inserted, no cost accumulated
- 429 with Retry-After header: confirms header value used in place of exponential backoff
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from runners.run_anthropic import (
    DEFAULT_MAX_RETRIES,
    _read_cost_so_far_gbp,
    call_anthropic_with_retry,
    run_one,
    start_run,
)
from runners.schema import Prompt, PromptInput, PromptMetadata, Scoring, Tier1Scoring

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_429(retry_after: str | None = None) -> anthropic.RateLimitError:
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    fake_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    fake_response = httpx.Response(status_code=429, headers=headers, request=fake_request)
    return anthropic.RateLimitError(
        message="429 rate limit",
        response=fake_response,
        body={"error": {"type": "rate_limit_error", "message": "rate_limit"}},
    )


def _fake_success(input_tokens: int = 157, output_tokens: int = 78) -> MagicMock:
    text_block = MagicMock(type="text", text='{"category": "billing", "reply": "ack"}')
    msg = MagicMock(content=[text_block], model="claude-sonnet-4-6")
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_read_input_tokens = 0
    return msg


def _test_prompt() -> Prompt:
    return Prompt(
        prompt_id="cs-test",
        task_category="customer_support",
        complexity="easy",
        input=PromptInput(system="sys", user="user"),
        scoring=Scoring(tier_1_deterministic=Tier1Scoring(expected={"category": "billing"})),
        metadata=PromptMetadata(input_tokens_approx=10, notes=None),
    )


def _fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "results.db"
    schema = (REPO_ROOT / "data" / "schema.sql").read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
    return db_path


def test_retry_then_success(monkeypatch):
    """429 → 429 → success: 3 attempts, 2 sleeps with exponential delays (1s, 2s)."""
    sleep_calls: list[float] = []
    monkeypatch.setattr("runners.run_anthropic.time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("runners.run_anthropic.random.uniform", lambda lo, hi: 0.0)

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = [
        _fake_429(),
        _fake_429(),
        _fake_success(input_tokens=157, output_tokens=78),
    ]

    result = call_anthropic_with_retry(
        mock_client, _test_prompt(), "claude-sonnet-4-6",
        max_tokens=1024, max_retries=DEFAULT_MAX_RETRIES, base_delay=1.0,
    )

    assert mock_client.messages.create.call_count == 3
    assert sleep_calls == [1.0, 2.0]  # base × 2^0, base × 2^1
    assert result["input_tokens"] == 157
    assert result["output_tokens"] == 78
    assert result["model_version"] == "claude-sonnet-4-6"


def test_retry_exhausted_inserts_error_row(tmp_path, monkeypatch):
    """429 × 6 (1 initial + 5 retries): error row inserted, runs.cost_so_far_gbp stays 0."""
    monkeypatch.setattr("runners.run_anthropic.time.sleep", lambda s: None)
    monkeypatch.setattr("runners.run_anthropic.random.uniform", lambda lo, hi: 0.0)
    monkeypatch.setattr("runners.run_anthropic.count_input_tokens", lambda *a, **kw: 157)

    db_path = _fresh_db(tmp_path)
    rid = start_run(cost_cap_gbp=300.0, db_path=db_path)

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = [_fake_429() for _ in range(DEFAULT_MAX_RETRIES + 1)]

    result = run_one(
        _test_prompt(), model="claude-sonnet-4-6", lever="baseline",
        run_id=rid, cap_gbp=300.0, completed=0, planned=1,
        client=mock_client, db_path=db_path,
        max_retries=DEFAULT_MAX_RETRIES, base_delay=1.0,
    )

    assert mock_client.messages.create.call_count == DEFAULT_MAX_RETRIES + 1
    assert result["error"] is not None
    assert "RateLimitError" in result["error"]
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert result["cost_usd"] == 0.0
    assert result["output_format_valid"] == 0

    with sqlite3.connect(db_path) as conn:
        assert _read_cost_so_far_gbp(conn, rid) == 0.0
        n = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        assert n == 1
        err = conn.execute("SELECT error FROM results").fetchone()[0]
        assert err is not None and "RateLimitError" in err


def test_retry_after_header_honoured(monkeypatch):
    """429 with Retry-After: 7 → sleep 7s (not the exponential 1s)."""
    sleep_calls: list[float] = []
    monkeypatch.setattr("runners.run_anthropic.time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("runners.run_anthropic.random.uniform", lambda lo, hi: 0.0)

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = [
        _fake_429(retry_after="7"),
        _fake_success(),
    ]

    call_anthropic_with_retry(
        mock_client, _test_prompt(), "claude-sonnet-4-6",
        max_tokens=1024, max_retries=DEFAULT_MAX_RETRIES, base_delay=1.0,
    )

    assert mock_client.messages.create.call_count == 2
    assert sleep_calls == [7.0]
