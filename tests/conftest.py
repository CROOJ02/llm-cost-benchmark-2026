"""Shared fixtures and helpers for the test suite.

Layer 1 (unit): runner invariants — cost reservation, skip-if-exists,
schema migration. See tests/test_runner_invariants.py.

Layer 3 (integration): orchestrator phase-transition + idempotency +
budget-gate decision logic. See tests/test_orchestrator.py.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from runners.schema import (
    Prompt,
    PromptInput,
    PromptMetadata,
    Scoring,
    Tier1Scoring,
    Tier2Scoring,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_prompt(
    prompt_id: str = "test-001",
    *,
    task_category: str = "summarisation",
    complexity: str = "easy",
    system: str = "You are a test assistant.",
    user: str = "Please respond with the literal string OK.",
    expected: dict[str, Any] | None = None,
    judge_criteria: str | None = None,
) -> Prompt:
    """Build a Prompt object for tests. Defaults satisfy the schema's at-least-one-tier
    rule via tier_1_deterministic; pass judge_criteria to add a tier_2_judge."""
    scoring_kwargs: dict[str, Any] = {}
    if expected is not None or judge_criteria is None:
        scoring_kwargs["tier_1_deterministic"] = Tier1Scoring(expected=expected or {"ok": True})
    if judge_criteria is not None:
        scoring_kwargs["tier_2_judge"] = Tier2Scoring(criteria=judge_criteria)
    return Prompt(
        prompt_id=prompt_id,
        task_category=task_category,  # type: ignore[arg-type]
        complexity=complexity,  # type: ignore[arg-type]
        input=PromptInput(system=system, user=user),
        scoring=Scoring(**scoring_kwargs),
        metadata=PromptMetadata(input_tokens_approx=10, notes=None),
    )


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Return a path to a freshly-migrated SQLite DB (WAL mode, all tables)."""
    db_path = tmp_path / "results.db"
    schema = (REPO_ROOT / "data" / "schema.sql").read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
    return db_path


@pytest.fixture
def temp_phase_log(tmp_path: Path) -> Path:
    return tmp_path / "phase_log.jsonl"


def read_phase_log(log_path: Path) -> list[dict[str, Any]]:
    """Parse all entries from a phase_log.jsonl file. Returns [] if absent."""
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
