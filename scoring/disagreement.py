"""Tier-2 disagreement detection + canonical-score computation.

Pairs each (prompt_id, model, lever) row's judge scores and:
  - flags the row when judges disagree (per the methodology rules below)
  - computes the canonical score as median
  - emits a CSV of disagreement cases for Day 11 human arbitration

Methodology rules (see docs/methodology/prompt_design_decisions.md
§ "Three-judge disagreement methodology"):

  - 2 judges (legacy 2-judge sweeps): disagreement = |a - b| > 0.2.
    Canonical = median (= mean for 2 values).
  - 3+ judges (Day 11 revision): canonical = median of valid scores;
    disagreement = ANY judge deviates from median by > 0.2. Generalises
    cleanly: with 2 judges, "any deviates from median by > 0.2" would mean
    |a - midpoint| > 0.2 i.e. |a - b| > 0.4, which is twice as strict as
    the existing 2-judge rule — so the 2-judge case is special-cased to
    preserve historical semantics.
  - 1 score available: no disagreement (need at least 2 to compare).
  - 0 scores: no disagreement (no data).

Public surface:
  - DISAGREEMENT_THRESHOLD (= 0.2)
  - canonical_score(*scores) -> float | None
  - is_disagreement(*scores) -> bool
  - emit_disagreement_csv(rows, out_path) -> int (count written)
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

DISAGREEMENT_THRESHOLD = 0.2


@dataclass
class JudgePair:
    """One row's pair of judge scores (legacy 2-judge shape; preserved for
    callers that haven't migrated to N-judge yet). Either may be None on
    judge_error."""
    prompt_id: str
    model: str
    lever: str
    judge_a_score: float | None  # opus
    judge_b_score: float | None  # mistral (legacy) or gpt55 (post Day 11)
    response_text: str = ""
    tier_2_criteria: str = ""


def is_disagreement(*scores: float | None) -> bool:
    """True iff judges disagree per the methodology rules.

    Variadic *args API supports both legacy 2-judge calls (preserves
    backward-compat — the 2-judge semantic is unchanged) and 3+-judge
    calls (median-deviation rule).

    None scores are excluded (judge_error rows route to a separate path,
    not arbitration). Need ≥ 2 valid scores to flag disagreement."""
    valid = [s for s in scores if s is not None]
    if len(valid) < 2:
        return False
    if len(valid) == 2:
        # Legacy 2-judge semantic: |a - b| > 0.2.
        return abs(valid[0] - valid[1]) > DISAGREEMENT_THRESHOLD
    # N ≥ 3: any judge deviates from median by > 0.2.
    median_v = statistics.median(valid)
    return any(abs(s - median_v) > DISAGREEMENT_THRESHOLD for s in valid)


def canonical_score(*scores: float | None) -> float | None:
    """Median of available judge scores; None if no judge scored.

    - All None → None (judge_error on every side)
    - One available → that value (best-available signal)
    - Two+ available → median
    """
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return statistics.median(valid)


def emit_disagreement_csv(rows: list[JudgePair], out_path: Path) -> int:
    """Write a CSV of disagreement rows for Day 11 arbitration. Returns the
    count of rows written. Schema: prompt_id, model, lever, judge_a_score,
    judge_b_score, delta, tier_2_criteria, response_text, human_score.

    NOTE: this writer is the legacy 2-judge shape. When the 3-judge re-grade
    lands, day_10.py emits the CSV directly with judge_a/b/c columns instead
    of going through this function. Kept here for the existing 2-judge
    test fixtures and any future 2-judge analysis.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "prompt_id", "model", "lever",
            "judge_a_score", "judge_b_score", "delta",
            "tier_2_criteria", "response_text",
            "human_score",  # operator fills this in during Day 11
        ])
        for r in rows:
            if not is_disagreement(r.judge_a_score, r.judge_b_score):
                continue
            delta = abs(r.judge_a_score - r.judge_b_score)  # type: ignore[operator]
            w.writerow([
                r.prompt_id, r.model, r.lever,
                f"{r.judge_a_score:.3f}", f"{r.judge_b_score:.3f}", f"{delta:.3f}",
                r.tier_2_criteria, r.response_text,
                "",
            ])
            n += 1
    return n
