"""Tier-2 disagreement detection + canonical-score computation.

Pairs each (prompt_id, model, lever) row's two judge scores and:
  - flags the row when |judge_a - judge_b| > 0.2 (per PRD §7)
  - computes the canonical score as median of the two on agreement
  - emits a CSV of disagreement cases for Day 11 human arbitration

Public surface:
  - DISAGREEMENT_THRESHOLD (= 0.2)
  - canonical_score(judge_a, judge_b) -> float | None
  - is_disagreement(judge_a, judge_b) -> bool
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
    """One row's pair of judge scores. Either may be None on judge_error."""
    prompt_id: str
    model: str
    lever: str
    judge_a_score: float | None  # opus
    judge_b_score: float | None  # mistral
    response_text: str = ""
    tier_2_criteria: str = ""


def is_disagreement(judge_a: float | None, judge_b: float | None) -> bool:
    """Returns True iff both judges produced a score AND |a-b| > 0.2.

    A None on either side means no judge_disagreement_flag fires (we can't
    arbitrate against a missing score; that case is judge_error and routes
    differently). Threshold is strict `>` per PRD §7 line 319 ("agree
    within 0.2" means |a-b| <= 0.2 agrees; > 0.2 disagrees)."""
    if judge_a is None or judge_b is None:
        return False
    return abs(judge_a - judge_b) > DISAGREEMENT_THRESHOLD


def canonical_score(judge_a: float | None, judge_b: float | None) -> float | None:
    """Median of the two judge scores when both present and they agree.

    - Both None → None (judge_error on both sides)
    - One None → the other (best available signal)
    - Both present and agree (|Δ|<=0.2) → median (= mean for 2 values)
    - Both present and disagree (|Δ|>0.2) → median anyway (the disagreement
      flag is set separately; final_score may later be overridden by human
      arbitration in Day 11)
    """
    if judge_a is None and judge_b is None:
        return None
    if judge_a is None:
        return judge_b
    if judge_b is None:
        return judge_a
    return statistics.median([judge_a, judge_b])


def emit_disagreement_csv(rows: list[JudgePair], out_path: Path) -> int:
    """Write a CSV of disagreement rows for Day 11 arbitration. Returns the
    count of rows written. Schema: prompt_id, model, lever, judge_a_score,
    judge_b_score, delta, tier_2_criteria, response_text, human_score (blank).
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
