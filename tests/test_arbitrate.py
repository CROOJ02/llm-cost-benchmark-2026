"""Layer 1 unit tests for the Day 11 arbitration CLI core.

Covers (per the Day 11 arbitration requirements):
  - Resume detection: identifies first unscored case in active ordering
  - Immediate save: human_score + human_note write back to CSV per case
  - Score range validation: accepts [0.0, 1.0]; rejects out-of-range / non-numeric
  - Ordering: compression-first default; cluster filter; by-category; shuffle determinism
  - Idempotent CSV upgrade: older CSV without human_note column gains it on first save

Tests do NOT exercise the interactive loop (input() / print()). The interactive
layer is a thin orchestration over the testable core; smoke-checked by the
operator on first run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scoring.arbitrate import (
    CSV_COLUMNS,
    Case,
    apply_ordering,
    auto_resolve_sub_threshold,
    find_resume_index,
    load_cases,
    save_case,
    validate_score,
)


def _write_fixture_csv(path: Path, rows: list[dict], include_human_note: bool = True) -> None:
    cols = list(CSV_COLUMNS) if include_human_note else [c for c in CSV_COLUMNS if c != "human_note"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _row(prompt_id: str, model: str = "claude-sonnet-4-6", lever: str = "compression",
         judge_a: float = 0.2, judge_b: float = 0.6, human_score: str = "",
         judge_a_reasoning: str = "opus says wrong",
         judge_b_reasoning: str = "mistral says partial",
         human_note: str = "") -> dict:
    return {
        "prompt_id": prompt_id, "model": model, "lever": lever,
        "judge_a_score": f"{judge_a:.3f}", "judge_b_score": f"{judge_b:.3f}",
        "delta": f"{abs(judge_a - judge_b):.3f}",
        "tier_2_criteria": "fake criteria text",
        "response_text": "fake response text",
        "human_score": human_score,
        "judge_a_reasoning": judge_a_reasoning,
        "judge_b_reasoning": judge_b_reasoning,
        "human_note": human_note,
    }


# ---------- score validation ----------

def test_validate_score_accepts_valid_floats():
    assert validate_score("0.0") == 0.0
    assert validate_score("1.0") == 1.0
    assert validate_score("0.5") == 0.5
    assert validate_score("0.75") == 0.75
    assert validate_score(" 0.5 ") == 0.5  # whitespace tolerated


def test_validate_score_accepts_integer_form():
    assert validate_score("0") == 0.0
    assert validate_score("1") == 1.0


def test_validate_score_rejects_negative():
    with pytest.raises(ValueError, match="out of range"):
        validate_score("-0.1")


def test_validate_score_rejects_above_one():
    with pytest.raises(ValueError, match="out of range"):
        validate_score("1.1")


def test_validate_score_rejects_non_numeric():
    with pytest.raises(ValueError, match="not a number"):
        validate_score("foo")
    with pytest.raises(ValueError, match="not a number"):
        validate_score("0.5x")


def test_validate_score_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_score("")
    with pytest.raises(ValueError, match="empty"):
        validate_score("   ")


# ---------- resume detection ----------

def test_resume_detects_first_unscored_in_ordering(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001", human_score="0.300"),  # scored
        _row("rea-002", human_score="0.500"),  # scored
        _row("rea-003"),                       # NOT scored — resume here
        _row("rea-004"),
        _row("rea-005"),
    ])
    cases = load_cases(csv_path)
    ordering = list(range(len(cases)))  # in original order
    assert find_resume_index(cases, ordering) == 2


def test_resume_returns_total_when_all_scored(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001", human_score="0.500"),
        _row("rea-002", human_score="0.700"),
    ])
    cases = load_cases(csv_path)
    assert find_resume_index(cases, [0, 1]) == 2


def test_resume_respects_active_ordering_not_csv_order(tmp_path):
    """If ordering is reversed, resume position is the first unscored in the
    reversed order, not the first unscored in CSV row order."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001", human_score="0.300"),  # scored, but at end of reversed
        _row("rea-002"),                       # not scored, at middle of reversed
        _row("rea-003", human_score="0.500"),  # scored, but at start of reversed
    ])
    cases = load_cases(csv_path)
    reversed_ordering = [2, 1, 0]
    # Position 0 (idx 2) is scored; position 1 (idx 1) is not — resume there
    assert find_resume_index(cases, reversed_ordering) == 1


def test_resume_skips_already_scored_cases_in_middle(tmp_path):
    """Edge case: someone scored case [0], skipped [1], scored [2]. Resume should
    return position 1 (the unscored gap), not position 3 (after the last scored)."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("a", human_score="0.5"),
        _row("b"),  # gap
        _row("c", human_score="0.7"),
    ])
    cases = load_cases(csv_path)
    assert find_resume_index(cases, [0, 1, 2]) == 1


# ---------- immediate save ----------

def test_save_case_persists_score_to_csv_immediately(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001"),
        _row("rea-002"),
    ])
    save_case(csv_path, "rea-001", "claude-sonnet-4-6", "compression",
              score=0.3, note="content failure per principle 2")
    # Re-read from disk to verify persistence
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    target = next(r for r in rows if r["prompt_id"] == "rea-001")
    assert target["human_score"] == "0.300"
    assert target["human_note"] == "content failure per principle 2"
    # Other rows untouched
    other = next(r for r in rows if r["prompt_id"] == "rea-002")
    assert other["human_score"] == ""
    assert other["human_note"] == ""


def test_save_case_writes_atomically(tmp_path):
    """Writes go through a .tmp + os.replace — verify no .tmp remains after."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [_row("rea-001")])
    save_case(csv_path, "rea-001", "claude-sonnet-4-6", "compression", score=0.5)
    assert csv_path.exists()
    assert not csv_path.with_suffix(".csv.tmp").exists()


def test_save_case_raises_on_unknown_row(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [_row("rea-001")])
    with pytest.raises(KeyError, match="no row matching"):
        save_case(csv_path, "rea-999", "claude-sonnet-4-6", "compression", score=0.5)


def test_save_case_upgrades_csv_lacking_human_note_column(tmp_path):
    """Older CSV (pre-Day-11 reasoning re-fire) lacks human_note. First save
    should add the column without losing existing data."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001", human_score="0.300"),
        _row("rea-002"),
    ], include_human_note=False)
    # Confirm baseline: no human_note column
    with csv_path.open() as f:
        header = next(csv.reader(f))
    assert "human_note" not in header

    save_case(csv_path, "rea-002", "claude-sonnet-4-6", "compression",
              score=0.7, note="upgraded on first save")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert all("human_note" in r for r in rows)
    # Existing scored row preserved
    r1 = next(r for r in rows if r["prompt_id"] == "rea-001")
    assert r1["human_score"] == "0.300"
    assert r1["human_note"] == ""
    # New score persisted
    r2 = next(r for r in rows if r["prompt_id"] == "rea-002")
    assert r2["human_score"] == "0.700"
    assert r2["human_note"] == "upgraded on first save"


# ---------- ordering ----------

def test_compression_first_ordering_puts_compression_before_others(tmp_path):
    """Default ordering: all compression cases (by descending Δ) before any
    non-compression case."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001", lever="baseline",    judge_a=0.2, judge_b=0.6),  # Δ=0.4
        _row("rea-002", lever="compression", judge_a=0.3, judge_b=0.6),  # Δ=0.3
        _row("rea-003", lever="output_cap",  judge_a=0.4, judge_b=0.7),  # Δ=0.3
        _row("rea-004", lever="compression", judge_a=0.1, judge_b=0.6),  # Δ=0.5 ← largest compression
        _row("rea-005", lever="batch",       judge_a=0.5, judge_b=0.9),  # Δ=0.4
    ])
    cases = load_cases(csv_path)
    ord_ = apply_ordering(cases, mode="compression-first")
    # First two must be compression cases, in descending |Δ|
    assert cases[ord_[0]].lever == "compression"
    assert cases[ord_[1]].lever == "compression"
    assert cases[ord_[0]].delta >= cases[ord_[1]].delta
    # Subsequent must be non-compression, descending |Δ|
    for pos in range(2, 5):
        assert cases[ord_[pos]].lever != "compression"
    assert cases[ord_[2]].delta >= cases[ord_[3]].delta >= cases[ord_[4]].delta


def test_cluster_lever_filter_returns_only_matching_lever(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001", lever="baseline"),
        _row("rea-002", lever="compression"),
        _row("rea-003", lever="compression"),
        _row("rea-004", lever="output_cap"),
    ])
    cases = load_cases(csv_path)
    ord_ = apply_ordering(cases, mode="cluster", cluster="compression")
    assert len(ord_) == 2
    assert all(cases[i].lever == "compression" for i in ord_)


def test_cluster_category_filter_returns_only_matching_category(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("cs-001",  lever="compression"),
        _row("rag-001", lever="compression"),
        _row("rea-001", lever="compression"),
        _row("sum-001", lever="compression"),
    ])
    cases = load_cases(csv_path)
    ord_ = apply_ordering(cases, mode="cluster", cluster="rag")
    assert len(ord_) == 1
    assert cases[ord_[0]].prompt_id == "rag-001"


def test_cluster_category_lever_combo(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("cs-001",  lever="compression"),
        _row("cs-002",  lever="baseline"),
        _row("rag-001", lever="compression"),
    ])
    cases = load_cases(csv_path)
    ord_ = apply_ordering(cases, mode="cluster", cluster="cs-compression")
    assert len(ord_) == 1
    assert cases[ord_[0]].prompt_id == "cs-001"


def test_by_category_iterates_in_given_order(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("rea-001", judge_a=0.1, judge_b=0.5),  # Δ=0.4
        _row("cs-001",  judge_a=0.2, judge_b=0.6),  # Δ=0.4
        _row("sum-001", judge_a=0.3, judge_b=0.6),  # Δ=0.3
    ])
    cases = load_cases(csv_path)
    ord_ = apply_ordering(cases, mode="by-category", by_category=["sum", "cs", "rea"])
    assert cases[ord_[0]].prompt_id == "sum-001"
    assert cases[ord_[1]].prompt_id == "cs-001"
    assert cases[ord_[2]].prompt_id == "rea-001"


def test_shuffle_is_deterministic_for_same_seed(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [_row(f"r-{i:02d}") for i in range(10)])
    cases = load_cases(csv_path)
    a = apply_ordering(cases, mode="shuffle", shuffle=True, seed=42)
    b = apply_ordering(cases, mode="shuffle", shuffle=True, seed=42)
    assert a == b


def test_shuffle_differs_across_seeds(tmp_path):
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [_row(f"r-{i:02d}") for i in range(10)])
    cases = load_cases(csv_path)
    a = apply_ordering(cases, mode="shuffle", shuffle=True, seed=1)
    b = apply_ordering(cases, mode="shuffle", shuffle=True, seed=2)
    assert a != b


def test_ordering_covers_all_cases_no_duplicates(tmp_path):
    """Default compression-first must not lose or duplicate any case."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row(f"rea-{i:03d}", lever="compression" if i % 2 else "baseline")
        for i in range(20)
    ])
    cases = load_cases(csv_path)
    ord_ = apply_ordering(cases, mode="compression-first")
    assert sorted(ord_) == list(range(20))


# ---------- load_cases compatibility ----------

def test_load_cases_tolerates_missing_human_note_column(tmp_path):
    """load_cases must handle older CSV files without human_note (pre-Day-11)."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [_row("rea-001", human_score="0.5")],
                       include_human_note=False)
    cases = load_cases(csv_path)
    assert cases[0].human_note == ""
    assert cases[0].is_scored


# ---------- auto_resolve_sub_threshold boundary semantics ----------
#
# REGRESSION: Day 11 run with --min-delta 0.3 left 2 cases at delta=0.3
# exact "in limbo" — the auto_resolve filter was `delta >= min_delta` (skip
# if >=) and the interactive filter was `delta > min_delta` (keep if >).
# Both filters EXCLUDED boundary cases. Fix: auto_resolve_sub_threshold
# now uses `delta > min_delta` (skip if strictly greater), making the two
# filters complementary: every case either auto-resolves or shows in
# interactive, never falls through.

def test_auto_resolve_includes_boundary_case_at_exact_threshold(tmp_path):
    """delta == min_delta exact must be auto-resolved (the case does NOT
    exceed the threshold). Without this fix, a case at delta=0.3 with
    --min-delta=0.3 would be skipped by both auto-resolve (delta >= 0.3
    skipped) and interactive filter (delta > 0.3 excluded) — stuck in limbo.

    Fixture uses (0.0, 0.3) for the boundary because `abs(0.0 - 0.3)` is
    exactly 0.3 in float; `abs(0.5 - 0.8)` evaluates to 0.30000000000000004
    (epsilon over) which would NOT trigger the boundary semantic. The actual
    production data had cases like rea-016 baseline at ja=0.0, jb=0.3."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("at-boundary",  judge_a=0.0, judge_b=0.3),    # delta=0.3 EXACT (float)
        _row("just-above",   judge_a=0.4, judge_b=0.75),   # delta=0.35 — interactive
        _row("just-below",   judge_a=0.5, judge_b=0.75),   # delta=0.25 — auto
    ])
    cases = load_cases(csv_path)
    n_auto = auto_resolve_sub_threshold(csv_path, cases, min_delta=0.3)
    assert n_auto == 2, (
        f"expected 2 cases auto-resolved (boundary + below); got {n_auto}. "
        f"If 1, the boundary case is stuck in limbo (Day 11 bug)."
    )

    # Verify the boundary case has the median canonical written
    cases_after = load_cases(csv_path)
    boundary = next(c for c in cases_after if c.prompt_id == "at-boundary")
    assert boundary.arbitration_method == "median_canonical_auto"
    assert boundary.human_score == "0.150"  # median(0.0, 0.3) = 0.15

    # just-above must NOT be auto-resolved
    above = next(c for c in cases_after if c.prompt_id == "just-above")
    assert above.arbitration_method == ""
    assert above.human_score == ""


def test_auto_resolve_complementary_filter_no_limbo(tmp_path):
    """Compound check: every case in the CSV after auto-resolve is EITHER
    auto-resolved OR (delta > min_delta). None in limbo. The interactive
    filter `delta > min_delta` plus the auto-resolve `delta <= min_delta`
    must form a complete partition of unresolved cases."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("c1",  judge_a=0.1, judge_b=0.9),    # delta=0.8 — interactive
        _row("c2",  judge_a=0.5, judge_b=0.8),    # delta=0.3 boundary — auto
        _row("c3",  judge_a=0.5, judge_b=0.85),   # delta=0.35 — interactive
        _row("c4",  judge_a=0.5, judge_b=0.75),   # delta=0.25 — auto
        _row("c5",  judge_a=0.7, judge_b=0.6),    # delta=0.1 — auto
    ])
    cases = load_cases(csv_path)
    auto_resolve_sub_threshold(csv_path, cases, min_delta=0.3)
    cases_after = load_cases(csv_path)

    interactive = [c for c in cases_after
                   if not c.arbitration_method and c.delta > 0.3]
    auto_resolved = [c for c in cases_after
                     if c.arbitration_method == "median_canonical_auto"]
    limbo = [c for c in cases_after
             if not c.arbitration_method and not (c.delta > 0.3)]
    assert len(limbo) == 0, (
        f"limbo cases (not auto-resolved AND not in interactive ordering): "
        f"{[(c.prompt_id, c.delta) for c in limbo]}"
    )
    assert len(auto_resolved) + len(interactive) == len(cases_after)


def test_auto_resolve_is_idempotent(tmp_path):
    """Re-running auto_resolve on an already-resolved CSV must not double-
    write or change any state."""
    csv_path = tmp_path / "d.csv"
    _write_fixture_csv(csv_path, [
        _row("c1", judge_a=0.5, judge_b=0.65),  # delta=0.15 — auto
        _row("c2", judge_a=0.5, judge_b=0.6),   # delta=0.1 — auto
    ])
    cases = load_cases(csv_path)
    n_first = auto_resolve_sub_threshold(csv_path, cases, min_delta=0.3)
    assert n_first == 2
    # Re-run; everything is already resolved
    cases_after = load_cases(csv_path)
    n_second = auto_resolve_sub_threshold(csv_path, cases_after, min_delta=0.3)
    assert n_second == 0, "second run must skip already-resolved cases"
