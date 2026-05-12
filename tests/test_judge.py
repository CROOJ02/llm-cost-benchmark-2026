"""Layer 1 unit tests for Tier-2 dual-judge scoring.

Covers (per the Day 10 judge implementation requirements):
  - Position randomisation produces consistent seed-based output
  - Anonymisation correctly hides model identity
  - JSON validation catches malformed judge responses (parse + range)
  - Disagreement detection correctly flags |Δ| > 0.2 and not |Δ| ≤ 0.2
  - Median canonical score handles edge cases
  - Reference-answer asymmetry per category (RAG/reasoning include; cs/sum don't)

Note: tests do NOT call the judge APIs. Network-touching paths are exercised
by the Day 10 dry-run script, not by Layer 1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import openai
import pytest

from scoring.disagreement import (
    DISAGREEMENT_THRESHOLD,
    JudgePair,
    canonical_score,
    emit_disagreement_csv,
    is_disagreement,
)
from scoring.judge import (
    LABELS,
    SCORE_HI,
    SCORE_LO,
    assemble_judge_call,
    position_seed,
    randomised_position_map,
    _parse_judge_response,
)
from tests.conftest import make_prompt


TEST_MODELS_4 = ["claude-sonnet-4-6", "claude-haiku-4-5", "gpt-5.4-2026-03-05", "gpt-5.4-mini-2026-03-17"]


# ---------- position randomisation: deterministic per (prompt, lever) ----------

def test_position_seed_is_deterministic():
    a = position_seed("rea-001", "baseline")
    b = position_seed("rea-001", "baseline")
    assert a == b


def test_position_seed_differs_by_prompt_and_lever():
    s1 = position_seed("rea-001", "baseline")
    s2 = position_seed("rea-002", "baseline")
    s3 = position_seed("rea-001", "compression")
    assert s1 != s2
    assert s1 != s3
    assert s2 != s3


def test_randomised_position_map_is_seed_stable():
    seed = position_seed("rea-001", "baseline")
    m1 = randomised_position_map(TEST_MODELS_4, seed)
    m2 = randomised_position_map(TEST_MODELS_4, seed)
    assert m1 == m2


def test_randomised_position_map_input_order_does_not_matter():
    seed = position_seed("rea-001", "baseline")
    m1 = randomised_position_map(TEST_MODELS_4, seed)
    m2 = randomised_position_map(list(reversed(TEST_MODELS_4)), seed)
    assert m1 == m2, "different input orders must produce same mapping at same seed"


def test_randomised_position_map_uses_all_4_labels_and_models():
    seed = position_seed("rea-001", "baseline")
    m = randomised_position_map(TEST_MODELS_4, seed)
    assert set(m.keys()) == set(LABELS)
    assert set(m.values()) == set(TEST_MODELS_4)


def test_randomised_position_map_rejects_wrong_arity():
    with pytest.raises(ValueError):
        randomised_position_map(TEST_MODELS_4[:3], position_seed("p", "l"))


def test_position_distribution_balances_across_calls():
    """Smoke check: across 100 (prompt, lever) seeds, no model gets locked
    into one position. Each model should appear in each slot on the order of
    25 times out of 100."""
    from collections import Counter
    counts: dict[str, Counter[str]] = {m: Counter() for m in TEST_MODELS_4}
    for i in range(100):
        seed = position_seed(f"prompt-{i}", "baseline")
        m = randomised_position_map(TEST_MODELS_4, seed)
        for label, model in m.items():
            counts[model][label] += 1
    for model, c in counts.items():
        for label in LABELS:
            assert 10 <= c[label] <= 40, f"{model} in slot {label}: {c[label]}/100 looks lopsided"


# ---------- anonymisation: judge prompt MUST hide model identity ----------

def test_assembled_call_does_not_leak_model_names():
    """The assembler must not inject model identity into the call. Stubs
    deliberately use neutral text so any leak would come from the assembler
    itself (header strings, position-to-model dump, etc.), not from the
    response_text content."""
    prompt = make_prompt(
        prompt_id="cs-001", task_category="customer_support",
        expected={"category": "billing"},
        judge_criteria="Reply acknowledges issue without committing to specific outcomes.",
    )
    responses = {m: f"neutral stub response number {i}"
                 for i, m in enumerate(TEST_MODELS_4)}
    call = assemble_judge_call(prompt, responses, lever="baseline")
    msg = call.user_message
    for model in TEST_MODELS_4:
        assert model not in msg, (
            f"model name {model!r} leaked into judge prompt at position "
            f"{msg.find(model)}"
        )
    # also assert no provider name leaks at the assembler layer
    for token in ("anthropic", "openai", "claude", "gpt"):
        assert token not in msg.lower(), f"provider/family token {token!r} leaked"


def test_assembled_call_does_not_leak_lever_name():
    prompt = make_prompt(
        prompt_id="cs-001", task_category="customer_support",
        expected={"category": "billing"},
        judge_criteria="x",
    )
    responses = {m: f"r-{m}" for m in TEST_MODELS_4}
    for lever in ["baseline", "batch", "compression", "output_cap"]:
        call = assemble_judge_call(prompt, responses, lever=lever)
        assert lever not in call.user_message.lower(), (
            f"lever name {lever!r} leaked into judge prompt"
        )


def test_assembled_call_uses_position_labels():
    prompt = make_prompt(
        prompt_id="cs-001", task_category="customer_support",
        expected={"category": "billing"},
        judge_criteria="x",
    )
    responses = {m: f"r-{m}" for m in TEST_MODELS_4}
    call = assemble_judge_call(prompt, responses, lever="baseline")
    for label in LABELS:
        assert f"--- Response {label} ---" in call.user_message


# ---------- reference-answer asymmetry per category ----------

def test_rag_qa_call_includes_reference_answer():
    prompt = make_prompt(
        prompt_id="rag-001", task_category="rag_qa",
        expected={"answer": "2014", "supporting_sentences": [4]},
        judge_criteria="Answer states the year 2014 concisely.",
    )
    responses = {m: f"r-{m}" for m in TEST_MODELS_4}
    call = assemble_judge_call(prompt, responses, lever="baseline")
    assert "REFERENCE ANSWER" in call.user_message
    assert "2014" in call.user_message


def test_reasoning_call_includes_reference_final_answer():
    prompt = make_prompt(
        prompt_id="rea-001", task_category="reasoning",
        expected={"final_answer": "£14.00"},
        judge_criteria="Reasoning must compute daily rate then multiply by unused days.",
    )
    responses = {m: f"r-{m}" for m in TEST_MODELS_4}
    call = assemble_judge_call(prompt, responses, lever="baseline")
    assert "REFERENCE ANSWER" in call.user_message
    assert "£14.00" in call.user_message


def test_customer_support_call_excludes_reference_block():
    prompt = make_prompt(
        prompt_id="cs-001", task_category="customer_support",
        expected={"category": "billing"},
        judge_criteria="Reply acknowledges issue.",
    )
    responses = {m: f"r-{m}" for m in TEST_MODELS_4}
    call = assemble_judge_call(prompt, responses, lever="baseline")
    assert "REFERENCE ANSWER" not in call.user_message


def test_summarisation_call_excludes_reference_block():
    prompt = make_prompt(
        prompt_id="sum-001", task_category="summarisation",
        judge_criteria="Must cover Q1 financial result and German market entry.",
    )
    responses = {m: f"r-{m}" for m in TEST_MODELS_4}
    call = assemble_judge_call(prompt, responses, lever="baseline")
    assert "REFERENCE ANSWER" not in call.user_message


# ---------- judge response JSON validation ----------

def test_parse_clean_json_response():
    raw = json.dumps({
        "A": 0.9, "B": 0.5, "C": 0.7, "D": 0.2,
        "reasoning": {"A": "good", "B": "partial", "C": "ok", "D": "wrong"},
    })
    scores, reasoning, err = _parse_judge_response(raw)
    assert err is None
    assert scores == {"A": 0.9, "B": 0.5, "C": 0.7, "D": 0.2}
    assert reasoning == {"A": "good", "B": "partial", "C": "ok", "D": "wrong"}


def test_parse_strips_markdown_fence():
    raw = '```json\n{"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.7}\n```'
    scores, _, err = _parse_judge_response(raw)
    assert err is None
    assert scores == {"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.7}


def test_parse_strips_pre_json_preamble():
    raw = (
        "Here is my evaluation of the four responses:\n\n"
        '{"A": 0.6, "B": 0.6, "C": 0.6, "D": 0.6}'
    )
    scores, _, err = _parse_judge_response(raw)
    assert err is None
    assert all(v == 0.6 for v in scores.values())


def test_parse_rejects_score_above_1():
    raw = '{"A": 1.5, "B": 0.5, "C": 0.5, "D": 0.5}'
    scores, _, err = _parse_judge_response(raw)
    assert scores is None
    assert "out of range" in err


def test_parse_rejects_negative_score():
    raw = '{"A": -0.1, "B": 0.5, "C": 0.5, "D": 0.5}'
    scores, _, err = _parse_judge_response(raw)
    assert scores is None
    assert "out of range" in err


def test_parse_rejects_missing_label():
    raw = '{"A": 0.5, "B": 0.5, "C": 0.5}'  # D missing
    scores, _, err = _parse_judge_response(raw)
    assert scores is None
    assert "missing label 'D'" in err


def test_parse_rejects_string_score():
    raw = '{"A": "0.5", "B": 0.5, "C": 0.5, "D": 0.5}'
    scores, _, err = _parse_judge_response(raw)
    assert scores is None
    assert "not a number" in err


def test_parse_rejects_bool_score():
    """`isinstance(True, int)` is True in Python — guard against it."""
    raw = '{"A": true, "B": 0.5, "C": 0.5, "D": 0.5}'
    scores, _, err = _parse_judge_response(raw)
    assert scores is None
    assert "not a number" in err


def test_parse_handles_no_json():
    raw = "I cannot evaluate these responses."
    scores, _, err = _parse_judge_response(raw)
    assert scores is None


def test_parse_score_at_exact_bounds_is_accepted():
    raw = '{"A": 0.0, "B": 1.0, "C": 0.0, "D": 1.0}'
    scores, _, err = _parse_judge_response(raw)
    assert err is None
    assert scores == {"A": SCORE_LO, "B": SCORE_HI, "C": SCORE_LO, "D": SCORE_HI}


# ---------- disagreement detection ----------

def test_disagreement_flags_above_threshold():
    assert is_disagreement(0.9, 0.6) is True   # delta 0.3
    assert is_disagreement(0.4, 0.0) is True   # delta 0.4
    assert is_disagreement(1.0, 0.0) is True   # delta 1.0


def test_no_disagreement_below_or_at_threshold():
    assert is_disagreement(0.5, 0.5) is False  # delta 0
    assert is_disagreement(0.7, 0.5) is False  # delta 0.2 (== threshold; PRD: agree)
    assert is_disagreement(0.6, 0.4) is False  # delta 0.2
    assert is_disagreement(0.9, 0.7001) is False  # delta ~0.1999


def test_disagreement_at_exact_threshold_does_not_fire():
    """PRD says |a-b| <= 0.2 agrees, > 0.2 disagrees. Boundary is in 'agree'."""
    assert is_disagreement(0.5, 0.7) is False
    assert is_disagreement(0.5, 0.7000001) is True


def test_disagreement_with_none_does_not_fire():
    """Missing scores route to judge_error, not to disagreement arbitration."""
    assert is_disagreement(None, 0.5) is False
    assert is_disagreement(0.5, None) is False
    assert is_disagreement(None, None) is False


# ---------- canonical score (median) ----------

def test_canonical_score_median_on_agreement():
    assert canonical_score(0.5, 0.5) == 0.5
    assert canonical_score(0.7, 0.5) == 0.6
    assert canonical_score(1.0, 1.0) == 1.0
    assert canonical_score(0.0, 0.0) == 0.0


def test_canonical_score_uses_available_when_one_judge_errored():
    assert canonical_score(0.8, None) == 0.8
    assert canonical_score(None, 0.4) == 0.4


def test_canonical_score_none_when_both_judges_errored():
    assert canonical_score(None, None) is None


def test_canonical_score_still_computed_on_disagreement():
    """Disagreement is flagged separately. The canonical score is still the
    median; final_score may be overridden by human arbitration in Day 11."""
    assert canonical_score(0.9, 0.3) == 0.6


# ---------- N-judge disagreement (Day 11 3-judge methodology) ----------
#
# Methodology: canonical = median; disagreement fires when ANY judge deviates
# from median by > 0.2. This generalises the existing 2-judge rule and is
# special-cased to preserve the historical |a-b| > 0.2 semantic for N=2.

def test_three_judges_consensus_break_example_from_methodology_doc():
    """Methodology doc § 'Three-judge disagreement methodology' worked example:
    scores (0.3, 0.5, 0.7) have median 0.5; max deviation = 0.2 (Opus and
    Gemini both at exactly threshold); NOT > 0.2 → no disagreement flag."""
    assert is_disagreement(0.3, 0.5, 0.7) is False
    assert canonical_score(0.3, 0.5, 0.7) == 0.5


def test_three_judges_disagreement_outlier():
    """One judge way off the consensus → disagreement flag fires."""
    # median = 0.5; deviations 0.3, 0, 0.4 → max 0.4 > 0.2
    assert is_disagreement(0.2, 0.5, 0.9) is True
    assert canonical_score(0.2, 0.5, 0.9) == 0.5


def test_three_judges_unanimous_no_disagreement():
    """All three at the same score → no disagreement, canonical is that score."""
    assert is_disagreement(0.7, 0.7, 0.7) is False
    assert canonical_score(0.7, 0.7, 0.7) == 0.7


def test_three_judges_with_one_None_falls_back_to_two_judge_rule():
    """One judge errored → fall back to the 2-judge semantic (|a-b| > 0.2),
    not the 3-judge median-deviation rule."""
    # 2 valid scores 0.3 and 0.7 → |Δ|=0.4 > 0.2 → disagreement
    assert is_disagreement(0.3, 0.7, None) is True
    assert canonical_score(0.3, 0.7, None) == 0.5
    # 2 valid scores 0.5 and 0.7 → |Δ|=0.2, not > 0.2 → no disagreement
    assert is_disagreement(0.5, None, 0.7) is False
    assert canonical_score(0.5, None, 0.7) == 0.6


def test_three_judges_all_None_no_disagreement():
    """All judges errored → no disagreement (no data), canonical is None."""
    assert is_disagreement(None, None, None) is False
    assert canonical_score(None, None, None) is None


def test_three_judges_one_score_no_disagreement():
    """Only one judge produced a score → no disagreement (need 2 to compare),
    canonical is that single score."""
    assert is_disagreement(0.6, None, None) is False
    assert canonical_score(0.6, None, None) == 0.6


def test_three_judges_at_exact_threshold_does_not_fire():
    """Boundary: median ± exactly 0.2 → not > 0.2 → no disagreement.
    Mirrors the 2-judge boundary behavior (PRD § 7: |Δ| ≤ 0.2 agrees)."""
    # median = 0.5; deviation = 0.2 exactly → no disagreement
    assert is_disagreement(0.3, 0.5, 0.7) is False
    # nudge above threshold → disagreement fires
    assert is_disagreement(0.29, 0.5, 0.7) is True


def test_two_judge_semantic_unchanged_after_n_judge_refactor():
    """Regression: existing 2-judge call sites must produce identical results
    to pre-refactor. The refactor introduced *args API but the 2-judge
    semantic (|a-b| > 0.2) is preserved verbatim."""
    # Equivalent to pre-refactor is_disagreement(0.7, 0.5)
    assert is_disagreement(0.7, 0.5) is False  # |Δ|=0.2, not > 0.2
    assert is_disagreement(0.71, 0.5) is True  # |Δ|=0.21 > 0.2
    assert is_disagreement(1.0, 0.0) is True
    # Empty / all-None
    assert is_disagreement() is False
    assert is_disagreement(None) is False
    assert is_disagreement(None, None) is False
    # Single score
    assert is_disagreement(0.5) is False


# ---------- disagreement CSV emission ----------

def test_emit_disagreement_csv_writes_only_disagreed_rows(tmp_path):
    rows = [
        JudgePair("p1", "m1", "baseline", 0.5, 0.5, "resp1", "crit"),  # agree
        JudgePair("p2", "m2", "baseline", 0.9, 0.3, "resp2", "crit"),  # disagree
        JudgePair("p3", "m3", "baseline", 0.7, 0.5, "resp3", "crit"),  # agree (delta 0.2)
        JudgePair("p4", "m4", "baseline", 0.8, 0.4, "resp4", "crit"),  # disagree (delta 0.4)
        JudgePair("p5", "m5", "baseline", None, 0.5, "resp5", "crit"),  # judge_error, not disagreement
    ]
    out = tmp_path / "disagreements.csv"
    n = emit_disagreement_csv(rows, out)
    assert n == 2
    with out.open() as f:
        reader = csv.DictReader(f)
        written = list(reader)
    assert len(written) == 2
    assert {r["prompt_id"] for r in written} == {"p2", "p4"}
    assert all("human_score" in r and r["human_score"] == "" for r in written)


def test_emit_disagreement_csv_writes_header_even_when_empty(tmp_path):
    out = tmp_path / "disagreements.csv"
    n = emit_disagreement_csv([], out)
    assert n == 0
    assert out.exists()
    header = out.read_text().strip().split(",")
    assert "prompt_id" in header
    assert "human_score" in header


# ---------- rate-limit + transient retry (Day 10 recovery) ----------
#
# Day 10 production hit Mistral 429s at concurrency 4 because the judge module
# had no retry handling. The patch mirrors the Day 7 retry hardening pattern
# (catch RateLimit/InternalServer/APIConnection + Mistral SDKError 429/5xx,
# honour Retry-After, exponential backoff, max 4 retries). Mocks the raw
# call function and exercises the wrapper directly.

def _fake_mistral_sdk_error(status_code: int, retry_after: float | None = None):
    """Construct an SDKError instance matching the real Mistral SDK shape.

    The real SDKError takes (message, raw_response, body=None) where
    raw_response is an httpx.Response. status_code, headers, and body
    are then exposed as attributes on the error. Build a real
    httpx.Response so the test exercises the production code path.
    """
    import httpx
    from mistralai import models as mistral_models
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    req = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    resp = httpx.Response(status_code, headers=headers, content=b"{}", request=req)
    return mistral_models.SDKError("rate limited", resp, body="{}")


def test_judge_retries_on_mistral_429_then_succeeds(monkeypatch):
    """Mistral 429 → wait → retry → success. Mirrors the Day 7 retry pattern."""
    from scoring import judge

    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_mistral_sdk_error(429)
        return ("ok", 100, 50, 1234)

    text, in_tok, out_tok, lat = judge._call_with_retry(raw, judge="mistral")
    assert calls["n"] == 2
    assert text == "ok"
    assert sleeps == [judge.JUDGE_RATE_LIMIT_DEFAULT_DELAY]


def test_judge_retries_on_mistral_5xx_then_succeeds(monkeypatch):
    from scoring import judge

    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _fake_mistral_sdk_error(503)
        return ("ok", 1, 1, 1)

    text, *_ = judge._call_with_retry(raw, judge="mistral")
    assert calls["n"] == 3
    assert text == "ok"
    # First 5xx retry uses delays[0]=5, second uses delays[1]=10
    assert sleeps == [judge.JUDGE_TRANSIENT_5XX_DELAYS[0], judge.JUDGE_TRANSIENT_5XX_DELAYS[1]]


def test_judge_retry_exhaustion_raises(monkeypatch):
    """All attempts fail → re-raises the last error."""
    from scoring import judge

    monkeypatch.setattr(judge.time, "sleep", lambda s: None)

    def raw():
        raise _fake_mistral_sdk_error(429)

    with pytest.raises(Exception) as exc_info:
        judge._call_with_retry(raw, judge="mistral", max_retries=2)
    # Should have been called max_retries + 1 = 3 times total
    assert "rate limited" in str(exc_info.value).lower() or "rate" in str(exc_info.value).lower()


def test_judge_honours_retry_after_header_on_429(monkeypatch):
    """Retry-After header overrides the default backoff."""
    from scoring import judge

    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_mistral_sdk_error(429, retry_after=12.0)
        return ("ok", 1, 1, 1)

    judge._call_with_retry(raw, judge="mistral")
    assert sleeps == [12.0], f"expected Retry-After=12 to be honoured, got sleeps={sleeps}"


def test_judge_does_not_retry_on_non_retriable_4xx(monkeypatch):
    """Mistral 400 (validation) or 401 (auth) must NOT trigger retries — those
    aren't transient and re-firing wastes API credit."""
    from scoring import judge

    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        raise _fake_mistral_sdk_error(400)

    with pytest.raises(Exception):
        judge._call_with_retry(raw, judge="mistral")
    assert calls["n"] == 1, f"non-retriable 4xx must not retry; got {calls['n']} attempts"
    assert sleeps == []


# ---------- Gemini retry (Day 11 3-judge integration) ----------

def _fake_gemini_error(status_code: int):
    """Construct a google.genai.errors.ClientError or ServerError matching the
    real SDK shape — for Layer 1 retry tests without firing API calls.
    Mirrors _fake_mistral_sdk_error from Day 10."""
    import httpx
    from google.genai import errors as genai_errors
    body = {"error": {"code": status_code, "message": "fake error",
                      "status": "RESOURCE_EXHAUSTED" if status_code == 429 else "ERROR"}}
    req = httpx.Request("POST",
                        "https://generativelanguage.googleapis.com/v1beta/"
                        "models/gemini-3.1-pro-preview:generateContent")
    resp = httpx.Response(status_code, content=str(body).encode(), request=req)
    if 400 <= status_code < 500:
        return genai_errors.ClientError(status_code, body, resp)
    return genai_errors.ServerError(status_code, body, resp)


def test_judge_retries_on_gemini_429_then_succeeds(monkeypatch):
    """Gemini 429 (RESOURCE_EXHAUSTED) → wait → retry → success."""
    from scoring import judge
    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_gemini_error(429)
        return ("ok", 1, 1, 1)
    text, *_ = judge._call_with_retry(raw, judge="gemini")
    assert calls["n"] == 2
    assert text == "ok"
    assert sleeps == [judge.JUDGE_RATE_LIMIT_DEFAULT_DELAY]


def test_judge_retries_on_gemini_503_then_succeeds(monkeypatch):
    """Gemini 503 (UNAVAILABLE) → wait with exponential backoff → retry → success."""
    from scoring import judge
    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _fake_gemini_error(503)
        return ("ok", 1, 1, 1)
    text, *_ = judge._call_with_retry(raw, judge="gemini")
    assert calls["n"] == 3
    assert sleeps == [judge.JUDGE_TRANSIENT_5XX_DELAYS[0],
                      judge.JUDGE_TRANSIENT_5XX_DELAYS[1]]


def test_judge_does_not_retry_on_gemini_400(monkeypatch):
    """Gemini 400 (INVALID_ARGUMENT, e.g. malformed config) is non-retriable —
    re-firing won't fix it, just wastes API budget."""
    from scoring import judge
    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        raise _fake_gemini_error(400)
    with pytest.raises(Exception):
        judge._call_with_retry(raw, judge="gemini")
    assert calls["n"] == 1
    assert sleeps == []


def test_judge_does_not_retry_on_gemini_403(monkeypatch):
    """Gemini 403 (PERMISSION_DENIED, e.g. billing not enabled) is non-retriable.
    Specifically guards against the smoke-test failure mode observed Day 11
    pre-billing where free-tier quota was 0 — retrying would just hit the same
    wall and burn time."""
    from scoring import judge
    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        raise _fake_gemini_error(403)
    with pytest.raises(Exception):
        judge._call_with_retry(raw, judge="gemini")
    assert calls["n"] == 1
    assert sleeps == []


# ---------- OpenAI judge (GPT-5.5) retry tests ----------
#
# The retry classifier already catches openai.RateLimitError,
# openai.InternalServerError, and openai.APIConnectionError (Day 7 hardening
# for test-model OpenAI calls). The judge calls share the same SDK and
# exception types — these tests confirm the existing retry path applies
# correctly to the new judge dispatch.

def _fake_openai_rate_limit(retry_after: float | None = None):
    """Construct an openai.RateLimitError matching the real SDK shape."""
    import httpx
    body = {"error": {"message": "rate limit", "type": "rate_limit_error"}}
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, headers=headers, content=str(body).encode(), request=req)
    return openai.RateLimitError("rate limited", response=resp, body=body)


def _fake_openai_5xx(status_code: int):
    """Construct an openai.InternalServerError for transient 5xx retry tests."""
    import httpx
    body = {"error": {"message": "server error", "type": "server_error"}}
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(status_code, content=str(body).encode(), request=req)
    return openai.InternalServerError("server error", response=resp, body=body)


def test_judge_retries_on_openai_429_then_succeeds(monkeypatch):
    """OpenAI 429 → wait → retry → success. Same retry path as test-model
    OpenAI calls; judge dispatch reuses it transparently."""
    from scoring import judge
    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_openai_rate_limit()
        return ("ok", 1, 1, 1)
    text, *_ = judge._call_with_retry(raw, judge="gpt55")
    assert calls["n"] == 2
    assert text == "ok"
    assert sleeps == [judge.JUDGE_RATE_LIMIT_DEFAULT_DELAY]


def test_judge_retries_on_openai_500_then_succeeds(monkeypatch):
    """OpenAI 500 (InternalServerError) → wait with exponential backoff → retry → success."""
    from scoring import judge
    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _fake_openai_5xx(500)
        return ("ok", 1, 1, 1)
    text, *_ = judge._call_with_retry(raw, judge="gpt55")
    assert calls["n"] == 3
    assert sleeps == [judge.JUDGE_TRANSIENT_5XX_DELAYS[0],
                      judge.JUDGE_TRANSIENT_5XX_DELAYS[1]]


def test_judge_retries_on_openai_apiconnection_error(monkeypatch):
    """openai.APIConnectionError (transient network) is in the retry classifier
    and treated as a 5xx — exponential backoff."""
    from scoring import judge
    sleeps: list[float] = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        if calls["n"] == 1:
            raise openai.APIConnectionError(request=None)
        return ("ok", 1, 1, 1)
    text, *_ = judge._call_with_retry(raw, judge="gpt55")
    assert calls["n"] == 2
    assert text == "ok"


def test_judge_retry_exhaustion_raises_for_openai(monkeypatch):
    """Persistent 429s exhaust the retry budget and re-raise."""
    from scoring import judge
    monkeypatch.setattr(judge.time, "sleep", lambda s: None)
    calls = {"n": 0}
    def raw():
        calls["n"] += 1
        raise _fake_openai_rate_limit()
    with pytest.raises(openai.RateLimitError):
        judge._call_with_retry(raw, judge="gpt55", max_retries=2)
    assert calls["n"] == 3, f"expected 1 + max_retries=2 attempts; got {calls['n']}"


# ---------- partial-result handling in score_one_batch (Day 10 recovery) ----------
#
# Pre-recovery: when one judge raised, the OTHER judge's successful work was
# discarded along with the cost it incurred. Refactor: each judge runs
# independently; a failure on one side returns judge_error rows for that
# side only, while the successful side's rows are preserved.

def test_score_one_batch_returns_partial_on_one_judge_failure(monkeypatch):
    """Opus succeeds, Mistral fails → returns Opus rows with scores +
    Mistral rows with judge_error populated. The successful judge's cost
    is NOT discarded."""
    from scoring import judge as judge_mod

    prompt = make_prompt(
        prompt_id="rea-001", task_category="reasoning",
        expected={"final_answer": "£14.00"},
        judge_criteria="Reasoning must compute daily rate then multiply.",
    )
    responses = {m: f"stub response {i}" for i, m in enumerate(TEST_MODELS_4)}

    fake_opus_resp = judge_mod.JudgeResponse(
        judge="opus",
        raw_text='{"A":0.9,"B":0.8,"C":0.7,"D":0.6}',
        scores={"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6},
        reasoning={"A": "x", "B": "x", "C": "x", "D": "x"},
        input_tokens=2000, output_tokens=200, latency_ms=4500,
        parse_error=None,
    )

    def fake_call_judge(name, call, **kw):
        if name == "opus":
            return fake_opus_resp
        raise _fake_mistral_sdk_error(429)

    monkeypatch.setattr(judge_mod, "call_judge", fake_call_judge)

    call, rows = judge_mod.score_one_batch(prompt, responses, lever="baseline")
    opus_rows = [r for r in rows if r.judge == "opus"]
    mistral_rows = [r for r in rows if r.judge == "mistral"]

    assert len(opus_rows) == 4, "Opus side must return 4 successful row scores"
    assert all(r.score is not None for r in opus_rows), "Opus scores must be present"
    assert all(r.judge_error is None for r in opus_rows), "Opus rows must not be flagged judge_error"

    assert len(mistral_rows) == 4, "Mistral side must return 4 error-marked rows"
    assert all(r.score is None for r in mistral_rows), "Mistral scores must be None on failure"
    assert all(r.judge_error is not None for r in mistral_rows), "Mistral rows must carry judge_error"
    assert all("SDKError" in r.judge_error or "rate" in r.judge_error.lower()
               for r in mistral_rows)

    # Per-row Opus cost must be > 0 (the work was paid for and is not discarded)
    assert sum(r.cost_usd for r in opus_rows) > 0


def test_score_one_batch_populates_reasoning_when_judge_returns_it(monkeypatch):
    """Day 11 invariant: JudgeRowScore.reasoning carries the judge's per-response
    one-sentence reasoning. The Day 10 day_10.py persistence step relies on
    this — it writes JudgeRowScore.reasoning into judge_a_reasoning / judge_b_reasoning.
    Lock in the contract so a future judge.py refactor can't silently drop reasoning
    without breaking this test."""
    from scoring import judge as judge_mod

    prompt = make_prompt(
        prompt_id="rea-001", task_category="reasoning",
        expected={"final_answer": "£14.00"},
        judge_criteria="Reasoning must compute daily rate then multiply.",
    )
    responses = {m: f"stub {i}" for i, m in enumerate(TEST_MODELS_4)}

    fake_resp = judge_mod.JudgeResponse(
        judge="opus",
        raw_text="{}",
        scores={"A": 0.9, "B": 0.5, "C": 0.7, "D": 0.2},
        reasoning={"A": "fully correct", "B": "missed substep b",
                   "C": "ok with caveat", "D": "wrong final"},
        input_tokens=100, output_tokens=80, latency_ms=2000,
        parse_error=None,
    )
    monkeypatch.setattr(judge_mod, "call_judge", lambda *a, **kw: fake_resp)

    call, rows = judge_mod.score_one_batch(prompt, responses, lever="baseline",
                                           judge_names=("opus",))
    rows_by_label = {r.position_label: r for r in rows}
    for label, expected in [("A", "fully correct"), ("B", "missed substep b"),
                            ("C", "ok with caveat"), ("D", "wrong final")]:
        assert rows_by_label[label].reasoning == expected


def test_score_one_batch_reasoning_is_none_when_judge_omits_it(monkeypatch):
    """If the judge response has no reasoning block, JudgeRowScore.reasoning must
    be None (not an empty string, not KeyError) — persistence layer relies on
    None to skip writing the column under COALESCE semantics."""
    from scoring import judge as judge_mod

    prompt = make_prompt(
        prompt_id="cs-001", task_category="customer_support",
        expected={"category": "billing"},
        judge_criteria="x",
    )
    responses = {m: f"stub {i}" for i, m in enumerate(TEST_MODELS_4)}

    fake_resp = judge_mod.JudgeResponse(
        judge="mistral",
        raw_text='{"A":0.5,"B":0.5,"C":0.5,"D":0.5}',
        scores={"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5},
        reasoning={},  # judge returned no reasoning block
        input_tokens=100, output_tokens=20, latency_ms=1000,
        parse_error=None,
    )
    monkeypatch.setattr(judge_mod, "call_judge", lambda *a, **kw: fake_resp)

    _, rows = judge_mod.score_one_batch(prompt, responses, lever="baseline",
                                        judge_names=("mistral",))
    for r in rows:
        assert r.reasoning is None


def test_score_one_batch_with_judge_subset_skips_unrequested_judges(monkeypatch):
    """When called with judge_names=('opus',) only, Mistral is never invoked
    and only 4 Opus rows are returned. This is the path --missing-only takes
    when one judge side is already filled in the DB."""
    from scoring import judge as judge_mod

    prompt = make_prompt(
        prompt_id="cs-001", task_category="customer_support",
        expected={"category": "billing"},
        judge_criteria="Reply acknowledges issue.",
    )
    responses = {m: f"stub {i}" for i, m in enumerate(TEST_MODELS_4)}

    fake_opus = judge_mod.JudgeResponse(
        judge="opus", raw_text="{}",
        scores={"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5},
        reasoning={"A": "x", "B": "x", "C": "x", "D": "x"},
        input_tokens=100, output_tokens=50, latency_ms=1000,
        parse_error=None,
    )

    invocations: list[str] = []
    def fake_call_judge(name, call, **kw):
        invocations.append(name)
        return fake_opus

    monkeypatch.setattr(judge_mod, "call_judge", fake_call_judge)

    call, rows = judge_mod.score_one_batch(
        prompt, responses, lever="baseline", judge_names=("opus",),
    )
    assert invocations == ["opus"], "Mistral must not be called when not in judge_names"
    assert len(rows) == 4
    assert {r.judge for r in rows} == {"opus"}
