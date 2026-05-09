"""Layer 1 unit tests for Tier-1 deterministic scoring (Day 9).

One test per check defined in scoring/tier_1.py, plus the two refinement-
driven cases the user called out at sign-off:

  - output_cap edge: response at exactly the cap that parses → pass; response
    at the cap that fails to parse → truncated.
  - unwrap rule: must not accept malformed-but-wrapped responses such as
    `{"result": "not the expected shape"}`.

Production response fixtures (the full strings observed in run-d1a9c980 on
2026-05-09) are inlined where they exercise normalisation steps that don't
fire on synthetic minimal cases.
"""

from __future__ import annotations

import json

import pytest

from scoring.tier_1 import (
    OUTPUT_CAP_TOKENS,
    ScoredRow,
    normalise_response,
    score_row,
)
from tests.conftest import make_prompt


# ---------- normalisation pipeline ----------

def test_normalise_strips_markdown_fence_observed_on_haiku():
    text = '```json\n{\n  "name": "Customer A",\n  "email": "a@example.com",\n  "amount": 249.00\n}\n```'
    out, steps = normalise_response(text, expects_json=True)
    assert out == '{\n  "name": "Customer A",\n  "email": "a@example.com",\n  "amount": 249.00\n}'
    assert "fence_strip" in steps
    assert "preamble_strip" not in steps


def test_normalise_strips_pre_json_preamble_observed_on_sonnet_rea_001():
    text = (
        "I need to find the unused portion of April after cancellation on 16 April.\n"
        "**Step 1: Determine used days**\n"
        "- Used days: 1 April through 16 April = 16 days\n\n"
        '{"reasoning": "...", "final_answer": "£14.00"}'
    )
    out, steps = normalise_response(text, expects_json=True)
    assert out.startswith("{") and out.endswith("}")
    assert json.loads(out)["final_answer"] == "£14.00"
    assert "preamble_strip" in steps


def test_normalise_strips_smart_quotes():
    text = '{“name”: “Customer A”}'
    out, steps = normalise_response(text, expects_json=True)
    assert json.loads(out) == {"name": "Customer A"}
    assert "smart_quote_normalise" in steps


def test_normalise_summarisation_skips_json_steps():
    text = "```json\n• Bullet one\n• Bullet two\n```"
    out, steps = normalise_response(text, expects_json=False)
    assert out == text
    assert steps == []


def test_normalise_records_only_steps_that_actually_modified_input():
    text = '{"category": "billing"}'
    out, steps = normalise_response(text, expects_json=True)
    assert out == text
    assert steps == []


# ---------- extraction ----------

def test_extraction_pass_with_exact_match():
    prompt = make_prompt(
        prompt_id="ext-001", task_category="extraction",
        expected={"name": "Customer A", "email": "a@example.com", "amount": 249.0},
    )
    row = _row(response='{"name": "Customer A", "email": "a@example.com", "amount": 249.00}')
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "pass"
    assert sr.rubric_score == 1.0


def test_extraction_float_tolerance_accepts_trailing_zero():
    prompt = make_prompt(prompt_id="ext-001", task_category="extraction",
                         expected={"amount": 249.0})
    sr = score_row(_row(response='{"amount": 249}'), prompt)
    assert sr.tier_1_status == "pass"


def test_extraction_bool_distinct_from_int():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"marketing_opt_in": True})
    sr = score_row(_row(response='{"marketing_opt_in": 1}'), prompt)
    assert sr.tier_1_status == "fail_schema"


def test_extraction_int_field_rejects_bool():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"attendee_count": 6})
    sr = score_row(_row(response='{"attendee_count": true}'), prompt)
    assert sr.tier_1_status == "fail_schema"


def test_extraction_list_order_insensitive():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"attendees": ["Alice", "Bob", "Carol"]})
    sr = score_row(_row(response='{"attendees": ["Carol", "Alice", "Bob"]}'), prompt)
    assert sr.tier_1_status == "pass"


def test_extraction_string_case_insensitive_whitespace_collapse():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"name": "Customer A"})
    sr = score_row(_row(response='{"name": "  customer  a  "}'), prompt)
    assert sr.tier_1_status == "pass"


def test_extraction_missing_key_is_fail_schema():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"name": "X", "email": "x@example.com"})
    sr = score_row(_row(response='{"name": "X"}'), prompt)
    assert sr.tier_1_status == "fail_schema"


def test_extraction_value_mismatch_is_fail_content():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"name": "Customer A"})
    sr = score_row(_row(response='{"name": "Customer B"}'), prompt)
    assert sr.tier_1_status == "fail_content"


# ---------- customer_support ----------

def test_cs_pass_on_exact_category():
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    sr = score_row(_row(response='{"category": "billing", "reply": "thank you..."}'), prompt)
    assert sr.tier_1_status == "pass"


def test_cs_extra_reply_field_is_tolerated():
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "technical"})
    response = '```json\n{"category": "technical", "reply": "We will look into it."}\n```'
    sr = score_row(_row(response=response), prompt)
    assert sr.tier_1_status == "pass"


def test_cs_off_enum_category_fails():
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    sr = score_row(_row(response='{"category": "payments"}'), prompt)
    assert sr.tier_1_status == "fail_content"


def test_cs_wrong_category_fails():
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    sr = score_row(_row(response='{"category": "complaint"}'), prompt)
    assert sr.tier_1_status == "fail_content"


# ---------- rag_qa ----------

def test_rag_short_answer_substring_in_sentence_passes():
    prompt = make_prompt(prompt_id="rag-001", task_category="rag_qa",
                         expected={"answer": "2014", "supporting_sentences": [4]})
    sr = score_row(_row(response='{"answer": "The company was founded in 2014.", "supporting_sentences": [4]}'), prompt)
    assert sr.tier_1_status == "pass"


def test_rag_short_answer_truncated_does_not_pass_bidirectionally():
    prompt = make_prompt(prompt_id="rag-x", task_category="rag_qa",
                         expected={"answer": "Sara Patel", "supporting_sentences": [6]})
    sr = score_row(_row(response='{"answer": "Sara", "supporting_sentences": [6]}'), prompt)
    assert sr.tier_1_status == "fail_content"


def test_rag_phrasal_answer_skips_content_check():
    prompt = make_prompt(
        prompt_id="rag-018", task_category="rag_qa",
        expected={
            "answer": "A misconfigured connection pool size (50 instead of 500) on a database proxy, which had been silently degrading for six hours before saturation",
            "supporting_sentences": [11, 12],
        },
    )
    sr = score_row(_row(response='{"answer": "The pool was misconfigured at 50 connections, which silently degraded for hours.", "supporting_sentences": [12, 11]}'), prompt)
    assert sr.tier_1_status == "pass"
    assert sr.detail["bucket"] == "phrasal"
    assert sr.detail["answer_check"] == "not_applicable"


def test_rag_phrasal_wrong_citations_fails():
    prompt = make_prompt(
        prompt_id="rag-018", task_category="rag_qa",
        expected={"answer": "A misconfigured connection pool size (50 instead of 500)",
                  "supporting_sentences": [11, 12]},
    )
    sr = score_row(_row(response='{"answer": "X", "supporting_sentences": [3, 4]}'), prompt)
    assert sr.tier_1_status == "fail_content"
    assert sr.detail["citations_check"] == "fail"


def test_rag_short_correct_answer_wrong_citations_fails():
    prompt = make_prompt(prompt_id="rag-001", task_category="rag_qa",
                         expected={"answer": "2014", "supporting_sentences": [4]})
    sr = score_row(_row(response='{"answer": "2014", "supporting_sentences": [3]}'), prompt)
    assert sr.tier_1_status == "fail_content"
    assert sr.detail["answer_check"] == "pass"
    assert sr.detail["citations_check"] == "fail"


def test_rag_supporting_sentences_must_be_ints():
    """Wrong element type in supporting_sentences is a schema fault, not content."""
    prompt = make_prompt(prompt_id="rag-001", task_category="rag_qa",
                         expected={"answer": "2014", "supporting_sentences": [4]})
    sr = score_row(_row(response='{"answer": "2014", "supporting_sentences": ["4"]}'), prompt)
    assert sr.tier_1_status == "fail_schema"


# ---------- reasoning ----------

def test_reasoning_currency_equivalent_forms_match():
    prompt = make_prompt(prompt_id="rea-001", task_category="reasoning",
                         expected={"final_answer": "£14.00"})
    for form in ['"£14.00"', '"£14"', '"£14.0"']:
        sr = score_row(_row(response=f'{{"reasoning": "...", "final_answer": {form}}}'), prompt)
        assert sr.tier_1_status == "pass", f"failed for {form}"


def test_reasoning_currency_wrong_amount_fails():
    prompt = make_prompt(prompt_id="rea-001", task_category="reasoning",
                         expected={"final_answer": "£14.00"})
    sr = score_row(_row(response='{"reasoning": "...", "final_answer": "£15.00"}'), prompt)
    assert sr.tier_1_status == "fail_content"


def test_reasoning_currency_wrong_symbol_fails():
    prompt = make_prompt(prompt_id="rea-001", task_category="reasoning",
                         expected={"final_answer": "£14.00"})
    sr = score_row(_row(response='{"reasoning": "...", "final_answer": "$14.00"}'), prompt)
    assert sr.tier_1_status == "fail_content"


def test_reasoning_time_24h_equivalent_to_12h():
    prompt = make_prompt(prompt_id="rea-003", task_category="reasoning",
                         expected={"final_answer": "14:00"})
    for form in ['"14:00"', '"14:00:00"', '"2:00 PM"', '"2:00 pm"']:
        sr = score_row(_row(response=f'{{"reasoning": "x", "final_answer": {form}}}'), prompt)
        assert sr.tier_1_status == "pass", f"failed for {form}"


def test_reasoning_enum_word_case_insensitive():
    prompt = make_prompt(prompt_id="rea-002", task_category="reasoning",
                         expected={"final_answer": "Eligible"})
    sr = score_row(_row(response='{"reasoning": "x", "final_answer": "eligible"}'), prompt)
    assert sr.tier_1_status == "pass"


def test_reasoning_compound_answer_both_parts_required():
    prompt = make_prompt(prompt_id="rea-019", task_category="reasoning",
                         expected={"final_answer": "Greendot, £12,960.00"})
    sr = score_row(_row(response='{"reasoning": "x", "final_answer": "Greendot, £12,960.00"}'), prompt)
    assert sr.tier_1_status == "pass"
    sr = score_row(_row(response='{"reasoning": "x", "final_answer": "Greendot, £13,000.00"}'), prompt)
    assert sr.tier_1_status == "fail_content"


# ---------- summarisation ----------

def test_summarisation_is_not_applicable():
    prompt = make_prompt(prompt_id="sum-001", task_category="summarisation",
                         judge_criteria="must cover X, Y, Z")
    sr = score_row(_row(response="• one\n• two\n• three"), prompt)
    assert sr.tier_1_status == "not_applicable"
    assert sr.rubric_score is None


# ---------- lever-aware: output_cap edge case (refinement 1) ----------

def test_output_cap_at_exact_cap_that_parses_is_genuine_pass():
    """Refinement 1: output_tokens == 200 AND parses cleanly → pass, NOT truncated."""
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    row = _row(
        response='{"category": "billing", "reply": "Thanks."}',
        lever="output_cap", output_tokens=OUTPUT_CAP_TOKENS,
    )
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "pass"
    assert sr.truncated_due_to_cap == 0


def test_output_cap_at_exact_cap_that_fails_to_parse_is_truncated():
    """Refinement 1: output_tokens == 200 AND format-fails → truncated."""
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    row = _row(
        response='{"category": "billing", "reply": "Thanks for reaching out, we are looking into',
        lever="output_cap", output_tokens=OUTPUT_CAP_TOKENS,
    )
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "truncated"
    assert sr.truncated_due_to_cap == 1
    assert sr.output_format_valid == 0


def test_output_cap_below_cap_failing_to_parse_is_genuine_format_fail():
    """A short malformed response under the cap is not 'truncated' — it's a real failure."""
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    row = _row(
        response='{"category": "billing"',
        lever="output_cap", output_tokens=15,
    )
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "fail_format"
    assert sr.truncated_due_to_cap == 0


# ---------- lever-aware: compression ----------

def test_compression_unavailable_short_circuits_to_status():
    prompt = make_prompt(prompt_id="ext-008", task_category="extraction",
                         expected={"name": "X"})
    cfg = json.dumps({
        "compression_status": "unavailable",
        "skip_reason": "no Anthropic-counted reduction at rate=0.5",
    })
    row = _row(response="(unused)", lever="compression", optimisation_config=cfg)
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "compression_unavailable"
    assert sr.rubric_score is None


def test_compression_engaged_runs_full_check():
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    cfg = json.dumps({"original_input_tokens": 200, "compressed_input_tokens": 147})
    row = _row(response='{"category": "billing"}', lever="compression",
               optimisation_config=cfg)
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "pass"


# ---------- lever-aware: error rows ----------

def test_error_row_is_marked_error_regardless_of_response_text():
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    row = _row(response="", lever="baseline", error="anthropic.RateLimitError after 3 retries")
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "error"


# ---------- normalisation: unwrap rule (refinement 2) ----------

def test_unwrap_accepts_legitimately_wrapped_response():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"name": "Customer A"})
    row = _row(response='{"result": {"name": "Customer A"}}')
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "pass"
    assert "unwrap_result" in sr.normalisation_steps_applied


def test_unwrap_does_not_accept_malformed_but_wrapped_response():
    """Refinement 2: `{"result": "not the expected shape"}` must still fail."""
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"name": "Customer A"})
    row = _row(response='{"result": "not the expected shape"}')
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "fail_schema"
    assert "unwrap_result" not in sr.normalisation_steps_applied


def test_unwrap_does_not_accept_wrapped_dict_missing_expected_keys():
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"name": "X", "email": "x@example.com"})
    row = _row(response='{"result": {"unrelated": "field"}}')
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "fail_schema"
    assert "unwrap_result" not in sr.normalisation_steps_applied


def test_unwrap_capped_at_one_level():
    """{"result": {"result": {...}}} must NOT recurse to find the inner shape.

    The unwrap rule looks one level deep and asks 'does this inner dict have
    any expected keys?'. A doubly-wrapped value reveals only another wrapper
    key, so the rule MUST decline to unwrap (and the row fails schema).
    Recursive unwrapping would silently accept arbitrarily-nested model
    output, which is exactly the bias risk refinement 2 was about.
    """
    prompt = make_prompt(prompt_id="ext-x", task_category="extraction",
                         expected={"name": "Customer A"})
    row = _row(response='{"result": {"result": {"name": "Customer A"}}}')
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "fail_schema"
    assert "unwrap_result" not in sr.normalisation_steps_applied


# ---------- audit trail (refinement 2 — visibility) ----------

def test_audit_trail_lists_only_steps_that_actually_fired():
    prompt = make_prompt(prompt_id="cs-x", task_category="customer_support",
                         expected={"category": "billing"})
    row = _row(response='```json\n{"category": "billing"}\n```')
    sr = score_row(row, prompt)
    assert sr.normalisation_steps_applied == ["fence_strip"]


def test_audit_trail_records_combined_steps_in_order():
    prompt = make_prompt(prompt_id="rea-001", task_category="reasoning",
                         expected={"final_answer": "£14.00"})
    response = (
        '   ```json\n'
        'Some preamble before the JSON\n'
        '{"reasoning": "x", "final_answer": "£14.00"}\n'
        '```'
    )
    row = _row(response=response)
    sr = score_row(row, prompt)
    assert sr.tier_1_status == "pass"
    assert "whitespace_strip" in sr.normalisation_steps_applied
    assert "fence_strip" in sr.normalisation_steps_applied
    assert "preamble_strip" in sr.normalisation_steps_applied
    assert sr.normalisation_steps_applied.index("fence_strip") < sr.normalisation_steps_applied.index("preamble_strip")


# ---------- helper ----------

def _row(*, response: str, lever: str = "baseline", output_tokens: int = 50,
         optimisation_config: str | None = None, error: str | None = None) -> dict:
    return {
        "response_text": response,
        "optimisation_lever": lever,
        "optimisation_config": optimisation_config,
        "output_tokens": output_tokens,
        "error": error,
    }
