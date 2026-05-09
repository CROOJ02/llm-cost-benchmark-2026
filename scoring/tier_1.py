"""Tier-1 deterministic scoring.

Per docs/PRD.md §7 and the Day 9 design doc:

  - Criteria are derived from each prompt's `tier_1_deterministic.expected`
    block — never from inspecting the model's response.
  - A single normalisation pipeline is applied to every response regardless
    of provider/model. Steps that fired are recorded in
    `normalisation_steps_applied` for audit (so any provider-style asymmetry
    is visible in Day 12 analysis rather than hidden).
  - Lever-aware pre-checks distinguish design-induced failure (output_cap
    truncation, compression_unavailable rows) from genuine model failure.

Public surface:
  - normalise_response(text, expects_json) -> (normalised, steps)
  - score_row(row_dict, prompt) -> ScoredRow
  - OUTPUT_CAP_TOKENS, RAG_SHORT_WORDS  (tunables, exposed for tests)

`row_dict` keys read by score_row:
  optimisation_lever, optimisation_config, output_tokens, response_text, error

The scorer never raises on row content — every code path produces a
ScoredRow with a well-defined tier_1_status.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from runners.schema import Prompt

OUTPUT_CAP_TOKENS = 200
RAG_SHORT_WORDS = 3
RAG_PHRASAL_WORDS = 7
FLOAT_TOLERANCE = 0.005
CS_CATEGORIES = {"billing", "technical", "feature_request", "complaint", "other"}
UNWRAP_KEYS = {"result", "output", "response", "data"}

_FENCE_RE = re.compile(r"^```(?:json|JSON|json5)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm|AM|PM)?\s*$")
_CURRENCY_RE = re.compile(r"^\s*([£$€])?\s*(-?\d[\d,]*(?:\.\d+)?)\s*([A-Za-z]{3})?\s*$")
_SMART_QUOTES = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'",
})


@dataclass
class ScoredRow:
    tier_1_status: str
    output_format_valid: int
    truncated_due_to_cap: int = 0
    response_parsed: str | None = None
    normalisation_steps_applied: list[str] = field(default_factory=list)
    rubric_score: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "tier_1_status": self.tier_1_status,
            "output_format_valid": self.output_format_valid,
            "truncated_due_to_cap": self.truncated_due_to_cap,
            "response_parsed": self.response_parsed,
            "normalisation_steps_applied": json.dumps(self.normalisation_steps_applied),
            "rubric_score": self.rubric_score,
        }


def normalise_response(text: str, *, expects_json: bool) -> tuple[str, list[str]]:
    """Apply the universal normalisation pipeline.

    Returns the normalised string plus the ordered list of steps that
    actually modified the input. Steps that no-op are not recorded.

    For expects_json=False (summarisation), only step 1 (whitespace strip)
    runs. The remaining JSON-oriented steps would over-mutate prose.
    """
    steps: list[str] = []
    s = text

    stripped = s.strip()
    if stripped != s:
        steps.append("whitespace_strip")
    s = stripped

    if not expects_json:
        return s, steps

    fence_m = _FENCE_RE.match(s)
    if fence_m:
        s = fence_m.group(1).strip()
        steps.append("fence_strip")

    sliced = _slice_to_outer_json(s)
    if sliced is not None and sliced != s:
        s = sliced
        steps.append("preamble_strip")

    # Smart-quote normalisation is conditional: only apply (and only record)
    # if the string contains smart quotes AND the initial parse fails. This
    # avoids inflating the audit count with cases where curly quotes appear
    # inside string content (which json.loads handles natively). When the
    # step fires and recovers a parse, that's a real provider-style quirk
    # worth surfacing in Day 12 analysis.
    if any(c in s for c in "“”‘’"):
        try:
            json.loads(s)
        except json.JSONDecodeError:
            translated = s.translate(_SMART_QUOTES)
            try:
                json.loads(translated)
                s = translated
                steps.append("smart_quote_normalise")
            except json.JSONDecodeError:
                pass

    return s, steps


def _slice_to_outer_json(s: str) -> str | None:
    """Locate the outermost {...} or [...] in `s` and return that slice.

    Tolerates leading prose ("Step 1: ...\\n{...}") and trailing prose. If
    the string contains neither a `{` nor a `[`, returns None.
    """
    obj_start = s.find("{")
    arr_start = s.find("[")
    candidates: list[tuple[int, str, str]] = []
    if obj_start != -1:
        candidates.append((obj_start, "{", "}"))
    if arr_start != -1:
        candidates.append((arr_start, "[", "]"))
    if not candidates:
        return None
    candidates.sort()
    start, _, close = candidates[0]
    end = s.rfind(close)
    if end <= start:
        return None
    return s[start : end + 1]


def _try_unwrap(parsed: Any, expected: dict[str, Any]) -> tuple[Any, bool]:
    """If parsed is a single-key dict whose key is in UNWRAP_KEYS and whose
    value matches the expected schema shape, unwrap one level. Capped at
    one unwrap. Returns (maybe-unwrapped, did_unwrap)."""
    if not isinstance(parsed, dict) or len(parsed) != 1:
        return parsed, False
    only_key = next(iter(parsed))
    if only_key not in UNWRAP_KEYS:
        return parsed, False
    inner = parsed[only_key]
    if not isinstance(inner, dict):
        return parsed, False
    if not any(k in inner for k in expected):
        return parsed, False
    return inner, True


def _norm_str(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def _values_equal(model_v: Any, expected_v: Any) -> bool:
    """Per-type comparison rule from the Day 9 design."""
    if expected_v is None:
        return model_v is None
    if isinstance(expected_v, bool):
        return isinstance(model_v, bool) and model_v == expected_v
    if isinstance(expected_v, int) and not isinstance(expected_v, bool):
        if isinstance(model_v, bool):
            return False
        if isinstance(model_v, int):
            return model_v == expected_v
        if isinstance(model_v, float):
            return abs(model_v - expected_v) < FLOAT_TOLERANCE and float(model_v).is_integer()
        return False
    if isinstance(expected_v, float):
        if isinstance(model_v, (int, float)) and not isinstance(model_v, bool):
            return abs(float(model_v) - expected_v) < FLOAT_TOLERANCE
        return False
    if isinstance(expected_v, str):
        if not isinstance(model_v, str):
            return False
        return _norm_str(model_v) == _norm_str(expected_v)
    if isinstance(expected_v, list):
        if not isinstance(model_v, list):
            return False
        if len(model_v) != len(expected_v):
            return False
        used = [False] * len(model_v)
        for ev in expected_v:
            for i, mv in enumerate(model_v):
                if not used[i] and _values_equal(mv, ev):
                    used[i] = True
                    break
            else:
                return False
        return True
    return model_v == expected_v


def _types_match(model_v: Any, expected_v: Any) -> bool:
    """Schema check: model value's type matches expected value's type.

    bool is treated distinct from int (Python's `bool is int` quirk would
    otherwise let `True` masquerade as `1` in extraction integer fields).
    """
    if expected_v is None:
        return model_v is None or isinstance(model_v, str) is False
    if isinstance(expected_v, bool):
        return isinstance(model_v, bool)
    if isinstance(expected_v, int) and not isinstance(expected_v, bool):
        return isinstance(model_v, int) and not isinstance(model_v, bool)
    if isinstance(expected_v, float):
        return isinstance(model_v, (int, float)) and not isinstance(model_v, bool)
    if isinstance(expected_v, str):
        return isinstance(model_v, str)
    if isinstance(expected_v, list):
        return isinstance(model_v, list)
    if isinstance(expected_v, dict):
        return isinstance(model_v, dict)
    return type(model_v) is type(expected_v)


def _parse_currency(s: str) -> tuple[float, str] | None:
    s2 = s.replace(",", "")
    m = _CURRENCY_RE.match(s2)
    if not m:
        return None
    sym, num, code = m.group(1), m.group(2), m.group(3)
    if sym is None and code is None:
        return None
    currency = sym or {"GBP": "£", "USD": "$", "EUR": "€"}.get((code or "").upper(), code)
    try:
        return float(num), currency
    except ValueError:
        return None


def _parse_time_24h(s: str) -> str | None:
    m = _TIME_RE.match(s)
    if not m:
        return None
    hh, mm, mer = int(m.group(1)), m.group(2), m.group(3)
    if mer:
        mer = mer.lower()
        if not (1 <= hh <= 12):
            return None
        if mer == "am" and hh == 12:
            hh = 0
        elif mer == "pm" and hh != 12:
            hh += 12
    if not (0 <= hh <= 23 and 0 <= int(mm) <= 59):
        return None
    return f"{hh:02d}:{mm}"


def _reasoning_answer_equal(model: str, expected: str) -> bool:
    """Match reasoning final_answer values per the Day 9 shape rules:
    currency / time / enum-word / compound. Falls back to normalised string
    equality.
    """
    e = expected.strip()
    m = model.strip()

    if "," in e:
        e_parts = [p.strip() for p in e.split(",")]
        m_parts = [p.strip() for p in m.split(",")]
        if len(e_parts) != len(m_parts):
            return False
        return all(_reasoning_answer_equal(mp, ep) for mp, ep in zip(m_parts, e_parts))

    e_cur = _parse_currency(e)
    if e_cur is not None:
        m_cur = _parse_currency(m)
        if m_cur is None:
            return False
        return abs(m_cur[0] - e_cur[0]) < FLOAT_TOLERANCE and m_cur[1] == e_cur[1]

    e_time = _parse_time_24h(e)
    if e_time is not None:
        m_time = _parse_time_24h(m)
        return m_time == e_time

    return _norm_str(m).rstrip(".!?") == _norm_str(e).rstrip(".!?")


def _check_extraction(parsed: Any, expected: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(parsed, dict):
        return "fail_schema", {"reason": "not an object"}
    missing = [k for k in expected if k not in parsed]
    if missing:
        return "fail_schema", {"reason": "missing keys", "missing": missing}
    bad_type = [k for k in expected if not _types_match(parsed[k], expected[k])]
    if bad_type:
        return "fail_schema", {"reason": "type mismatch", "fields": bad_type}
    bad_value = [k for k in expected if not _values_equal(parsed[k], expected[k])]
    if bad_value:
        return "fail_content", {"reason": "value mismatch", "fields": bad_value}
    return "pass", {}


def _check_customer_support(parsed: Any, expected: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(parsed, dict):
        return "fail_schema", {"reason": "not an object"}
    if "category" not in parsed:
        return "fail_schema", {"reason": "missing category"}
    if not isinstance(parsed["category"], str):
        return "fail_schema", {"reason": "category not a string"}
    cat = parsed["category"].strip().lower()
    if cat not in CS_CATEGORIES:
        return "fail_content", {"reason": "category not in enum", "got": cat}
    if cat != _norm_str(expected["category"]):
        return "fail_content", {"reason": "category mismatch", "got": cat, "expected": expected["category"]}
    return "pass", {}


def _check_rag_qa(parsed: Any, expected: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(parsed, dict):
        return "fail_schema", {"reason": "not an object"}
    for k, t in [("answer", str), ("supporting_sentences", list)]:
        if k not in parsed:
            return "fail_schema", {"reason": f"missing {k}"}
        if not isinstance(parsed[k], t):
            return "fail_schema", {"reason": f"{k} wrong type"}
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in parsed["supporting_sentences"]):
        return "fail_schema", {"reason": "supporting_sentences must be ints"}

    expected_ans = expected["answer"]
    expected_cites = set(expected["supporting_sentences"])
    model_cites = set(parsed["supporting_sentences"])
    cites_ok = model_cites == expected_cites

    n_words = len(expected_ans.split())
    if n_words <= RAG_SHORT_WORDS:
        ans_ok = _norm_str(expected_ans) in _norm_str(parsed["answer"])
        detail = {
            "bucket": "short",
            "answer_check": "pass" if ans_ok else "fail",
            "citations_check": "pass" if cites_ok else "fail",
            "expected_cites": sorted(expected_cites),
            "model_cites": sorted(model_cites),
        }
        return ("pass" if (cites_ok and ans_ok) else "fail_content"), detail
    elif n_words >= RAG_PHRASAL_WORDS:
        detail = {
            "bucket": "phrasal",
            "answer_check": "not_applicable",
            "citations_check": "pass" if cites_ok else "fail",
            "expected_cites": sorted(expected_cites),
            "model_cites": sorted(model_cites),
        }
        return ("pass" if cites_ok else "fail_content"), detail
    else:
        raise AssertionError(
            f"rag_qa expected.answer landed in the 4-6 word gap "
            f"(words={n_words}); update RAG_SHORT_WORDS / RAG_PHRASAL_WORDS "
            f"explicitly. answer={expected_ans!r}"
        )


def _check_reasoning(parsed: Any, expected: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(parsed, dict):
        return "fail_schema", {"reason": "not an object"}
    if "final_answer" not in parsed:
        return "fail_schema", {"reason": "missing final_answer"}
    if not isinstance(parsed["final_answer"], str):
        return "fail_schema", {"reason": "final_answer not a string"}
    if _reasoning_answer_equal(parsed["final_answer"], expected["final_answer"]):
        return "pass", {"got": parsed["final_answer"], "expected": expected["final_answer"]}
    return "fail_content", {"reason": "final_answer mismatch", "got": parsed["final_answer"], "expected": expected["final_answer"]}


_CHECKERS = {
    "extraction": _check_extraction,
    "customer_support": _check_customer_support,
    "rag_qa": _check_rag_qa,
    "reasoning": _check_reasoning,
}


def _is_compression_unavailable(optimisation_config: str | None) -> bool:
    if not optimisation_config:
        return False
    try:
        cfg = json.loads(optimisation_config)
    except (json.JSONDecodeError, TypeError):
        return False
    return cfg.get("compression_status") == "unavailable"


def score_row(row: dict[str, Any], prompt: Prompt) -> ScoredRow:
    """Score a single result row against its prompt.

    Lever-aware pre-checks fire first; then format → schema → content. The
    scorer never raises on row content (errors in the row itself, malformed
    JSON, type mismatches all produce a well-defined tier_1_status).
    """
    if row.get("error"):
        return ScoredRow(tier_1_status="error", output_format_valid=0)

    lever = row["optimisation_lever"]
    if lever == "compression" and _is_compression_unavailable(row.get("optimisation_config")):
        return ScoredRow(tier_1_status="compression_unavailable", output_format_valid=0)

    if prompt.task_category == "summarisation":
        normalised, steps = normalise_response(row["response_text"], expects_json=False)
        return ScoredRow(
            tier_1_status="not_applicable",
            output_format_valid=1,
            response_parsed=None,
            normalisation_steps_applied=steps,
            detail={"reason": "summarisation has no tier_1_deterministic block"},
        )

    normalised, steps = normalise_response(row["response_text"], expects_json=True)

    try:
        parsed = json.loads(normalised)
        parse_error = None
    except json.JSONDecodeError as e:
        parsed = None
        parse_error = str(e)

    if parsed is None:
        out_tokens = row.get("output_tokens") or 0
        if lever == "output_cap" and out_tokens >= OUTPUT_CAP_TOKENS:
            return ScoredRow(
                tier_1_status="truncated",
                output_format_valid=0,
                truncated_due_to_cap=1,
                response_parsed=None,
                normalisation_steps_applied=steps,
                detail={"parse_error": parse_error, "output_tokens": out_tokens},
            )
        return ScoredRow(
            tier_1_status="fail_format",
            output_format_valid=0,
            response_parsed=None,
            normalisation_steps_applied=steps,
            detail={"parse_error": parse_error, "output_tokens": out_tokens},
        )

    expected = prompt.scoring.tier_1_deterministic.expected
    parsed_after_unwrap, did_unwrap = _try_unwrap(parsed, expected)
    if did_unwrap:
        steps.append("unwrap_result")

    checker = _CHECKERS[prompt.task_category]
    status, detail = checker(parsed_after_unwrap, expected)

    return ScoredRow(
        tier_1_status=status,
        output_format_valid=1,
        response_parsed=json.dumps(parsed_after_unwrap),
        normalisation_steps_applied=steps,
        rubric_score=(1.0 if status == "pass" else 0.0),
        detail=detail,
    )
