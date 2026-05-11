"""Tier-2 dual-judge blind evaluation.

Day 10 of the benchmark. See docs/methodology/prompt_design_decisions.md
§ "Tier-2 dual-judge scoring (Day 10)".

Design (locked at Day 10 sign-off):
  - Two judges: Claude Opus 4.6 (Anthropic) + mistral-large-2512 (Mistral)
  - Per-(prompt, lever) batched call: 4 model responses anonymised A/B/C/D
    in deterministic-seed-randomised order; lever is structurally hidden
    because all 4 responses in a single call come from the same lever
  - Reference answer shown for RAG (expected.answer) and reasoning
    (expected.final_answer); not shown for cs/sum (no canonical reference)
  - 0.0–1.0 continuous scale with PRD partial-credit anchors
  - JSON response schema validated; one retry on parse/range failure;
    second failure marks the row(s) as judge_error

Public surface:
  - JudgeName: 'opus' | 'mistral'
  - assemble_judge_call(prompt, responses_by_model, judge_seed) -> JudgeCall
  - call_judge(judge_name, judge_call) -> JudgeResponse
  - score_one_batch(prompt, responses_by_model, judge_name, ...) -> list[JudgeRowScore]
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from mistralai import Mistral
from mistralai import models as mistral_models

from runners._base import retry_after_from_error_body
from runners.budget import estimate_cost_usd, usd_to_gbp
from runners.schema import Prompt

# Day 11 revision: 3-judge panel. Mistral kept in JudgeName for the archive
# path (judge_b_mistral_* columns); GPT-5.5 + Gemini added as new judges.
# See methodology doc § "Scope and rigor positioning".
JudgeName = Literal["opus", "mistral", "gpt55", "gemini"]

OPUS_MODEL = "claude-opus-4-6"
MISTRAL_MODEL = "mistral-large-2512"
# GPT-5.5 judge model (released April 2026). Uses chat.completions API with
# response_format={"type": "json_object"} for forced JSON output. Reasoning
# tokens count toward max_completion_tokens (similar to o-series and Gemini)
# at the default reasoning_effort='medium', so we use GPT55_MAX_TOKENS=4096
# (vs JUDGE_MAX_TOKENS=1024 for Opus and Mistral) to leave room for both
# reasoning + JSON output. completion_tokens reported by the API already
# INCLUDES reasoning tokens, so cost accounting reads it directly without
# the visible+thinking summing needed for Gemini.
GPT55_MODEL = "gpt-5.5"
GPT55_MAX_TOKENS = 4096
# Gemini judge model: gemini-2.5-pro (GA, released June 2025). The Day 11
# original choice of gemini-3.1-pro-preview was abandoned after smoke-testing
# revealed it produced ~14 min wall time per call (operationally infeasible
# for the 320-batch sweep). 2.5 Pro requires thinking mode too (cannot set
# thinking_budget=0 — same `400: This model only works in thinking mode` as
# 3.1 Pro), but its thinking is much faster (~5–10s per call vs ~14 min).
# See methodology doc § "Gemini judge model selection".
GEMINI_MODEL = "gemini-2.5-pro"

# Gemini-specific max-tokens cap. Both 2.5 Pro and 3.1 Pro require thinking
# mode; thinking tokens are billed against the candidates_token_count and
# share the max_output_tokens budget with visible output. JUDGE_MAX_TOKENS
# (1024) was insufficient — smoke-test showed thinking alone consumed all
# 1024 leaving zero for the JSON response. Bumped 4096 → 8192 on Day 11
# after the validation set's case 1 (rea-015 sonnet compression — complex
# multi-step proration reasoning) hit the 4096 cap and returned an empty
# response after 58s. 8192 leaves headroom for demanding reasoning prompts.
GEMINI_MAX_TOKENS = 8192
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024
SCORE_LO, SCORE_HI = 0.0, 1.0
LABELS = ("A", "B", "C", "D")

# Rate-limit + transient-error retry config for judge API calls. Mirrors the
# Day 7 retry hardening in runners/run_*.py — same pattern, judge-specific
# delays. Day 10 production hit Mistral 429s at concurrency 4 (TPM-bound,
# not RPM-bound); this catches them and recovers. Catches both the Mistral
# SDKError 429/5xx surface and the Anthropic exception classes.
JUDGE_MAX_RETRIES = 4
JUDGE_TRANSIENT_5XX_DELAYS = (5.0, 10.0, 20.0, 40.0)
JUDGE_RATE_LIMIT_DEFAULT_DELAY = 5.0
_TRANSIENT_HTTP_CODES = {500, 502, 503, 504}

# Categories that have a canonical reference answer worth showing the judges
_REF_ANSWER_FIELD = {
    "rag_qa": "answer",
    "reasoning": "final_answer",
}

_SYSTEM_PROMPT = """You are an expert evaluator of LLM responses. You will see a task prompt, evaluation criteria, an optional reference answer, and 4 candidate responses labelled A, B, C, D. Score each response on a 0.0–1.0 continuous scale per the rubric below.

Rubric (PRD partial-credit anchors):
- 1.0  — satisfies all criteria fully and accurately
- 0.5–0.7  — covers part of the criteria correctly but missing one or more components
- 0.0–0.3  — wrong on the central facts or misleading
- intermediate values where appropriate; do not collapse to only 0 or 1

Score each response independently. Do NOT score relatively (e.g. "B is better than A"); score each one against the criteria as if it were the only response.

Output ONLY a JSON object of the form:
{"A": <number 0.0-1.0>, "B": <number 0.0-1.0>, "C": <number 0.0-1.0>, "D": <number 0.0-1.0>, "reasoning": {"A": "<one short sentence>", "B": "<one short sentence>", "C": "<one short sentence>", "D": "<one short sentence>"}}

No prose outside the JSON. No markdown fences."""


@dataclass
class JudgeCall:
    """One assembled judge call covering a single (prompt, lever).

    `position_to_model` records which test model is at each A/B/C/D slot for
    this call. Persisted alongside the result rows so Day 12's position-bias
    audit can run from the DB without replaying API calls.
    """
    prompt_id: str
    lever: str
    user_message: str
    position_to_model: dict[str, str]  # {"A": "claude-sonnet-4-6", ...}
    seed: int


@dataclass
class JudgeResponse:
    """Raw + parsed result of one judge API call. `parse_error` is None on
    success; on retry-exhausted failure it carries the last error string."""
    judge: JudgeName
    raw_text: str
    scores: dict[str, float] | None  # {"A": 0.8, "B": 0.5, ...} or None on parse failure
    reasoning: dict[str, str] | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    parse_error: str | None = None


@dataclass
class JudgeRowScore:
    """One scored row — judge_a or judge_b score for a single
    (prompt_id, model, lever) combination."""
    prompt_id: str
    model: str
    lever: str
    judge: JudgeName
    score: float | None  # None on judge_error
    position_label: str  # 'A' / 'B' / 'C' / 'D'
    reasoning: str | None
    judge_error: str | None = None
    cost_usd: float = 0.0
    latency_ms: int = 0


def position_seed(prompt_id: str, lever: str) -> int:
    """Deterministic per-(prompt, lever) seed for A/B/C/D ordering. Same seed
    is used by both judges so they see the same order on the same call —
    important for cross-judge comparability of per-position effects."""
    h = hashlib.sha256(f"{prompt_id}|{lever}".encode()).hexdigest()
    return int(h[:8], 16)


def randomised_position_map(models: list[str], seed: int) -> dict[str, str]:
    """Return {position_label: model} mapping. Order of `models` does not
    affect output as long as the set is the same — sort first for determinism."""
    if len(models) != len(LABELS):
        raise ValueError(
            f"randomised_position_map requires exactly {len(LABELS)} models; got {len(models)}"
        )
    rng = random.Random(seed)
    sorted_models = sorted(models)
    shuffled = list(sorted_models)
    rng.shuffle(shuffled)
    return dict(zip(LABELS, shuffled))


def _reference_answer_block(prompt: Prompt) -> str | None:
    """Return the reference-answer block to show the judge, or None if this
    category has no canonical reference (cs, sum)."""
    field_name = _REF_ANSWER_FIELD.get(prompt.task_category)
    if field_name is None:
        return None
    t1 = prompt.scoring.tier_1_deterministic
    if t1 is None:
        return None
    expected = t1.expected
    if field_name not in expected:
        return None
    return f"Reference answer (canonical): {expected[field_name]!r}"


def assemble_judge_call(
    prompt: Prompt,
    responses_by_model: dict[str, str],
    lever: str,
) -> JudgeCall:
    """Build the user-message string + position mapping for a (prompt, lever).

    `responses_by_model` must contain exactly 4 entries (one per test model).
    The same JudgeCall is sent to both judges — they see identical content
    in identical position order.
    """
    if not prompt.scoring.tier_2_judge:
        raise ValueError(f"prompt {prompt.prompt_id} has no tier_2_judge.criteria")

    seed = position_seed(prompt.prompt_id, lever)
    pos_to_model = randomised_position_map(list(responses_by_model), seed)

    parts: list[str] = []
    parts.append("=== TASK PROMPT ===")
    parts.append(f"[system]\n{prompt.input.system}")
    parts.append(f"[user]\n{prompt.input.user}")

    ref = _reference_answer_block(prompt)
    if ref is not None:
        parts.append("=== REFERENCE ANSWER ===")
        parts.append(ref)

    parts.append("=== EVALUATION CRITERIA ===")
    parts.append(prompt.scoring.tier_2_judge.criteria)

    parts.append("=== CANDIDATE RESPONSES ===")
    for label in LABELS:
        model = pos_to_model[label]
        text = responses_by_model[model]
        parts.append(f"--- Response {label} ---\n{text}")

    parts.append(
        "=== YOUR TASK ===\n"
        "Score each of A, B, C, D on the 0.0–1.0 scale against the criteria. "
        "Output the JSON object as specified."
    )

    user_message = "\n\n".join(parts)
    return JudgeCall(
        prompt_id=prompt.prompt_id,
        lever=lever,
        user_message=user_message,
        position_to_model=pos_to_model,
        seed=seed,
    )


_FENCE_STRIP_RE = re.compile(r"^```(?:json|JSON)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _parse_judge_response(raw: str) -> tuple[dict[str, float] | None, dict[str, str] | None, str | None]:
    """Parse the judge's JSON response. Returns (scores, reasoning, error_or_None).

    Tolerates markdown fence wrapping (Opus tendency) and pre-JSON preamble.
    Validates: all 4 labels present, all scores in [0.0, 1.0]."""
    s = raw.strip()
    m = _FENCE_STRIP_RE.match(s)
    if m:
        s = m.group(1).strip()
    obj_start = s.find("{")
    obj_end = s.rfind("}")
    if obj_start == -1 or obj_end <= obj_start:
        return None, None, "no JSON object found"
    s = s[obj_start : obj_end + 1]

    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        return None, None, f"json decode failed: {e}"

    if not isinstance(parsed, dict):
        return None, None, "top-level not an object"

    scores: dict[str, float] = {}
    for label in LABELS:
        if label not in parsed:
            return None, None, f"missing label {label!r}"
        v = parsed[label]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None, None, f"score for {label!r} not a number: {v!r}"
        v = float(v)
        if not (SCORE_LO <= v <= SCORE_HI):
            return None, None, f"score for {label!r} out of range [0.0, 1.0]: {v}"
        scores[label] = v

    reasoning: dict[str, str] = {}
    raw_reasoning = parsed.get("reasoning")
    if isinstance(raw_reasoning, dict):
        for label in LABELS:
            v = raw_reasoning.get(label)
            if isinstance(v, str):
                reasoning[label] = v
    return scores, reasoning, None


def _call_opus_raw(client: anthropic.Anthropic, user_message: str) -> tuple[str, int, int, int]:
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    return text, msg.usage.input_tokens, msg.usage.output_tokens, latency_ms


def _call_mistral_raw(client: Mistral, user_message: str) -> tuple[str, int, int, int]:
    t0 = time.perf_counter()
    resp = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens, latency_ms


def _mistral_status_code(err: Exception) -> int | None:
    """Mistral SDK raises SDKError for HTTP failures; raw_status_code lives on
    the exception. Returns None when not a Mistral SDKError."""
    if not isinstance(err, mistral_models.SDKError):
        return None
    return getattr(err, "status_code", None) or getattr(err, "raw_status_code", None)


def _gemini_status_code(err: Exception) -> int | None:
    """google.genai.errors.APIError (parent of ClientError + ServerError)
    carries .code as the HTTP status. Returns None when not a Gemini error."""
    if not isinstance(err, genai_errors.APIError):
        return None
    return getattr(err, "code", None)


def _retry_after_seconds(err: Exception, default: float) -> float:
    """Honour Retry-After header on rate-limit responses where present.

    Anthropic's RateLimitError exposes .response.headers; Mistral's SDKError
    exposes .headers (when set). Falls through to default when missing.
    """
    headers = None
    resp = getattr(err, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
    if headers is None:
        headers = getattr(err, "headers", None)
    if headers is not None:
        try:
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra is not None:
                return float(ra)
        except (TypeError, ValueError, AttributeError):
            pass
    return retry_after_from_error_body(err, default)


def _call_with_retry(
    raw_call,
    *,
    judge: JudgeName,
    max_retries: int = JUDGE_MAX_RETRIES,
) -> tuple[str, int, int, int]:
    """Call wrapper with rate-limit + transient-5xx retry. Mirrors the Day 7
    retry-hardening pattern in runners/run_anthropic.py + run_openai.py.

    Catches and retries:
      - anthropic.RateLimitError + anthropic.InternalServerError + APIConnectionError
      - openai.RateLimitError + openai.InternalServerError + openai.APIConnectionError
      - mistralai SDKError with status in {429, 500, 502, 503, 504}
      - google.genai.errors.APIError with status in {429, 500, 502, 503, 504}
    Re-raises any other exception immediately. Re-raises after max_retries.
    """
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return raw_call()
        except anthropic.RateLimitError as e:
            last_err = e
            if attempt >= max_retries:
                break
            delay = _retry_after_seconds(e, JUDGE_RATE_LIMIT_DEFAULT_DELAY)
            time.sleep(delay)
        except (anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            delay = JUDGE_TRANSIENT_5XX_DELAYS[min(attempt, len(JUDGE_TRANSIENT_5XX_DELAYS) - 1)]
            delay = _retry_after_seconds(e, delay)
            time.sleep(delay)
        except openai.RateLimitError as e:
            last_err = e
            if attempt >= max_retries:
                break
            delay = _retry_after_seconds(e, JUDGE_RATE_LIMIT_DEFAULT_DELAY)
            time.sleep(delay)
        except (openai.InternalServerError, openai.APIConnectionError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            delay = JUDGE_TRANSIENT_5XX_DELAYS[min(attempt, len(JUDGE_TRANSIENT_5XX_DELAYS) - 1)]
            delay = _retry_after_seconds(e, delay)
            time.sleep(delay)
        except mistral_models.SDKError as e:
            code = _mistral_status_code(e)
            if code == 429:
                last_err = e
                if attempt >= max_retries:
                    break
                delay = _retry_after_seconds(e, JUDGE_RATE_LIMIT_DEFAULT_DELAY)
                time.sleep(delay)
            elif code in _TRANSIENT_HTTP_CODES:
                last_err = e
                if attempt >= max_retries:
                    break
                delay = JUDGE_TRANSIENT_5XX_DELAYS[min(attempt, len(JUDGE_TRANSIENT_5XX_DELAYS) - 1)]
                time.sleep(delay)
            else:
                # 4xx other than 429 (auth, validation) is non-retriable.
                raise
        except genai_errors.APIError as e:
            code = _gemini_status_code(e)
            if code == 429:
                last_err = e
                if attempt >= max_retries:
                    break
                # Gemini puts retry hint in error.details RetryInfo, not headers.
                # _retry_after_seconds will fall through to default for now;
                # could be extended to parse RetryInfo if production hits 429s often.
                delay = _retry_after_seconds(e, JUDGE_RATE_LIMIT_DEFAULT_DELAY)
                time.sleep(delay)
            elif code in _TRANSIENT_HTTP_CODES:
                last_err = e
                if attempt >= max_retries:
                    break
                delay = JUDGE_TRANSIENT_5XX_DELAYS[min(attempt, len(JUDGE_TRANSIENT_5XX_DELAYS) - 1)]
                time.sleep(delay)
            else:
                # 4xx other than 429 (auth, INVALID_ARGUMENT, PERMISSION_DENIED
                # incl. billing-not-enabled) is non-retriable.
                raise
    raise last_err  # type: ignore[misc]


def _call_opus(client: anthropic.Anthropic, user_message: str) -> tuple[str, int, int, int]:
    """Retry-wrapped Opus call. Returns (raw_text, input_tokens, output_tokens, latency_ms)."""
    return _call_with_retry(lambda: _call_opus_raw(client, user_message), judge="opus")


def _call_mistral(client: Mistral, user_message: str) -> tuple[str, int, int, int]:
    """Retry-wrapped Mistral call."""
    return _call_with_retry(lambda: _call_mistral_raw(client, user_message), judge="mistral")


def _call_gemini_raw(client: genai.Client, user_message: str) -> tuple[str, int, int, int]:
    """Returns (raw_text, input_tokens, output_tokens, latency_ms).

    Gemini supports response_mime_type='application/json' which forces valid
    JSON server-side, reducing parse-failure risk vs Opus (markdown fences)
    and Mistral (occasional prose preamble). System instruction goes via the
    config rather than as a separate message.

    Thinking is enabled implicitly (Gemini 2.5 Pro REQUIRES thinking — cannot
    set thinking_budget=0). Thinking tokens are billed at the output rate
    and consume the max_output_tokens budget alongside visible output, so
    we use GEMINI_MAX_TOKENS (4096) instead of JUDGE_MAX_TOKENS (1024) and
    sum thinking + visible into out_tok for cost accounting.
    """
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=JUDGE_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_TOKENS,
            response_mime_type="application/json",
        ),
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = resp.text or ""
    in_tok = resp.usage_metadata.prompt_token_count
    visible_out = resp.usage_metadata.candidates_token_count or 0
    thinking_out = getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0
    out_tok = visible_out + thinking_out
    return text, in_tok, out_tok, latency_ms


def _call_gemini(client: genai.Client, user_message: str) -> tuple[str, int, int, int]:
    """Retry-wrapped Gemini call."""
    return _call_with_retry(lambda: _call_gemini_raw(client, user_message), judge="gemini")


def _call_gpt55_raw(client: openai.OpenAI, user_message: str) -> tuple[str, int, int, int]:
    """Returns (raw_text, input_tokens, output_tokens, latency_ms).

    GPT-5.5 is a reasoning model — reasoning tokens count toward the
    max_completion_tokens budget (default reasoning_effort='medium'). We use
    GPT55_MAX_TOKENS=4096 to leave room for both reasoning + the ~300-token
    JSON response. response_format={"type":"json_object"} forces valid JSON
    server-side, reducing parse-failure risk.

    completion_tokens reported by the API already INCLUDES reasoning tokens
    (per OpenAI docs); read it directly for cost accounting (no visible+thinking
    summing needed, unlike Gemini).

    NOTE on temperature: GPT-5.5 only supports the default temperature=1
    (verified 2026-05-10 — API returns `400: Unsupported value: 'temperature'
    does not support 0 with this model. Only the default (1) value is supported.`
    for any explicit non-1 value). This is stricter than GPT-5.4 family which
    accepts temp=0 at the default reasoning_effort='medium'. We omit the
    temperature parameter entirely (default 1 applies). Judge non-determinism
    at temp=1 is documented in the methodology section and is the same kind of
    constraint Gemini and Mistral exhibit at any temperature setting.
    """
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=GPT55_MODEL,
        max_completion_tokens=GPT55_MAX_TOKENS,
        # temperature omitted — API-mandated default of 1; see docstring.
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    return text, in_tok, out_tok, latency_ms


def _call_gpt55(client: openai.OpenAI, user_message: str) -> tuple[str, int, int, int]:
    """Retry-wrapped GPT-5.5 call. The retry classifier already covers
    openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError
    (added during Day 7 retry hardening for the test-model OpenAI calls;
    judges share the same exception types via the same SDK)."""
    return _call_with_retry(lambda: _call_gpt55_raw(client, user_message), judge="gpt55")


def call_judge(
    judge_name: JudgeName,
    judge_call: JudgeCall,
    *,
    opus_client: anthropic.Anthropic | None = None,
    mistral_client: Mistral | None = None,
    gemini_client: "genai.Client | None" = None,
    gpt55_client: openai.OpenAI | None = None,
) -> JudgeResponse:
    """Fire a judge API call. Single retry on parse/range failure (one extra
    API call); a second parse failure returns JudgeResponse with parse_error
    populated and scores=None."""
    if judge_name == "opus":
        if opus_client is None:
            opus_client = anthropic.Anthropic()
        call_fn = lambda: _call_opus(opus_client, judge_call.user_message)
    elif judge_name == "mistral":
        if mistral_client is None:
            mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        call_fn = lambda: _call_mistral(mistral_client, judge_call.user_message)
    elif judge_name == "gemini":
        if gemini_client is None:
            gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        call_fn = lambda: _call_gemini(gemini_client, judge_call.user_message)
    elif judge_name == "gpt55":
        if gpt55_client is None:
            # max_retries=0 disables the OpenAI SDK's internal retry loop —
            # we control retry via _call_with_retry, mirroring the test-model
            # OpenAI adapter pattern in runners/run_openai.py.
            gpt55_client = openai.OpenAI(max_retries=0)
        call_fn = lambda: _call_gpt55(gpt55_client, judge_call.user_message)
    else:
        raise ValueError(f"unknown judge {judge_name!r}")

    last_err: str | None = None
    last_raw = ""
    last_in = last_out = last_lat = 0
    for attempt in range(2):
        raw, in_tok, out_tok, lat = call_fn()
        last_raw, last_in, last_out, last_lat = raw, in_tok, out_tok, lat
        scores, reasoning, err = _parse_judge_response(raw)
        if err is None:
            return JudgeResponse(
                judge=judge_name, raw_text=raw, scores=scores, reasoning=reasoning,
                input_tokens=in_tok, output_tokens=out_tok, latency_ms=lat,
                parse_error=None,
            )
        last_err = err

    return JudgeResponse(
        judge=judge_name, raw_text=last_raw, scores=None, reasoning=None,
        input_tokens=last_in, output_tokens=last_out, latency_ms=last_lat,
        parse_error=last_err,
    )


_JUDGE_MODEL_BY_NAME: dict[str, str] = {
    "opus": OPUS_MODEL,
    "mistral": MISTRAL_MODEL,
    "gemini": GEMINI_MODEL,
    "gpt55": GPT55_MODEL,
}


def _judge_model_id(judge_name: JudgeName) -> str:
    """Map judge slot name to the API model ID. Used for cost lookup in
    runners/budget.py.estimate_cost_usd."""
    return _JUDGE_MODEL_BY_NAME[judge_name]


def score_one_batch(
    prompt: Prompt,
    responses_by_model: dict[str, str],
    lever: str,
    judge_names: tuple[JudgeName, ...] = ("opus", "mistral"),
    *,
    opus_client: anthropic.Anthropic | None = None,
    mistral_client: Mistral | None = None,
    gemini_client: "genai.Client | None" = None,
    gpt55_client: openai.OpenAI | None = None,
) -> tuple[JudgeCall, list[JudgeRowScore]]:
    """Run one (prompt, lever) batch through both judges. Returns the
    JudgeCall (for audit/persistence) plus per-(model, judge) row scores.

    Partial-result behaviour: each judge is fired independently. If one
    judge raises (after retry exhaustion) the OTHER judge's results are
    still returned — caller persists what succeeded and re-fires only the
    failing side later. Pre-Day-10-recovery this method discarded the
    successful judge's work whenever the other judge raised, wasting £4.7
    of Opus calls. See methodology doc § 'Day 10 dry-run validation'.
    """
    call = assemble_judge_call(prompt, responses_by_model, lever)
    rows: list[JudgeRowScore] = []
    for judge_name in judge_names:
        try:
            resp = call_judge(
                judge_name, call,
                opus_client=opus_client, mistral_client=mistral_client,
                gemini_client=gemini_client, gpt55_client=gpt55_client,
            )
        except Exception as e:
            # Retry-exhausted judge call. Emit error rows so the caller can
            # persist what they have for this side and re-fire later via
            # --missing-only mode. The other judge's successful rows in this
            # batch are still returned alongside.
            err_msg = f"{type(e).__name__}: {e}"
            for label in LABELS:
                rows.append(JudgeRowScore(
                    prompt_id=call.prompt_id,
                    model=call.position_to_model[label],
                    lever=lever, judge=judge_name, score=None,
                    position_label=label, reasoning=None,
                    judge_error=err_msg,
                    cost_usd=0.0, latency_ms=0,
                ))
            continue

        usd = estimate_cost_usd(
            _judge_model_id(judge_name),
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            cached_tokens=0, cache_creation_tokens=0,
        )
        for label in LABELS:
            model = call.position_to_model[label]
            score = None if resp.scores is None else resp.scores[label]
            reasoning = None if resp.reasoning is None else resp.reasoning.get(label)
            rows.append(JudgeRowScore(
                prompt_id=call.prompt_id, model=model, lever=lever,
                judge=judge_name, score=score, position_label=label,
                reasoning=reasoning,
                judge_error=resp.parse_error,
                cost_usd=usd / len(LABELS),  # amortise across the 4 scored rows
                latency_ms=resp.latency_ms,
            ))
    return call, rows
