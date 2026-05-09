"""OpenAI provider runner.

Provides OpenAI-specific bits — input-token counting via tiktoken (OpenAI
has no count_tokens API), the raw chat-completions call, retry-on-429 with
Retry-After honoured, and the OpenAIAdapter wiring those into the
provider-agnostic core in `runners._base`. Public `run_one` / `run_many` /
`start_run` are thin wrappers.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import openai
import tiktoken
from dotenv import load_dotenv

from runners import _base
from runners._base import (
    DB_PATH,
    DEFAULT_BASE_DELAY,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    REPO_ROOT,
    _concurrency,
    _read_cost_so_far_gbp,
    mark_run_aborted_cost,
    start_run,
)
from runners.budget import CostCapExceeded, usd_to_gbp
from runners.schema import Prompt, load_prompts

PROVIDER = "openai"

# Per-provider input-token cache. Keys are (prompt_id, model).
_input_token_cache: dict[tuple[str, str], int] = {}

# tiktoken encoder cache — encoders are heavy to construct.
_encoder_cache: dict[str, Any] = {}

# Chat-envelope token overhead. Each chat message adds ~3 tokens of role-tag
# wrapping that tiktoken's raw text encoding doesn't account for; with 2
# messages (system + user) plus the assistant turn marker, ~9-10 tokens of
# fixed overhead. Slightly pessimistic is fine for the cap pre-check.
_CHAT_ENVELOPE_OVERHEAD = 10


def _get_encoder(model: str):
    """Return a tiktoken encoder for the model. Falls back to o200k_base
    (used by both the GPT-4o and GPT-5.x families) when tiktoken doesn't
    recognise the dated snapshot or alias."""
    if model not in _encoder_cache:
        try:
            _encoder_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _encoder_cache[model] = tiktoken.get_encoding("o200k_base")
    return _encoder_cache[model]


def count_input_tokens(client: openai.OpenAI, prompt: Prompt, model: str) -> int:
    """Count input tokens for the cap pre-check via tiktoken locally.

    Falls back to char count at ~4 chars/token if tiktoken errors. The
    `client` argument is unused but kept for parity with the Anthropic
    adapter's signature so the ProviderAdapter Protocol stays clean.
    """
    cache_key = (prompt.prompt_id, model)
    if cache_key in _input_token_cache:
        return _input_token_cache[cache_key]
    try:
        enc = _get_encoder(model)
        n = (
            len(enc.encode(prompt.input.system))
            + len(enc.encode(prompt.input.user))
            + _CHAT_ENVELOPE_OVERHEAD
        )
    except Exception:
        n = max(1, (len(prompt.input.system) + len(prompt.input.user)) // 4)
    _input_token_cache[cache_key] = n
    return n


def model_is_gpt_5_4_family(model: str) -> bool:
    """GPT-5.4 family (gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.4-pro and
    their dated snapshots). Used by the annotation helper below to tag result
    rows with the API-default reasoning_effort the call ran at."""
    return model.startswith("gpt-5.4")


def annotate_optimisation_config_for_reasoning_effort(
    optimisation_config: dict[str, Any] | None, model: str,
) -> dict[str, Any] | None:
    """Record reasoning_effort='medium' in the row's optimisation_config when
    calling a GPT-5.4-family model. We do NOT pass reasoning_effort to the API
    (so the API default applies); for the GPT-5.4 family that default is
    'medium', so we annotate to record what actually ran. Returns
    optimisation_config unchanged for non-GPT-5.4 models. Levers and the
    orchestrator call this just before passing optimisation_config to
    `_base.run_one` / `_base.run_many` so config_hash includes the reasoning
    dimension and Day 12 analysis can JOIN on it.

    Day 6+ decision (see methodology doc § "GPT-5.4 reasoning_effort and
    temperature interaction"): the benchmark uses reasoning_effort=medium +
    temperature=0 for cross-provider symmetry. Setting reasoning_effort='low'
    explicitly forces temperature=1 (API constraint); the empirical 2×2
    decomposition showed temperature is the dominant cost variable, so the
    cheaper available regime is medium+temp=0."""
    if not model_is_gpt_5_4_family(model):
        return optimisation_config
    cfg = dict(optimisation_config) if optimisation_config else {}
    cfg["reasoning_effort"] = "medium"
    return cfg


def call_openai(
    client: openai.OpenAI,
    prompt: Prompt,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    # OpenAI renamed `max_tokens` to `max_completion_tokens` for the GPT-5.x
    # family; the GPT-4o family is silently being deprecated to legacy support.
    # We emit `max_completion_tokens` universally — both families accept it as
    # of 2026-05. The internal parameter name (`max_tokens`) is kept for
    # provider-agnostic call-site symmetry with run_anthropic.
    # Per methodology doc § "GPT-5.4 reasoning_effort and temperature
    # interaction": the benchmark uses temperature=0 across all models for
    # reproducibility, and lets reasoning_effort default (API default is
    # 'medium' for GPT-5.4 reasoning models). The empirical decomposition
    # showed temperature dominates cost; pinning temperature=0 + leaving
    # reasoning_effort at API default is the cheapest regime available
    # within cross-provider symmetry constraints. (Setting reasoning_effort
    # explicitly to 'low' would force temperature=1 — API constraint — and
    # cost 43-46% MORE on the cs-001..005 controlled comparison.)
    started = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt.input.system},
            {"role": "user", "content": prompt.input.user},
        ],
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    response_text = resp.choices[0].message.content or ""
    # OpenAI surfaces cache hits via usage.prompt_tokens_details.cached_tokens.
    # Field may be absent on older API versions; default to 0.
    cached = 0
    details = getattr(resp.usage, "prompt_tokens_details", None) if resp.usage else None
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "response_text": response_text,
        # Normalised input_tokens = UNCACHED portion. OpenAI's resp.usage.prompt_tokens
        # INCLUDES cached tokens; we subtract here so estimate_cost_usd's
        # (input_tokens × base + cached_tokens × cache_read_mult) formula matches the
        # provider's actual billing without double-counting.
        "input_tokens": resp.usage.prompt_tokens - cached,
        "output_tokens": resp.usage.completion_tokens,
        "cached_tokens": cached,
        # OpenAI does not expose or separately bill cache creation; always 0.
        "cache_creation_tokens": 0,
        "model_version": resp.model,
        "latency_ms": latency_ms,
        # Truncation signal — OpenAI calls this 'finish_reason'; normalised to
        # 'stop_reason' for cross-provider symmetry. 'length' indicates max_tokens
        # truncation; 'stop' indicates natural completion.
        "stop_reason": resp.choices[0].finish_reason,
    }


def _retry_after_seconds(err: openai.RateLimitError) -> float | None:
    response = getattr(err, "response", None)
    if response is None:
        return None
    val = response.headers.get("retry-after")
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def call_openai_with_retry(
    client: openai.OpenAI,
    prompt: Prompt,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> dict[str, Any]:
    """Wraps call_openai with retry on three transient error classes:
      - RateLimitError (429): Retry-After header honoured, else exponential
                              backoff base_delay × 2^attempt + jitter
      - InternalServerError (5xx): Cloudflare retry_after body hint honoured
                                   (via `retry_after_from_error_body`), else
                                   DEFAULT_TRANSIENT_5XX_DELAY (120s) default
      - Connection errors: DEFAULT_TRANSIENT_5XX_DELAY (120s) default

    All three classes share the same `max_retries` budget (default 3 — so
    worst case 4 attempts before raising). After exhaustion the original
    last exception is raised so the caller (`_base.run_one`) can decide."""
    from runners._base import (
        DEFAULT_TRANSIENT_5XX_DELAY, retry_after_from_error_body,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_openai(client, prompt, model, max_tokens=max_tokens)
        except openai.RateLimitError as e:
            last_err = e
            if attempt == max_retries:
                break
            retry_after = _retry_after_seconds(e)
            if retry_after is not None:
                delay = retry_after
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay * 0.1)
            time.sleep(delay)
        except (openai.InternalServerError, openai.APIConnectionError) as e:
            last_err = e
            if attempt == max_retries:
                break
            delay = retry_after_from_error_body(e, default=DEFAULT_TRANSIENT_5XX_DELAY)
            time.sleep(delay)
    assert last_err is not None
    raise last_err


class _OpenAIAdapter:
    """Wires OpenAI-specific bits into the provider-agnostic core in _base."""

    name = PROVIDER
    rate_limit_error = openai.RateLimitError

    def make_client(self) -> openai.OpenAI:
        return openai.OpenAI(max_retries=0)

    def count_input_tokens(self, client: openai.OpenAI, prompt: Prompt, model: str) -> int:
        return count_input_tokens(client, prompt, model)

    def call_with_retry(
        self, client: openai.OpenAI, prompt: Prompt, model: str,
        max_tokens: int, max_retries: int, base_delay: float,
        *,
        optimisation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # OpenAI's prompt caching is automatic and requires no opt-in; the
        # optimisation_config is ignored here. The lever still exercises caching
        # on this provider by virtue of making repeated calls within the cache TTL.
        return call_openai_with_retry(
            client, prompt, model,
            max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay,
        )


OPENAI_ADAPTER = _OpenAIAdapter()


def run_one(
    prompt: Prompt,
    model: str,
    lever: str,
    *,
    run_id: str,
    cap_gbp: float,
    completed: int,
    planned: int,
    optimisation_config: dict[str, Any] | None = None,
    force_new_attempt: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    db_path: Path = DB_PATH,
    client: openai.OpenAI | None = None,
) -> dict[str, Any]:
    return _base.run_one(
        OPENAI_ADAPTER, prompt, model, lever,
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config=optimisation_config, force_new_attempt=force_new_attempt,
        max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay,
        db_path=db_path, client=client,
    )


def run_many(
    prompts: list[Prompt],
    model: str,
    lever: str,
    *,
    run_id: str,
    cap_gbp: float,
    optimisation_config: dict[str, Any] | None = None,
    force_new_attempt: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    db_path: Path = DB_PATH,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    return _base.run_many(
        OPENAI_ADAPTER, prompts, model, lever,
        run_id=run_id, cap_gbp=cap_gbp,
        optimisation_config=optimisation_config, force_new_attempt=force_new_attempt,
        max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay,
        db_path=db_path, concurrency=concurrency,
    )


if __name__ == "__main__":
    """Day 5 OpenAI smoke driver. Args: <cap_gbp> <n_prompts> [offset] [--force]."""
    load_dotenv(REPO_ROOT / ".env")
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    offset = 0
    force = False
    for arg in sys.argv[3:]:
        if arg == "--force":
            force = True
        else:
            offset = int(arg)
    prompts = load_prompts(REPO_ROOT / "prompts" / "customer_support.json")
    targets = prompts[offset:offset + n]
    rid = start_run(cost_cap_gbp=cap)
    workers = _concurrency()
    print(f"run_id={rid} cap=£{cap:.4f} n={n} offset={offset} force={force} workers={workers}")
    started = time.perf_counter()
    results = run_many(
        targets, model="gpt-5.4-2026-03-05", lever="baseline",
        run_id=rid, cap_gbp=cap, force_new_attempt=force,
    )
    wall_ms = int((time.perf_counter() - started) * 1000)
    sum_latency_ms = sum(r.get("latency_ms", 0) or 0 for r in results)
    n_ran = sum(1 for r in results if not r.get("skipped") and not r.get("aborted") and not r.get("error"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_errored = sum(1 for r in results if r.get("error"))
    for r in results:
        tag = "SKIP" if r.get("skipped") else ("ERR " if r.get("error") else "RAN ")
        cost_g = usd_to_gbp(r.get("cost_usd") or 0.0)
        print(
            f"  [{tag}] {r['prompt_id']} attempt={r.get('run_attempt', '?')} "
            f"cost_gbp=£{cost_g:.6f} latency={r.get('latency_ms', 0)}ms"
        )
    print(
        f"WALL={wall_ms}ms  SUM_LATENCY={sum_latency_ms}ms  "
        f"SPEEDUP={(sum_latency_ms / max(wall_ms, 1)):.2f}x  "
        f"ran={n_ran} skipped={n_skipped} errored={n_errored} workers={workers}"
    )
