"""Anthropic provider runner.

Provides Anthropic-specific bits — input-token counting via count_tokens API
(char-count fallback), the raw API call, retry-on-429 with Retry-After
honoured, and the AnthropicAdapter wiring those into the provider-agnostic
core in `runners._base`. Public `run_one` / `run_many` / `start_run` are
thin wrappers so callers (and existing tests) see the same surface as
before the refactor.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
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

PROVIDER = "anthropic"

# Per-provider input-token cache. Keys are (prompt_id, model). Concurrent
# dict reads/writes are OK in CPython for this purpose — duplicate
# count_tokens calls are harmless.
_input_token_cache: dict[tuple[str, str], int] = {}


def count_input_tokens(client: anthropic.Anthropic, prompt: Prompt, model: str) -> int:
    """Count input tokens for the cap pre-check.

    Primary: client.beta.messages.count_tokens (free, server-side, exact).
    Fallback: char count at ~4 chars/token (English average).
    """
    cache_key = (prompt.prompt_id, model)
    if cache_key in _input_token_cache:
        return _input_token_cache[cache_key]
    try:
        result = client.beta.messages.count_tokens(
            model=model,
            system=prompt.input.system,
            messages=[{"role": "user", "content": prompt.input.user}],
        )
        n = int(result.input_tokens)
    except Exception:
        n = max(1, (len(prompt.input.system) + len(prompt.input.user)) // 4)
    _input_token_cache[cache_key] = n
    return n


def call_anthropic(
    client: anthropic.Anthropic,
    prompt: Prompt,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    started = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=prompt.input.system,
        messages=[{"role": "user", "content": prompt.input.user}],
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    # TODO(Day 9): Tier 1 scorer must defensively strip markdown fences from
    # response_text before JSON parsing. Sonnet 4.6 wraps JSON in ```json ```
    # despite system-prompt instructions to "respond ONLY with a JSON object".
    response_text = "".join(b.text for b in resp.content if b.type == "text")
    return {
        "response_text": response_text,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cached_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "model_version": resp.model,
        "latency_ms": latency_ms,
    }


def _retry_after_seconds(err: anthropic.RateLimitError) -> float | None:
    if err.response is None:
        return None
    val = err.response.headers.get("retry-after")
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def call_anthropic_with_retry(
    client: anthropic.Anthropic,
    prompt: Prompt,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> dict[str, Any]:
    """Wraps call_anthropic with exponential-backoff retry on 429s.
    Honors Retry-After when present; otherwise base_delay × 2^attempt + jitter."""
    last_err: anthropic.RateLimitError | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_anthropic(client, prompt, model, max_tokens=max_tokens)
        except anthropic.RateLimitError as e:
            last_err = e
            if attempt == max_retries:
                break
            retry_after = _retry_after_seconds(e)
            if retry_after is not None:
                delay = retry_after
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay * 0.1)
            time.sleep(delay)
    assert last_err is not None
    raise last_err


class _AnthropicAdapter:
    """Wires Anthropic-specific bits into the provider-agnostic core in _base."""

    name = PROVIDER
    rate_limit_error = anthropic.RateLimitError

    def make_client(self) -> anthropic.Anthropic:
        # max_retries=0 disables the SDK's own retry loop so this layer
        # is the single source of truth for 429 handling.
        return anthropic.Anthropic(max_retries=0)

    def count_input_tokens(self, client: anthropic.Anthropic, prompt: Prompt, model: str) -> int:
        # Delegates to the module-level function so monkeypatching
        # `runners.run_anthropic.count_input_tokens` continues to work.
        return count_input_tokens(client, prompt, model)

    def call_with_retry(
        self, client: anthropic.Anthropic, prompt: Prompt, model: str,
        max_tokens: int, max_retries: int, base_delay: float,
    ) -> dict[str, Any]:
        return call_anthropic_with_retry(
            client, prompt, model,
            max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay,
        )


ANTHROPIC_ADAPTER = _AnthropicAdapter()


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
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    return _base.run_one(
        ANTHROPIC_ADAPTER, prompt, model, lever,
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
        ANTHROPIC_ADAPTER, prompts, model, lever,
        run_id=run_id, cap_gbp=cap_gbp,
        optimisation_config=optimisation_config, force_new_attempt=force_new_attempt,
        max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay,
        db_path=db_path, concurrency=concurrency,
    )


if __name__ == "__main__":
    """Test driver. Args: <cap_gbp> <n_prompts> [offset] [--force]."""
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
        targets, model="claude-sonnet-4-6", lever="baseline",
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
