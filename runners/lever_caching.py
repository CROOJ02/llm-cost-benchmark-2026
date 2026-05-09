"""Caching lever (Day 5).

Orchestrates the three-call test sequence per (model, prompt) to measure
prompt-caching's cost and latency multipliers:

  1. baseline call    — no cache_control                (lever="baseline")
  2. cache-write call — with cache_control, cold cache  (lever="caching", config={cache_phase: "write"})
  3. cache-read call  — with cache_control, warm cache  (lever="caching", config={cache_phase: "read"})

Engagement assertions per the methodology spec:
  - Anthropic write: cache_creation_input_tokens > 0 AND cache_read_input_tokens == 0
  - Anthropic read:  cache_read_input_tokens > 0
  - OpenAI read:     usage.prompt_tokens_details.cached_tokens > 0

Failures raise CachingEngagementError loudly rather than silently logging
misleading results. If a (model, prompt) pair is below the model's caching
threshold (per `budget.cache_min_tokens_for`), the lever runs the baseline
call only and reports caching as "unavailable at this prompt size" without
attempting the write/read calls or asserting engagement.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from runners import _base, run_anthropic, run_openai
from runners._base import (
    DB_PATH,
    REPO_ROOT,
    _concurrency,
    start_run,
)
from runners.budget import CostCapExceeded, cache_min_tokens_for, usd_to_gbp
from runners.schema import Prompt, load_prompts


class CachingEngagementError(AssertionError):
    """Raised when a caching call fails its engagement assertion. The runner
    treats this as fatal — silently logging a non-engaged caching call would
    pollute the Day 12 multiplier analysis with misleading data."""


def _assert_cache_write_engaged(adapter_name: str, row: dict[str, Any]) -> None:
    """Anthropic-only write-side assertion. OpenAI's auto-cache is opaque on
    the write side (cached_tokens may be 0 or >0 depending on prior state),
    so OpenAI's write call has no engagement assertion — the read assertion
    is sufficient evidence that caching engaged for the (model, prompt)."""
    if adapter_name != "anthropic":
        return
    creation = row.get("cache_creation_tokens", 0) or 0
    read = row.get("cached_tokens", 0) or 0
    if creation == 0:
        raise CachingEngagementError(
            "CACHING ENGAGEMENT FAILURE — Anthropic cache-write did not write to cache.\n"
            f"  prompt_id: {row.get('prompt_id')}\n"
            f"  model: {row.get('model')}\n"
            f"  cache_creation_input_tokens: {creation} (expected > 0)\n"
            f"  cache_read_input_tokens: {read}\n"
            f"  input_tokens (uncached): {row.get('input_tokens')}\n"
            "Possible causes: prompt below model threshold (Sonnet 4.6=2048, Haiku 4.5=4096), "
            "cache_control not applied to the request, or message structure mismatch."
        )
    if read != 0:
        raise CachingEngagementError(
            "CACHING ENGAGEMENT FAILURE — Anthropic cache-write unexpectedly read from cache.\n"
            f"  prompt_id: {row.get('prompt_id')}, model: {row.get('model')}\n"
            f"  cache_read_input_tokens: {read} (expected 0; cache should have been cold)\n"
            "A prior run within the 5-minute TTL has polluted state. Wait for TTL or use --force-new."
        )


def _assert_cache_read_engaged(adapter_name: str, row: dict[str, Any]) -> None:
    """Both providers must report cached_tokens > 0 on the read call."""
    cached = row.get("cached_tokens", 0) or 0
    if cached == 0:
        raise CachingEngagementError(
            f"CACHING ENGAGEMENT FAILURE — {adapter_name} cache-read did not hit cache.\n"
            f"  prompt_id: {row.get('prompt_id')}, model: {row.get('model')}\n"
            f"  cached_tokens: {cached} (expected > 0)\n"
            f"  input_tokens (uncached): {row.get('input_tokens')}\n"
            "Possible causes: 5-minute TTL exceeded between write and read, prompt content "
            "differs from write call, or caching did not engage."
        )


def run_caching_for_prompt(
    adapter,
    prompt: Prompt,
    model: str,
    *,
    run_id: str,
    cap_gbp: float,
    completed: int,
    planned: int,
    force_new_attempt: bool = False,
    db_path: Path = DB_PATH,
    client: Any = None,
) -> dict[str, Any]:
    """Run baseline + cache_write + cache_read for one (model, prompt). Returns
    a dict with the three result rows plus availability metadata. Raises
    CachingEngagementError if any caching engagement assertion fails."""
    threshold = cache_min_tokens_for(model)
    if client is None:
        client = adapter.make_client()
    input_estimate = adapter.count_input_tokens(client, prompt, model)

    baseline = _base.run_one(
        adapter, prompt, model, lever="baseline",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        force_new_attempt=force_new_attempt,
        db_path=db_path, client=client,
    )

    if input_estimate < threshold:
        return {
            "prompt_id": prompt.prompt_id,
            "model": model,
            "baseline": baseline,
            "cache_write": None,
            "cache_read": None,
            "caching_available": False,
            "skip_reason": (
                f"input_tokens estimate {input_estimate} < model caching threshold {threshold}"
            ),
        }

    write = _base.run_one(
        adapter, prompt, model, lever="caching",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config={"cache_phase": "write", "enable_cache": True},
        force_new_attempt=force_new_attempt,
        db_path=db_path, client=client,
    )
    _assert_cache_write_engaged(adapter.name, write)

    read = _base.run_one(
        adapter, prompt, model, lever="caching",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config={"cache_phase": "read", "enable_cache": True},
        force_new_attempt=force_new_attempt,
        db_path=db_path, client=client,
    )
    _assert_cache_read_engaged(adapter.name, read)

    return {
        "prompt_id": prompt.prompt_id,
        "model": model,
        "baseline": baseline,
        "cache_write": write,
        "cache_read": read,
        "caching_available": True,
        "skip_reason": None,
    }


def run_caching_test(
    adapter,
    prompts: list[Prompt],
    model: str,
    *,
    run_id: str,
    cap_gbp: float,
    force_new_attempt: bool = False,
    db_path: Path = DB_PATH,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    """Run the caching lever across a list of prompts.

    Per-prompt the 3-call sequence runs sequentially (write must precede read
    within the cache TTL); across prompts up to INFEROPS_CONCURRENCY workers
    run in parallel. CachingEngagementError raised in any worker propagates."""
    n = len(prompts)
    n_workers = concurrency if concurrency is not None else _concurrency()
    results: list[dict[str, Any] | None] = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {
            ex.submit(
                run_caching_for_prompt, adapter, p, model,
                run_id=run_id, cap_gbp=cap_gbp, completed=i, planned=n,
                force_new_attempt=force_new_attempt,
                db_path=db_path,
            ): i
            for i, p in enumerate(prompts)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
    return [r for r in results if r is not None]


def _stats(xs: list[float]) -> dict[str, float]:
    """Min / median / max for a small sample. Per the methodology doc, multipliers
    are reported as a range across N=5 rather than a single point estimate so
    heterogeneity is visible to a reader extrapolating to production."""
    if not xs:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    s = sorted(xs)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {"min": s[0], "median": median, "max": s[-1]}


def summarise_multipliers(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute observed cost and latency multipliers from a caching run.

    For each available (model, prompt) result, computes per-prompt:
      cost_write_mult    = cache_write.cost_usd   / baseline.cost_usd
      cost_read_mult     = cache_read.cost_usd    / baseline.cost_usd
      latency_write_mult = cache_write.latency_ms / baseline.latency_ms
      latency_read_mult  = cache_read.latency_ms  / baseline.latency_ms
    Aggregates as min/median/max across prompts; per-prompt detail also returned.
    """
    available = [r for r in results if r["caching_available"]]
    unavailable = [r for r in results if not r["caching_available"]]
    if not available:
        return {
            "available_count": 0,
            "unavailable_count": len(unavailable),
            "skip_reasons": [r["skip_reason"] for r in unavailable],
        }
    cost_w = [r["cache_write"]["cost_usd"] / r["baseline"]["cost_usd"] for r in available]
    cost_r = [r["cache_read"]["cost_usd"] / r["baseline"]["cost_usd"] for r in available]
    lat_w = [r["cache_write"]["latency_ms"] / max(r["baseline"]["latency_ms"], 1) for r in available]
    lat_r = [r["cache_read"]["latency_ms"] / max(r["baseline"]["latency_ms"], 1) for r in available]
    return {
        "available_count": len(available),
        "unavailable_count": len(unavailable),
        "cost_write_mult": _stats(cost_w),
        "cost_read_mult": _stats(cost_r),
        "latency_write_mult": _stats(lat_w),
        "latency_read_mult": _stats(lat_r),
        "per_prompt": [
            {
                "prompt_id": r["prompt_id"],
                "cost_write_mult": r["cache_write"]["cost_usd"] / r["baseline"]["cost_usd"],
                "cost_read_mult": r["cache_read"]["cost_usd"] / r["baseline"]["cost_usd"],
                "latency_write_mult": r["cache_write"]["latency_ms"] / max(r["baseline"]["latency_ms"], 1),
                "latency_read_mult": r["cache_read"]["latency_ms"] / max(r["baseline"]["latency_ms"], 1),
            }
            for r in available
        ],
    }


_ADAPTERS_BY_NAME = {
    "anthropic": run_anthropic.ANTHROPIC_ADAPTER,
    "openai":    run_openai.OPENAI_ADAPTER,
}

_PROVIDER_FOR_MODEL = {
    "claude-sonnet-4-6":       "anthropic",
    "claude-haiku-4-5":        "anthropic",
    "claude-opus-4-6":         "anthropic",
    "gpt-5.4":                 "openai",
    "gpt-5.4-2026-03-05":      "openai",
    "gpt-5.4-mini":            "openai",
    "gpt-5.4-mini-2026-03-17": "openai",
}


def _adapter_for_model(model: str):
    provider = _PROVIDER_FOR_MODEL.get(model)
    if provider is None:
        raise ValueError(f"unknown model {model!r}")
    return _ADAPTERS_BY_NAME[provider]


if __name__ == "__main__":
    """Day 5 caching lever driver. Args: <cap_gbp> [--force].
    Runs the caching test on sum-001..005 across all 4 test models."""
    load_dotenv(REPO_ROOT / ".env")
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    force = "--force" in sys.argv[2:]
    # The 5 longest hards (per methodology doc § Prompt subset selection):
    # 3,348–3,838 input tokens, all clearing Sonnet 4.6's 2048 floor by 1,300+
    # tokens; 0 of the 20 summarisation prompts clear Haiku 4.5's 4096 floor.
    SUBSET = {"sum-015", "sum-016", "sum-017", "sum-018", "sum-020"}
    summary_prompts = load_prompts(REPO_ROOT / "prompts" / "summarisation.json")
    targets = [p for p in summary_prompts if p.prompt_id in SUBSET]
    rid = start_run(cost_cap_gbp=cap)
    print(f"run_id={rid} cap=£{cap:.4f} prompts={[p.prompt_id for p in targets]} force={force}")

    overall: dict[str, dict[str, Any]] = {}
    for model in ["claude-sonnet-4-6", "claude-haiku-4-5", "gpt-5.4-2026-03-05", "gpt-5.4-mini-2026-03-17"]:
        adapter = _adapter_for_model(model)
        print(f"\n=== {model} ({adapter.name}) ===")
        started = time.perf_counter()
        try:
            results = run_caching_test(
                adapter, targets, model,
                run_id=rid, cap_gbp=cap, force_new_attempt=force,
            )
        except CachingEngagementError as e:
            print(f"  ENGAGEMENT FAILURE: {e}", file=sys.stderr)
            sys.exit(2)
        wall_ms = int((time.perf_counter() - started) * 1000)
        summary = summarise_multipliers(results)
        overall[model] = summary
        for r in results:
            if r["caching_available"]:
                base_cost = r["baseline"]["cost_usd"]
                write_cost = r["cache_write"]["cost_usd"]
                read_cost = r["cache_read"]["cost_usd"]
                base_lat = r["baseline"]["latency_ms"]
                write_lat = r["cache_write"]["latency_ms"]
                read_lat = r["cache_read"]["latency_ms"]
                print(
                    f"  {r['prompt_id']:8s}  "
                    f"baseline=${base_cost:.6f}/{base_lat}ms  "
                    f"write=${write_cost:.6f}/{write_lat}ms ({write_cost/base_cost:.2f}× cost, {write_lat/max(base_lat,1):.2f}× lat)  "
                    f"read=${read_cost:.6f}/{read_lat}ms ({read_cost/base_cost:.2f}× cost, {read_lat/max(base_lat,1):.2f}× lat)"
                )
            else:
                print(f"  {r['prompt_id']:8s}  UNAVAILABLE — {r['skip_reason']}")
        if summary["available_count"] > 0:
            cw, cr = summary["cost_write_mult"], summary["cost_read_mult"]
            lw, lr = summary["latency_write_mult"], summary["latency_read_mult"]
            print(
                f"  -- min/median/max  (n={summary['available_count']})\n"
                f"     cost_write    : {cw['min']:.3f}× / {cw['median']:.3f}× / {cw['max']:.3f}×\n"
                f"     cost_read     : {cr['min']:.3f}× / {cr['median']:.3f}× / {cr['max']:.3f}×\n"
                f"     latency_write : {lw['min']:.3f}× / {lw['median']:.3f}× / {lw['max']:.3f}×\n"
                f"     latency_read  : {lr['min']:.3f}× / {lr['median']:.3f}× / {lr['max']:.3f}×"
            )
        print(f"  wall={wall_ms}ms")

    print("\n=== SUMMARY (min / median / max across N=5 prompts) ===")
    for model, s in overall.items():
        if s.get("available_count", 0) > 0:
            cw, cr = s["cost_write_mult"], s["cost_read_mult"]
            lw, lr = s["latency_write_mult"], s["latency_read_mult"]
            print(f"  {model}")
            print(f"    cost   : write {cw['min']:.3f}/{cw['median']:.3f}/{cw['max']:.3f}×    read {cr['min']:.3f}/{cr['median']:.3f}/{cr['max']:.3f}×")
            print(f"    latency: write {lw['min']:.3f}/{lw['median']:.3f}/{lw['max']:.3f}×    read {lr['min']:.3f}/{lr['median']:.3f}/{lr['max']:.3f}×")
        else:
            print(f"  {model}  caching unavailable across all {s['unavailable_count']} prompts")
