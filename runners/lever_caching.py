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
import sqlite3
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
    """Reserved for unrecoverable caching protocol violations (e.g. malformed
    response). The everyday non-engagement cases (cache-write didn't write,
    cache-read returned cached_tokens=0) are NOT raised — they're recorded as
    `lever='caching_unavailable'` rows in `results` instead. Day 6+ finding:
    GPT-5.4 caching engagement is flaky in some configurations (1/5 prompts
    on a re-measurement showed cache-read returning cached=0 despite a
    successful write); treating those as fatal stalled the run. Treating
    them as data points lets Day 12 quantify caching reliability across
    prompts/models/conditions."""


def _diagnose_cache_write(adapter_name: str, row: dict[str, Any]) -> str | None:
    """Return a skip_reason string if the cache_write call did not engage as
    expected, or None if engagement looks healthy. Anthropic-specific (OpenAI's
    auto-cache is opaque on the write side; OpenAI write engagement is judged
    via the subsequent read call hitting the cache)."""
    if adapter_name != "anthropic":
        return None
    creation = row.get("cache_creation_tokens", 0) or 0
    read = row.get("cached_tokens", 0) or 0
    if creation == 0:
        return (
            f"Anthropic cache-write did not write to cache: "
            f"cache_creation_input_tokens=0 (expected > 0). "
            f"Possible causes: prompt below model threshold, cache_control not applied, "
            f"or message structure mismatch."
        )
    if read != 0:
        return (
            f"Anthropic cache-write unexpectedly read from cache: "
            f"cache_read_input_tokens={read} (expected 0; cache should have been cold). "
            f"A prior run within the 5-minute TTL has polluted state."
        )
    return None


def _diagnose_cache_read(adapter_name: str, row: dict[str, Any]) -> str | None:
    """Return a skip_reason string if the cache_read call did not engage,
    or None if cached_tokens > 0. Both providers."""
    cached = row.get("cached_tokens", 0) or 0
    if cached == 0:
        return (
            f"{adapter_name} cache-read did not hit cache: cached_tokens=0 "
            f"(expected > 0; input_tokens uncached={row.get('input_tokens')}). "
            f"Possible causes: 5-minute TTL exceeded between write and read, "
            f"prompt content differs from write call, or caching did not engage."
        )
    return None


def _insert_caching_unavailable_row(
    *,
    adapter,
    prompt: Prompt,
    model: str,
    run_id: str,
    db_path: Path,
    write_row: dict[str, Any],
    read_row: dict[str, Any] | None,
    skip_reason: str,
    force_new_attempt: bool,
) -> dict[str, Any]:
    """Persist a `lever='caching_unavailable'` marker row when the caching
    write/read engagement check fails. Mirrors `compression_unavailable` —
    the actual write/read API calls already inserted their rows under
    `lever='caching'`; this marker is a Day 12-friendly signal that the
    `(prompt, model)` pair did not engage cleanly. Cost is zero (no extra
    API call)."""
    optimisation_config: dict[str, Any] = {
        "caching_status": "unavailable",
        "observed_write_cached_tokens":   write_row.get("cached_tokens", 0) or 0,
        "observed_write_creation_tokens": write_row.get("cache_creation_tokens", 0) or 0,
        "observed_read_cached_tokens":    None if read_row is None else (read_row.get("cached_tokens", 0) or 0),
        "skip_reason": skip_reason,
    }
    optimisation_config = run_openai.annotate_optimisation_config_for_reasoning_effort(
        optimisation_config, model,
    )
    config_hash = _base._config_hash("caching_unavailable", optimisation_config)
    run_attempt = 1
    if force_new_attempt:
        with sqlite3.connect(db_path) as conn:
            run_attempt = _base._next_run_attempt(
                conn, prompt.prompt_id, model, "caching_unavailable", config_hash,
            )
    else:
        with sqlite3.connect(db_path) as conn:
            existing = _base._existing_successful_row(
                conn, prompt.prompt_id, model, "caching_unavailable", config_hash, run_attempt,
                run_id,
            )
        if existing is not None:
            return {**existing, "skipped": True}

    row = _base._new_row(
        prompt=prompt, model=model, provider=adapter.name, lever="caching_unavailable",
        config_hash=config_hash, optimisation_config=optimisation_config,
        run_id=run_id, run_attempt=run_attempt,
    )
    # Numeric fields stay 0 (no API call attached to this marker; the actual
    # write/read API calls have their own rows under lever='caching').
    with sqlite3.connect(db_path) as conn:
        _base._insert_row(conn, row)
        conn.commit()
    return {**row, "skipped": False}


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

    baseline_cfg = run_openai.annotate_optimisation_config_for_reasoning_effort(None, model)
    baseline = _base.run_one(
        adapter, prompt, model, lever="baseline",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config=baseline_cfg,
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

    write_cfg = run_openai.annotate_optimisation_config_for_reasoning_effort(
        {"cache_phase": "write", "enable_cache": True}, model,
    )
    write = _base.run_one(
        adapter, prompt, model, lever="caching",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config=write_cfg,
        force_new_attempt=force_new_attempt,
        db_path=db_path, client=client,
    )
    write_skip_reason = _diagnose_cache_write(adapter.name, write)
    if write_skip_reason is not None:
        # Soft-fail: the write call's row is already inserted under lever='caching'.
        # Add a marker row + return unavailable. Don't crash the sweep.
        unavailable = _insert_caching_unavailable_row(
            adapter=adapter, prompt=prompt, model=model, run_id=run_id, db_path=db_path,
            write_row=write, read_row=None, skip_reason=write_skip_reason,
            force_new_attempt=force_new_attempt,
        )
        return {
            "prompt_id": prompt.prompt_id, "model": model,
            "baseline": baseline, "cache_write": write, "cache_read": None,
            "caching_unavailable_row": unavailable,
            "caching_available": False, "skip_reason": write_skip_reason,
        }

    read_cfg = run_openai.annotate_optimisation_config_for_reasoning_effort(
        {"cache_phase": "read", "enable_cache": True}, model,
    )
    read = _base.run_one(
        adapter, prompt, model, lever="caching",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config=read_cfg,
        force_new_attempt=force_new_attempt,
        db_path=db_path, client=client,
    )
    read_skip_reason = _diagnose_cache_read(adapter.name, read)
    if read_skip_reason is not None:
        # Soft-fail: read row exists with cached_tokens=0 under lever='caching'.
        # Add a marker row capturing the observed write+read state.
        unavailable = _insert_caching_unavailable_row(
            adapter=adapter, prompt=prompt, model=model, run_id=run_id, db_path=db_path,
            write_row=write, read_row=read, skip_reason=read_skip_reason,
            force_new_attempt=force_new_attempt,
        )
        return {
            "prompt_id": prompt.prompt_id, "model": model,
            "baseline": baseline, "cache_write": write, "cache_read": read,
            "caching_unavailable_row": unavailable,
            "caching_available": False, "skip_reason": read_skip_reason,
        }

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
