"""Output-cap lever (Day 6).

Bounds output cost by passing `max_tokens=N` to the provider. Models often
generate longer responses than the task requires; capping output is the
cheapest lever in the matrix to apply (one parameter, no extra calls), but
the cost-quality tradeoff varies by task category — short categories
(customer_support, rag_qa) are unaffected at N=200; long categories
(reasoning, summarisation) can lose answers mid-sentence.

Per-call shape: lever='output_cap', optimisation_config={'max_tokens': N}.
The N value is part of the config_hash so output_cap@200 and output_cap@500
are distinct rows under skip-if-exists. Engagement assertion is a structural
sanity check (output_tokens ≤ max_tokens) — fails loudly if max_tokens
wasn't plumbed through to the provider.
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
from runners.budget import CostCapExceeded, usd_to_gbp
from runners.schema import Prompt, load_prompts


DEFAULT_OUTPUT_CAP = 200


class OutputCapEngagementError(AssertionError):
    """Raised when an output-cap call's output_tokens exceeds the requested
    max_tokens. Structurally this should never happen — the provider enforces
    the cap server-side — so it indicates a plumbing bug (max_tokens not
    forwarded). Treated as fatal to keep corrupted rows out of the analysis."""


def _assert_cap_engaged(row: dict[str, Any], max_tokens: int) -> None:
    out = row.get("output_tokens", 0) or 0
    if out > max_tokens:
        raise OutputCapEngagementError(
            "OUTPUT-CAP ENGAGEMENT FAILURE — output_tokens exceeded requested max.\n"
            f"  prompt_id: {row.get('prompt_id')}, model: {row.get('model')}\n"
            f"  output_tokens: {out} (expected ≤ {max_tokens})\n"
            "Likely cause: max_tokens not forwarded to the provider call. "
            "Check adapter.call_with_retry signature and lever wiring."
        )


def run_output_cap_for_prompt(
    adapter,
    prompt: Prompt,
    model: str,
    *,
    run_id: str,
    cap_gbp: float,
    completed: int,
    planned: int,
    max_tokens: int = DEFAULT_OUTPUT_CAP,
    force_new_attempt: bool = False,
    db_path: Path = DB_PATH,
    client: Any = None,
) -> dict[str, Any]:
    optimisation_config = run_openai.annotate_optimisation_config_for_reasoning_effort(
        {"max_tokens": max_tokens}, model,
    )
    row = _base.run_one(
        adapter, prompt, model, lever="output_cap",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config=optimisation_config,
        max_tokens=max_tokens,
        force_new_attempt=force_new_attempt,
        db_path=db_path, client=client,
    )
    if not row.get("skipped") and not row.get("error"):
        _assert_cap_engaged(row, max_tokens)
    return row


def run_output_cap_test(
    adapter,
    prompts: list[Prompt],
    model: str,
    *,
    run_id: str,
    cap_gbp: float,
    max_tokens: int = DEFAULT_OUTPUT_CAP,
    force_new_attempt: bool = False,
    db_path: Path = DB_PATH,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    """Run the output-cap lever across a list of prompts. Each prompt is one
    call (vs caching's three) so this is a thin wrapper over a worker pool;
    OutputCapEngagementError raised in any worker propagates."""
    n = len(prompts)
    n_workers = concurrency if concurrency is not None else _concurrency()
    results: list[dict[str, Any] | None] = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {
            ex.submit(
                run_output_cap_for_prompt, adapter, p, model,
                run_id=run_id, cap_gbp=cap_gbp, completed=i, planned=n,
                max_tokens=max_tokens,
                force_new_attempt=force_new_attempt,
                db_path=db_path,
            ): i
            for i, p in enumerate(prompts)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
    return [r for r in results if r is not None]


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
    """Layer 2 smoke driver. Args: <cap_gbp> [prompt_id] [model] [max_tokens] [--force].
    Defaults: cs-001 on claude-sonnet-4-6 with max_tokens=200.

    Runs baseline first (skip-if-exists) then output_cap@<max_tokens> on the
    same prompt. Prints output_tokens, cost, latency, stop_reason, and a
    JSON-validity check on the response_text so truncation is verifiable
    end-to-end.
    """
    import json as _json

    load_dotenv(REPO_ROOT / ".env")

    positional: list[str] = []
    force = False
    for arg in sys.argv[1:]:
        if arg == "--force":
            force = True
        else:
            positional.append(arg)
    cap = float(positional[0]) if len(positional) > 0 else 5.0
    pid = positional[1] if len(positional) > 1 else "cs-001"
    model = positional[2] if len(positional) > 2 else "claude-sonnet-4-6"
    max_tokens_cap = int(positional[3]) if len(positional) > 3 else DEFAULT_OUTPUT_CAP

    prefix_to_file = {
        "cs-":  "customer_support.json",
        "ext-": "extraction.json",
        "rag-": "rag_qa.json",
        "sum-": "summarisation.json",
        "rea-": "reasoning.json",
    }
    fname = next((f for pre, f in prefix_to_file.items() if pid.startswith(pre)), None)
    if fname is None:
        raise SystemExit(f"unknown prompt_id prefix: {pid!r}")
    prompts = load_prompts(REPO_ROOT / "prompts" / fname)
    target = next((p for p in prompts if p.prompt_id == pid), None)
    if target is None:
        raise SystemExit(f"prompt {pid!r} not found in {fname}")

    adapter = _adapter_for_model(model)
    rid = start_run(cost_cap_gbp=cap)
    print(
        f"run_id={rid} cap=£{cap:.4f} prompt={pid} model={model} "
        f"max_tokens={max_tokens_cap} force={force}"
    )

    started = time.perf_counter()
    baseline = _base.run_one(
        adapter, target, model, lever="baseline",
        run_id=rid, cap_gbp=cap, completed=0, planned=2,
        force_new_attempt=force,
    )
    capped = run_output_cap_for_prompt(
        adapter, target, model,
        run_id=rid, cap_gbp=cap, completed=1, planned=2,
        max_tokens=max_tokens_cap,
        force_new_attempt=force,
    )
    wall_ms = int((time.perf_counter() - started) * 1000)

    def _tag(r: dict[str, Any]) -> str:
        if r.get("skipped"):
            return "SKIP"
        if r.get("error"):
            return "ERR "
        return "RAN "

    def _json_parse_status(text: str) -> str:
        try:
            _json.loads(text)
            return "valid"
        except _json.JSONDecodeError as e:
            return f"INVALID ({type(e).__name__}: {e.msg})"

    base_cost_g = usd_to_gbp(baseline.get("cost_usd") or 0.0)
    cap_cost_g  = usd_to_gbp(capped.get("cost_usd")  or 0.0)
    base_out = baseline.get("output_tokens", 0)
    cap_out  = capped.get("output_tokens", 0)
    base_lat = baseline.get("latency_ms", 0)
    cap_lat  = capped.get("latency_ms", 0)
    base_stop = baseline.get("stop_reason")
    cap_stop  = capped.get("stop_reason")

    print(
        f"  [{_tag(baseline)}] baseline             "
        f"output_tokens={base_out:4d}  cost_gbp=£{base_cost_g:.6f}  "
        f"latency={base_lat}ms  stop_reason={base_stop!r}"
    )
    print(
        f"  [{_tag(capped)}]  output_cap@{max_tokens_cap:<4d}    "
        f"output_tokens={cap_out:4d}  cost_gbp=£{cap_cost_g:.6f}  "
        f"latency={cap_lat}ms  stop_reason={cap_stop!r}"
    )
    if base_cost_g > 0:
        print(
            f"  cost_mult={cap_cost_g / base_cost_g:.3f}×  "
            f"output_mult={cap_out / max(base_out, 1):.3f}×  "
            f"wall={wall_ms}ms"
        )
    print(
        f"  baseline JSON parse:    {_json_parse_status(baseline.get('response_text', ''))}\n"
        f"  output_cap JSON parse:  {_json_parse_status(capped.get('response_text', ''))}"
    )
    print(
        f"  optimisation_lever={capped.get('optimisation_lever')!r}  "
        f"optimisation_config={capped.get('optimisation_config')!r}  "
        f"cap_engaged={cap_out <= max_tokens_cap}"
    )
    print("\n  --- baseline response_text ---")
    print(f"  {baseline.get('response_text', '')!r}")
    print("\n  --- output_cap response_text ---")
    print(f"  {capped.get('response_text', '')!r}")
