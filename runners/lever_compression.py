"""Compression lever (Day 6).

Runtime LLMLingua-2 prompt compression. Compresses `prompt.input.user` before
the API call (system prompt is preserved un-compressed — it carries task
instructions that risk degrading if rewritten). The compressed user text
replaces the original in a synthetic `Prompt` and the call proceeds via the
provider-agnostic `_base.run_one`.

Both `original_input_tokens` and `compressed_input_tokens` recorded in
`optimisation_config` are measured by Anthropic's `count_tokens` API — NOT
by LLMLingua-2's BERT tokenizer counts. This is a binding requirement (see
docs/methodology/prompt_design_decisions.md § "Compression timing"): the
compression ratio Day 12 analyses must reflect the billable token count, not
LLMLingua-2's internal tokenization which differs from the model's.

LLMLingua-2 model load is ~8.6s; cached at module level via a lock-protected
singleton so the cost is paid once per process. Per-prompt warm-path
compression is ~1.6s on Mac CPU (per Day 5 measurements on sum-001/sum-020).
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from runners import _base, run_anthropic, run_openai
from runners._base import (
    DB_PATH,
    REPO_ROOT,
    _concurrency,
    start_run,
)
from runners.budget import usd_to_gbp
from runners.schema import Prompt, PromptInput, load_prompts


# Day 5 timing measurements (sum-001 cold 1.50s / sum-020 warm 1.59s, init 8.6s) were
# against the bert-base-multilingual variant; pinning to the same model keeps the
# methodology doc's numbers reproducible. The xlm-roberta-large variant is ~2× slower
# at inference and ~2× larger (1.4GB) — switching to it would invalidate the doc.
_LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
_LLMLINGUA_DEVICE = "cpu"  # Day 5 timing measurements were on Mac CPU; pin for reproducibility.
DEFAULT_RATE = 0.5

_compressor: Any = None
_compressor_lock = threading.Lock()


def _get_compressor():
    """Lazy singleton — model load is ~8.6s, paid once per process."""
    global _compressor
    if _compressor is None:
        with _compressor_lock:
            if _compressor is None:
                from llmlingua import PromptCompressor
                _compressor = PromptCompressor(
                    model_name=_LLMLINGUA_MODEL,
                    device_map=_LLMLINGUA_DEVICE,
                    use_llmlingua2=True,
                )
    return _compressor


_DEFAULT_COUNT_REFERENCE_MODEL = "claude-sonnet-4-6"


def _reference_model_for_count(model: str) -> str:
    """Which Anthropic model to pass to count_tokens. For Anthropic call models
    use the actual model so the count matches what's billed. For non-Anthropic
    call models (OpenAI), fall back to a fixed Anthropic reference model so
    the cross-provider compression ratio remains uniformly measured (the
    binding requirement is Anthropic's count_tokens, irrespective of which
    provider runs the actual call)."""
    if model.startswith("claude-"):
        return model
    return _DEFAULT_COUNT_REFERENCE_MODEL


def _count_tokens_direct(
    client: anthropic.Anthropic, system: str, user: str, model: str,
) -> int:
    """Anthropic `count_tokens`, bypassing the per-(prompt_id, model) cache in
    run_anthropic. Required here because compression mutates user content under
    the same prompt_id — the cache would otherwise return the stale original
    count for the compressed variant."""
    result = client.beta.messages.count_tokens(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return int(result.input_tokens)


class CompressionEngagementError(AssertionError):
    """Reserved for unrecoverable compression failures (e.g. negative token
    counts, malformed compressor output). The no-reduction case (compressed
    >= original) is NOT raised — it's recorded as a `compression_unavailable`
    row in `results` instead, mirroring caching's `caching_available=False`
    pattern. Day 6 finding: short or structured prompts (extraction schemas,
    code) routinely re-tokenise larger in Anthropic's tokenizer than they
    started, even when LLMLingua-2 reduces them in BERT counts. Treating
    those as a fatal error stalled the dry-run on prompt ext-008. Treating
    them as data points lets Day 12 analyse compression effectiveness across
    prompt types instead."""


def compress_user_text(text: str, rate: float = DEFAULT_RATE) -> dict[str, Any]:
    """Compress one user-message string via LLMLingua-2. Returns the
    compressed string plus LLMLingua-2's BERT-based metadata (NOT used for
    cost accounting — kept only for diagnostic/timing purposes)."""
    compressor = _get_compressor()
    started = time.perf_counter()
    result = compressor.compress_prompt([text], rate=rate, force_tokens=["\n"])
    compress_ms = int((time.perf_counter() - started) * 1000)
    return {
        "compressed_text": result["compressed_prompt"],
        "compress_ms": compress_ms,
        "llmlingua_origin_tokens": result.get("origin_tokens"),
        "llmlingua_compressed_tokens": result.get("compressed_tokens"),
        "llmlingua_rate_actual": result.get("rate"),
    }


def _insert_compression_unavailable_row(
    *,
    adapter,
    prompt: Prompt,
    model: str,
    run_id: str,
    rate: float,
    original_tokens: int,
    compressed_tokens: int,
    db_path: Path,
    force_new_attempt: bool,
) -> dict[str, Any]:
    """Persist a `compression_unavailable` data point as a result row when
    LLMLingua-2 produces no Anthropic-counted reduction. Mirrors caching's
    `caching_available=False` pattern at the lever-return level, but with
    a distinct shape: caching returns metadata only (no row inserted),
    whereas compression PERSISTS a row so Day 12 analysis can `JOIN` and
    distinguish "didn't try compression" from "tried, tokeniser asymmetry
    made it unavailable."

    The row carries the unavailability evidence (`compression_status`,
    `original_input_tokens`, `compressed_input_tokens`, `skip_reason`) in
    `optimisation_config` and has zero cost / empty response_text — it's a
    measurement of unavailability, not a failed call.
    """
    skip_reason = (
        f"no Anthropic-counted reduction at rate={rate}: "
        f"{compressed_tokens} >= {original_tokens} "
        f"(compressed/original = {compressed_tokens / max(original_tokens, 1):.3f}×)"
    )
    optimisation_config = {
        "compression_status": "unavailable",
        "original_input_tokens": original_tokens,
        "compressed_input_tokens": compressed_tokens,
        "compression_ratio_anthropic": round(compressed_tokens / max(original_tokens, 1), 4),
        "llmlingua_rate": rate,
        "llmlingua_model": _LLMLINGUA_MODEL,
        "skip_reason": skip_reason,
    }
    optimisation_config = run_openai.annotate_optimisation_config_for_reasoning_effort(
        optimisation_config, model,
    )
    config_hash = _base._config_hash("compression", optimisation_config)
    run_attempt = 1
    if force_new_attempt:
        with sqlite3.connect(db_path) as conn:
            run_attempt = _base._next_run_attempt(
                conn, prompt.prompt_id, model, "compression", config_hash,
            )
    else:
        with sqlite3.connect(db_path) as conn:
            existing = _base._existing_successful_row(
                conn, prompt.prompt_id, model, "compression", config_hash, run_attempt,
                run_id,
            )
        if existing is not None:
            return {
                **existing, "skipped": True,
                "compression_available": False, "skip_reason": skip_reason,
            }

    row = _base._new_row(
        prompt=prompt, model=model, provider=adapter.name, lever="compression",
        config_hash=config_hash, optimisation_config=optimisation_config,
        run_id=run_id, run_attempt=run_attempt,
    )
    # All numeric fields stay 0 (no API call → no tokens billed, no latency).
    # response_text='' is fine; output_format_valid stays at default 1.
    with sqlite3.connect(db_path) as conn:
        _base._insert_row(conn, row)
        conn.commit()
    return {
        **row, "skipped": False,
        "compression_available": False, "skip_reason": skip_reason,
        "original_input_tokens": original_tokens,
        "compressed_input_tokens": compressed_tokens,
    }


def run_compression_for_prompt(
    adapter,
    prompt: Prompt,
    model: str,
    *,
    run_id: str,
    cap_gbp: float,
    completed: int,
    planned: int,
    rate: float = DEFAULT_RATE,
    force_new_attempt: bool = False,
    db_path: Path = DB_PATH,
    client: Any = None,
) -> dict[str, Any]:
    """Compress prompt.input.user via LLMLingua-2, then call the model with the
    compressed prompt. The system prompt is left intact.

    Records both `original_input_tokens` and `compressed_input_tokens` in
    `optimisation_config`, both measured by Anthropic's count_tokens API
    (binding requirement — not LLMLingua-2's BERT counts).

    If LLMLingua-2 produces no Anthropic-counted reduction (compressed >=
    original — typically because the prompt is short or structured), a
    `compression_unavailable` row is inserted instead of running the API
    call. Returns a result dict with `compression_available: False` in
    that case. See `_insert_compression_unavailable_row` for the row shape.
    """
    if client is None:
        # Anthropic client used here for the count_tokens calls regardless of
        # the call adapter. For OpenAI runs, the count is still useful as a
        # billable-token estimate; cross-provider count comparison is a known
        # caveat documented in the methodology doc.
        anth_client = anthropic.Anthropic()
        client = adapter.make_client()
    else:
        anth_client = anthropic.Anthropic()

    cresult = compress_user_text(prompt.input.user, rate=rate)
    compressed_text = cresult["compressed_text"]

    # Anthropic-counted tokens — NOT LLMLingua-2's BERT counts (binding requirement).
    ref_model = _reference_model_for_count(model)
    original_tokens = _count_tokens_direct(
        anth_client, prompt.input.system, prompt.input.user, model=ref_model,
    )
    compressed_tokens = _count_tokens_direct(
        anth_client, prompt.input.system, compressed_text, model=ref_model,
    )

    if compressed_tokens >= original_tokens:
        # No reduction in Anthropic's counts — record as unavailable, no API call.
        result = _insert_compression_unavailable_row(
            adapter=adapter, prompt=prompt, model=model, run_id=run_id,
            rate=rate, original_tokens=original_tokens,
            compressed_tokens=compressed_tokens, db_path=db_path,
            force_new_attempt=force_new_attempt,
        )
        result["compression_metadata"] = cresult
        return result

    compressed_input = PromptInput(system=prompt.input.system, user=compressed_text)
    compressed_prompt = prompt.model_copy(update={"input": compressed_input})

    optimisation_config = {
        "original_input_tokens":      original_tokens,
        "compressed_input_tokens":    compressed_tokens,
        "compression_ratio_anthropic": round(compressed_tokens / max(original_tokens, 1), 4),
        "llmlingua_rate":             rate,
        "llmlingua_model":            _LLMLINGUA_MODEL,
    }
    optimisation_config = run_openai.annotate_optimisation_config_for_reasoning_effort(
        optimisation_config, model,
    )

    row = _base.run_one(
        adapter, compressed_prompt, model, lever="compression",
        run_id=run_id, cap_gbp=cap_gbp, completed=completed, planned=planned,
        optimisation_config=optimisation_config,
        force_new_attempt=force_new_attempt,
        db_path=db_path, client=client,
    )
    return {
        **row,
        "compression_metadata": cresult,
        "compression_available": True,
        "original_input_tokens": original_tokens,
        "compressed_input_tokens": compressed_tokens,
    }


def run_compression_test(
    adapter,
    prompts: list[Prompt],
    model: str,
    *,
    run_id: str,
    cap_gbp: float,
    rate: float = DEFAULT_RATE,
    force_new_attempt: bool = False,
    db_path: Path = DB_PATH,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    """Run the compression lever across a list of prompts. Compression is
    CPU-bound (PyTorch on CPU); the GIL serialises it in practice but PyTorch
    BLAS releases the GIL during numerical work, so threading still gives
    some overlap with the I/O-bound provider call."""
    n = len(prompts)
    n_workers = concurrency if concurrency is not None else _concurrency()
    results: list[dict[str, Any] | None] = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {
            ex.submit(
                run_compression_for_prompt, adapter, p, model,
                run_id=run_id, cap_gbp=cap_gbp, completed=i, planned=n,
                rate=rate, force_new_attempt=force_new_attempt,
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
    """Layer 2 smoke driver. Args: <cap_gbp> [prompt_id] [model] [rate] [--force].
    Defaults: cap=£5, prompt=sum-015, model=claude-sonnet-4-6, rate=0.5.

    Runs baseline (skip-if-exists; existing sum-015 row is reused) then
    compression on the same prompt and prints the comparison.
    """
    load_dotenv(REPO_ROOT / ".env")

    positional: list[str] = []
    force = False
    for arg in sys.argv[1:]:
        if arg == "--force":
            force = True
        else:
            positional.append(arg)
    cap = float(positional[0]) if len(positional) > 0 else 5.0
    pid = positional[1] if len(positional) > 1 else "sum-015"
    model = positional[2] if len(positional) > 2 else "claude-sonnet-4-6"
    rate = float(positional[3]) if len(positional) > 3 else DEFAULT_RATE

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
        f"rate={rate} force={force}"
    )

    print("  loading LLMLingua-2 (cold init ~8.6s)...")
    init_started = time.perf_counter()
    _get_compressor()
    init_ms = int((time.perf_counter() - init_started) * 1000)
    print(f"  LLMLingua-2 ready in {init_ms}ms")

    started = time.perf_counter()
    baseline = _base.run_one(
        adapter, target, model, lever="baseline",
        run_id=rid, cap_gbp=cap, completed=0, planned=2,
        force_new_attempt=force,
    )
    compressed = run_compression_for_prompt(
        adapter, target, model,
        run_id=rid, cap_gbp=cap, completed=1, planned=2,
        rate=rate, force_new_attempt=force,
    )
    wall_ms = int((time.perf_counter() - started) * 1000)

    def _tag(r: dict[str, Any]) -> str:
        if r.get("skipped"):
            return "SKIP"
        if r.get("error"):
            return "ERR "
        return "RAN "

    base_cost_g = usd_to_gbp(baseline.get("cost_usd") or 0.0)
    comp_cost_g = usd_to_gbp(compressed.get("cost_usd") or 0.0)
    base_in   = baseline.get("input_tokens", 0)
    comp_in   = compressed.get("input_tokens", 0)
    base_out  = baseline.get("output_tokens", 0)
    comp_out  = compressed.get("output_tokens", 0)
    base_lat  = baseline.get("latency_ms", 0)
    comp_lat  = compressed.get("latency_ms", 0)
    cmeta = compressed.get("compression_metadata", {})

    print(
        f"  [{_tag(baseline)}] baseline    "
        f"input_tokens={base_in:5d}  output_tokens={base_out:4d}  "
        f"cost_gbp=£{base_cost_g:.6f}  latency={base_lat}ms"
    )
    print(
        f"  [{_tag(compressed)}]  compression@{rate}  "
        f"input_tokens={comp_in:5d}  output_tokens={comp_out:4d}  "
        f"cost_gbp=£{comp_cost_g:.6f}  latency={comp_lat}ms"
    )
    if base_cost_g > 0:
        print(
            f"  cost_mult={comp_cost_g / base_cost_g:.3f}×  "
            f"input_mult={comp_in / max(base_in, 1):.3f}×  "
            f"wall={wall_ms}ms  compress_ms={cmeta.get('compress_ms')}"
        )
    print(
        f"  optimisation_lever={compressed.get('optimisation_lever')!r}\n"
        f"  optimisation_config={compressed.get('optimisation_config')!r}"
    )
    print(
        f"  Anthropic-counted: original={compressed.get('original_input_tokens')}  "
        f"compressed={compressed.get('compressed_input_tokens')}  "
        f"ratio={compressed.get('compressed_input_tokens') / max(compressed.get('original_input_tokens', 1), 1):.3f}"
    )
    print(
        f"  LLMLingua-2 BERT counts: origin={cmeta.get('llmlingua_origin_tokens')}  "
        f"compressed={cmeta.get('llmlingua_compressed_tokens')}  "
        f"rate_actual={cmeta.get('llmlingua_rate_actual')}"
    )

    print("\n  --- response_text (compression run) ---")
    print(f"  {compressed.get('response_text', '')!r}")
