"""Batch lever (Day 6).

Submits a list of prompts to a provider's batch API in one request. Provider
batch APIs offer ~50% discount on token costs in exchange for asynchronous
processing (1–24h SLA). The lever splits into two phases:

  Day 7: submit_batch — submits N prompts as a single batch, writes one
                        batch_jobs row with status='submitted', batch_id
                        populated, prompt_ids JSON correct. Returns the
                        batch_id within seconds; does NOT wait for results.

  Day 8: retrieve_batch — separate function (not built today; lands with the
                          orchestrator integration). Polls batch_jobs for
                          in-flight batches and pulls completed results into
                          the results table.

Per testing_strategy.md Layer 2, submit-only is the smoke; retrieval is
exercised end-to-end by the Layer 4 dry-run.

Idempotency: the (run_id, provider, model, lever) tuple skips re-submission
if a batch_jobs row already exists. Critical for surviving script restarts
during the provider-side queue — re-submitting would forfeit the batch
discount on the still-processing batch and double-charge.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import anthropic
import openai
from dotenv import load_dotenv

T = TypeVar("T")

# Transient batch-submission errors worth retrying. 5xx (incl. Cloudflare 504
# from origin overload — the failure mode that crashed Day 7's first attempt)
# and connection errors (DNS / TLS / TCP transients). Excludes 4xx (bad
# request, auth, etc.) which are non-recoverable from our side.
_TRANSIENT_SUBMIT_ERRORS: tuple = (
    anthropic.InternalServerError, anthropic.APIConnectionError,
    openai.InternalServerError, openai.APIConnectionError,
)


def _retry_batch_submit(
    call_fn: Callable[[], T], *,
    max_retries: int = 3,
    default_delay_s: float = 120.0,
) -> T:
    """Run a batch-submission call, retrying on transient 5xx / connection
    errors. Honours Cloudflare's `retry_after` body hint when present;
    otherwise sleeps `default_delay_s` between attempts. Max 3 retries
    (worst-case ~6 minutes wall before exhausting). On exhaustion, raises
    the last transient exception so the orchestrator can decide what to do
    (currently bubbles up to the operator).

    Used to wrap `client.messages.batches.create` (Anthropic) and
    `client.files.create` + `client.batches.create` (OpenAI) in
    `_submit_anthropic_batch` / `_submit_openai_batch`. Same robustness
    pattern as `retrieve_batches`'s per-poll catch — catches OpenAI's batch
    queue overload at submit time so a single transient gateway timeout
    doesn't crash the whole Day 7 sweep mid-batch_submit phase."""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except _TRANSIENT_SUBMIT_ERRORS as e:
            last_err = e
            if attempt == max_retries:
                break
            from runners._base import retry_after_from_error_body
            delay = retry_after_from_error_body(e, default=default_delay_s)
            time.sleep(delay)
    assert last_err is not None
    raise last_err

from runners import run_anthropic, run_openai
from runners._base import (
    DB_PATH,
    DEFAULT_MAX_TOKENS,
    REPO_ROOT,
    _now_iso,
    start_run,
)
from runners.schema import Prompt, load_prompts


# ---- provider-specific submission ----

def _submit_anthropic_batch(
    client: anthropic.Anthropic,
    prompts: list[Prompt],
    model: str,
    max_tokens: int,
) -> str:
    """Submit prompts to Anthropic's Message Batches API. Returns batch_id.

    Each prompt is one request with custom_id=prompt.prompt_id; that custom_id
    is what links per-prompt results back at retrieval time.

    Wrapped in `_retry_batch_submit` so transient 5xx / connection errors
    (incl. Cloudflare 504 origin overload) trigger up to 3 retries with the
    Cloudflare-provided retry_after hint or a 120s default backoff.
    """
    requests = [
        {
            "custom_id": p.prompt_id,
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": p.input.system,
                "messages": [{"role": "user", "content": p.input.user}],
            },
        }
        for p in prompts
    ]

    def _do_create() -> Any:
        return client.messages.batches.create(requests=requests)

    batch = _retry_batch_submit(_do_create)
    if not batch.id:
        raise RuntimeError(
            f"Anthropic batch submission returned empty id; batch object = {batch!r}"
        )
    return batch.id


def _submit_openai_batch(
    client: openai.OpenAI,
    prompts: list[Prompt],
    model: str,
    max_tokens: int,
) -> str:
    """Submit prompts to OpenAI's Batch API. Returns batch_id.

    OpenAI's batch flow is two-step: upload a JSONL file (purpose='batch'),
    then create a batch from the file. The custom_id field on each line links
    per-prompt results back at retrieval time.
    """
    # Mirror the sync-call decision: temperature=0 across all models, and
    # let reasoning_effort default to API default ('medium' for GPT-5.4
    # reasoning models). See run_openai.call_openai for rationale.
    lines = [
        json.dumps({
            "custom_id": p.prompt_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                # GPT-5.x family requires `max_completion_tokens`; `max_tokens`
                # is rejected with a 400 BadRequest. Same rename as in run_openai.
                "max_completion_tokens": max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": p.input.system},
                    {"role": "user", "content": p.input.user},
                ],
            },
        })
        for p in prompts
    ]
    payload = ("\n".join(lines)).encode("utf-8")

    # Both file upload and batch creation can transient 5xx (Cloudflare 504
    # caught Day 7 attempt 1 on batches.create). Each wrapped in retry; on
    # exhaustion the last error bubbles up. File upload retries build a fresh
    # BytesIO each attempt because the previous read may have advanced position.
    def _do_upload() -> Any:
        file_obj = io.BytesIO(payload)
        file_obj.name = "batch.jsonl"  # OpenAI requires a filename on upload
        return client.files.create(file=file_obj, purpose="batch")

    uploaded = _retry_batch_submit(_do_upload)

    def _do_create() -> Any:
        return client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    batch = _retry_batch_submit(_do_create)
    if not batch.id:
        raise RuntimeError(
            f"OpenAI batch submission returned empty id; batch object = {batch!r}"
        )
    return batch.id


# ---- adapter dispatch ----

_PROVIDER_FOR_MODEL = {
    "claude-sonnet-4-6":       "anthropic",
    "claude-haiku-4-5":        "anthropic",
    "claude-opus-4-6":         "anthropic",
    "gpt-5.4":                 "openai",
    "gpt-5.4-2026-03-05":      "openai",
    "gpt-5.4-mini":            "openai",
    "gpt-5.4-mini-2026-03-17": "openai",
}

# Each provider entry: (submit_function, sync_adapter_for_make_client)
_BATCH_SUBMITTERS = {
    "anthropic": (_submit_anthropic_batch, run_anthropic.ANTHROPIC_ADAPTER),
    "openai":    (_submit_openai_batch,    run_openai.OPENAI_ADAPTER),
}


def _provider_for_model(model: str) -> str:
    p = _PROVIDER_FOR_MODEL.get(model)
    if p is None:
        raise ValueError(f"unknown model {model!r}")
    return p


# ---- batch_jobs DB helpers ----

def _existing_batch_job(
    db_path: Path, run_id: str, provider: str, model: str, lever: str,
) -> dict[str, Any] | None:
    """Return the most recent batch_jobs row for this (run_id, provider, model,
    lever) combo whose status is one we should NOT re-submit over: an in-flight
    or successfully-completed batch. Cancelled / failed / expired rows are
    explicitly skipped here so the caller can re-submit a replacement (the
    audit row stays in DB; the new submission gets its own row)."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """SELECT * FROM batch_jobs
               WHERE run_id = ? AND provider = ? AND model = ? AND lever = ?
                 AND status IN ('submitted', 'in_progress', 'completed')
               ORDER BY submitted_at DESC
               LIMIT 1""",
            (run_id, provider, model, lever),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))


def _insert_batch_job(db_path: Path, row: dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT INTO batch_jobs ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
        conn.commit()


# ---- public submission API ----

def submit_batch(
    prompts: list[Prompt],
    model: str,
    *,
    run_id: str,
    lever: str = "batch",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    db_path: Path = DB_PATH,
    client: Any = None,
) -> dict[str, Any]:
    """Submit prompts to the provider's batch API. Returns within seconds —
    does NOT wait for batch completion (1–24h SLA).

    Default lever='batch' (NOT 'baseline'). Batch processing is a distinct
    lever in PRD §5's lever availability matrix; result rows produced by
    retrieving this batch are tagged optimisation_lever='batch' so they
    coexist with sync baseline rows for the same (prompt_id, model) pair
    under the schema's UNIQUE(prompt_id, model, lever, config_hash, attempt)
    constraint. Day 12 analysis computes batch's cost ratio as
    cost(batch) / cost(baseline) per (prompt, model).

    Returns a dict shaped like a batch_jobs row, plus a `skipped` flag.
    Idempotency: if a batch_jobs row exists for (run_id, provider, model,
    lever), returns the existing row marked skipped=True without re-submitting.
    """
    provider = _provider_for_model(model)

    existing = _existing_batch_job(db_path, run_id, provider, model, lever)
    if existing is not None:
        return {**existing, "skipped": True}

    submit_fn, adapter = _BATCH_SUBMITTERS[provider]
    if client is None:
        client = adapter.make_client()

    batch_id = submit_fn(client, prompts, model, max_tokens)

    row = {
        "batch_id": batch_id,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "lever": lever,
        "status": "submitted",
        "submitted_at": _now_iso(),
        "retrieved_at": None,
        "completed_at": None,
        "prompt_ids": json.dumps([p.prompt_id for p in prompts]),
        "request_count": len(prompts),
        "error": None,
    }
    _insert_batch_job(db_path, row)
    return {**row, "skipped": False}


# ---- smoke driver ----

if __name__ == "__main__":
    """Layer 2 smoke driver. Args: <cap_gbp> [model] [n_prompts].
    Defaults: cap=£5, model=claude-sonnet-4-6, n=2.

    Submits a batch of `n_prompts` summarisation prompts, prints the
    batch_id and the persisted batch_jobs row. Does NOT poll for results.
    """
    load_dotenv(REPO_ROOT / ".env")
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    model = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-4-6"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    prompts = load_prompts(REPO_ROOT / "prompts" / "summarisation.json")
    targets = prompts[:n]

    rid = start_run(cost_cap_gbp=cap)
    print(
        f"run_id={rid} cap=£{cap:.4f} model={model} "
        f"n_prompts={n} prompts={[p.prompt_id for p in targets]}"
    )

    started = time.perf_counter()
    result = submit_batch(targets, model=model, run_id=rid)
    wall_ms = int((time.perf_counter() - started) * 1000)

    tag = "SKIP" if result.get("skipped") else "RAN "
    print(
        f"  [{tag}] batch_id={result['batch_id']!r}  status={result['status']!r}  "
        f"request_count={result['request_count']}  wall={wall_ms}ms"
    )
    print(
        f"  provider={result['provider']!r}  model={result['model']!r}  "
        f"lever={result['lever']!r}"
    )
    print(
        f"  submitted_at={result['submitted_at']!r}  "
        f"prompt_ids={result['prompt_ids']!r}"
    )

    print("\n  --- DB row (re-read for verification) ---")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT * FROM batch_jobs WHERE batch_id = ?", (result["batch_id"],)
        )
        cols = [d[0] for d in cur.description]
        db_row = dict(zip(cols, cur.fetchone()))
    for k, v in db_row.items():
        print(f"    {k}: {v!r}")
