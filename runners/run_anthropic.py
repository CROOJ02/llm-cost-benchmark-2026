"""Anthropic runner.

Step A: basic call + result row.
Step B: cost-cap enforcement (GBP-canonical, with pre-call check per PRD §10).
Step C: per-category output-token estimates, accurate input-token counting via
  count_tokens API (char-count fallback), skip-if-exists logic on
  (prompt_id, model, lever, config_hash, run_attempt) per PRD §10, and
  force-new-attempt support.
Step D: exponential-backoff retry on 429s with Retry-After honoured; on
  retry exhaustion, log an error row (no cost accumulator update).
Step E: ThreadPoolExecutor concurrency via INFEROPS_CONCURRENCY env var
  (default 4); cost reservation pattern (estimate-then-reconcile) gated by
  a process-wide lock so the cap stays accurate under parallel calls.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from runners.budget import (
    CostCapExceeded,
    check_cap,
    estimate_cost_gbp,
    estimate_cost_usd,
    output_estimate_for,
    usd_to_gbp,
)
from runners.schema import Prompt, load_prompts

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "results.db"
PROVIDER = "anthropic"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_CONCURRENCY = 4

# In-process cache for input-token counts. Concurrent dict reads/writes are
# OK in CPython for our purposes — worst case is a duplicate count_tokens call.
_input_token_cache: dict[tuple[str, str], int] = {}

# Process-wide lock for the cap-check + reservation sequence. Ensures that
# under ThreadPoolExecutor, parallel calls cannot both pass the cap check
# while reading the same pre-update cost_so_far value.
_cost_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(lever: str, config: dict[str, Any] | None) -> str:
    payload = json.dumps({"lever": lever, "config": config or {}}, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _concurrency() -> int:
    return int(os.environ.get("INFEROPS_CONCURRENCY", str(DEFAULT_CONCURRENCY)))


def count_input_tokens(client: anthropic.Anthropic, prompt: Prompt, model: str) -> int:
    """Count input tokens for the cap pre-check.

    Primary: client.beta.messages.count_tokens (free, server-side, exact
    tokenizer). Fallback: char count at ~4 chars/token (English average).
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
    """Wraps call_anthropic with exponential-backoff retry on 429s."""
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


def _read_cost_so_far_gbp(conn: sqlite3.Connection, run_id: str) -> float:
    row = conn.execute("SELECT cost_so_far_gbp FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run_id {run_id!r} not found in runs table")
    return float(row[0] or 0.0)


def _check_and_reserve(
    *, run_id: str, estimated_gbp: float, cap_gbp: float,
    completed: int, planned: int, db_path: Path,
) -> None:
    """Atomically check the cap and reserve the estimate against runs.cost_so_far_gbp.

    Held under _cost_lock so parallel callers can't both pass the check while
    reading the same pre-update cost_so_far value. After the call completes,
    the caller MUST call either _reconcile_actual (success) or
    _release_reservation (failure) to correct the temporary inflation.
    """
    with _cost_lock:
        with sqlite3.connect(db_path) as conn:
            current = _read_cost_so_far_gbp(conn, run_id)
            check_cap(
                cost_so_far_gbp=current,
                estimated_call_gbp_value=estimated_gbp,
                cap_gbp=cap_gbp,
                completed=completed,
                planned=planned,
            )
            conn.execute(
                "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp + ? WHERE run_id = ?",
                (estimated_gbp, run_id),
            )
            conn.commit()


def _reconcile_actual(
    *, run_id: str, estimated_gbp: float, actual_gbp: float, db_path: Path,
) -> None:
    """Replace the reservation with the actual cost (delta = actual − estimate)."""
    delta = actual_gbp - estimated_gbp
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp + ? WHERE run_id = ?",
            (delta, run_id),
        )
        conn.commit()


def _release_reservation(*, run_id: str, estimated_gbp: float, db_path: Path) -> None:
    """Release the reservation when the API call ultimately fails (no actual cost incurred)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp - ? WHERE run_id = ?",
            (estimated_gbp, run_id),
        )
        conn.commit()


def _existing_successful_row(
    conn: sqlite3.Connection,
    prompt_id: str, model: str, lever: str, config_hash: str, run_attempt: int,
) -> dict[str, Any] | None:
    cur = conn.execute(
        """SELECT * FROM results
           WHERE prompt_id = ? AND model = ? AND optimisation_lever = ?
             AND config_hash = ? AND run_attempt = ? AND error IS NULL
           LIMIT 1""",
        (prompt_id, model, lever, config_hash, run_attempt),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], row))


def _next_run_attempt(
    conn: sqlite3.Connection,
    prompt_id: str, model: str, lever: str, config_hash: str,
) -> int:
    row = conn.execute(
        """SELECT MAX(run_attempt) FROM results
           WHERE prompt_id = ? AND model = ? AND optimisation_lever = ? AND config_hash = ?""",
        (prompt_id, model, lever, config_hash),
    ).fetchone()
    return int(row[0] or 0) + 1


def _new_row(
    *, prompt: Prompt, model: str, lever: str, config_hash: str,
    optimisation_config: dict[str, Any] | None, run_id: str, run_attempt: int,
) -> dict[str, Any]:
    return {
        "result_id": str(uuid.uuid4()),
        "run_id": run_id,
        "timestamp": _now_iso(),
        "run_attempt": run_attempt,
        "prompt_id": prompt.prompt_id,
        "task_category": prompt.task_category,
        "complexity": prompt.complexity,
        "model": model,
        "provider": PROVIDER,
        "optimisation_lever": lever,
        "optimisation_config": json.dumps(optimisation_config) if optimisation_config else None,
        "config_hash": config_hash,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "latency_ms": 0,
        "cost_usd": 0.0,
        "response_text": "",
        "response_parsed": None,
        "output_format_valid": 1,
        "rubric_score": None,
        "judge_a_score": None,
        "judge_b_score": None,
        "judge_disagreement_flag": 0,
        "human_score": None,
        "final_score": None,
        "score_recomputed_at": None,
        "model_version": None,
        "temperature": 0.0,
        "error": None,
    }


def _insert_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    conn.execute(f"INSERT INTO results ({cols}) VALUES ({placeholders})", list(row.values()))


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
    """Run one (prompt × model × lever), persist a result row, return the row."""
    client = client or anthropic.Anthropic(max_retries=0)
    config_hash = _config_hash(lever, optimisation_config)

    if force_new_attempt:
        with sqlite3.connect(db_path) as conn:
            run_attempt = _next_run_attempt(conn, prompt.prompt_id, model, lever, config_hash)
    else:
        run_attempt = 1
        with sqlite3.connect(db_path) as conn:
            existing = _existing_successful_row(
                conn, prompt.prompt_id, model, lever, config_hash, run_attempt
            )
        if existing is not None:
            return {**existing, "skipped": True}

    input_tokens_est = count_input_tokens(client, prompt, model)
    output_tokens_est = output_estimate_for(prompt.task_category)
    est_gbp = estimate_cost_gbp(model, input_tokens_est, output_tokens_est)

    _check_and_reserve(
        run_id=run_id, estimated_gbp=est_gbp, cap_gbp=cap_gbp,
        completed=completed, planned=planned, db_path=db_path,
    )

    row = _new_row(
        prompt=prompt, model=model, lever=lever, config_hash=config_hash,
        optimisation_config=optimisation_config, run_id=run_id, run_attempt=run_attempt,
    )

    try:
        m = call_anthropic_with_retry(
            client, prompt, model, max_tokens=max_tokens,
            max_retries=max_retries, base_delay=base_delay,
        )
    except anthropic.RateLimitError as e:
        _release_reservation(run_id=run_id, estimated_gbp=est_gbp, db_path=db_path)
        row["error"] = f"RateLimitError after {max_retries + 1} attempts: {e}"
        row["output_format_valid"] = 0
        with sqlite3.connect(db_path) as conn:
            _insert_row(conn, row)
            conn.commit()
        return {**row, "skipped": False}

    cost_usd = estimate_cost_usd(model, m["input_tokens"], m["output_tokens"], m["cached_tokens"])
    cost_gbp = usd_to_gbp(cost_usd)
    _reconcile_actual(
        run_id=run_id, estimated_gbp=est_gbp, actual_gbp=cost_gbp, db_path=db_path,
    )
    row.update({
        "input_tokens": m["input_tokens"],
        "output_tokens": m["output_tokens"],
        "cached_tokens": m["cached_tokens"],
        "latency_ms": m["latency_ms"],
        "cost_usd": cost_usd,
        "response_text": m["response_text"],
        "model_version": m["model_version"],
    })

    with sqlite3.connect(db_path) as conn:
        _insert_row(conn, row)
        conn.commit()
    return {**row, "skipped": False}


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
    """Run a batch of prompts in parallel via ThreadPoolExecutor.

    Worker count from `concurrency` arg, else INFEROPS_CONCURRENCY env var,
    else DEFAULT_CONCURRENCY (4). Cap enforcement is process-wide thread-safe
    via _cost_lock + DB reservations. Results are returned in input order.
    """
    n_workers = concurrency if concurrency is not None else _concurrency()
    n = len(prompts)
    results: list[dict[str, Any] | None] = [None] * n
    aborted = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {
            ex.submit(
                run_one, p, model, lever,
                run_id=run_id, cap_gbp=cap_gbp, completed=i, planned=n,
                optimisation_config=optimisation_config,
                force_new_attempt=force_new_attempt,
                max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay,
                db_path=db_path,
            ): i
            for i, p in enumerate(prompts)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = fut.result()
            except CostCapExceeded as e:
                aborted = True
                results[i] = {
                    "prompt_id": prompts[i].prompt_id,
                    "skipped": False,
                    "aborted": True,
                    "error": str(e),
                }
    if aborted:
        mark_run_aborted_cost(run_id, db_path)
    return [r for r in results if r is not None]


def start_run(cost_cap_gbp: float = 300.0, db_path: Path = DB_PATH) -> str:
    run_id = f"run-{uuid.uuid4()}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO runs (run_id, started_at, cost_cap_gbp, status) VALUES (?, ?, ?, ?)",
            (run_id, _now_iso(), cost_cap_gbp, "running"),
        )
        conn.commit()
    return run_id


def mark_run_aborted_cost(run_id: str, db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status = 'aborted_cost', completed_at = ? WHERE run_id = ?",
            (_now_iso(), run_id),
        )
        conn.commit()


if __name__ == "__main__":
    """Step E test driver. Args: <cap_gbp> <n_prompts> [offset] [--force].

    Reads INFEROPS_CONCURRENCY env var (default 4) for worker count.

    Examples:
      INFEROPS_CONCURRENCY=1 python -m runners.run_anthropic 300 5 2          # serial cs-003..007
      INFEROPS_CONCURRENCY=4 python -m runners.run_anthropic 300 5 2 --force  # parallel, force new attempt
    """
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
