"""Provider-agnostic core of the runner.

The Anthropic and OpenAI provider modules are thin shims around this module:
each supplies a `ProviderAdapter` describing how to call its API and count
input tokens, and delegates to `run_one` / `run_many` here for the shared
concerns — skip-if-exists, cap reservation under concurrency, error-row
insertion on retry exhaustion, and ThreadPoolExecutor parallelism. The
retry mechanism itself stays in each provider module so that exception
classes and Retry-After parsing remain provider-specific.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from runners.budget import (
    CostCapExceeded,
    check_cap,
    estimate_cost_gbp,
    estimate_cost_usd,
    output_estimate_for,
    usd_to_gbp,
)
from runners.schema import Prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "results.db"

DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_CONCURRENCY = 4
# Default backoff (seconds) for transient 5xx / connection errors when the
# provider's response body doesn't include a Cloudflare-style `retry_after`
# hint. Aligned with `lever_batch._retry_batch_submit`'s default.
DEFAULT_TRANSIENT_5XX_DELAY = 120.0


def retry_after_from_error_body(err: Exception, default: float) -> float:
    """Extract Cloudflare's `retry_after` hint from the error response body
    when present. Cloudflare 5xx responses include a `retry_after` field in
    the JSON body (not a header). Falls back to `default` when missing or
    malformed. Provider-agnostic — works for both Anthropic and OpenAI
    InternalServerError instances. Used by run_anthropic / run_openai's
    sync-call retry wrappers and lever_batch's submit retry wrapper."""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        ra = body.get("retry_after")
        if ra is not None:
            try:
                return float(ra)
            except (TypeError, ValueError):
                pass
    return default

# Process-wide lock around (cap-check + reservation). Shared across providers
# so that mixed-provider concurrent runs (Day 7+) still serialise the cap gate.
_cost_lock = threading.Lock()


class ProviderAdapter(Protocol):
    """Contract implemented by each provider runner module."""

    name: str
    rate_limit_error: type[Exception]

    def make_client(self) -> Any: ...
    def count_input_tokens(self, client: Any, prompt: Prompt, model: str) -> int: ...
    def call_with_retry(
        self,
        client: Any, prompt: Prompt, model: str,
        max_tokens: int, max_retries: int, base_delay: float,
        *,
        optimisation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(lever: str, config: dict[str, Any] | None) -> str:
    payload = json.dumps({"lever": lever, "config": config or {}}, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _concurrency() -> int:
    return int(os.environ.get("INFEROPS_CONCURRENCY", str(DEFAULT_CONCURRENCY)))


def _read_cost_so_far_gbp(conn: sqlite3.Connection, run_id: str) -> float:
    row = conn.execute("SELECT cost_so_far_gbp FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run_id {run_id!r} not found in runs table")
    return float(row[0] or 0.0)


def _check_and_reserve(
    *, run_id: str, estimated_gbp: float, cap_gbp: float,
    completed: int, planned: int, db_path: Path,
) -> None:
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
    delta = actual_gbp - estimated_gbp
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp + ? WHERE run_id = ?",
            (delta, run_id),
        )
        conn.commit()


def _release_reservation(*, run_id: str, estimated_gbp: float, db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp - ? WHERE run_id = ?",
            (estimated_gbp, run_id),
        )
        conn.commit()


def _existing_successful_row(
    conn: sqlite3.Connection,
    prompt_id: str, model: str, lever: str, config_hash: str, run_attempt: int,
    run_id: str,
) -> dict[str, Any] | None:
    """Skip-if-exists check, scoped to a single run_id.

    Each `run_id` is a fresh measurement: skip-if-exists must not block a
    new run from re-firing a prompt+model+lever just because a prior run
    happens to have a row at the same config_hash and run_attempt. The
    prior bug was a missing `run_id` predicate here, which caused (e.g.)
    Day 7's baseline phase to silently inherit Day 6 dry-run rows. See
    methodology doc, Day 9 audit.
    """
    cur = conn.execute(
        """SELECT * FROM results
           WHERE prompt_id = ? AND model = ? AND optimisation_lever = ?
             AND config_hash = ? AND run_attempt = ? AND run_id = ?
             AND error IS NULL
           LIMIT 1""",
        (prompt_id, model, lever, config_hash, run_attempt, run_id),
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
    *, prompt: Prompt, model: str, provider: str, lever: str, config_hash: str,
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
        "provider": provider,
        "optimisation_lever": lever,
        "optimisation_config": json.dumps(optimisation_config) if optimisation_config else None,
        "config_hash": config_hash,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
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


def run_one(
    adapter: ProviderAdapter,
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
    client: Any = None,
) -> dict[str, Any]:
    """Provider-agnostic single-prompt runner. Adapter supplies the call/count behaviour."""
    if client is None:
        client = adapter.make_client()
    config_hash = _config_hash(lever, optimisation_config)

    if force_new_attempt:
        with sqlite3.connect(db_path) as conn:
            run_attempt = _next_run_attempt(conn, prompt.prompt_id, model, lever, config_hash)
    else:
        run_attempt = 1
        with sqlite3.connect(db_path) as conn:
            existing = _existing_successful_row(
                conn, prompt.prompt_id, model, lever, config_hash, run_attempt,
                run_id,
            )
        if existing is not None:
            return {**existing, "skipped": True}

    input_tokens_est = adapter.count_input_tokens(client, prompt, model)
    output_tokens_est = output_estimate_for(prompt.task_category)
    est_gbp = estimate_cost_gbp(model, input_tokens_est, output_tokens_est)

    _check_and_reserve(
        run_id=run_id, estimated_gbp=est_gbp, cap_gbp=cap_gbp,
        completed=completed, planned=planned, db_path=db_path,
    )

    row = _new_row(
        prompt=prompt, model=model, provider=adapter.name, lever=lever,
        config_hash=config_hash, optimisation_config=optimisation_config,
        run_id=run_id, run_attempt=run_attempt,
    )

    # Phase 1 — API call. On rate-limit exhaustion, write an error row and
    # release the reservation. On any other exception (auth error, network,
    # adapter bug), release the reservation and re-raise.
    try:
        m = adapter.call_with_retry(
            client, prompt, model,
            max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay,
            optimisation_config=optimisation_config,
        )
    except adapter.rate_limit_error as e:
        try:
            row["error"] = f"RateLimitError after {max_retries + 1} attempts: {e}"
            row["output_format_valid"] = 0
            with sqlite3.connect(db_path) as conn:
                _insert_row(conn, row)
                conn.commit()
        finally:
            _release_reservation(run_id=run_id, estimated_gbp=est_gbp, db_path=db_path)
        return {**row, "skipped": False, "stop_reason": None}
    except Exception:
        _release_reservation(run_id=run_id, estimated_gbp=est_gbp, db_path=db_path)
        raise

    # Phase 2 — accounting + insert. Wrapped so that any failure releases the
    # original reservation rather than leaving phantom reserved cost in the
    # cap accumulator. The reconcile UPDATE is folded into the same SQLite
    # transaction as the row insert so we never end up with a row inserted
    # but the cost not booked (or vice versa).
    try:
        cache_creation_tokens = m.get("cache_creation_tokens", 0) or 0
        cost_usd = estimate_cost_usd(
            model, m["input_tokens"], m["output_tokens"],
            cached_tokens=m["cached_tokens"],
            cache_creation_tokens=cache_creation_tokens,
        )
        cost_gbp = usd_to_gbp(cost_usd)
        delta_gbp = cost_gbp - est_gbp
        row.update({
            "input_tokens": m["input_tokens"],
            "output_tokens": m["output_tokens"],
            "cached_tokens": m["cached_tokens"],
            "cache_creation_tokens": cache_creation_tokens,
            "latency_ms": m["latency_ms"],
            "cost_usd": cost_usd,
            "response_text": m["response_text"],
            "model_version": m["model_version"],
        })
        with sqlite3.connect(db_path) as conn:
            _insert_row(conn, row)
            conn.execute(
                "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp + ? WHERE run_id = ?",
                (delta_gbp, run_id),
            )
            conn.commit()
    except Exception:
        _release_reservation(run_id=run_id, estimated_gbp=est_gbp, db_path=db_path)
        raise

    # stop_reason is not a DB column — threaded through the return dict for
    # callers (the output_cap smoke and Day 9 truncation diagnostics) only.
    return {**row, "skipped": False, "stop_reason": m.get("stop_reason")}


def run_many(
    adapter: ProviderAdapter,
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
    n_workers = concurrency if concurrency is not None else _concurrency()
    n = len(prompts)
    results: list[dict[str, Any] | None] = [None] * n
    aborted = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {
            ex.submit(
                run_one, adapter, p, model, lever,
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
