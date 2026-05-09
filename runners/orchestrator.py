"""Phase-driven matrix runner.

Consumes the existing `run_anthropic` / `run_openai` adapters and the four
lever modules (`lever_caching`, `lever_output_cap`, `lever_batch`,
`lever_compression`). Phases run in a defined sequence with idempotent
re-entry: each phase relies on the existing skip-if-exists logic in the
runners (`results` table) plus the `batch_jobs` status state machine to
avoid redoing work on script restart.

State persists across Day 7 → Day 8 script invocations through the `runs`,
`results`, and `batch_jobs` tables, plus an append-only structured log at
`data/phase_log.jsonl`.

Phase sequence:
  Day 7 (in this order — `batch_submit` first so the 24h provider-side SLA
         clock starts as early as possible):
    1. batch_submit   — submit batch jobs to provider batch APIs; writes
                        batch_jobs rows with status='submitted'; returns
                        immediately
    2. baseline       — sync calls, all 102 prompts × 4 models (lever='baseline')
    3. caching        — 3-call sequence on sum-015..020 × engaging models
    4. output_cap     — sync calls with max_tokens=200, all 102 × 4 models

  [Day 7 script ends. Provider-side queue runs 1–24h.]

  Day 8:
    5. batch_retrieve — poll all submitted batch_jobs, pull completed results
                        into the results table; idempotent over the
                        batch_jobs.status state machine
    6. compression_decide — pick tier from headroom per the §9 Day 8 ladder
    7. compression_run — runtime LLMLingua-2 on the chosen prompt subset
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import openai

from runners import (
    _base,
    lever_batch,
    lever_caching,
    lever_compression,
    lever_output_cap,
    run_anthropic,
    run_openai,
)
from runners._base import DB_PATH, REPO_ROOT, _now_iso, _read_cost_so_far_gbp
from runners.budget import estimate_cost_usd, usd_to_gbp
from runners.schema import Prompt, load_prompts

PHASE_LOG_PATH = REPO_ROOT / "data" / "phase_log.jsonl"

TEST_MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
]

# Subset for the caching lever — long-enough summarisation prompts that clear
# Sonnet 4.6's 2048-token threshold (per Day 5 caching-engagement work).
CACHING_SUBSET_IDS: set[str] = {"sum-015", "sum-016", "sum-017", "sum-018", "sum-020"}

# Per the testing strategy: max_tokens=200 for the output_cap phase across the matrix.
OUTPUT_CAP_TOKENS = 200

# Tier → subset size, per PRD §5 stratified-subset rule.
COMPRESSION_TIER_SIZES = {
    "full": None,        # use all 102 prompts (no stratification needed)
    "60-subset": 60,
    "30-subset": 30,
    "operator-call": 15,
}

# §9 Day 8 ladder boundaries — proportional to cap, not absolute GBP. The PRD
# originally stated £180/£120/£80/£40 at a £300 cap; expressing those as
# percentages (60%/40%/27%/13%) makes the ladder scale-invariant so the same
# decision logic applies at any cap (e.g. the dry-run's £5 cap, or a future
# campaign with a different headroom budget). At cap=£300 the percentages
# reproduce the PRD numbers within ~£1 (27%×300=£81 vs PRD £80; 13%×300=£39
# vs PRD £40); the same tier comes out for every parameterised test point.
def _tier_for_headroom(headroom_gbp: float, cap_gbp: float) -> str:
    pct = headroom_gbp / cap_gbp if cap_gbp > 0 else 0.0
    if pct > 0.60:
        return "full"
    if pct > 0.40:
        return "60-subset"
    if pct > 0.27:
        return "30-subset"
    if pct > 0.13:
        return "operator-call"
    return "skip"


_PROVIDER_FOR_MODEL = {
    "claude-sonnet-4-6":       "anthropic",
    "claude-haiku-4-5":        "anthropic",
    "claude-opus-4-6":         "anthropic",
    "gpt-5.4":                 "openai",
    "gpt-5.4-2026-03-05":      "openai",
    "gpt-5.4-mini":            "openai",
    "gpt-5.4-mini-2026-03-17": "openai",
}

_ADAPTERS_BY_NAME = {
    "anthropic": run_anthropic.ANTHROPIC_ADAPTER,
    "openai":    run_openai.OPENAI_ADAPTER,
}


def _adapter_for_model(model: str):
    provider = _PROVIDER_FOR_MODEL.get(model)
    if provider is None:
        raise ValueError(f"unknown model {model!r}")
    return _ADAPTERS_BY_NAME[provider]


_PROMPT_FILES = [
    "customer_support.json",
    "rag_qa.json",
    "extraction.json",
    "summarisation.json",
    "reasoning.json",
]


def load_all_prompts() -> list[Prompt]:
    """Load every prompt file in `prompts/` and return a flat list."""
    out: list[Prompt] = []
    for fname in _PROMPT_FILES:
        out.extend(load_prompts(REPO_ROOT / "prompts" / fname))
    return out


class Orchestrator:
    def __init__(
        self, run_id: str, cap_gbp: float,
        db_path: Path = DB_PATH,
        phase_log_path: Path = PHASE_LOG_PATH,
    ):
        self.run_id = run_id
        self.cap_gbp = cap_gbp
        self.db_path = db_path
        self.phase_log_path = phase_log_path

    # ---- Day 7 phases ----

    def run_baseline(self, prompts: list[Prompt], models: list[str]) -> list[dict[str, Any]]:
        all_results: list[dict[str, Any]] = []
        for model in models:
            self._phase_log("baseline", "start", {"model": model, "n_prompts": len(prompts)})
            adapter = _adapter_for_model(model)
            results = _base.run_many(
                adapter, prompts, model, lever="baseline",
                run_id=self.run_id, cap_gbp=self.cap_gbp, db_path=self.db_path,
            )
            # Cache-contamination check (methodology audit signal). OpenAI's
            # auto-caching is account-level with a 5–10 min TTL; any prior
            # call against the same prompt+model can leave cache state that
            # contaminates this baseline measurement. We can't prevent it
            # (caching is server-side), but we log it so Day 12 analysis can
            # flag affected rows. See methodology doc § "Day 6 finding:
            # OpenAI auto-caching is account-level".
            for r in results:
                if r.get("skipped") or r.get("error"):
                    continue
                cached = r.get("cached_tokens") or 0
                if cached > 0:
                    self._phase_log(
                        "baseline", "warning",
                        {
                            "warning": "baseline_cache_contamination",
                            "model": model,
                            "prompt_id": r.get("prompt_id"),
                            "cached_tokens": cached,
                            "input_tokens_uncached": r.get("input_tokens"),
                            "explanation": (
                                "baseline call hit non-zero cached_tokens; "
                                "OpenAI auto-caching from a prior run within "
                                "the cache TTL has contaminated this "
                                "measurement. Cost is artificially low; "
                                "treat with caveat in Day 12 analysis."
                            ),
                        },
                    )
            all_results.extend(results)
            self._phase_log("baseline", "complete", {"model": model, "n_results": len(results)})
        return all_results

    def run_caching(self, prompt_subset: list[Prompt], models: list[str]) -> list[dict[str, Any]]:
        """3-call caching sequence per (model, prompt). Below-threshold pairs
        short-circuit to baseline-only inside `lever_caching.run_caching_for_prompt`."""
        all_results: list[dict[str, Any]] = []
        for model in models:
            self._phase_log("caching", "start", {"model": model, "n_prompts": len(prompt_subset)})
            adapter = _adapter_for_model(model)
            results = lever_caching.run_caching_test(
                adapter, prompt_subset, model,
                run_id=self.run_id, cap_gbp=self.cap_gbp, db_path=self.db_path,
            )
            all_results.extend(results)
            self._phase_log(
                "caching", "complete",
                {"model": model, "n_results": len(results),
                 "n_available": sum(1 for r in results if r.get("caching_available")),
                 "n_unavailable": sum(1 for r in results if not r.get("caching_available"))},
            )
        return all_results

    def run_output_cap(self, prompts: list[Prompt], models: list[str]) -> list[dict[str, Any]]:
        all_results: list[dict[str, Any]] = []
        for model in models:
            self._phase_log("output_cap", "start", {"model": model, "n_prompts": len(prompts)})
            adapter = _adapter_for_model(model)
            results = lever_output_cap.run_output_cap_test(
                adapter, prompts, model,
                run_id=self.run_id, cap_gbp=self.cap_gbp,
                max_tokens=OUTPUT_CAP_TOKENS, db_path=self.db_path,
            )
            all_results.extend(results)
            self._phase_log("output_cap", "complete", {"model": model, "n_results": len(results)})
        return all_results

    def submit_batches(self, prompts: list[Prompt], models: list[str]) -> list[str]:
        """Submit one batch per (provider, model) tagged lever='batch'.
        Returns the list of batch_ids submitted (or already-existing ids when
        idempotent skip kicked in). Batch result rows produced at retrieve
        time coexist with sync baseline rows for the same (prompt, model)
        pair via the distinct lever value (PRD §5 — batch is its own lever)."""
        batch_ids: list[str] = []
        for model in models:
            self._phase_log("batch_submit", "start", {"model": model, "n_prompts": len(prompts)})
            result = lever_batch.submit_batch(
                prompts, model=model,
                run_id=self.run_id, db_path=self.db_path,
            )
            batch_ids.append(result["batch_id"])
            self._phase_log(
                "batch_submit", "complete",
                {"model": model, "batch_id": result["batch_id"],
                 "skipped": result.get("skipped", False)},
            )
        return batch_ids

    # ---- Day 8 phases ----

    def retrieve_batches(
        self, *, poll_interval_s: int = 60, timeout_s: int = 86_400,
        per_batch_timeout_s: int = 1800,
    ) -> list[dict[str, Any]]:
        """Poll all batch_jobs with status in {'submitted','in_progress'};
        pull completed results into the results table; mark each batch's
        terminal status.

        per_batch_timeout_s: maximum age (seconds since submitted_at) a single
        batch can stay in_flight before being marked 'timed_out' and dropped
        from the poll set. Default 1800s (30 min) — caps Day 6+ exposure to
        single-batch tail-latency outliers without forfeiting other batches'
        results. The batch is not cancelled provider-side; it remains in the
        provider queue and may complete eventually, but the orchestrator
        proceeds without waiting on it. The 'timed_out' row is preserved in
        batch_jobs as audit / future-recovery anchor.

        Idempotent — terminal-state batches (including 'timed_out') are filtered
        out of the in-flight set on re-entry, so re-running is a no-op."""
        started = time.time()
        retrieved: list[dict[str, Any]] = []
        prompts_by_id = {p.prompt_id: p for p in load_all_prompts()}
        self._phase_log("batch_retrieve", "start",
                        {"per_batch_timeout_s": per_batch_timeout_s})

        while True:
            in_flight = self._read_in_flight_batch_jobs()
            # Apply per-batch age cap: any batch older than the timeout is
            # marked 'timed_out' and dropped from the poll set this iteration.
            now = datetime.now(timezone.utc)
            active: list[dict[str, Any]] = []
            for job in in_flight:
                submitted = datetime.fromisoformat(job["submitted_at"])
                age_s = (now - submitted).total_seconds()
                if age_s > per_batch_timeout_s:
                    self._update_batch_status(
                        job["batch_id"], "timed_out",
                        completed_at=_now_iso(),
                        error=(
                            f"per-batch timeout exceeded "
                            f"(age={age_s:.0f}s > limit={per_batch_timeout_s}s)"
                        ),
                    )
                    self._phase_log(
                        "batch_retrieve", "decision",
                        {"batch_id": job["batch_id"], "model": job["model"],
                         "marked": "timed_out", "age_s": int(age_s)},
                    )
                else:
                    active.append(job)
            in_flight = active

            if not in_flight:
                self._phase_log("batch_retrieve", "complete", {"n_retrieved": len(retrieved)})
                return retrieved

            for job in in_flight:
                # Transient network errors during multi-hour polling shouldn't kill
                # the whole wait — catch per-poll, log a phase event, continue. The
                # outer loop will retry on the next poll cycle.
                try:
                    if job["provider"] == "anthropic":
                        n = self._poll_anthropic_batch(job, prompts_by_id)
                    elif job["provider"] == "openai":
                        n = self._poll_openai_batch(job, prompts_by_id)
                    else:
                        raise ValueError(f"unknown provider {job['provider']!r}")
                    if n > 0:
                        retrieved.append({"batch_id": job["batch_id"], "n_results": n})
                except (anthropic.APIConnectionError, openai.APIConnectionError) as e:
                    self._phase_log(
                        "batch_retrieve", "error",
                        {"transient": True, "batch_id": job["batch_id"], "error": str(e)},
                    )
                    continue

            # If still in-flight after the polling pass, sleep then retry.
            still_in_flight = self._read_in_flight_batch_jobs()
            if not still_in_flight:
                self._phase_log("batch_retrieve", "complete", {"n_retrieved": len(retrieved)})
                return retrieved
            if time.time() - started > timeout_s:
                self._phase_log(
                    "batch_retrieve", "error",
                    {"reason": "timeout", "still_in_flight": [j["batch_id"] for j in still_in_flight]},
                )
                raise TimeoutError(
                    f"Batch retrieval timed out after {timeout_s}s; "
                    f"still in-flight: {[j['batch_id'] for j in still_in_flight]}"
                )
            time.sleep(poll_interval_s)

    def decide_compression_tier(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cost_so_far = _read_cost_so_far_gbp(conn, self.run_id)
        headroom = self.cap_gbp - cost_so_far
        tier = _tier_for_headroom(headroom, self.cap_gbp)
        decision = {
            "tier": tier,
            "headroom_gbp": round(headroom, 4),
            "cost_so_far_gbp": round(cost_so_far, 4),
            "cap_gbp": self.cap_gbp,
            "rationale": (
                f"headroom £{headroom:.2f} (cap £{self.cap_gbp:.2f} − spent £{cost_so_far:.2f}) "
                f"→ tier '{tier}' per §9 Day 8 ladder"
            ),
        }
        self._phase_log("compression_decide", "decision", decision)
        return decision

    def run_compression(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        tier = decision["tier"]
        if tier == "skip":
            self._phase_log("compression_run", "complete", {"tier": "skip", "n_results": 0})
            return []

        all_prompts = load_all_prompts()
        subset_size = COMPRESSION_TIER_SIZES.get(tier)
        if subset_size is None:
            # 'full' tier — every prompt
            subset = all_prompts
        else:
            subset = self._stratified_subset(all_prompts, subset_size)

        self._phase_log(
            "compression_run", "start",
            {"tier": tier, "n_prompts": len(subset)},
        )
        all_results: list[dict[str, Any]] = []
        for model in TEST_MODELS:
            adapter = _adapter_for_model(model)
            results = lever_compression.run_compression_test(
                adapter, subset, model,
                run_id=self.run_id, cap_gbp=self.cap_gbp, db_path=self.db_path,
            )
            all_results.extend(results)
        self._phase_log(
            "compression_run", "complete",
            {"tier": tier, "n_results": len(all_results)},
        )
        return all_results

    # ---- Top-level entry points ----

    def run_day_7(self, prompts: list[Prompt], models: list[str]) -> dict[str, Any]:
        """Phases 1–4 (batch_submit FIRST so the 24h SLA clock starts early).
        Returns when sync phases finish; does not wait for batch completion."""
        self._phase_log(
            "day_7", "start",
            {"n_prompts": len(prompts), "models": models},
        )
        batch_ids = self.submit_batches(prompts, models)
        baseline = self.run_baseline(prompts, models)
        caching_subset = [p for p in prompts if p.prompt_id in CACHING_SUBSET_IDS]
        caching = self.run_caching(caching_subset, models)
        output_cap = self.run_output_cap(prompts, models)
        summary = {
            "phase": "day_7_complete",
            "batch_ids": batch_ids,
            "n_baseline": len(baseline),
            "n_caching": len(caching),
            "n_output_cap": len(output_cap),
        }
        self._phase_log("day_7", "complete", summary)
        return summary

    def run_day_8(self) -> dict[str, Any]:
        """Phases 5–7. Reads in-flight batch_jobs from DB; no prompts/models
        argument needed (state is already persisted from Day 7)."""
        self._phase_log("day_8", "start", {})
        retrieved = self.retrieve_batches()
        decision = self.decide_compression_tier()
        compression = self.run_compression(decision)
        summary = {
            "phase": "day_8_complete",
            "n_retrieved_batches": len(retrieved),
            "compression_decision": decision,
            "n_compression_results": len(compression),
        }
        self._phase_log("day_8", "complete", summary)
        return summary

    # ---- Internals ----

    def _phase_log(self, phase: str, event: str, payload: dict[str, Any]) -> None:
        """Append one JSON line to the orchestrator's phase log path. JSONL was
        chosen over a SQLite phase_log table because it's append-only, has no
        schema-evolution cost, and is sufficient for Day 12 analysis (which reads
        it as a stream of events). The file is created on first append; safe to
        read concurrently with writes (line-atomic on POSIX).

        `event` ∈ {'start', 'progress', 'decision', 'complete', 'error'}.
        """
        self.phase_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "phase": phase,
            "event": event,
            "payload": payload,
        }
        with self.phase_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _stratified_subset(self, prompts: list[Prompt], n: int) -> list[Prompt]:
        """Per PRD §5: subsets stratified across the 5 task categories so each
        category contributes to the compression signal. Within each category,
        pick deterministically (sorted by complexity then prompt_id)."""
        if n % 5 != 0:
            raise ValueError(f"stratified subset size {n} must be divisible by 5")
        per_cat = n // 5
        by_cat: dict[str, list[Prompt]] = defaultdict(list)
        for p in prompts:
            by_cat[p.task_category].append(p)
        complexity_order = {"easy": 0, "medium": 1, "hard": 2}
        subset: list[Prompt] = []
        for cat in sorted(by_cat.keys()):
            in_cat = sorted(
                by_cat[cat],
                key=lambda p: (complexity_order.get(p.complexity, 99), p.prompt_id),
            )
            subset.extend(in_cat[:per_cat])
        return subset

    # ---- Batch retrieval helpers (testable seams) ----

    def _read_in_flight_batch_jobs(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """SELECT * FROM batch_jobs
                   WHERE run_id = ? AND status IN ('submitted', 'in_progress')""",
                (self.run_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _update_batch_status(
        self, batch_id: str, status: str, *,
        retrieved_at: str | None = None, completed_at: str | None = None,
        error: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE batch_jobs SET
                       status = ?,
                       retrieved_at = COALESCE(?, retrieved_at),
                       completed_at = COALESCE(?, completed_at),
                       error = COALESCE(?, error)
                   WHERE batch_id = ?""",
                (status, retrieved_at, completed_at, error, batch_id),
            )
            conn.commit()

    def _insert_batch_result_row(
        self, *, prompt: Prompt, model: str, provider: str, lever: str,
        message_meta: dict[str, Any],
    ) -> bool:
        """Insert one batch result row into `results`. Returns True if inserted,
        False if a row already existed (idempotent re-entry). Also bumps
        runs.cost_so_far_gbp by the actual cost."""
        config_hash = _base._config_hash(lever, None)
        run_attempt = 1
        with sqlite3.connect(self.db_path) as conn:
            existing = _base._existing_successful_row(
                conn, prompt.prompt_id, model, lever, config_hash, run_attempt
            )
            if existing is not None:
                return False
            cost_usd = estimate_cost_usd(
                model,
                input_tokens=message_meta["input_tokens"],
                output_tokens=message_meta["output_tokens"],
                cached_tokens=message_meta.get("cached_tokens", 0),
                cache_creation_tokens=message_meta.get("cache_creation_tokens", 0),
                batch_discount=True,
            )
            cost_gbp = usd_to_gbp(cost_usd)
            row = _base._new_row(
                prompt=prompt, model=model, provider=provider, lever=lever,
                config_hash=config_hash, optimisation_config=None,
                run_id=self.run_id, run_attempt=run_attempt,
            )
            row.update({
                "input_tokens":          message_meta["input_tokens"],
                "output_tokens":         message_meta["output_tokens"],
                "cached_tokens":         message_meta.get("cached_tokens", 0),
                "cache_creation_tokens": message_meta.get("cache_creation_tokens", 0),
                "latency_ms":            0,  # batch has no per-request latency
                "cost_usd":              cost_usd,
                "response_text":         message_meta["response_text"],
                "model_version":         message_meta.get("model_version") or model,
            })
            _base._insert_row(conn, row)
            conn.execute(
                "UPDATE runs SET cost_so_far_gbp = cost_so_far_gbp + ? WHERE run_id = ?",
                (cost_gbp, self.run_id),
            )
            conn.commit()
        return True

    def _poll_anthropic_batch(
        self, job: dict[str, Any], prompts_by_id: dict[str, Prompt],
    ) -> int:
        """Poll one Anthropic batch. Returns number of results inserted this call."""
        client = anthropic.Anthropic()
        batch = client.messages.batches.retrieve(job["batch_id"])
        if batch.processing_status != "ended":
            if job["status"] == "submitted":
                self._update_batch_status(job["batch_id"], "in_progress")
            return 0
        n_inserted = 0
        for entry in client.messages.batches.results(job["batch_id"]):
            custom_id = entry.custom_id
            prompt = prompts_by_id.get(custom_id)
            if prompt is None:
                continue
            if entry.result.type != "succeeded":
                continue
            msg = entry.result.message
            response_text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            inserted = self._insert_batch_result_row(
                prompt=prompt, model=job["model"], provider="anthropic", lever=job["lever"],
                message_meta={
                    "input_tokens":          msg.usage.input_tokens,
                    "output_tokens":         msg.usage.output_tokens,
                    "cached_tokens":         getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                    "cache_creation_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
                    "response_text":         response_text,
                    "model_version":         msg.model,
                },
            )
            if inserted:
                n_inserted += 1
        self._update_batch_status(
            job["batch_id"], "completed",
            retrieved_at=_now_iso(), completed_at=_now_iso(),
        )
        return n_inserted

    def _poll_openai_batch(
        self, job: dict[str, Any], prompts_by_id: dict[str, Prompt],
    ) -> int:
        """Poll one OpenAI batch. Returns number of results inserted this call."""
        client = openai.OpenAI()
        batch = client.batches.retrieve(job["batch_id"])
        terminal = {"completed", "failed", "expired", "cancelled"}
        if batch.status not in terminal:
            if job["status"] == "submitted" and batch.status == "in_progress":
                self._update_batch_status(job["batch_id"], "in_progress")
            return 0
        if batch.status != "completed" or not batch.output_file_id:
            self._update_batch_status(
                job["batch_id"], batch.status,
                completed_at=_now_iso(),
                error=str(getattr(batch, "errors", None)) if batch.status != "completed" else None,
            )
            return 0
        file_content = client.files.content(batch.output_file_id)
        n_inserted = 0
        for line in file_content.text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id")
            prompt = prompts_by_id.get(custom_id)
            if prompt is None:
                continue
            response = entry.get("response") or {}
            if response.get("status_code") != 200:
                continue
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens", 0) or 0
            choices = body.get("choices") or [{}]
            response_text = (choices[0].get("message") or {}).get("content") or ""
            inserted = self._insert_batch_result_row(
                prompt=prompt, model=job["model"], provider="openai", lever=job["lever"],
                message_meta={
                    "input_tokens":          (usage.get("prompt_tokens", 0) or 0) - cached,
                    "output_tokens":         usage.get("completion_tokens", 0) or 0,
                    "cached_tokens":         cached,
                    "cache_creation_tokens": 0,
                    "response_text":         response_text,
                    "model_version":         body.get("model") or job["model"],
                },
            )
            if inserted:
                n_inserted += 1
        self._update_batch_status(
            job["batch_id"], "completed",
            retrieved_at=_now_iso(), completed_at=_now_iso(),
        )
        return n_inserted
