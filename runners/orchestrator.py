"""Phase-driven matrix runner.

The orchestrator consumes the existing `run_anthropic` / `run_openai` adapters
and lever modules (`lever_caching` already; `lever_output_cap`, `lever_batch`,
`lever_compression` to land in Day 6 work). It runs phases in a defined
sequence with idempotent re-entry — each phase relies on the existing
skip-if-exists logic in the runners (`results` table) plus the `batch_jobs`
status state machine to avoid redoing work on script restart.

State persists across Day 7 → Day 8 script invocations through the `runs`,
`results`, and `batch_jobs` tables, plus an append-only structured log at
`data/phase_log.jsonl`.

Phase sequence:
  Day 7:
    1. baseline       — sync calls, all 102 prompts × 4 models (lever='baseline')
    2. caching        — 3-call sequence on sum-015..020 × engaging models
    3. output_cap     — sync calls with max_tokens=200, all 102 × 4 models
    4. batch_submit   — submit batch jobs to provider batch APIs; writes batch_jobs
                        rows with status='submitted'; returns immediately

  [Day 7 script ends. Provider-side queue runs 1–24h.]

  Day 8:
    5. batch_retrieve — poll all submitted batch_jobs, pull completed results
                        into the results table; idempotent over the
                        batch_jobs.status state machine
    6. budget_check   — compute headroom = cap_gbp − cost_so_far_gbp
    7. compression_decide — pick tier from headroom per the §9 Day 8 ladder
    8. compression_run — runtime LLMLingua-2 on the chosen prompt subset

This file is a SKELETON committed as part of the Day 6 prep work; concrete
phase implementations land in the Day 6 commit alongside the lever modules.
Methods raise NotImplementedError so the file imports cleanly but any call
fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners._base import DB_PATH
from runners.schema import Prompt


PHASE_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "phase_log.jsonl"


class Orchestrator:
    def __init__(self, run_id: str, cap_gbp: float, db_path: Path = DB_PATH):
        self.run_id = run_id
        self.cap_gbp = cap_gbp
        self.db_path = db_path

    # ---- Day 7 phases ----

    def run_baseline(self, prompts: list[Prompt], models: list[str]) -> list[dict[str, Any]]:
        """Sync baseline pass: lever='baseline', no special config."""
        raise NotImplementedError("Day 6 build")

    def run_caching(self, prompt_subset: list[Prompt], models: list[str]) -> list[dict[str, Any]]:
        """3-call caching sequence (delegates to lever_caching). Engaging models
        only; below-threshold (model, prompt) pairs short-circuit to baseline-only
        per the threshold check in lever_caching.run_caching_for_prompt."""
        raise NotImplementedError("Day 6 build")

    def run_output_cap(self, prompts: list[Prompt], models: list[str]) -> list[dict[str, Any]]:
        """Sync calls with max_tokens=200 (delegates to lever_output_cap)."""
        raise NotImplementedError("Day 6 build")

    def submit_batches(self, prompts: list[Prompt], models: list[str]) -> list[str]:
        """Submit batch jobs to each provider's batch API; writes one batch_jobs
        row per (provider, model) batch with status='submitted'. Returns the
        list of batch_ids. Does not wait for completion — Day 7 ends here."""
        raise NotImplementedError("Day 6 build")

    # ---- Day 8 phases ----

    def retrieve_batches(
        self, *, poll_interval_s: int = 60, timeout_s: int = 86_400,
    ) -> list[dict[str, Any]]:
        """Poll all batch_jobs with status in {'submitted','in_progress'};
        pull completed results into the results table; mark each batch's
        terminal status (completed/failed/expired). Idempotent — safe to
        re-invoke after partial completion or script restart."""
        raise NotImplementedError("Day 6 build")

    def decide_compression_tier(self) -> dict[str, Any]:
        """Read runs.cost_so_far_gbp; return:
            {tier, headroom_gbp, rationale}
        Tier ∈ {'full', '60-subset', '30-subset', 'operator-call', 'skip'}
        per the §9 Day 8 ladder. Logs the decision via _phase_log so the
        Day 12 analysis can cite which tier ran (and why)."""
        raise NotImplementedError("Day 6 build")

    def run_compression(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        """Iterate the prompt subset implied by decision['tier'] and run the
        compression lever (delegates to lever_compression). The 'skip' tier
        early-returns with []. Stratified subsets ('60-subset', '30-subset')
        are computed via _stratified_subset."""
        raise NotImplementedError("Day 6 build")

    # ---- Top-level entry points ----

    def run_day_7(
        self, prompts: list[Prompt], models: list[str],
    ) -> dict[str, Any]:
        """Phases 1–4. Returns when batch jobs are submitted (does not wait
        for batch completion). Each phase logs to phase_log.jsonl on entry,
        progress, and completion."""
        raise NotImplementedError("Day 6 build")

    def run_day_8(self) -> dict[str, Any]:
        """Phases 5–8. Reads in-flight batch_jobs from DB; no prompts/models
        argument needed (state is already persisted from Day 7)."""
        raise NotImplementedError("Day 6 build")

    # ---- Internals ----

    def _phase_log(self, phase: str, payload: dict[str, Any]) -> None:
        """Append one JSON line to data/phase_log.jsonl describing this phase
        entry. JSONL was chosen over a SQLite phase_log table because it has
        lower ceremony for an append-only audit trail and is sufficient for
        Day 12 analysis (which reads it as a stream of events). Each line:

            {"ts": ISO8601, "run_id": str, "phase": str, "event": str,
             "payload": {...phase-specific...}}

        `event` ∈ {'start', 'progress', 'decision', 'complete', 'error'}.
        File is created on first append; safe to read concurrently with writes."""
        raise NotImplementedError("Day 6 build")

    def _stratified_subset(self, prompts: list[Prompt], n: int) -> list[Prompt]:
        """Per PRD §5: subsets stratified across the 5 task categories so each
        category contributes to the compression signal. Used by run_compression
        when the chosen tier is '60-subset' or '30-subset'."""
        raise NotImplementedError("Day 6 build")
