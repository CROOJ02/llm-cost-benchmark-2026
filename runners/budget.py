"""Pricing, GBP/USD conversion, output-token estimates, and cost-cap enforcement.

GBP is the canonical operational unit for the benchmark — the cap, soft warning,
and `runs.cost_so_far_gbp` accumulator are all in GBP. Provider per-call costs
are computed in USD (the provider's native unit) and converted at the boundary.
The per-row `results.cost_usd` column stays in USD so per-call figures cross-check
cleanly against provider invoices.
"""

from __future__ import annotations

# Exchange rate: GBP/USD as of 2026-05-02, Bank of England daily reference (£1 = $1.27).
# Intentionally FIXED, not live — pinning the rate keeps the cap and reported costs
# reproducible across re-runs that may span days. Live FX would introduce drift that
# obscures whether changes in cost_so_far reflect actual API usage or rate movement.
# Methodology footnote when written up.
GBP_USD_RATE: float = 1.27

# USD per million tokens, indexed by the model alias the runner calls with.
# These are best estimates as of 2026-05; verify against the live provider
# pricing pages before kicking off the full benchmark run.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # Test set
    "claude-sonnet-4-6":  {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":   {"input": 1.00, "output": 5.00},
    "gpt-4o":             {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":        {"input": 0.15, "output": 0.60},
    # Judges (Day 10)
    "claude-opus-4-6":    {"input": 15.00, "output": 75.00},
    "mistral-large-2512": {"input": 2.00, "output": 6.00},
}

# Per-category output-token estimates for the pre-call cap gate. Each value is
# sized to include a 30–50% margin over typical observed output for that task,
# so the gate triggers predictably before overspend rather than chronically
# after. Not used as the actual API max_tokens parameter — that's set per-lever.
OUTPUT_TOKEN_ESTIMATES: dict[str, int] = {
    "customer_support": 200,  # 2-sentence reply + category in JSON ≈ 50–100 typical
    "extraction":       300,  # JSON object, varies by schema; line-items hardest
    "rag_qa":           200,  # answer + supporting_sentences indices ≈ 50–100 typical
    "summarisation":    400,  # 3 bullets, 1–2 sentences each ≈ 150–250 typical
    "reasoning":        600,  # explicit reasoning chain + final_answer ≈ 200–500 typical
}
DEFAULT_OUTPUT_ESTIMATE: int = 500


def output_estimate_for(task_category: str) -> int:
    return OUTPUT_TOKEN_ESTIMATES.get(task_category, DEFAULT_OUTPUT_ESTIMATE)


def usd_to_gbp(usd: float) -> float:
    return usd / GBP_USD_RATE


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Cost in USD from token counts. Cache-discount handling deferred to lever_caching."""
    if model not in PRICING_USD_PER_MTOK:
        raise ValueError(f"unknown model {model!r}; add to PRICING_USD_PER_MTOK before running")
    p = PRICING_USD_PER_MTOK[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000.0


def estimate_cost_gbp(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    return usd_to_gbp(estimate_cost_usd(model, input_tokens, output_tokens, cached_tokens))


class CostCapExceeded(RuntimeError):
    """Raised when the next call would breach the GBP cap. Carries enough info
    for the caller to update runs.status and print the PRD §10 abort message."""

    def __init__(self, message: str, *, completed: int, planned: int, cap_gbp: float):
        super().__init__(message)
        self.completed = completed
        self.planned = planned
        self.cap_gbp = cap_gbp


def _fmt_gbp(v: float) -> str:
    return f"£{v:,.2f}" if v >= 1 else f"£{v:.4f}"


def check_cap(
    *,
    cost_so_far_gbp: float,
    estimated_call_gbp_value: float,
    cap_gbp: float,
    completed: int,
    planned: int,
) -> None:
    """Raise CostCapExceeded if the next call's upper-bound cost would breach cap.

    Message format pinned to PRD §10: it must be exactly recognisable in stderr
    so an operator can grep for it. The (e.g. --cost-cap=350) example is from
    the PRD verbatim — not derived from cap_gbp — to keep the message
    consistent regardless of the test cap used.
    """
    if cost_so_far_gbp + estimated_call_gbp_value > cap_gbp:
        msg = (
            f"Cost cap of {_fmt_gbp(cap_gbp)} reached. "
            f"Completed {completed} of {planned} planned runs. "
            f"Raise --cost-cap (e.g. --cost-cap=350) and re-run with --force-resume "
            f"to continue. Skip-if-exists logic prevents redoing completed work."
        )
        raise CostCapExceeded(msg, completed=completed, planned=planned, cap_gbp=cap_gbp)
