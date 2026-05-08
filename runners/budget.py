"""Pricing, GBP/USD conversion, output-token estimates, and cost-cap enforcement.

GBP is the canonical operational unit for the benchmark — the cap, soft warning,
and `runs.cost_so_far_gbp` accumulator are all in GBP. Provider per-call costs
are computed in USD (the provider's native unit) and converted at the boundary.
The per-row `results.cost_usd` column stays in USD so per-call figures cross-check
cleanly against provider invoices.

Cache pricing (from Day 5 caching threshold verification, see methodology doc):
  - Anthropic: cache writes are 1.25× input rate, cache reads are 0.1× input rate
  - OpenAI: cache reads are 0.5× input rate (per the GPT-4o pricing page); cache
    creation is not separately billed or surfaced (treated as 1.0× / no premium).
Each model carries explicit cache_read_mult and cache_creation_mult so the cost
function stays generic across providers.
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
    "claude-sonnet-4-6":  {"input": 3.00, "output": 15.00, "cache_read_mult": 0.10, "cache_creation_mult": 1.25},
    "claude-haiku-4-5":   {"input": 1.00, "output":  5.00, "cache_read_mult": 0.10, "cache_creation_mult": 1.25},
    "gpt-4o":             {"input": 2.50, "output": 10.00, "cache_read_mult": 0.50, "cache_creation_mult": 1.00},
    "gpt-4o-mini":        {"input": 0.15, "output":  0.60, "cache_read_mult": 0.50, "cache_creation_mult": 1.00},
    # Judges (Day 10)
    "claude-opus-4-6":    {"input": 15.00, "output": 75.00, "cache_read_mult": 0.10, "cache_creation_mult": 1.25},
    "mistral-large-2512": {"input":  2.00, "output":  6.00, "cache_read_mult": 1.00, "cache_creation_mult": 1.00},
}

# Per-category output-token estimates for the pre-call cap gate. Each value is
# sized to include a 30–50% margin over typical observed output for that task.
OUTPUT_TOKEN_ESTIMATES: dict[str, int] = {
    "customer_support": 200,
    "extraction":       300,
    "rag_qa":           200,
    "summarisation":    400,
    "reasoning":        600,
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
    cache_creation_tokens: int = 0,
) -> float:
    """Cost in USD from token counts.

    `input_tokens` here is the UNCACHED portion of the prompt (charged at base
    rate). `cached_tokens` is the cache-read portion (charged at cache_read_mult).
    `cache_creation_tokens` is tokens written to cache this call (charged at
    cache_creation_mult; Anthropic only — always 0 for OpenAI). Each runner is
    responsible for normalising provider-specific usage shapes into this triple.
    """
    if model not in PRICING_USD_PER_MTOK:
        raise ValueError(f"unknown model {model!r}; add to PRICING_USD_PER_MTOK before running")
    p = PRICING_USD_PER_MTOK[model]
    return (
        input_tokens * p["input"]
        + cached_tokens * p["input"] * p["cache_read_mult"]
        + cache_creation_tokens * p["input"] * p["cache_creation_mult"]
        + output_tokens * p["output"]
    ) / 1_000_000.0


def estimate_cost_gbp(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    return usd_to_gbp(estimate_cost_usd(model, input_tokens, output_tokens, cached_tokens, cache_creation_tokens))


# Per-model minimum cacheable prompt length (tokens). Below this, cache_control
# is silently ignored by Anthropic / automatic caching does not engage on OpenAI.
# Used by lever_caching to decide whether to attempt the cache test or record
# the (model, prompt) as "caching unavailable at this prompt size".
# Source: methodology doc § "Anthropic prompt-caching minimum lengths" and
# § "OpenAI prompt-caching specs" (verified 2026-05-04).
CACHE_MIN_TOKENS: dict[str, int] = {
    "claude-sonnet-4-6": 2048,
    "claude-haiku-4-5":  4096,
    "claude-opus-4-6":   4096,
    "gpt-4o":            1024,
    "gpt-4o-mini":       1024,
}


def cache_min_tokens_for(model: str) -> int:
    """Minimum prompt length at which caching engages for this model.
    Returns a large number for unknown models so caching is conservatively skipped."""
    return CACHE_MIN_TOKENS.get(model, 1_000_000)


class CostCapExceeded(RuntimeError):
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
    if cost_so_far_gbp + estimated_call_gbp_value > cap_gbp:
        msg = (
            f"Cost cap of {_fmt_gbp(cap_gbp)} reached. "
            f"Completed {completed} of {planned} planned runs. "
            f"Raise --cost-cap (e.g. --cost-cap=350) and re-run with --force-resume "
            f"to continue. Skip-if-exists logic prevents redoing completed work."
        )
        raise CostCapExceeded(msg, completed=completed, planned=planned, cap_gbp=cap_gbp)
