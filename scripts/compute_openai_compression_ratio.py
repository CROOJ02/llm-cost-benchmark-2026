"""Post-dry-run helper: compute OpenAI tiktoken-counted compression ratio for
sum-015 against GPT-5.4-mini (the dry-run's OpenAI mini model), for the
methodology paragraph update.

The dry-run records:
  - LLMLingua-2 BERT counts (in stdout, not persisted)
  - Anthropic count_tokens (in optimisation_config)
  - actual API-billed counts (in results.input_tokens for Anthropic models)

This script computes the third missing measurement: OpenAI's tiktoken count
on the same compressed text, so the methodology paragraph can report the
full cross-tokeniser picture (BERT vs Anthropic vs OpenAI).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
DRYRUN_DB_PATH = REPO_ROOT / "data" / "dryrun_results.db"

from runners.schema import load_prompts  # noqa: E402


def main() -> None:
    summary_prompts = load_prompts(REPO_ROOT / "prompts" / "summarisation.json")
    sum_015 = next(p for p in summary_prompts if p.prompt_id == "sum-015")

    # tiktoken's `encoding_for_model` may not recognise dated GPT-5.x snapshots
    # yet; o200k_base is the GPT-5.x family encoder so falling back is safe.
    try:
        enc = tiktoken.encoding_for_model("gpt-5.4-mini")
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")

    # Original text (system + user, exactly as the runner sends to the API).
    original_text = sum_015.input.system + "\n" + sum_015.input.user
    original_tiktoken = len(enc.encode(original_text))

    # Compressed text — pull from the dry-run's GPT-4o compression result row.
    # results.input_tokens for that row is the OpenAI billed count of the
    # compressed prompt; we recompute from response_text input via tiktoken
    # by pulling the compressed_input_tokens from optimisation_config and
    # using the API's billed count which is itself tiktoken-derived.
    with sqlite3.connect(DRYRUN_DB_PATH) as conn:
        row = conn.execute(
            "SELECT optimisation_config, input_tokens FROM results "
            "WHERE prompt_id='sum-015' AND model LIKE 'gpt-5.4-mini%' "
            "AND optimisation_lever='compression' LIMIT 1",
        ).fetchone()
    if row is None:
        raise SystemExit("no compression row for sum-015/gpt-5.4-mini in dryrun DB; run dry-run first")
    cfg = json.loads(row[0])
    billed_compressed_tiktoken = row[1]  # API-returned, tiktoken-counted

    print(f"=== sum-015 compression: cross-tokeniser comparison ===")
    print(f"LLMLingua-2 BERT (from sum-015 step-3 smoke): claimed ~47.7% reduction")
    print(f"Anthropic count_tokens:")
    print(f"  original   = {cfg['original_input_tokens']:>5}")
    print(f"  compressed = {cfg['compressed_input_tokens']:>5}")
    print(
        f"  ratio      = {cfg['compressed_input_tokens']/cfg['original_input_tokens']:.4f} "
        f"({(1 - cfg['compressed_input_tokens']/cfg['original_input_tokens'])*100:.1f}% reduction)"
    )
    print(f"OpenAI tiktoken (encoder: {enc.name}):")
    print(f"  original (system+user, locally counted)       = {original_tiktoken:>5}")
    print(f"  compressed (API-billed prompt_tokens, gpt-5.4-mini) = {billed_compressed_tiktoken:>5}")
    if original_tiktoken > 0:
        ratio = billed_compressed_tiktoken / original_tiktoken
        print(
            f"  ratio = {ratio:.4f} "
            f"({(1 - ratio)*100:.1f}% reduction)"
        )


if __name__ == "__main__":
    main()
