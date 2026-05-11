"""Day 11 Tier-3 human arbitration CLI.

Walks through the 185 disagreement cases one at a time, showing the prompt,
criteria, reference (if applicable), response_text, and both judges' scores +
reasoning. Operator types a score in [0.0, 1.0] (or s/b/q/h commands). Each
case writes back to scoring/disagreements.csv immediately so progress is never
lost. Resume detection picks up at the first unscored case in the active
ordering on relaunch.

Default ordering: compression cluster (76 cases) first, then descending |Δ|
across the remaining 109. The compression block enables consistent application
of arbitration Principle 2 (content-vs-wrapper distinction). See methodology
doc § "Day 11 arbitration principles".

Usage:
  poetry run python -m scoring.arbitrate
  poetry run python -m scoring.arbitrate --cluster compression
  poetry run python -m scoring.arbitrate --by-category cs,rag,rea,sum
  poetry run python -m scoring.arbitrate --shuffle --seed 42
  poetry run python -m scoring.arbitrate --start-from 50

Public surface (testable, no I/O):
  - load_cases(csv_path) -> list[Case]
  - apply_ordering(cases, mode, **kwargs) -> list[int]
  - find_resume_index(cases, ordered_indices) -> int
  - validate_score(input_str) -> float
  - save_case(csv_path, prompt_id, model, lever, score, note) -> None
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.orchestrator import load_all_prompts  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "scoring" / "disagreements.csv"

CSV_COLUMNS = [
    "prompt_id", "model", "lever",
    "judge_a_score", "judge_b_score", "delta",
    "tier_2_criteria", "response_text", "human_score",
    "judge_a_reasoning", "judge_b_reasoning",
    "human_note", "arbitration_method",
]

# arbitration_method values:
#   ''                       — not yet resolved
#   'human'                  — operator entered the score interactively
#   'median_canonical_auto'  — sub-threshold case (|Δ| < min-delta); canonical
#                              score = median(judge_a_score, judge_b_score)
#                              auto-written without human input. Used when the
#                              operator selects --min-delta to skip arbitration
#                              on minor disagreements.

CATEGORY_FOR_PREFIX = {"cs": "customer_support", "rag": "rag_qa",
                       "rea": "reasoning", "sum": "summarisation"}
REF_FIELD = {"rag_qa": "answer", "reasoning": "final_answer"}


@dataclass
class Case:
    prompt_id: str
    model: str
    lever: str
    judge_a_score: float
    judge_b_score: float
    delta: float
    tier_2_criteria: str
    response_text: str
    human_score: str  # empty string when unscored
    judge_a_reasoning: str
    judge_b_reasoning: str
    human_note: str = ""
    arbitration_method: str = ""  # '' | 'human' | 'median_canonical_auto'

    @property
    def category(self) -> str:
        return CATEGORY_FOR_PREFIX.get(self.prompt_id.split("-")[0], "?")

    @property
    def is_scored(self) -> bool:
        return self.human_score.strip() != ""


def load_cases(csv_path: Path) -> list[Case]:
    """Read the disagreements CSV. Tolerates files with or without the
    human_note + arbitration_method columns (older CSVs lack them; the
    on-first-save upgrade adds them transparently).

    delta is recomputed from raw judge scores rather than read from the CSV
    column — the CSV stores delta rounded to 3 decimals, which loses
    precision at the threshold boundary. Specifically: floats like
    abs(0.7 - 1.0) = 0.30000000000000004 round to '0.300' in CSV, and a
    --min-delta 0.3 filter using the rounded value would silently drop the
    boundary case from BOTH the auto-resolve (delta < 0.3 fails) and the
    interactive ordering (delta > 0.3 fails). Recompute from raw scores.
    """
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    cases: list[Case] = []
    for r in rows:
        ja = float(r["judge_a_score"])
        jb = float(r["judge_b_score"])
        cases.append(Case(
            prompt_id=r["prompt_id"], model=r["model"], lever=r["lever"],
            judge_a_score=ja, judge_b_score=jb,
            delta=abs(ja - jb),
            tier_2_criteria=r["tier_2_criteria"], response_text=r["response_text"],
            human_score=r.get("human_score", ""),
            judge_a_reasoning=r.get("judge_a_reasoning", ""),
            judge_b_reasoning=r.get("judge_b_reasoning", ""),
            human_note=r.get("human_note", ""),
            arbitration_method=r.get("arbitration_method", ""),
        ))
    return cases


def auto_resolve_sub_threshold(
    csv_path: Path, cases: list[Case], min_delta: float,
) -> int:
    """Write median(judge_a, judge_b) as canonical for cases at or below
    min_delta. Mark them arbitration_method='median_canonical_auto'.
    Idempotent — already-resolved cases (any non-empty arbitration_method)
    are skipped. Returns the count of cases newly auto-resolved.

    Boundary semantic: cases at delta == min_delta exactly are auto-resolved
    (they do not EXCEED the threshold). The interactive filter in main()
    uses `delta > min_delta`, so the two filters are complementary —
    every case either auto-resolves or appears in interactive ordering,
    never falls into limbo. Earlier version used `case.delta >= min_delta`
    (skip if >=) which left boundary cases stuck. See methodology note
    on float-precision at the threshold.
    """
    n = 0
    for case in cases:
        if case.delta > min_delta:
            continue
        if case.arbitration_method:
            continue  # already resolved (human or auto from prior run)
        median_score = (case.judge_a_score + case.judge_b_score) / 2.0
        save_case(
            csv_path, case.prompt_id, case.model, case.lever,
            score=median_score,
            note=f"AUTO: median canonical (|Δ|={case.delta:.3f} ≤ {min_delta} threshold)",
            arbitration_method="median_canonical_auto",
        )
        case.human_score = f"{median_score:.3f}"
        case.human_note = f"AUTO: median canonical (|Δ|={case.delta:.3f} ≤ {min_delta} threshold)"
        case.arbitration_method = "median_canonical_auto"
        n += 1
    return n


def apply_ordering(
    cases: list[Case],
    mode: str = "compression-first",
    *,
    cluster: str | None = None,
    by_category: list[str] | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[int]:
    """Return indices into `cases` in the order arbitration should visit them.

    Modes:
      - 'compression-first' (default): all 76 compression cases, then the
        remaining 109 ordered by descending |Δ|. Within each block, secondary
        sort is descending |Δ| then prompt_id for stability.
      - 'cluster': only cases matching `cluster` (e.g. 'compression',
        'cs-compression', 'compression', 'baseline'); ordered by descending |Δ|.
      - 'by-category': iterate categories in the order given; within each,
        descending |Δ|.
      - 'shuffle': random.Random(seed) shuffle; deterministic given seed.
    """
    indexed = list(enumerate(cases))

    if shuffle:
        rng = random.Random(seed)
        idxs = [i for i, _ in indexed]
        rng.shuffle(idxs)
        return idxs

    if cluster is not None:
        # Cluster can be a lever name ('compression') or category-lever
        # ('rag-compression') or category alone ('rag').
        def matches(c: Case) -> bool:
            if "-" in cluster:
                cat_prefix, lev = cluster.split("-", 1)
                cat = CATEGORY_FOR_PREFIX.get(cat_prefix, cat_prefix)
                return c.category == cat and c.lever == lev
            if cluster in {"baseline", "batch", "compression", "output_cap"}:
                return c.lever == cluster
            return c.category == CATEGORY_FOR_PREFIX.get(cluster, cluster)
        filtered = [(i, c) for i, c in indexed if matches(c)]
        filtered.sort(key=lambda t: (-t[1].delta, t[1].prompt_id, t[1].model))
        return [i for i, _ in filtered]

    if by_category is not None:
        cats = [CATEGORY_FOR_PREFIX.get(c, c) for c in by_category]
        out: list[int] = []
        for cat in cats:
            block = [(i, c) for i, c in indexed if c.category == cat]
            block.sort(key=lambda t: (-t[1].delta, t[1].prompt_id, t[1].model))
            out.extend(i for i, _ in block)
        # Append anything not in the listed categories at the end (defensive)
        listed = set(out)
        rest = [i for i, _ in indexed if i not in listed]
        rest.sort(key=lambda i: (-cases[i].delta, cases[i].prompt_id, cases[i].model))
        return out + rest

    # Default: compression-first then descending |Δ|
    compression = [(i, c) for i, c in indexed if c.lever == "compression"]
    other = [(i, c) for i, c in indexed if c.lever != "compression"]
    compression.sort(key=lambda t: (-t[1].delta, t[1].prompt_id, t[1].model))
    other.sort(key=lambda t: (-t[1].delta, t[1].prompt_id, t[1].model))
    return [i for i, _ in compression] + [i for i, _ in other]


def find_resume_index(cases: list[Case], ordered: list[int]) -> int:
    """Return the position in `ordered` of the first unscored case, or
    len(ordered) if every case in the ordering is already scored."""
    for pos, idx in enumerate(ordered):
        if not cases[idx].is_scored:
            return pos
    return len(ordered)


def validate_score(raw: str) -> float:
    """Parse a score input; raise ValueError on anything not in [0.0, 1.0]."""
    s = raw.strip()
    if not s:
        raise ValueError("empty input")
    try:
        v = float(s)
    except ValueError as e:
        raise ValueError(f"not a number: {s!r}") from e
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"out of range [0.0, 1.0]: {v}")
    return v


def save_case(
    csv_path: Path, prompt_id: str, model: str, lever: str,
    score: float, note: str = "", arbitration_method: str = "human",
) -> None:
    """Read CSV, mutate the matching row's human_score + human_note +
    arbitration_method, write back.

    arbitration_method defaults to 'human' (the interactive-loop common case).
    auto_resolve_sub_threshold() passes 'median_canonical_auto' for the
    sub-threshold bulk write.

    Idempotent: writes the full file every call, ensuring header upgrades
    (adding new columns) take effect on first save. Crash-safe via
    write-to-temp + atomic rename.
    """
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    target = None
    for r in rows:
        if r["prompt_id"] == prompt_id and r["model"] == model and r["lever"] == lever:
            target = r
            break
    if target is None:
        raise KeyError(f"no row matching ({prompt_id}, {model}, {lever})")
    target["human_score"] = f"{score:.3f}"
    target["human_note"] = note
    target["arbitration_method"] = arbitration_method

    # Ensure all rows have the full column set (upgrade older CSVs)
    for r in rows:
        for col in CSV_COLUMNS:
            r.setdefault(col, "")

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp_path, csv_path)


# ----------------------------------------------------------------------
# Interactive layer (not unit-tested; covered by manual run + smoke check)
# ----------------------------------------------------------------------

def _render_case(case: Case, prompts_by_id: dict, position: int, total: int,
                 elapsed_s: float, scored_so_far: int) -> str:
    """Format one case for terminal display. Pure function — testable."""
    prompt = prompts_by_id.get(case.prompt_id)
    parts: list[str] = []
    parts.append("=" * 78)
    rate = scored_so_far / elapsed_s if elapsed_s > 0 else 0
    eta_s = (total - position - 1) / rate if rate > 0 else 0
    parts.append(f"  [{position + 1:>3} / {total}]  {case.prompt_id:8s}  "
                 f"{case.model:32s}  {case.lever:12s}  Δ={case.delta:.2f}")
    parts.append(f"  scored this session: {scored_so_far}  elapsed: {elapsed_s/60:.1f} min  "
                 f"ETA remaining: {eta_s/60:.1f} min")
    parts.append("=" * 78)

    if prompt is not None:
        parts.append("\nPROMPT (system):")
        parts.append(f"  {prompt.input.system}")
        parts.append("\nPROMPT (user):")
        for line in prompt.input.user.split("\n"):
            parts.append(f"  {line}")

        ref_field = REF_FIELD.get(prompt.task_category)
        if ref_field and prompt.scoring.tier_1_deterministic:
            ref_val = prompt.scoring.tier_1_deterministic.expected.get(ref_field)
            if ref_val is not None:
                parts.append(f"\nREFERENCE ({ref_field}):  {ref_val!r}")
        else:
            parts.append("\nREFERENCE:  (none — criteria-only category)")

    parts.append("\nCRITERIA:")
    parts.append(f"  {case.tier_2_criteria}")

    parts.append("\nRESPONSE (anonymised position label hidden in arbitration):")
    for line in case.response_text.split("\n"):
        parts.append(f"  {line}")

    parts.append(f"\nJUDGE A (Opus 4.6):     {case.judge_a_score:.2f}")
    if case.judge_a_reasoning:
        parts.append(f"  reasoning:  {case.judge_a_reasoning}")
    # NOTE for v2 maintainers: the judge_b_score column was Mistral large 2512
    # in Day 10; replaced with GPT-5.5 in the Day 11 panel revision (Mistral
    # exhibited quality issues — see methodology doc § "Day 11 panel revision").
    # The DB column name `judge_b_score` is retained for schema continuity; the
    # archived Mistral data lives in `judge_b_mistral_score`.
    parts.append(f"\nJUDGE B (GPT-5.5):      {case.judge_b_score:.2f}")
    if case.judge_b_reasoning:
        parts.append(f"  reasoning:  {case.judge_b_reasoning}")
    parts.append("")
    return "\n".join(parts)


def _print_help() -> None:
    print("""
COMMANDS:
  <number>   score the case (0.0–1.0; e.g. 0.5, 0.75, 1.0)
  s          skip — leave human_score empty, advance to next case
  b          back — return to previous case (re-arbitrate)
  q          quit — save state and exit
  h          help — print this message
  n: <text>  add an arbitration note (use after score on next prompt)
""")


def _arbitration_loop(
    csv_path: Path, cases: list[Case], ordered: list[int],
    prompts_by_id: dict, start_pos: int,
) -> None:
    pos = start_pos
    total = len(ordered)
    t0 = time.perf_counter()
    scored_this_session = 0

    if pos >= total:
        print(f"All {total} cases in this ordering are already scored.")
        return

    while 0 <= pos < total:
        idx = ordered[pos]
        case = cases[idx]
        elapsed = time.perf_counter() - t0
        print(_render_case(case, prompts_by_id, pos, total, elapsed, scored_this_session))

        if case.is_scored:
            print(f"  (already scored: {case.human_score})")

        try:
            raw = input("Score (number, or s/b/q/h): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting (state saved per case).")
            return

        if raw == "h":
            _print_help()
            continue
        if raw == "q":
            print(f"\nQuit. {scored_this_session} cases scored this session.")
            return
        if raw == "s":
            pos += 1
            continue
        if raw == "b":
            pos = max(0, pos - 1)
            continue

        try:
            score = validate_score(raw)
        except ValueError as e:
            print(f"  invalid: {e}. Try again.")
            continue

        note = input("Optional note (Enter to skip): ").strip()

        try:
            save_case(csv_path, case.prompt_id, case.model, case.lever, score, note)
        except Exception as e:
            print(f"  ERROR saving: {e}. Score NOT persisted; try again.")
            continue

        case.human_score = f"{score:.3f}"
        case.human_note = note
        scored_this_session += 1
        pos += 1

    print(f"\nReached end of ordering. {scored_this_session} cases scored this session.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 11 Tier-3 arbitration CLI")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help=f"Disagreements CSV path. Default: {DEFAULT_CSV}")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--cluster", metavar="NAME",
                   help="Only arbitrate cases matching cluster: 'compression', "
                        "'baseline', etc, or category like 'rag', or 'rag-compression'.")
    g.add_argument("--by-category", metavar="cs,rag,rea,sum",
                   help="Iterate categories in the listed order; within each, descending Δ.")
    g.add_argument("--shuffle", action="store_true",
                   help="Shuffle order (deterministic via --seed).")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    p.add_argument("--start-from", type=int, default=None, metavar="N",
                   help="Skip to the Nth case (1-indexed in the active ordering).")
    p.add_argument("--min-delta", type=float, default=0.0, metavar="X",
                   help="Only arbitrate cases where |Δ| > X. Sub-threshold cases are "
                        "auto-resolved by writing canonical_score = median(judge_a, "
                        "judge_b) and marking arbitration_method='median_canonical_auto'. "
                        "Default 0.0 means arbitrate all disagreements. Day 11 H2 used "
                        "0.3 to focus 16 substantive disagreements while auto-handling "
                        "64 minor ones.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")

    cases = load_cases(args.csv)
    prompts_by_id = {p.prompt_id: p for p in load_all_prompts()}

    # If --min-delta is set, auto-resolve sub-threshold cases via median
    # canonical, then filter the ordering to above-threshold cases only.
    if args.min_delta > 0:
        n_auto = auto_resolve_sub_threshold(args.csv, cases, args.min_delta)
        # Re-load to pick up the just-written human_score / arbitration_method
        cases = load_cases(args.csv)
        n_above = sum(1 for c in cases if c.delta > args.min_delta)
        print(f"\n--min-delta {args.min_delta} active:")
        print(f"  auto-resolved (median canonical): {n_auto} sub-threshold cases this run")
        print(f"  to arbitrate interactively:        {n_above} above-threshold cases")

    if args.cluster:
        ordered = apply_ordering(cases, mode="cluster", cluster=args.cluster)
    elif args.by_category:
        cats = [s.strip() for s in args.by_category.split(",")]
        ordered = apply_ordering(cases, mode="by-category", by_category=cats)
    elif args.shuffle:
        ordered = apply_ordering(cases, mode="shuffle", shuffle=True, seed=args.seed)
    else:
        ordered = apply_ordering(cases, mode="compression-first")

    # When --min-delta is set, restrict the active ordering to above-threshold
    # cases. Sub-threshold ones already have human_score populated and would
    # be skipped by find_resume_index anyway, but filtering here makes the
    # progress display "[N/16]" rather than "[N/80]".
    if args.min_delta > 0:
        ordered = [i for i in ordered if cases[i].delta > args.min_delta]

    if args.start_from is not None:
        start = max(0, args.start_from - 1)
    else:
        start = find_resume_index(cases, ordered)

    n_already = sum(1 for c in cases if c.is_scored)
    print(f"\nDay 11 arbitration — {len(cases)} cases total in CSV")
    if args.min_delta > 0:
        print(f"  threshold filter:  arbitrating cases with |Δ| > {args.min_delta} — {len(ordered)} cases")
    print(f"  ordering scope:   {len(ordered)} cases")
    print(f"  already scored:   {n_already} cases")
    print(f"  resume position:  {start + 1} / {len(ordered)}")
    if start >= len(ordered):
        print("  All cases in this ordering are scored. Pass --start-from 1 to revisit.")
        return
    print()
    _arbitration_loop(args.csv, cases, ordered, prompts_by_id, start)


if __name__ == "__main__":
    main()
