"""Day 12 step 2 — populate canonical_score column on results.

Two-rule population:
  flag = 0 (agreement)    → canonical_score = mean(judge_a_score, judge_b_score)
  flag = 1 (disagreement) → canonical_score = human_score from scoring/disagreements.csv
                            joined on (prompt_id, model, lever)

The 80 disagreement-CSV rows are 16 human-arbitrated + 64 median_canonical_auto;
both kinds carry a final canonical value in the human_score column.
"""

from __future__ import annotations

import csv
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "results.db"
CSV = Path(__file__).resolve().parents[1] / "scoring" / "disagreements.csv"


def load_csv_map() -> dict[tuple[str, str, str], tuple[float, str]]:
    out: dict[tuple[str, str, str], tuple[float, str]] = {}
    with CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["prompt_id"], row["model"], row["lever"])
            out[key] = (float(row["human_score"]), row["arbitration_method"])
    return out


def main() -> None:
    csv_map = load_csv_map()
    print(f"[csv] loaded {len(csv_map)} disagreement-resolution rows")
    methods = Counter(m for _, m in csv_map.values())
    print(f"[csv] arbitration_method: {dict(methods)}")

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- agreement rows ---
    cur.execute(
        """
        UPDATE results
        SET canonical_score = (judge_a_score + judge_b_score) / 2.0
        WHERE judge_disagreement_flag = 0
          AND judge_a_score IS NOT NULL
          AND judge_b_score IS NOT NULL
        """
    )
    agreement_updated = cur.rowcount
    print(f"[update] agreement rows: {agreement_updated} updated")

    # --- disagreement rows (joined to CSV by (prompt_id, model, lever)) ---
    cur.execute(
        """
        SELECT result_id, prompt_id, model, optimisation_lever
        FROM results
        WHERE judge_disagreement_flag = 1
        """
    )
    disagreement_rows = cur.fetchall()
    print(f"[db] disagreement rows in DB: {len(disagreement_rows)}")

    missing: list[tuple[str, str, str]] = []
    matched = 0
    for r in disagreement_rows:
        key = (r["prompt_id"], r["model"], r["optimisation_lever"])
        if key not in csv_map:
            missing.append(key)
            continue
        human_score, _ = csv_map[key]
        cur.execute(
            "UPDATE results SET canonical_score = ? WHERE result_id = ?",
            (human_score, r["result_id"]),
        )
        matched += 1

    print(f"[update] disagreement rows updated from CSV: {matched}")
    if missing:
        print(f"[WARN] {len(missing)} DB disagreement rows have no CSV match:")
        for k in missing[:10]:
            print(f"   {k}")

    # CSV keys not present in DB disagreement rows
    db_keys = {(r["prompt_id"], r["model"], r["optimisation_lever"]) for r in disagreement_rows}
    csv_extras = [k for k in csv_map if k not in db_keys]
    if csv_extras:
        print(f"[WARN] {len(csv_extras)} CSV rows not in DB disagreements:")
        for k in csv_extras[:10]:
            print(f"   {k}")

    conn.commit()

    # --- verification ---
    # Tier-2 rows are the population target — Tier-1-only rows (no judge scoring)
    # legitimately stay NULL on canonical_score.
    cur.execute("SELECT COUNT(*) FROM results")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM results "
        "WHERE judge_a_score IS NOT NULL AND judge_b_score IS NOT NULL"
    )
    tier_2 = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM results "
        "WHERE judge_a_score IS NOT NULL AND judge_b_score IS NOT NULL "
        "AND canonical_score IS NULL"
    )
    tier_2_nulls = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM results WHERE canonical_score IS NOT NULL")
    populated = cur.fetchone()[0]
    print(f"\n[verify] total rows: {total}")
    print(f"[verify] tier-2 rows (judge_a/b not null): {tier_2}")
    print(f"[verify] tier-2 rows with NULL canonical_score: {tier_2_nulls}  (target: 0)")
    print(f"[verify] canonical_score populated: {populated}  (target: {tier_2})")
    print(f"[verify] tier-1-only rows left NULL: {total - populated}  (expected: {total - tier_2})")

    cur.execute(
        "SELECT MIN(canonical_score), MAX(canonical_score), AVG(canonical_score) "
        "FROM results WHERE canonical_score IS NOT NULL"
    )
    mn, mx, mean = cur.fetchone()
    print(f"[stats] min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}")

    # by flag (tier-2 only)
    for flag in (0, 1):
        cur.execute(
            "SELECT MIN(canonical_score), MAX(canonical_score), AVG(canonical_score), COUNT(*) "
            "FROM results WHERE judge_disagreement_flag = ? AND canonical_score IS NOT NULL",
            (flag,),
        )
        mn, mx, mean, n = cur.fetchone()
        label = "agreement" if flag == 0 else "disagreement"
        print(
            f"[stats:{label}] n={n} min={mn:.3f} max={mx:.3f} mean={mean:.3f}"
        )

    # histogram (populated rows only)
    cur.execute("SELECT canonical_score FROM results WHERE canonical_score IS NOT NULL")
    scores = [row[0] for row in cur.fetchall()]
    buckets = Counter()
    for s in scores:
        b = min(int(s * 10), 9)  # 0.0–0.1 .. 0.9–1.0
        buckets[b] += 1
    print("\n[hist] canonical_score distribution (10 buckets):")
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        bar = "#" * (buckets[b] // 10)
        print(f"   {lo:.1f}–{hi:.1f}  {buckets[b]:4d}  {bar}")
    print(f"   median={statistics.median(scores):.3f}  stdev={statistics.stdev(scores):.3f}")

    # spot-check: 3 agreement + 2 disagreement
    print("\n[spot] 3 agreement rows:")
    cur.execute(
        """
        SELECT prompt_id, model, optimisation_lever, judge_a_score, judge_b_score,
               canonical_score
        FROM results
        WHERE judge_disagreement_flag = 0
          AND judge_a_score IS NOT NULL
          AND judge_b_score IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 3
        """
    )
    for row in cur.fetchall():
        expected = (row["judge_a_score"] + row["judge_b_score"]) / 2.0
        ok = abs(row["canonical_score"] - expected) < 1e-9
        flag = "OK" if ok else "MISMATCH"
        print(
            f"   [{flag}] {row['prompt_id']} {row['model']} {row['optimisation_lever']}  "
            f"a={row['judge_a_score']:.2f} b={row['judge_b_score']:.2f} "
            f"→ canon={row['canonical_score']:.3f}  expected={expected:.3f}"
        )

    print("\n[spot] 2 disagreement rows:")
    cur.execute(
        """
        SELECT prompt_id, model, optimisation_lever, judge_a_score, judge_b_score,
               canonical_score
        FROM results
        WHERE judge_disagreement_flag = 1
        ORDER BY RANDOM()
        LIMIT 2
        """
    )
    for row in cur.fetchall():
        key = (row["prompt_id"], row["model"], row["optimisation_lever"])
        csv_score, method = csv_map[key]
        ok = abs(row["canonical_score"] - csv_score) < 1e-9
        flag = "OK" if ok else "MISMATCH"
        print(
            f"   [{flag}] {row['prompt_id']} {row['model']} {row['optimisation_lever']}  "
            f"a={row['judge_a_score']:.2f} b={row['judge_b_score']:.2f} "
            f"→ canon={row['canonical_score']:.3f}  csv_human={csv_score:.3f} "
            f"({method})"
        )

    # Confirm Day 10/11 columns untouched: pick 3 rows and show they still hold
    # judge_a_*, judge_b_*, judge_b_mistral_* values.
    print("\n[audit] judge columns untouched on 3 sample rows:")
    cur.execute(
        """
        SELECT prompt_id, model, optimisation_lever,
               judge_a_score, judge_b_score, judge_b_mistral_score,
               judge_a_reasoning IS NOT NULL AS a_has_reasoning,
               judge_b_reasoning IS NOT NULL AS b_has_reasoning,
               judge_b_mistral_reasoning IS NOT NULL AS m_has_reasoning
        FROM results
        ORDER BY RANDOM()
        LIMIT 3
        """
    )
    for row in cur.fetchall():
        print(
            f"   {row['prompt_id']} {row['model']} {row['optimisation_lever']}: "
            f"a={row['judge_a_score']} b={row['judge_b_score']} "
            f"mistral={row['judge_b_mistral_score']}  "
            f"reasoning a/b/m={row['a_has_reasoning']}/{row['b_has_reasoning']}/{row['m_has_reasoning']}"
        )

    conn.close()
    print("\n[done] canonical_score populated.")


if __name__ == "__main__":
    main()
