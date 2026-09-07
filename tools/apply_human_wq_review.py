#!/usr/bin/env python3
"""Merge independent human WQ ratings from one or more candidate pools.

Default candidate pools:
    data/wq_calibration_candidates.jsonl
    data/wq_calibration_boundary_pairs_v2.jsonl  (used automatically if present)

Review file:
    data/wq_calibration_human_review.jsonl

Output:
    data/wq_calibration_candidates_human_verified.jsonl

The source candidate files are never modified. Candidate IDs must be unique
across the supplied pools. Reviews may be for any of those candidates.

Examples:
    python tools/apply_human_wq_review.py
    python tools/apply_human_wq_review.py --candidates data/foo.jsonl data/bar.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

# These paths were previously sourced from data/config/wq_review_paths.json,
# which has been removed (it was the only file in data/config/ that any code
# actually read; everything else there was unused/duplicate). Inlined here
# since this is the sole consumer.
DEFAULT_CANDIDATES = [Path("data/wq_calibration_candidates.jsonl")]
OPTIONAL_CANDIDATES = [Path("data/wq_calibration_boundary_pairs_v2.jsonl")]
REVIEWS = Path("data/wq_calibration_human_review.jsonl")
OUTPUT = Path("data/wq_calibration_candidates_human_verified.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows: list[dict] = []
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path} line {line_no}: {exc}")
    return rows


def merge_candidate_pools(paths: Iterable[Path]) -> list[dict]:
    merged: list[dict] = []
    seen: dict[str, Path] = {}
    for path in paths:
        rows = load_jsonl(path)
        if not rows:
            raise SystemExit(f"No candidates found: {path}")
        for row in rows:
            cid = row.get("id")
            if not cid:
                raise SystemExit(f"Candidate missing id in {path}: {row}")
            if cid in seen:
                raise SystemExit(
                    f"Duplicate candidate ID {cid!r} found in {seen[cid]} and {path}"
                )
            seen[cid] = path
            merged.append(row)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge human WQ reviews into one or more candidate pools."
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=Path,
        help="Candidate JSONL files. If omitted, use the default candidate pool plus the boundary pool when present.",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=REVIEWS,
        help=f"Human review JSONL (default: {REVIEWS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help=f"Verified output JSONL (default: {OUTPUT})",
    )
    args = parser.parse_args()

    if args.candidates:
        candidate_paths = args.candidates
    else:
        candidate_paths = list(DEFAULT_CANDIDATES)
        candidate_paths.extend(p for p in OPTIONAL_CANDIDATES if p.exists())

    candidates = merge_candidate_pools(candidate_paths)
    reviews = load_jsonl(args.reviews)

    if not reviews:
        raise SystemExit(f"No human reviews found: {args.reviews}")

    ratings: dict[str, int] = {}
    for review in reviews:
        cid = review.get("id")
        score = review.get("human_quality_score")
        if not cid or score not in {1, 2, 3, 4, 5}:
            raise SystemExit(f"Invalid review row: {review}")
        if cid in ratings and ratings[cid] != score:
            raise SystemExit(f"Conflicting ratings for {cid}")
        ratings[cid] = score

    candidate_ids = {row["id"] for row in candidates}
    unknown = sorted(set(ratings) - candidate_ids)
    if unknown:
        raise SystemExit(
            "Reviews contain unknown candidate IDs. Make sure every candidate "
            "pool used by the review tool is included here:\n" + "\n".join(unknown)
        )

    missing = [row["id"] for row in candidates if row["id"] not in ratings]
    if missing:
        raise SystemExit(
            "Missing human ratings for candidate IDs:\n" + "\n".join(missing)
        )

    verified: list[dict] = []
    for original in candidates:
        row = dict(original)
        row["provisional_quality_target"] = row.get("intended_quality_band")
        row["gold_quality_score"] = ratings[row["id"]]
        row["wq_calibration_status"] = "human_verified"
        row["wq_calibration_eligible"] = True
        row["human_review_source"] = str(args.reviews)
        verified.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in verified:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Candidate pools: {len(candidate_paths)}")
    for path in candidate_paths:
        print(f"  {path}")
    print(f"Total verified candidates: {len(verified)}")
    print(f"Output: {args.output}")
    print("Human gold distribution:")
    for score in range(1, 6):
        count = sum(r["gold_quality_score"] == score for r in verified)
        print(f"  {score}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
