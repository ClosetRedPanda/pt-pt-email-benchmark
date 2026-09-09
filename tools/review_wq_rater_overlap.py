#!/usr/bin/env python3
"""Interactively rate the blinded Task 7 WQ overlap set.

Run this from the Task 7 project root, or pass explicit --template/--output paths.
The primary/gold score is never shown during rating. After each rating, the tool
writes a record containing the entered second score and the existing primary score
from the verified calibration dataset, so calibrate_wq.py can measure agreement.

This tool is for a REAL independent human rater. It does not fabricate ratings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# If copied into <project>/tools/, these resolve correctly.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_DIR / "data" / "wq_rater_agreement_template.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "wq_rater_agreement_log.jsonl"
DEFAULT_PRIMARY = PROJECT_DIR / "data" / "wq_calibration_candidates_human_verified.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
    return rows


def prompt_score(index: int, total: int, email: str) -> int | None:
    print("\n" + "=" * 72)
    print(f"INDEPENDENT WQ RATING {index}/{total}")
    print("=" * 72)
    print(email.strip())
    print("\nRate the writing itself, independently of any previous score.")
    print("1 = very poor")
    print("2 = poor")
    print("3 = acceptable / mixed")
    print("4 = good")
    print("5 = excellent")
    print("Enter 1-5, 's' to skip, or 'q' to quit.")

    while True:
        raw = input("\nIndependent human rating [1-5]: ").strip().lower()
        if raw == "q":
            return None
        if raw == "s":
            return -1
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("Please enter 1, 2, 3, 4, 5, s, or q.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    ap.add_argument("--rater-id", required=True, help="Independent human rater identifier")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output records")
    args = ap.parse_args()

    if args.rater_id.lower().startswith("ai"):
        print("ERROR: --rater-id must identify a human rater, not an AI adjudicator.", file=sys.stderr)
        return 2

    template_rows = load_jsonl(args.template)
    primary_rows = load_jsonl(args.primary)
    primary_by_id = {
        row["id"]: row.get("gold_quality_score")
        for row in primary_rows
        if row.get("id") is not None
    }

    # Resume only this rater's records; do not mix with prior AI records.
    completed: dict[str, dict] = {}
    if args.resume and args.output.exists():
        for row in load_jsonl(args.output):
            if row.get("rater_id") == args.rater_id and row.get("rater_type") == "human":
                completed[row["id"]] = row

    print("=" * 72)
    print("WQ INDEPENDENT HUMAN OVERLAP REVIEW")
    print("=" * 72)
    print(f"Template: {args.template}")
    print(f"Rater:    {args.rater_id}")
    print(f"Rows:     {len(template_rows)}")
    print(f"Already completed by this rater: {len(completed)}")
    print("\nThe primary/gold scores are intentionally hidden during rating.")
    print("Your entered ratings are saved after each item.")
    input("\nPress Enter to begin...")

    # Preserve existing non-human records (e.g. previous AI adjudication) and
    # replace/append this rater's human records as the tool progresses.
    existing_rows = load_jsonl(args.output) if args.output.exists() else []
    existing_other = [
        row for row in existing_rows
        if not (row.get("rater_id") == args.rater_id and row.get("rater_type") == "human")
    ]

    rated_count = len(completed)
    skipped_count = 0

    for index, item in enumerate(template_rows, 1):
        item_id = item.get("id")
        if not item_id or item_id not in primary_by_id:
            print(f"Skipping invalid/missing primary row: {item_id!r}")
            skipped_count += 1
            continue
        if item_id in completed:
            continue

        score = prompt_score(index, len(template_rows), item.get("email", ""))
        if score is None:
            print("\nStopped by user. Progress has been saved.")
            break
        if score == -1:
            skipped_count += 1
            continue

        record = {
            "id": item_id,
            "primary_score": int(primary_by_id[item_id]),
            "second_score": int(score),
            "rater_id": args.rater_id,
            "rater_type": "human",
            "independence_note": "Independent human rating entered without seeing the primary score.",
        }
        completed[item_id] = record
        rated_count += 1

        combined = existing_other + list(completed.values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for row in combined:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {rated_count}/{len(template_rows)} ratings to {args.output}")

    print("\n" + "=" * 72)
    print("REVIEW COMPLETE")
    print("=" * 72)
    print(f"Human ratings recorded for this rater: {len(completed)}")
    print(f"Skipped this run: {skipped_count}")
    print(f"Output: {args.output}")
    print("\nNext step:")
    print(f"  python tests/calibrate_wq.py --agreement {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
