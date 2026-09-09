#!/usr/bin/env python3
"""
Interactive human review tool for WQ calibration candidates.

Usage:
    python review_wq_calibration.py
    python review_wq_calibration.py data/wq_calibration_candidates.jsonl

The original candidate file is never modified.
Ratings are saved to data/wq_calibration_human_review.jsonl by default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


DEFAULT_INPUT = Path("data/wq_calibration_candidates.jsonl")
DEFAULT_OUTPUT = Path("data/wq_calibration_human_review.jsonl")


def load_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no}: {exc}")
            if "id" not in row or "email" not in row:
                raise SystemExit(
                    f"Candidate on line {line_no} must contain 'id' and 'email'."
                )
            rows.append(row)
    return rows


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    existing = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row
    return existing


def prompt_rating() -> int | None:
    while True:
        value = input(
            "\nYour rating [1-5], or 's' to skip, 'q' to quit: "
        ).strip().lower()
        if value == "q":
            return None
        if value == "s":
            return 0
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)
        print("Please enter 1, 2, 3, 4, 5, 's', or 'q'.")


def save(path: Path, reviewed: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in reviewed.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    )

    candidates = load_candidates(input_path)
    if not candidates:
        print("No candidates found.")
        return 1

    reviewed = load_existing(output_path)

    pending = [r for r in candidates if r["id"] not in reviewed]

    print("=" * 72)
    print("WQ HUMAN CALIBRATION REVIEW")
    print("=" * 72)
    print(f"Candidates: {len(candidates)}")
    print(f"Already reviewed: {len(candidates) - len(pending)}")
    print(f"Remaining: {len(pending)}")
    print()
    print("Rate the writing itself from 1 to 5.")
    print("Do NOT try to match any intended/provisional score.")
    print("1 = very poor")
    print("2 = poor")
    print("3 = acceptable / mixed")
    print("4 = good")
    print("5 = excellent")
    print()
    print("Your ratings are saved after every candidate.")
    print("The original candidate file is never modified.")
    input("Press Enter to begin...")

    for index, candidate in enumerate(pending, 1):
        print("\n" + "=" * 72)
        print(f"CANDIDATE {index}/{len(pending)}")
        print("=" * 72)
        print(f"ID: {candidate['id']}")
        print(f"Task type: {candidate.get('task_type', '')}")
        print(f"Language: {candidate.get('lang', '')}")
        print(f"Subject: {candidate.get('subject', '')}")
        print()
        print("-" * 72)
        print(candidate["email"])
        print("-" * 72)

        # Deliberately do not display intended_quality_band or reference_judgments.
        rating = prompt_rating()

        if rating is None:
            save(output_path, reviewed)
            print(f"\nStopped. Progress saved to {output_path}")
            return 0

        if rating == 0:
            print("Skipped; you can review it again later.")
            continue

        reviewed[candidate["id"]] = {
            "id": candidate["id"],
            "human_quality_score": rating,
            "review_status": "human_verified",
        }
        save(output_path, reviewed)
        print(f"Saved rating {rating} for {candidate['id']}.")

    print("\n" + "=" * 72)
    print("REVIEW COMPLETE")
    print("=" * 72)
    print(f"Rated: {len(reviewed)}")
    print(f"Saved to: {output_path}")
    print()
    print("The provisional/intended bands were not used as ratings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
