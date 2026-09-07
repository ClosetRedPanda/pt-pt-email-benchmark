#!/usr/bin/env python3
"""Prepare a blinded second-rater WQ overlap sheet.

This script intentionally does NOT generate second ratings. It samples a fixed,
reproducible set of human-verified rows and writes a blank template that a second
rater must complete independently.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "wq_calibration_candidates_human_verified.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "wq_rater_agreement_template.jsonl"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("wq_calibration_status") == "human_verified" and r.get("gold_quality_score") in {1,2,3,4,5}]
    rng = random.Random(args.seed)
    chosen = []
    # Prefer coverage across human bands, then fill deterministically.
    for band in range(1, 6):
        candidates = [r for r in rows if r.get("gold_quality_score") == band]
        if candidates:
            chosen.append(rng.choice(candidates))
    remaining = [r for r in rows if r not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[:max(0, args.n - len(chosen))])
    chosen = chosen[:args.n]

    out = []
    for r in chosen:
        out.append({
            "id": r["id"],
            "email": r["email"],
            "second_score": None,
            "rater_id": "",
            "instructions": "Rate independently from 1 to 5 without consulting the primary score.",
        })
    args.output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    print(f"Wrote blinded overlap template: {args.output} ({len(out)} rows)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
