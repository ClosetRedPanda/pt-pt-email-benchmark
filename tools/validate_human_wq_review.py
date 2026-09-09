#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

path = Path("data/wq_calibration_candidates_human_verified.jsonl")
rows = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]

assert rows, "No verified candidates"
assert all(r["wq_calibration_status"] == "human_verified" for r in rows)
assert all(r["wq_calibration_eligible"] is True for r in rows)
assert all(r.get("gold_quality_score") in {1,2,3,4,5} for r in rows)

print("Human-reviewed rows:", len(rows))
print("Gold distribution:", Counter(r["gold_quality_score"] for r in rows))
print("C3 human-review validation: PASS")
