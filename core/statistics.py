"""Small dependency-free paired comparison statistics for benchmark artifacts."""
from __future__ import annotations

import random
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_METRICS = (
    "instruction_adherence_pct",
    "semantic_preservation_pct",
    "euptvid_probability",
    "ptpt_compliance_pct",
    "ptbr_leakage_detected",
    "wf_score",
    "writing_quality_score",
)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _paired_differences(
    first_rows: Iterable[Dict[str, Any]],
    second_rows: Iterable[Dict[str, Any]],
    metric: str,
) -> List[float]:
    first = {str(row.get("id")): row for row in first_rows if row.get("id") is not None}
    second = {str(row.get("id")): row for row in second_rows if row.get("id") is not None}
    differences = []
    for row_id in sorted(set(first) & set(second)):
        left = first[row_id]
        right = second[row_id]
        if left.get("status", "success") != "success" or left.get("error"):
            continue
        if right.get("status", "success") != "success" or right.get("error"):
            continue
        left_value = _number(left.get(metric))
        right_value = _number(right.get(metric))
        if left_value is not None and right_value is not None:
            differences.append(right_value - left_value)
    return differences


def paired_bootstrap(
    first_rows: Iterable[Dict[str, Any]],
    second_rows: Iterable[Dict[str, Any]],
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    repetitions: int = 4000,
    seed: int = 20260907,
) -> Dict[str, Any]:
    """Return paired differences and deterministic percentile bootstrap intervals.

    Differences are always ``second artifact minus first artifact``. The method
    resamples task-level differences, preserving pairing between models.
    """
    first = list(first_rows)
    second = list(second_rows)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    report: Dict[str, Any] = {
        "definition": "second artifact minus first artifact",
        "method": "paired task bootstrap percentile interval",
        "seed": seed,
        "repetitions": repetitions,
        "metrics": {},
    }
    for metric in metrics:
        differences = _paired_differences(first, second, metric)
        if not differences:
            report["metrics"][metric] = {"n": 0, "unavailable": True}
            continue
        rng = random.Random(seed + sum(ord(char) for char in metric))
        bootstrap_means = [
            mean(rng.choice(differences) for _ in differences)
            for _ in range(repetitions)
        ]
        report["metrics"][metric] = {
            "n": len(differences),
            "mean_difference": mean(differences),
            "median_difference": median(differences),
            "wins": sum(value > 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "ci95": [
                _percentile(bootstrap_means, 2.5),
                _percentile(bootstrap_means, 97.5),
            ],
        }
    return report
