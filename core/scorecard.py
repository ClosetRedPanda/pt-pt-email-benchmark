"""Canonical Task 8 multidimensional scorecard helpers.

This module deliberately contains only aggregation/normalisation logic.  It does
not invent a composite "overall quality" score; the benchmark remains
multidimensional by design.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


SCORECARD_SECTIONS = {
    "understanding": (
        "schema_validity_pct",
        "category_acc_pct",
        "urgency_acc_pct",
        "action_req_acc_pct",
        "entity_f1",
    ),
    "generation": (
        "instruction_adherence_pct",
        "semantic_preservation_pct",
    ),
    "ptpt": (
        "euptvid_probability",
        "ptpt_compliance_pct",
        "ptbr_leakage_pct",
        "wf_score",
    ),
    "writing": (
        "grammar_errors_per_email",
        "spelling_errors_per_email",
        "structural_failures_pct",
        "repetition_pct",
        "local_writing_quality",
    ),
    "performance": (
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "throughput_emails_per_min",
        "tokens_per_second",
        "cost_per_1k_emails_usd",
    ),
}


def _observed_elapsed_ms(records: List[Dict[str, Any]]) -> float:
    """Return wall-clock span for concurrent runs; fallback to summed latency for legacy files."""
    starts = []
    finishes = []
    for record in records:
        try:
            if record.get("started_at") is not None and record.get("finished_at") is not None:
                starts.append(float(record["started_at"]))
                finishes.append(float(record["finished_at"]))
        except (TypeError, ValueError):
            continue
    if starts and finishes and len(starts) == len(records):
        return max(0.0, (max(finishes) - min(starts)) * 1000.0)
    return sum(max(0.0, float(r.get("latency_ms") or 0.0)) for r in records)


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    frac = k - lo
    return vals[lo] + (vals[hi] - vals[lo]) * frac


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _record_cost(record: Dict[str, Any]) -> Optional[float]:
    """Use exact catalog pricing, then a valid provider-reported cost."""
    value = record.get("cost_usd")
    if value is None:
        usage = (record.get("raw_response") or {}).get("usage", {})
        details = usage.get("cost_details", {}) if isinstance(usage, dict) else {}
        value = details.get("upstream_inference_cost", usage.get("cost") if isinstance(usage, dict) else None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def build_elaboration_scorecard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate generation/PT-PT/WQ/performance metrics for elaboration."""
    if not records:
        return {}

    successful = [r for r in records if r.get("status", "success") == "success" and not r.get("error")]
    latencies = [_safe_num(r.get("latency_ms")) for r in successful if r.get("latency_ms") is not None]
    # Unknown/missing cost must never be silently coerced to 0.0. Exact
    # catalog pricing is preferred, with trusted provider usage metadata as a
    # fallback; genuinely unknown costs remain explicitly unknown.
    known_costs = [cost for r in successful if (cost := _record_cost(r)) is not None]
    unknown_cost_count = sum(1 for r in successful if _record_cost(r) is None)
    prompt_tokens = sum(int(_safe_num(r.get("prompt_tokens"))) for r in successful)
    completion_tokens = sum(int(_safe_num(r.get("completion_tokens"))) for r in successful)
    wall_ms = _observed_elapsed_ms(records)
    total_cost = None if unknown_cost_count else (sum(known_costs) if known_costs else None)

    # PT-PT metrics only apply to outputs that were explicitly required to be PT-PT.
    # In elaboration records that requirement is carried by target_lang.
    ptpt = [r for r in successful if str(r.get("target_lang", "")).strip().lower() == "pt-pt"]
    wf_vals = [float(r["wf_score"]) for r in ptpt if r.get("wf_score") is not None]
    euptvid = [float(r["euptvid_probability"]) for r in ptpt if r.get("euptvid_probability") is not None]
    comp = [float(r["ptpt_compliance_pct"]) for r in ptpt if r.get("ptpt_compliance_pct") is not None]
    leakage_flags = [r.get("ptbr_leakage_detected") for r in ptpt if isinstance(r.get("ptbr_leakage_detected"), bool)]

    adh = [r.get("instruction_adherence_pct") for r in successful if r.get("instruction_adherence_pct") is not None]
    sem = [r.get("semantic_preservation_pct") for r in successful if r.get("semantic_preservation_pct") is not None]
    wq = [r.get("writing_quality_score") for r in successful if r.get("writing_quality_score") is not None]
    writing_details = [r.get("writing_quality") or {} for r in successful]
    grammar = [d.get("grammar_error_count") for d in writing_details if d.get("grammar_error_count") is not None]
    spelling = [d.get("spelling_error_count") for d in writing_details if d.get("spelling_error_count") is not None]
    repetition = [d.get("repetition_count") for d in writing_details if d.get("repetition_count") is not None]
    structural = [d.get("structural_issue_count") for d in writing_details if d.get("structural_issue_count") is not None]

    return {
        "samples": len(records),
        "requested_samples": len(records),
        "attempted_samples": len(records),
        "successful_samples": len(successful),
        "failed_samples": len(records) - len(successful),
        "ptpt_samples": len(ptpt),
        "instruction_adherence_pct": _mean(adh),
        "semantic_preservation_pct": _mean(sem),
        "euptvid_probability": _mean(euptvid),
        "ptpt_compliance_pct": _mean(comp),
        "ptbr_leakage_pct": (sum(bool(v) for v in leakage_flags) / len(leakage_flags) * 100.0) if leakage_flags else None,
        "wf_score": _mean(wf_vals),
        "local_writing_quality": _mean(wq),
        "avg_grammar_errors_per_email": _mean(grammar),
        "avg_spelling_errors_per_email": _mean(spelling),
        "structural_failures_pct": (sum(value > 0 for value in structural) / len(structural) * 100.0) if structural else None,
        "repetition_pct": (sum(value > 0 for value in repetition) / len(repetition) * 100.0) if repetition else None,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p90_ms": _percentile(latencies, 90),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
        "throughput_emails_per_min": (len(successful) / (wall_ms / 60000.0)) if wall_ms > 0 else 0.0,
        "tokens_per_second": ((prompt_tokens + completion_tokens) / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
        "cost_per_1k_emails_usd": (total_cost / len(successful) * 1000.0) if (successful and total_cost is not None) else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost_usd": total_cost,
        "unknown_cost_samples": unknown_cost_count,
        "denominators": {
            "instruction_adherence_pct": len(adh),
            "semantic_preservation_pct": len(sem),
            "euptvid_probability": len(euptvid),
            "ptpt_compliance_pct": len(comp),
            "ptbr_leakage_pct": len(leakage_flags),
            "wf_score": len(wf_vals),
            "local_writing_quality": len(wq),
            "grammar_errors_per_email": len(grammar),
            "spelling_errors_per_email": len(spelling),
            "structural_failures_pct": len(structural),
            "repetition_pct": len(repetition),
            "latency_ms": len(latencies),
            "cost_usd": len(known_costs),
        },
        "unavailable_metrics": sorted({
            metric for metric, count in {
                "instruction_adherence_pct": len(adh),
                "semantic_preservation_pct": len(sem),
                "euptvid_probability": len(euptvid),
                "ptpt_compliance_pct": len(comp),
                "ptbr_leakage_pct": len(leakage_flags),
                "wf_score": len(wf_vals),
                "local_writing_quality": len(wq),
            }.items() if count == 0
        }),
    }


def format_scorecard(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stable nested representation used by CLI/report/UI."""
    return {
        "understanding": {
            "schema_validity_pct": summary.get("schema_validity_pct"),
            "category_acc_pct": summary.get("category_acc_pct"),
            "urgency_acc_pct": summary.get("urgency_acc_pct"),
            "action_req_acc_pct": summary.get("action_req_acc_pct"),
            "entity_f1": summary.get("entity_f1"),
        },
        "generation": {
            "instruction_adherence_pct": summary.get("instruction_adherence_pct"),
            "semantic_preservation_pct": summary.get("semantic_preservation_pct"),
        },
        "ptpt": {
            "euptvid_probability": summary.get("euptvid_probability"),
            "ptpt_compliance_pct": summary.get("ptpt_compliance_pct"),
            "ptbr_leakage_pct": summary.get("ptbr_leakage_pct"),
            "wf_score": summary.get("wf_score"),
        },
        "writing": {
            "grammar_errors_per_email": summary.get("avg_grammar_errors_per_email"),
            "spelling_errors_per_email": summary.get("avg_spelling_errors_per_email"),
            "structural_failures_pct": summary.get("structural_failures_pct"),
            "repetition_pct": summary.get("repetition_pct"),
            "local_writing_quality": summary.get("local_writing_quality"),
        },
        "performance": {
            "latency_p50_ms": summary.get("latency_p50_ms"),
            "latency_p90_ms": summary.get("latency_p90_ms"),
            "latency_p95_ms": summary.get("latency_p95_ms"),
            "latency_p99_ms": summary.get("latency_p99_ms"),
            "throughput_emails_per_min": summary.get("throughput_emails_per_min"),
            "tokens_per_second": summary.get("tokens_per_second"),
            "cost_per_1k_emails_usd": summary.get("cost_per_1k_emails_usd"),
        },
    }
