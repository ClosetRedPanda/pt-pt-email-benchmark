"""Canonical Task 8 multidimensional scorecard helpers.

This module deliberately contains only aggregation/normalisation logic.  It does
not invent a composite "overall quality" score; the benchmark remains
multidimensional by design.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core._util import mean_or_none, percentile


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
        "ptpt_compliance_graded_pct",
        "ptbr_leakage_pct",
        "wf_score",
    ),
    "writing": (
        # Issue 3: the defect-only aggregate is the primary writing-quality metric
        # for model comparison; the calibrated aggregate follows as the
        # conservative headline. Listed in that order everywhere downstream.
        "local_writing_quality_defect_only",
        "local_writing_quality",
        "grammar_errors_per_email",
        "spelling_errors_per_email",
        "structural_failures_pct",
        "repetition_pct",
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


def _observed_elapsed_ms(records: List[Dict[str, Any]]) -> Optional[float]:
    """Return the observed wall-clock span of a run, in milliseconds.

    FIX (P0.1): the previous implementation fell back to ``sum(latency_ms)``
    whenever wall-clock timestamps were unavailable *or* whenever a single
    record was missing them. Summed latency is not wall-clock time: under
    concurrency N the sum overstates elapsed time by roughly N, so
    ``throughput_emails_per_min`` and ``tokens_per_second`` came out ~N times
    too low. Worse, the failure was silent and plausible-looking.

    Concurrency is not recorded per row (only in the run manifest), so elapsed
    time genuinely cannot be reconstructed from latencies alone. Rather than
    emit a confidently wrong number, return ``None`` and let callers publish
    the throughput metrics as unavailable.

    Records that carry timestamps are used even when *other* records do not:
    a failed row without timestamps must not discard the whole measurement.
    """
    starts: List[float] = []
    finishes: List[float] = []
    for record in records:
        started = record.get("started_at")
        finished = record.get("finished_at")
        if started is None or finished is None:
            continue
        try:
            started_f = float(started)
            finished_f = float(finished)
        except (TypeError, ValueError):
            continue
        starts.append(started_f)
        finishes.append(finished_f)
    if not starts:
        return None
    return max(0.0, (max(finishes) - min(starts)) * 1000.0)


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
    # P0.1: throughput is only meaningful when wall-clock time was observed.
    throughput_available = wall_ms is not None and wall_ms > 0
    total_cost = None if unknown_cost_count else (sum(known_costs) if known_costs else None)

    # FIX (P0.4): a single unknown cost previously nulled the entire run's cost
    # metric, so one provider hiccup destroyed all cost data. `total_cost` (and
    # therefore `cost_per_1k_emails_usd`) keeps its strict all-or-nothing
    # semantics, because a partial total is not a total and existing consumers
    # rely on that. The partial figure is exposed alongside it instead, scoped
    # to the samples whose cost is actually known and always paired with its
    # denominator so it can never be mistaken for a full-run number.
    cost_per_1k_known_only = (
        (sum(known_costs) / len(known_costs) * 1000.0) if known_costs else None
    )

    # PT-PT metrics only apply to outputs that were explicitly required to be PT-PT.
    # In elaboration records that requirement is carried by target_lang.
    ptpt = [r for r in successful if str(r.get("target_lang", "")).strip().lower() == "pt-pt"]
    wf_vals = [float(r["wf_score"]) for r in ptpt if r.get("wf_score") is not None]
    euptvid = [float(r["euptvid_probability"]) for r in ptpt if r.get("euptvid_probability") is not None]
    comp = [float(r["ptpt_compliance_pct"]) for r in ptpt if r.get("ptpt_compliance_pct") is not None]
    comp_graded = [float(r["ptpt_compliance_graded_pct"]) for r in ptpt if r.get("ptpt_compliance_graded_pct") is not None]
    leakage_flags = [r.get("ptbr_leakage_detected") for r in ptpt if isinstance(r.get("ptbr_leakage_detected"), bool)]

    adh = [r.get("instruction_adherence_pct") for r in successful if r.get("instruction_adherence_pct") is not None]
    sem = [r.get("semantic_preservation_pct") for r in successful if r.get("semantic_preservation_pct") is not None]
    wq = [r.get("writing_quality_score") for r in successful if r.get("writing_quality_score") is not None]
    # Issue 3: `writing_quality_score` is calibrated (40-tree ensemble) and capped
    # below 100 by design so a clean sample is never indistinguishable from a
    # formally perfect score; its observed top end for genuinely flawless prose is
    # therefore ~91-96, which makes model comparisons in the "clean writing" range
    # compressed. `wq_defect_only_score` is the defect-burden counterpart computed
    # from the same violations but *before* that ceiling: flawless text reaches
    # 100.0 there, so it preserves spread exactly where the calibrated score
    # compresses. Aggregate and surface both so the unambiguous one is comparable.
    wq_defect_only = [r.get("wq_defect_only_score") for r in successful if r.get("wq_defect_only_score") is not None]
    writing_details = [r.get("writing_quality") or {} for r in successful]
    # REL-07: grammar counts exist only when a LanguageTool backend responded for
    # at least one row. When it did not, every row carries a structural 0, so the
    # old behaviour printed "grammar_errors_per_email = 0" with a full denominator
    # — indistinguishable from genuinely error-free text and silently different
    # across environments. Grammar is reported as unavailable (N/A) unless LT
    # evidence exists, and `languagetool_available` is surfaced so reports warn.
    grammar_checked = any(
        isinstance(d, dict) and bool(d.get("languagetool_api_used")) for d in writing_details
    )
    grammar = [d.get("grammar_error_count") for d in writing_details if d.get("grammar_error_count") is not None]
    if not grammar_checked:
        grammar = []
    # REPRO-1: surface which writing-quality evaluator version produced the
    # rows, so an artifact rescored or regenerated with different evaluator code
    # is identifiable from the summary alone. Falls back to None for rows that
    # predate the version field (reported as N/A, never invented). REL-11: when
    # rows disagree on the version (e.g. an artifact spliced from two runs), the
    # summary must say so instead of silently attributing everything to the
    # first row's version.
    wq_versions = [
        d.get("wq_evaluator_version")
        for d in writing_details
        if isinstance(d, dict) and isinstance(d.get("wq_evaluator_version"), str)
    ]
    distinct_wq_versions = sorted(set(wq_versions))
    wq_version_reported = (
        "mixed"
        if len(distinct_wq_versions) > 1
        else (distinct_wq_versions[0] if distinct_wq_versions else None)
    )
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
        "instruction_adherence_pct": mean_or_none(adh),
        "semantic_preservation_pct": mean_or_none(sem),
        "euptvid_probability": mean_or_none(euptvid),
        "ptpt_compliance_pct": mean_or_none(comp),
        "ptpt_compliance_graded_pct": mean_or_none(comp_graded),
        "ptbr_leakage_pct": (sum(bool(v) for v in leakage_flags) / len(leakage_flags) * 100.0) if leakage_flags else None,
        "wf_score": mean_or_none(wf_vals),
        "local_writing_quality_defect_only": mean_or_none(wq_defect_only),
        "local_writing_quality": mean_or_none(wq),
        "languagetool_available": grammar_checked,
        "wq_evaluator_version": wq_version_reported,
        "avg_grammar_errors_per_email": mean_or_none(grammar),
        "avg_spelling_errors_per_email": mean_or_none(spelling),
        "structural_failures_pct": (sum(value > 0 for value in structural) / len(structural) * 100.0) if structural else None,
        "repetition_pct": (sum(value > 0 for value in repetition) / len(repetition) * 100.0) if repetition else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p90_ms": percentile(latencies, 90),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "throughput_emails_per_min": (len(successful) / (wall_ms / 60000.0)) if throughput_available else None,
        "tokens_per_second": ((prompt_tokens + completion_tokens) / (wall_ms / 1000.0)) if throughput_available else None,
        "cost_per_1k_emails_usd": (total_cost / len(successful) * 1000.0) if (successful and total_cost is not None) else None,
        "cost_per_1k_emails_usd_known_only": cost_per_1k_known_only,
        "known_cost_samples": len(known_costs),
        "observed_wall_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost_usd": total_cost,
        "unknown_cost_samples": unknown_cost_count,
        "denominators": {
            "instruction_adherence_pct": len(adh),
            "semantic_preservation_pct": len(sem),
            "euptvid_probability": len(euptvid),
            "ptpt_compliance_pct": len(comp),
            "ptpt_compliance_graded_pct": len(comp_graded),
            "ptbr_leakage_pct": len(leakage_flags),
            "wf_score": len(wf_vals),
            "local_writing_quality_defect_only": len(wq_defect_only),
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
                "local_writing_quality_defect_only": len(wq_defect_only),
                "grammar_errors_per_email": len(grammar),
            }.items() if count == 0
        } | (set() if throughput_available else {"throughput_emails_per_min", "tokens_per_second"})),
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
            "ptpt_compliance_graded_pct": summary.get("ptpt_compliance_graded_pct"),
            "ptbr_leakage_pct": summary.get("ptbr_leakage_pct"),
            "wf_score": summary.get("wf_score"),
        },
        "writing": {
            "local_writing_quality_defect_only": summary.get("local_writing_quality_defect_only"),
            "local_writing_quality": summary.get("local_writing_quality"),
            "grammar_errors_per_email": summary.get("avg_grammar_errors_per_email"),
            "spelling_errors_per_email": summary.get("avg_spelling_errors_per_email"),
            "structural_failures_pct": summary.get("structural_failures_pct"),
            "repetition_pct": summary.get("repetition_pct"),
        },
        "performance": {
            "latency_p50_ms": summary.get("latency_p50_ms"),
            "latency_p90_ms": summary.get("latency_p90_ms"),
            "latency_p95_ms": summary.get("latency_p95_ms"),
            "latency_p99_ms": summary.get("latency_p99_ms"),
            "throughput_emails_per_min": summary.get("throughput_emails_per_min"),
            "tokens_per_second": summary.get("tokens_per_second"),
            "cost_per_1k_emails_usd": summary.get("cost_per_1k_emails_usd"),
            "cost_per_1k_emails_usd_known_only": summary.get("cost_per_1k_emails_usd_known_only"),
            "unknown_cost_samples": summary.get("unknown_cost_samples"),
        },
    }
