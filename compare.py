"""Readable comparison helpers for already-produced result files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from core.scorecard import build_elaboration_scorecard
from core.statistics import paired_bootstrap
from core.pt_dialect import evaluate_pt_dialect
from core.wf_fidelity import compute_word_fidelity_from_dialect
from core.artifacts import ArtifactValidationError, load_manifest, validate_rows
from runner import read_jsonl, score_analysis
from config import ANALYSIS_REFERENCE, ELABORATION_PROMPTS, BENCHMARK_VERSION


SECTION_LABELS = {
    "understanding": "Understanding",
    "generation": "Generation",
    "ptpt": "PT-PT fidelity",
    "writing": "Writing quality",
    "performance": "Performance",
}

LABELS = {
    "schema_validity_pct": "Schema validity",
    "category_acc_pct": "Category accuracy",
    "urgency_acc_pct": "Urgency accuracy",
    "action_req_acc_pct": "Action-required accuracy",
    "sentiment_acc_pct": "Sentiment accuracy",
    "language_variant_acc_pct": "Language-variant accuracy",
    "entity_f1": "Entity F1",
    "instruction_adherence_pct": "Instruction adherence",
    "semantic_preservation_pct": "Semantic preservation",
    "euptvid_probability": "EUPTVID probability",
    "ptpt_compliance_pct": "PT-PT compliance",
    "ptpt_compliance_graded_pct": "PT-PT compliance (graded)",
    "ptbr_leakage_pct": "PT-BR leakage",
    "wf_score": "Word fidelity",
    "local_writing_quality": "Writing quality",
    "latency_p50_ms": "Latency p50",
    "latency_p90_ms": "Latency p90",
    "latency_p95_ms": "Latency p95",
    "latency_p99_ms": "Latency p99",
    "throughput_emails_per_min": "Throughput",
    "tokens_per_second": "Tokens / second",
    "cost_per_1k_emails_usd_known_only": "Cost / 1k (known-cost samples)",
    "unknown_cost_samples": "Unknown-cost samples",
    "cost_per_1k_emails_usd": "Cost / 1,000 emails",
}


def _format_value(key: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if key.endswith("_pct"):
        return f"{float(value):.1f}%"
    if key == "euptvid_probability":
        return f"{float(value):.3f}"
    if key == "entity_f1":
        return f"{float(value):.3f}"
    if key.endswith("_ms"):
        return f"{float(value):,.0f} ms"
    if key == "throughput_emails_per_min":
        return f"{float(value):,.1f} emails/min"
    if key == "tokens_per_second":
        return f"{float(value):,.1f} tokens/s"
    if key in ("cost_per_1k_emails_usd", "cost_per_1k_emails_usd_known_only"):
        return f"${float(value):,.4f}"
    if key in ("unknown_cost_samples", "known_cost_samples"):
        return f"{int(value):,d}"
    return f"{float(value):.1f}" if isinstance(value, (int, float)) else str(value)


def _render_sections(sections: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for section, metrics in sections.items():
        if not metrics:
            continue
        lines.append(f"  {SECTION_LABELS.get(section, section.title())}")
        for key, value in metrics.items():
            label = LABELS.get(key, key.replace("_", " ").title())
            lines.append(f"    {label:<28} {_format_value(key, value)}")
    return lines


def _enrich_generation_records(rows: List[Dict[str, Any]], *, use_languagetool: bool = False) -> List[Dict[str, Any]]:
    """Backfill evaluator fields for result files created before resource fixes."""
    enriched = []
    for row in rows:
        result = dict(row)
        raw_choices = (result.get("raw_response") or {}).get("choices") or []
        finish_reason = raw_choices[0].get("finish_reason") if raw_choices and isinstance(raw_choices[0], dict) else None
        provider_error = (result.get("raw_response") or {}).get("error")
        if result.get("status") is None and (finish_reason in {"error", "failed"} or provider_error):
            result["status"] = "error"
            result["error"] = provider_error or f"provider finish_reason={finish_reason!r}"
        if (
            result.get("status", "success") == "success"
            and not result.get("error")
            and str(result.get("target_lang", "")).lower() == "pt-pt"
            and result.get("content")
            and (
                result.get("ptpt_compliance_pct") is None
                or result.get("ptbr_leakage_detected") is None
                or result.get("wf_score") is None
            )
        ):
            dialect = evaluate_pt_dialect(str(result["content"]), use_languagetool=use_languagetool)
            wf = compute_word_fidelity_from_dialect(str(result["content"]), dialect)
            result.update({
                "pt_dialect_score": dialect.get("pt_dialect_score"),
                "euptvid_probability": dialect.get("euptvid_prob"),
                "ptpt_compliance_pct": dialect.get("ptpt_compliance_pct"),
                "ptbr_leakage_detected": dialect.get("ptbr_leakage_detected"),
                "wf_score": wf.get("wf_score"),
                "dialect_evaluation": dialect,
                "wf_evaluation": wf,
            })
        enriched.append(result)
    return enriched


def validate_comparison_artifacts(
    paths: List[Path],
    *,
    kind: str,
    allow_legacy: bool = False,
) -> List[Dict[str, Any]]:
    """Validate artifacts and ensure comparable manifest metadata."""
    manifests: List[Dict[str, Any]] = []
    expected_ids = None
    if kind == "generation":
        prompts = json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8"))
        expected_ids = [str(item["id"]) for item in prompts.get("prompts", [])]
    for path in paths:
        rows = read_jsonl(path)
        sidecar = path.with_suffix(".manifest.json")
        if not sidecar.is_file():
            if not allow_legacy:
                raise ArtifactValidationError(
                    f"{path}: missing manifest; pass --allow-legacy only for exploratory comparison"
                )
            validate_rows(rows, kind=kind, strict=False)
            manifests.append({"legacy": True, "result_path": path.name})
            continue
        manifest = load_manifest(path)
        if manifest.get("artifact_kind") != kind:
            raise ArtifactValidationError(f"{path}: manifest kind is {manifest.get('artifact_kind')!r}, expected {kind!r}")
        validate_rows(rows, kind=kind, expected_ids=expected_ids, strict=True)
        if manifest.get("row_count") != len(rows):
            raise ArtifactValidationError(f"{path}: manifest row_count does not match result rows")
        manifests.append(manifest)

    versioned = [item for item in manifests if not item.get("legacy")]
    if versioned:
        if len(versioned) != len(manifests):
            raise ArtifactValidationError("cannot mix manifest-backed and legacy artifacts")
        comparable_fields = (
            "artifact_schema_version", "benchmark_version", "input_hashes",
            "evaluator_versions", "resource_hashes", "dependency_versions",
        )
        for field in comparable_fields:
            values = {json.dumps(item.get(field), sort_keys=True) for item in versioned}
            if len(values) != 1:
                raise ArtifactValidationError(f"artifacts disagree on {field}")
        if versioned[0].get("benchmark_version") != BENCHMARK_VERSION:
            raise ArtifactValidationError(
                f"artifacts use benchmark {versioned[0].get('benchmark_version')!r}, expected {BENCHMARK_VERSION!r}"
            )
    return manifests


def _pretty_report(path: Path, summary: Dict[str, Any], kind: str) -> str:
    rows = read_jsonl(path)
    model_names = sorted({str(row["model"]) for row in rows if row.get("model")})
    model = ", ".join(model_names) if model_names else "unknown model"
    if kind == "generation":
        sections = {
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
                "local_writing_quality": summary.get("local_writing_quality"),
            },
            "performance": {
                key: summary.get(key)
                for key in (
                    "latency_p50_ms", "latency_p90_ms", "latency_p95_ms",
                    "latency_p99_ms", "throughput_emails_per_min",
                    "tokens_per_second", "cost_per_1k_emails_usd",
                )
            },
        }
        title = "Generation comparison"
        counts = (
            f"  Samples: {summary.get('samples', 0)} | "
            f"Successful: {summary.get('successful_samples', 0)} | "
            f"Failed: {summary.get('failed_samples', 0)}"
        )
    else:
        title = "Analysis comparison"
        sections = {
            "understanding": {
                key: summary.get(key)
                for key in (
                    "schema_validity_pct", "category_acc_pct", "urgency_acc_pct",
                    "action_req_acc_pct", "sentiment_acc_pct",
                    "language_variant_acc_pct", "entity_f1",
                )
            },
            "coverage": {
                "samples": summary.get("samples"),
                "missing_results": summary.get("missing_results"),
                "failed_results": summary.get("failed_results"),
            },
        }
        counts = f"  Samples: {summary.get('samples', 0)}"

    lines = [
        "=" * 72,
        f"{title}: {path.name}",
        f"  Model: {model}",
        f"  Provenance: {summary.get('artifact_provenance', 'unknown')}",
        counts,
        "-" * 72,
        *_render_sections(sections),
    ]
    denominators = summary.get("denominators") or {}
    if denominators:
        lines.extend(["", "  Metric denominators: " + ", ".join(
            f"{key}={value}" for key, value in sorted(denominators.items())
        )])
    if kind == "generation":
        dialect_statuses = {
            str((row.get("dialect_evaluation") or {}).get("language_adherence_status"))
            for row in rows
        }
        if (
            any("Hunspell dictionary unavailable" in status for status in dialect_statuses)
            and summary.get("ptpt_compliance_pct") is None
        ):
            lines.extend([
                "",
                "  Note: PT-PT compliance, PT-BR leakage, and Word fidelity are unavailable",
                "  because docs/pt_PT.dic and docs/pt_PT.aff are missing.",
            ])
    return "\n".join(lines)


def _pairwise_uncertainty(paths: List[Path], *, repetitions: int = 4000) -> List[Dict[str, Any]]:
    reports = []
    for first_index in range(len(paths)):
        for second_index in range(first_index + 1, len(paths)):
            first_path = paths[first_index]
            second_path = paths[second_index]
            reports.append({
                "first": first_path.name,
                "second": second_path.name,
                "statistics": paired_bootstrap(
                    read_jsonl(first_path),
                    read_jsonl(second_path),
                    repetitions=repetitions,
                ),
            })
    return reports


def _pretty_uncertainty(reports: List[Dict[str, Any]]) -> str:
    lines = ["", "=" * 72, "Paired uncertainty (second model minus first model)"]
    metric_order = ("instruction_adherence_pct", "semantic_preservation_pct", "euptvid_probability", "ptpt_compliance_pct", "ptbr_leakage_pct", "writing_quality_score")
    for report in reports:
        lines.extend(["-" * 72, f"  {report['first']}  ->  {report['second']}"])
        metrics = report["statistics"]["metrics"]
        for metric in metric_order:
            result = metrics.get(metric, {})
            if result.get("unavailable"):
                lines.append(f"    {metric:<32} N/A")
                continue
            ci = result["ci95"]
            lines.append(
                f"    {metric:<32} mean {result['mean_difference']:+.2f} "
                f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] "
                f"n={result['n']} W/L/T={result['wins']}/{result['losses']}/{result['ties']}"
            )
    return "\n".join(lines)


def _json_comparison(
    paths: List[Path],
    summaries: List[Dict[str, Any]],
    kind: str,
    *,
    bootstrap_repetitions: int = 4000,
    include_uncertainty: bool = True,
) -> Dict[str, Any]:
    """Return one machine-readable document for all compared artifacts."""
    artifacts = []
    for path, summary in zip(paths, summaries):
        model_names = sorted({str(row["model"]) for row in read_jsonl(path) if row.get("model")})
        artifacts.append({
            "artifact": path.name,
            "model": ", ".join(model_names) if model_names else "unknown model",
            "summary": summary,
        })

    metric_keys = (
        "instruction_adherence_pct", "semantic_preservation_pct",
        "euptvid_probability", "ptpt_compliance_pct", "ptbr_leakage_pct",
        "wf_score", "local_writing_quality", "latency_p50_ms",
        "latency_p90_ms", "throughput_emails_per_min", "tokens_per_second",
        "cost_per_1k_emails_usd",
    )
    deltas = {}
    if len(summaries) == 2:
        left, right = summaries
        for key in metric_keys:
            if left.get(key) is not None and right.get(key) is not None:
                deltas[key] = right[key] - left[key]

    report = {
        "kind": kind,
        "artifact_provenance": sorted({item["summary"].get("artifact_provenance", "unknown") for item in artifacts}),
        "artifacts": artifacts,
        "delta": {
            "definition": "second artifact minus first artifact",
            "values": deltas,
        },
    }
    if include_uncertainty and kind == "generation" and len(paths) >= 2:
        report["pairwise_uncertainty"] = _pairwise_uncertainty(
            paths, repetitions=bootstrap_repetitions,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark result artifacts")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--kind", choices=("analysis", "generation"), default="generation")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead")
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=4000,
        help="paired bootstrap repetitions for JSON comparison output",
    )
    parser.add_argument(
        "--no-uncertainty", "--no-paired-uncertainty",
        dest="no_uncertainty", action="store_true",
        help="omit the paired uncertainty section from the comparison",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="recompute missing legacy evaluator fields from stored content",
    )
    parser.add_argument(
        "--full-dialect-checks",
        action="store_true",
        help="use LanguageTool while backfilling legacy dialect fields (slower)",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="allow bare legacy JSONL artifacts for exploratory comparison",
    )
    args = parser.parse_args()
    manifests = validate_comparison_artifacts(
        args.results,
        kind=args.kind,
        allow_legacy=args.allow_legacy or args.rescore,
    )
    summaries = []
    if args.kind == "generation":
        for path, manifest in zip(args.results, manifests):
            rows = read_jsonl(path)
            if args.rescore:
                rows = _enrich_generation_records(rows, use_languagetool=args.full_dialect_checks)
            summary = build_elaboration_scorecard(rows)
            summary["artifact_provenance"] = (
                "legacy exploratory rescore" if args.rescore
                else "legacy exploratory" if manifest.get("legacy")
                else "manifest-backed"
            )
            summaries.append(summary)
            if not args.json:
                print(_pretty_report(path, summary, args.kind))
    else:
        truth = read_jsonl(ANALYSIS_REFERENCE)
        for path, manifest in zip(args.results, manifests):
            summary = score_analysis(truth, read_jsonl(path))
            summary["artifact_provenance"] = "legacy exploratory" if manifest.get("legacy") else "manifest-backed"
            summaries.append(summary)
            if not args.json:
                print(_pretty_report(path, summary, args.kind))
    if not args.json and not args.no_uncertainty and args.kind == "generation" and len(args.results) >= 2:
        print(_pretty_uncertainty(_pairwise_uncertainty(args.results)))
    if args.json:
        print(json.dumps(
            _json_comparison(
                args.results, summaries, args.kind,
                bootstrap_repetitions=args.bootstrap_repetitions,
                include_uncertainty=not args.no_uncertainty,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ))


if __name__ == "__main__":
    main()
