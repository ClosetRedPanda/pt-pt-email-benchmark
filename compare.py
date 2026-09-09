"""Readable comparison helpers for already-produced result files."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from core.scorecard import build_elaboration_scorecard
from core.statistics import paired_bootstrap
from core.pt_dialect import evaluate_pt_dialect
from core.wf_fidelity import compute_word_fidelity_from_dialect
from core.writing_quality import evaluate_writing_quality
from core.generation_evaluator import constraint_echo_vocabulary, evaluate_generation_output
from core.artifacts import (
    ArtifactValidationError,
    build_manifest,
    load_manifest,
    sha256_bytes,
    sha256_file,
    validate_rows,
    write_manifest,
)
from runner import load_constraints, read_jsonl, score_analysis
from config import (
    ANALYSIS_REFERENCE,
    BENCHMARK_VERSION,
    ELABORATION_CONSTRAINTS,
    ELABORATION_PROMPTS,
    SYSTEM_PROMPT_ELABORATION,
)


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
    # REL-08: word fidelity is a leak *density* (weighted penalties divided by
    # total words), so the label must not read like a percentage of clean mail.
    "wf_score": "Word fidelity (density)",
    # Issue 3: the defect-only score is the primary writing-quality metric for
    # model comparison; the calibrated score is the conservative headline.
    "local_writing_quality_defect_only": "Writing quality (primary)",
    "local_writing_quality": "Writing quality (calibrated)",
    "wq_evaluator_version": "WQ evaluator version",
    # Display gap: the scorecard computes these and the denominator block already
    # lists them, but the pretty report never printed the value -- so a full
    # denominator was shown for a metric whose number the reader could not see,
    # and "N/A" (backend silent) was indistinguishable from a measured zero.
    "avg_grammar_errors_per_email": "Grammar errors / email",
    "avg_spelling_errors_per_email": "Spelling errors / email",
    "structural_failures_pct": "Structural failures",
    "repetition_pct": "Repetition",
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

# The metrics a reader is invited to rank models on. Kept in one place so the
# resolution guard, the text report and the JSON document all describe the same
# set -- a guard applied to a different list than the one printed is decoration.
GENERATION_RANKING_KEYS = (
    "instruction_adherence_pct", "semantic_preservation_pct",
    "euptvid_probability", "ptpt_compliance_pct", "ptpt_compliance_graded_pct",
    "ptbr_leakage_pct", "avg_grammar_errors_per_email",
    "avg_spelling_errors_per_email", "structural_failures_pct", "repetition_pct",
    "wf_score", "local_writing_quality_defect_only", "local_writing_quality",
)


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
    if key.endswith("_errors_per_email"):
        # Two decimals: a sub-1 error rate is the whole point of these metrics,
        # and one decimal would render 0.04 and 0.0 identically.
        return f"{float(value):.2f}"
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


_ECHO_CONSTRAINT_MAP = None


def _echo_vocab_for_row(row: Dict[str, Any]) -> Optional[set]:
    """Task-echo vocabulary for a result row, from the row's own task id.

    Loads the bundled constraint map once; rows whose id is not a known task
    (or when the map cannot load) yield no exemption and stay strict.
    """
    global _ECHO_CONSTRAINT_MAP
    if _ECHO_CONSTRAINT_MAP is None:
        try:
            _ECHO_CONSTRAINT_MAP = load_constraints()
        except Exception:
            _ECHO_CONSTRAINT_MAP = {}
    spec = _ECHO_CONSTRAINT_MAP.get(str(row.get("id", "")), {})
    return constraint_echo_vocabulary(spec) if isinstance(spec, dict) else set()


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
            dialect = evaluate_pt_dialect(
                str(result["content"]),
                use_languagetool=use_languagetool,
                echo_vocab=_echo_vocab_for_row(result),
            )
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


def _wq_language_for(result: Dict[str, Any]) -> str:
    """Map a row's target language to the evaluator's language tag."""
    target = str(result.get("target_lang", "")).strip().lower()
    return "pt-PT" if target.startswith("pt") else "en-US"


def _full_rescore_generation_records(
    rows: List[Dict[str, Any]],
    *,
    use_languagetool: bool = False,
) -> List[Dict[str, Any]]:
    """Re-evaluate every scoreable row with the current evaluator code.

    This is the rescorer behind ``compare.py --rescore``. Unlike
    ``_enrich_generation_records`` — which only backfills *missing* dialect/WF
    fields and never touches writing quality — this recomputes, for every
    successful row with stored content:

    - PT dialect evaluation and word fidelity (PT-PT rows), replacing the frozen
      legacy values, so artifacts scored before the Issue 2 URL/protocol-masking
      fix get fresh compliance/leakage numbers;
    - writing quality in both flavours, writing both ``wq_defect_only_score``
      and ``writing_quality_score`` at the top level (and the nested
      ``writing_quality`` dict), so scorecards stop reporting the primary
      metric as unavailable on legacy rows.

    Failed rows and rows without content are copied unchanged. The method never
    calls an LLM or the network; only the local deterministic evaluators run.
    """
    enriched = []
    for row in rows:
        result = dict(row)
        content = result.get("content")
        if (
            result.get("status", "success") != "success"
            or result.get("error")
            or not isinstance(content, str)
            or not content.strip()
        ):
            enriched.append(result)
            continue
        text = str(content)
        is_ptpt = str(result.get("target_lang", "")).strip().lower() == "pt-pt"
        if is_ptpt:
            dialect = evaluate_pt_dialect(
                text,
                use_languagetool=use_languagetool,
                echo_vocab=_echo_vocab_for_row(result),
            )
            wf = compute_word_fidelity_from_dialect(text, dialect)
            result.update({
                "pt_dialect_score": dialect.get("pt_dialect_score"),
                "euptvid_probability": dialect.get("euptvid_prob"),
                "ptpt_compliance_pct": dialect.get("ptpt_compliance_pct"),
                "ptpt_compliance_graded_pct": dialect.get("ptpt_compliance_graded_pct"),
                "ptbr_leakage_detected": dialect.get("ptbr_leakage_detected"),
                "wf_score": wf.get("wf_score"),
                "dialect_evaluation": dialect,
                "wf_evaluation": wf,
            })
        wq = evaluate_writing_quality(
            text,
            language=_wq_language_for(result),
            use_languagetool=use_languagetool,
        )
        result.update({
            "writing_quality": wq,
            "writing_quality_score": wq.get("writing_quality_score"),
            "wq_defect_only_score": wq.get("wq_defect_only_score"),
        })
        enriched.append(result)
    return enriched


def _refresh_criteria_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-run the executable criteria against stored content.

    ``--rescore`` refreshes dialect, WF and WQ but deliberately leaves
    ``instruction_adherence_pct`` and ``semantic_preservation_pct`` frozen, because
    those are properties of the task definition, not of the evaluators. That is the
    right default: silently re-grading an old run under new criteria would make a
    before/after comparison look like a model change.

    This function is the explicit opt-in. It recomputes both from the stored text
    and the current constraint file, so a criteria retarget (e.g. closing a
    permissive pattern) can be applied to existing outputs without new model
    calls. The criteria are a pure function of stored content, so nothing here
    touches the network or an LLM.
    """
    prompts = {
        str(item["id"]): str(item.get("prompt") or "")
        for item in json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8")).get("prompts", [])
    }
    constraints = load_constraints()
    refreshed = []
    for row in rows:
        result = dict(row)
        content = result.get("content")
        if (
            result.get("status", "success") != "success"
            or result.get("error")
            or not isinstance(content, str)
            or not content.strip()
        ):
            refreshed.append(result)
            continue
        pid = str(result.get("id"))
        ev = evaluate_generation_output(content, constraints.get(pid, {}), source_text=prompts.get(pid, ""))
        result["instruction_adherence_pct"] = ev["instruction_adherence_score"]
        result["semantic_preservation_pct"] = ev["semantic_preservation_score"]
        refreshed.append(result)
    return refreshed


def _rescored_prompt_provenance(source_manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve the system-prompt provenance a re-derivation is allowed to claim.

    `runner.py` records `parameters.system_prompt_sha256` at run time, so a
    manifest-backed artifact *does* carry the hash of the prompt its text was
    generated under. A re-derivation must inherit that value, never hash the
    prompt currently in `config.py`: the stored text is the only evidence of what
    produced it, and it cannot be re-elicited. Hashing the live prompt instead
    would assert an input condition that was never verified, inside the one
    subsystem whose whole job is binding results to their actual inputs.
    """
    current_sha = sha256_bytes(SYSTEM_PROMPT_ELABORATION.encode("utf-8"))
    recorded = (source_manifest or {}).get("parameters", {}).get("system_prompt_sha256")
    if not isinstance(recorded, str) or not recorded:
        return {
            "system_prompt_sha256": None,
            "system_prompt_provenance": "unrecorded",
            "current_system_prompt_sha256": current_sha,
        }
    return {
        "system_prompt_sha256": recorded,
        "system_prompt_provenance": (
            "inherited-from-run" if recorded == current_sha else "inherited-diverges-from-current-config"
        ),
        "current_system_prompt_sha256": current_sha,
    }


def _stamp_rescored_manifest(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    source_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a sidecar for a fully re-derived artifact, recording current versions.

    Honest only because every scoreable field is recomputed here: a rescored
    dialect/WF/WQ artifact keeps its frozen criteria and must not be stamped.
    """
    root = Path(__file__).resolve().parent
    models = {str(r.get("model")) for r in rows}
    if len(models) != 1:
        raise ArtifactValidationError(f"rescore requires exactly one model, found {sorted(models)}")
    input_paths = [ELABORATION_PROMPTS, ELABORATION_CONSTRAINTS]
    evaluator_paths = [
        root / "core" / "generation_evaluator.py",
        root / "core" / "pt_dialect.py",
        root / "core" / "wf_fidelity.py",
        root / "core" / "writing_quality.py",
        root / "core" / "scorecard.py",
        root / "runner.py",
        root / "compare.py",
        ELABORATION_CONSTRAINTS,
    ]
    from core.resources import EUPTVID, HUNSPELL_RESOURCES

    resource_paths = [
        *(resource.path for resource in HUNSPELL_RESOURCES),
        EUPTVID.path,
        root / "data" / "wq_length_neutral_calibration.json",
    ]
    manifest = build_manifest(
        path,
        kind="generation",
        benchmark_version=BENCHMARK_VERSION,
        model=next(iter(models)),
        input_hashes={str(q.relative_to(root)): sha256_file(q) for q in input_paths},
        evaluator_versions={str(q.relative_to(root)): sha256_file(q) for q in evaluator_paths},
        resource_hashes={
            str(q.relative_to(root)): sha256_file(q) if q.is_file() else "UNAVAILABLE"
            for q in resource_paths
        },
        parameters={
            "rescore": True,
            "criteria_refreshed": True,
            "source_artifact": path.name,
            **_rescored_prompt_provenance(source_manifest),
        },
    )
    write_manifest(path, manifest)
    return manifest


def _persist_rescored(
    source: Path,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    *,
    source_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write re-derived rows and a manifest to a new artifact; never touch the source.

    Refuses if the destination exists, so a re-derivation cannot quietly replace
    the record of the run it was derived from. Returns the stamped manifest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / source.name
    if destination.exists():
        raise ArtifactValidationError(
            f"refusing to overwrite existing re-derivation artifact: {destination}"
        )
    expected_ids = [
        str(item["id"])
        for item in json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8")).get("prompts", [])
    ]
    validate_rows(rows, kind="generation", expected_ids=expected_ids, strict=True)
    destination.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    _stamp_rescored_manifest(destination, rows, source_manifest=source_manifest)
    print(f"[rescore] persisted re-derived artifact: {destination}", file=sys.stderr)
    return destination


def validate_comparison_artifacts(
    paths: List[Path],
    *,
    kind: str,
    allow_legacy: bool = False,
    rescore: bool = False,
) -> List[Dict[str, Any]]:
    """Validate artifacts and ensure comparable manifest metadata.

    ``rescore`` means the caller will recompute every scoreable field from stored
    content before scoring. A version mismatch is then a fact to report, not a
    reason to refuse: refusing is what left ``compare.py --rescore`` unusable on any
    artifact predating a criteria change, including the retarget that closed the
    boilerplate floor.
    """
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
            if not rescore:
                raise ArtifactValidationError(
                    f"artifacts use benchmark {versioned[0].get('benchmark_version')!r}, expected "
                    f"{BENCHMARK_VERSION!r}. Their criteria-verdict fields are frozen, so either re-run "
                    "generation, or pass --rescore to re-derive every score (including adherence) from "
                    "stored content and stamp the result as a new artifact."
                )
            print(
                f"[rescore] artifacts recorded under {versioned[0].get('benchmark_version')!r}; "
                f"re-scoring every field with {BENCHMARK_VERSION!r} criteria. This is exploratory: the "
                "sidecar still records the original run, and the numbers are not comparable with "
                "artifacts produced under other criteria.",
                file=sys.stderr,
            )
    return manifests


_CONSTRAINT_PROFILE = None


def _constraint_profile() -> Dict[str, Any]:
    """Cache whether the bundled task constraints exercise forbidden checks.

    Surfaces in reports whether the semantic metric can actually catch
    contradictions/hallucinations for this task set (REL-02): the check exists
    in code but is inert until tasks define ``forbidden_changes``.
    """
    global _CONSTRAINT_PROFILE
    if _CONSTRAINT_PROFILE is not None:
        return _CONSTRAINT_PROFILE
    profile: Dict[str, Any] = {"task_count": 0, "has_forbidden": False}
    try:
        from runner import load_constraints
        constraints = load_constraints()
        profile["task_count"] = len(constraints)
        profile["has_forbidden"] = any(
            isinstance(spec, dict)
            and (
                (spec.get("forbidden_changes") or [])
                or (spec.get("forbidden_facts") or [])
            )
            for spec in constraints.values()
        )
    except Exception:
        pass
    _CONSTRAINT_PROFILE = profile
    return profile


_GOLD_BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "gold-agreement.json"


def _detector_reliability_notes() -> List[str]:
    """Surface the leakage detector's measured limitations inside the scorecard.

    The compliance, leakage, graded-density and word-fidelity lines all read from
    the same dictionary-contrast detector. `tools/gold_agreement.py` measures that
    detector against the gold set and records the result in `baselines/
    gold-agreement.json`, including that its headline precision depends on a single
    suppression rule -- but that disclosure has never reached the report a reader
    actually looks at, so a clean leakage number read as evidence rather than as a
    floor. This is a display of already-frozen numbers, not a new judgement: it
    fails soft (no note) if the baseline is missing, and cites the file so the
    reader can check it.
    """
    try:
        baseline = json.loads(_GOLD_BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    detector = baseline.get("detector") or {}
    counts = baseline.get("counts") or {}
    blindspot = baseline.get("blindspot") or {}
    suppression = baseline.get("remediation_risk") or {}
    recall = detector.get("leakage_recall_pct")
    if not isinstance(recall, (int, float)):
        return []
    notes = [
        "  Note: the dictionary-contrast detector behind compliance/leakage/word-fidelity",
    ]
    agreement = detector.get("agreement_pct")
    notes.append(
        f"        was measured on the frozen gold set: {agreement}% overall agreement,"
        if isinstance(agreement, (int, float))
        else "        was measured on the frozen gold set with no agreement rate recorded,"
    )
    fn = counts.get("false_negative")
    invisible = blindspot.get("missed_rows_wholly_invisible_to_mechanism")
    notes.append(
        f"        recall {recall}% across {counts.get('true_positive', 0) + fn} gold leaks."
        if isinstance(fn, int)
        else f"        recall {recall}% of the gold-positive leaks."
    )
    if isinstance(invisible, int) and invisible:
        notes.append(
            f"        {invisible} of the misses cannot be represented by a dictionary difference at"
        )
        notes.append("        all, so they stay missed however the word lists grow.")
    suppressed = suppression.get("precision_without_suppression_pct")
    if isinstance(suppressed, (int, float)) and isinstance(detector.get("leakage_precision_pct"), (int, float)):
        notes.append(
            f"        Its {detector['leakage_precision_pct']}% precision rests on one capitalisation filter;"
        )
        notes.append(
            f"        without it precision is {suppressed}%. A low leakage number is a floor, not proof"
        )
        notes.append("        of none -- and it is the same instrument scoring every model in this run.")
    notes.append("        Source: baselines/gold-agreement.json.")
    return notes


def _generation_reading_notes(summary: Dict[str, Any]) -> List[str]:
    """Short 'how to read this' notes appended to generation reports.

    Each note states a documented measurement boundary so a headline number is
    not over-read: adherence/semantic are deterministic pattern-coverage; the
    current task set exercises no forbidden-change checks; compliance is
    per-email binary (compare the graded line); word fidelity is leak density;
    tokens/s is a whole-run aggregate; and the WQ evaluator version identifies
    the code that produced the writing scores.
    """
    profile = _constraint_profile()
    notes: List[str] = []
    notes.append(
        "  Note: instruction adherence and semantic preservation measure deterministic "
        "pattern coverage of each"
    )
    notes.append(
        "        task's required actions/facts, not holistic quality or general semantic "
        "equivalence."
    )
    if profile.get("task_count") and not profile.get("has_forbidden"):
        notes.append(
            "  Note: the current task set defines no forbidden_changes, so contradictions "
            "and invented facts are not"
        )
        notes.append(
            "        penalised by semantic preservation for these tasks."
        )
    notes.append(
        "  Note: PT-PT compliance is binary per email (one score-eligible violation zeroes "
        "the email); the"
    )
    notes.append(
        "        graded line above reports violation density as a percentage."
    )
    notes.append(
        "  Note: word fidelity is leak density (weighted penalties / words); one leak costs "
        "less in a longer"
    )
    notes.append(
        "        email, so compare it across runs only at similar lengths."
    )
    notes.append(
        "  Note: tokens/s divides total tokens by whole-run wall clock and includes "
        "concurrency and queueing."
    )
    # Only attach the detector caveat when a detector-backed line is on the page.
    if any(
        summary.get(key) is not None
        for key in ("ptpt_compliance_pct", "ptbr_leakage_pct", "wf_score", "ptpt_compliance_graded_pct")
    ):
        notes.extend(_detector_reliability_notes())
    version = summary.get("wq_evaluator_version")
    if version:
        notes.append(f"  Note: writing-quality evaluator version: {version}")
    return notes


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
                "ptpt_compliance_graded_pct": summary.get("ptpt_compliance_graded_pct"),
                "ptbr_leakage_pct": summary.get("ptbr_leakage_pct"),
                "wf_score": summary.get("wf_score"),
            },
            "writing": {
                "local_writing_quality_defect_only": summary.get("local_writing_quality_defect_only"),
                "local_writing_quality": summary.get("local_writing_quality"),
                # These four are printed rather than only counted in the
                # denominator block, so a measured zero and an unavailable
                # measurement are visually distinct ("0.00" vs "N/A").
                "avg_grammar_errors_per_email": summary.get("avg_grammar_errors_per_email"),
                "avg_spelling_errors_per_email": summary.get("avg_spelling_errors_per_email"),
                "structural_failures_pct": summary.get("structural_failures_pct"),
                "repetition_pct": summary.get("repetition_pct"),
                "wq_evaluator_version": summary.get("wq_evaluator_version"),
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
    if kind == "generation" and summary.get("languagetool_available") is False:
        lines.append("  Note: no LanguageTool backend responded, so grammar evidence was unavailable and")
        lines.append("        grammar_errors_per_email is reported as N/A (not as a measured zero).")
    denominators = summary.get("denominators") or {}
    if denominators:
        lines.extend(["", "  Metric denominators: " + ", ".join(
            f"{key}={value}" for key, value in sorted(denominators.items())
        )])
    if kind == "generation":
        lines.extend(["", *_generation_reading_notes(summary)])
    if kind == "generation":
        dialect_statuses = {            str((row.get("dialect_evaluation") or {}).get("language_adherence_status"))
            for row in rows
        }
        if (
            any("Hunspell dictionary unavailable" in status for status in dialect_statuses)
            and summary.get("ptpt_compliance_pct") is None
        ):
            lines.extend([
                "",
                "  Note: PT-PT compliance, PT-BR leakage, and Word fidelity are unavailable",
                "  because managed Hunspell files are unavailable; run `python runner.py setup`.",
            ])
    return "\n".join(lines)


def _metric_separability(
    summaries: List[Dict[str, Any]],
    metric_keys: Iterable[str],
) -> Dict[str, Any]:
    """Flag metrics whose cross-model spread is at or below one quantisation unit.

    With n=20 artifacts every binary per-email metric moves in exact 5.0-point
    steps, so 90.0 vs 85.0 is one email, not a measured difference. A density
    metric rounded to one decimal sits in the 99.8-99.9 band for every model at
    realistic email lengths. Neither supports a conclusion; a metric identical
    across models is not corroboration, it is an instrument that is not moving.
    """
    units = {
        "instruction_adherence_pct": 100.0,
        "semantic_preservation_pct": 100.0,
        "ptpt_compliance_pct": 100.0,
        "ptbr_leakage_pct": 100.0,
        "structural_failures_pct": 100.0,
        "repetition_pct": 100.0,
        # REL-08: a leak-density metric, not a per-email rate. At ~150 words a
        # single leak moves it ~0.7 points, so sub-point gaps cannot separate.
        "wf_score": 100.0,
        # Violation density rounded to one decimal in pt_dialect.py.
        "ptpt_compliance_graded_pct": 100.0,
        # A probability on 0..1, so the unit is a fraction, not a percentage.
        "euptvid_probability": 1.0,
        # Count per email, not a rate; one email contributes a non-integer count.
        "avg_grammar_errors_per_email": 1.0,
        "avg_spelling_errors_per_email": 1.0,
    }
    # `denominators` keys off the per-row field name for some metrics, so the
    # lookup needs an explicit alias rather than a string mangle.
    denom_keys = {
        "local_writing_quality_defect_only": "local_writing_quality_defect_only",
        "local_writing_quality": "local_writing_quality",
    }
    metrics: Dict[str, Any] = {}
    for key in metric_keys:
        lookup = denom_keys.get(key, key)
        denom = None
        for summary in summaries:
            candidate = (summary.get("denominators") or {}).get(lookup)
            if isinstance(candidate, int) and candidate > 0:
                denom = candidate
                break
        values = sorted(
            {
                round(float(summary[key]), 6)
                for summary in summaries
                if isinstance(summary.get(key), (int, float)) and not isinstance(summary.get(key), bool)
            }
        )
        unit = units.get(key, 100.0)
        record: Dict[str, Any] = {
            "values": values,
            "n": denom,
            "metric_unit": unit,
            "min_gap": 0.0,
            "quantisation_unit": (unit / denom) if denom else None,
            "separable": False,
            "reason": "",
        }
        if len(summaries) < 2:
            record["reason"] = "fewer than two artifacts"
        elif not values:
            record["reason"] = "no artifact reports this metric"
        elif len(values) == 1:
            record["reason"] = "identical across every artifact"
        else:
            record["min_gap"] = values[-1] - values[0]
            threshold = (unit / denom) if denom else unit
            record["quantisation_unit"] = threshold
            if record["min_gap"] > threshold:
                record["separable"] = True
                record["reason"] = "spread exceeds one quantisation unit"
            else:
                record["reason"] = "spread within one quantisation unit"
        metrics[key] = record
    return {
        "artifact_count": len(summaries),
        "rule": "a metric is not separable when max-min spread across models does not exceed one quantisation unit (1/n of the metric range)",
        "metrics": metrics,
    }


def _separability_lines(separability: Dict[str, Any]) -> List[str]:
    """Render the non-separable metrics for the text report."""
    blocked = sorted(
        key
        for key, item in (separability.get("metrics") or {}).items()
        if item.get("values") and item.get("reason") != "fewer than two artifacts"
        and not item.get("separable")
    )
    if not blocked:
        return []
    lines = [
        "",
        "-" * 72,
        f"  Resolution guard: {len(blocked)} of {len(separability.get('metrics') or {})} ranking metrics cannot",
        "  separate these models -- their max-min spread is no larger than one",
        "  quantisation unit (1 unit = the metric range divided by the sample size).",
    ]
    for key in blocked:
        item = separability["metrics"][key]
        unit = item.get("quantisation_unit")
        state = "identical across all artifacts" if item["min_gap"] == 0.0 else "one quantisation unit"
        lines.append(
            f"    {LABELS.get(key, key):<28} spread {item['min_gap']:.2f}"
            + (f"  (1 unit = {unit:.2f})" if isinstance(unit, (int, float)) else "")
            + f"  [{state}]"
        )
    lines.append(
        "  A flat or saturated metric is not corroboration. Read the paired bootstrap CI below."
    )
    return lines


def _pairwise_uncertainty(paths: List[Path], *, repetitions: int = 4000) -> List[Dict[str, Any]]:
    return _uncertainty_from_rows([read_jsonl(path) for path in paths], paths, repetitions=repetitions)


def _uncertainty_from_rows(
    row_sets: List[List[Dict[str, Any]]],
    paths: List[Path],
    *,
    repetitions: int = 4000,
) -> List[Dict[str, Any]]:
    """Paired bootstrap over artifact rows.

    Takes already-prepared row sets rather than paths because under --rescore the
    rows that were scored are not the rows on disk. Reading the files here made
    the confidence interval describe the frozen artifact while the report above it
    described re-derived values -- the uncertainty and the number it qualifies
    came from different measurements.
    """
    reports = []
    for first_index in range(len(paths)):
        for second_index in range(first_index + 1, len(paths)):
            reports.append({
                "first": paths[first_index].name,
                "second": paths[second_index].name,
                "statistics": paired_bootstrap(
                    row_sets[first_index],
                    row_sets[second_index],
                    repetitions=repetitions,
                ),
            })
    return reports


def _pretty_uncertainty(reports: List[Dict[str, Any]]) -> str:
    lines = ["", "=" * 72, "Paired uncertainty (second model minus first model)"]
    metric_order = ("instruction_adherence_pct", "semantic_preservation_pct", "euptvid_probability", "ptpt_compliance_pct", "ptpt_compliance_graded_pct", "ptbr_leakage_pct", "wq_defect_only_score", "writing_quality_score")
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
    row_sets: Optional[List[List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Return one machine-readable document for all compared artifacts."""
    # Under --rescore the scored rows are not the rows on disk; use whichever the
    # caller supplied for every read in this function, not just for the bootstrap.
    resolved_rows = row_sets if row_sets is not None else [read_jsonl(path) for path in paths]
    artifacts = []
    for path, summary, artifact_rows in zip(paths, summaries, resolved_rows):
        model_names = sorted({str(row["model"]) for row in artifact_rows if row.get("model")})
        artifacts.append({
            "artifact": path.name,
            "model": ", ".join(model_names) if model_names else "unknown model",
            "summary": summary,
        })

    # Delta and separability share one key list so a number cannot appear in one
    # and be silently absent from the other.
    metric_keys = GENERATION_RANKING_KEYS + (
        "latency_p50_ms", "latency_p90_ms", "throughput_emails_per_min",
        "tokens_per_second", "cost_per_1k_emails_usd",
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
    if kind == "generation" and len(summaries) >= 2:
        # Ranking metrics only: latency and cost are continuous measurements, and
        # calling a 0.5 ms difference "below one quantisation unit" would bury the
        # real failures under a guard that was never meant for them.
        report["separability"] = _metric_separability(summaries, GENERATION_RANKING_KEYS)
    if include_uncertainty and kind == "generation" and len(paths) >= 2:
        report["pairwise_uncertainty"] = _uncertainty_from_rows(
            resolved_rows, paths, repetitions=bootstrap_repetitions,
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
        help="re-evaluate stored content with current evaluators (dialect, word fidelity, and both writing-quality scores)",
    )
    parser.add_argument(
        "--full-dialect-checks",
        action="store_true",
        help="use LanguageTool while re-scoring legacy dialect fields (slower)",
    )
    parser.add_argument(
        "--rescore-output",
        type=Path,
        default=None,
        help="with --rescore, persist re-derived rows plus a manifest into this new "
             "directory instead of reporting only; existing artifacts are never modified",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="allow bare legacy JSONL artifacts for exploratory comparison",
    )
    args = parser.parse_args()
    # A re-derivation is exactly the case that needs an interval: its point
    # estimates are noisier than a normal run's because the content was produced
    # under different criteria. Honouring --no-uncertainty there left only the
    # bare numbers, which is how one-email differences came to be read as a
    # ranking. The section is forced back on; --json callers still get what they
    # asked for, since they can read the numbers programmatically.
    if args.rescore and args.no_uncertainty and not args.json:
        print(
            "[notice] --rescore overrides --no-uncertainty: re-derived point estimates are "
            "reported with their paired bootstrap interval.",
            file=sys.stderr,
        )
        args.no_uncertainty = False
    if args.rescore and not args.rescore_output:
        print(
            "[notice] reporting only: these numbers are re-derived, not run. To persist an "
            "auditable artifact (with inherited prompt provenance) pass --rescore-output DIR.",
            file=sys.stderr,
        )
    manifests = validate_comparison_artifacts(
        args.results,
        kind=args.kind,
        allow_legacy=args.allow_legacy or args.rescore,
        rescore=args.rescore,
    )
    summaries = []
    scored_rows: List[List[Dict[str, Any]]] = []
    if args.kind == "generation":
        for path, manifest in zip(args.results, manifests):
            if manifest.get("legacy") and not args.rescore:
                print(
                    f"[warning] {path.name}: legacy artifact with per-row values frozen at creation time. "
                    "Pass --rescore to re-evaluate stored content with the current evaluators.",
                    file=sys.stderr,
                )
            rows = read_jsonl(path)
            if args.rescore:
                rows = _full_rescore_generation_records(rows, use_languagetool=args.full_dialect_checks)
                rows = _refresh_criteria_records(rows)
                # A re-derivation never overwrites the record of what actually ran:
                # it is written to a NEW file under --rescore-output, with a manifest
                # stamped at the current version. Without the flag, the run stays
                # exploratory and the original sidecar is untouched.
                if args.rescore_output:
                    _persist_rescored(path, rows, args.rescore_output, source_manifest=manifest)
            scored_rows.append(rows)
            summary = build_elaboration_scorecard(rows)
            if args.rescore and manifest.get("legacy"):
                provenance = "legacy re-derived (exploratory)"
            elif args.rescore:
                # Manifest-backed: the sidecar still records the original run, while
                # these numbers were re-derived just now. Say so, or the report reads
                # as though the artifact itself had been updated.
                provenance = (
                    f"re-derived from stored content under {BENCHMARK_VERSION}; "
                    f"artifact recorded under {manifest.get('benchmark_version')}"
                )
            elif manifest.get("legacy"):
                provenance = "legacy exploratory"
            else:
                provenance = "manifest-backed"
            summary["artifact_provenance"] = provenance
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
    if not args.json and args.kind == "generation" and len(summaries) >= 2:
        # Printed once because it is a property of the *set*: whether these
        # artifacts differ by more than one quantisation unit is not answerable
        # per artifact.
        print("\n".join(_separability_lines(_metric_separability(summaries, GENERATION_RANKING_KEYS))))
        if not args.no_uncertainty:
            print(
                _pretty_uncertainty(
                    _uncertainty_from_rows(scored_rows, args.results) if scored_rows
                    else _pairwise_uncertainty(args.results)
                )
            )
    if args.json:
        print(json.dumps(
            _json_comparison(
                args.results, summaries, args.kind,
                bootstrap_repetitions=args.bootstrap_repetitions,
                include_uncertainty=not args.no_uncertainty,
                row_sets=scored_rows or None,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ))


if __name__ == "__main__":
    main()
