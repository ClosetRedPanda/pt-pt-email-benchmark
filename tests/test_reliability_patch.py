"""Tests for the reliability patch on compare.py.

Four behaviours, one per way a score was readable that the evidence did not
support:

  1. A persisted re-derivation may not claim a system prompt it never verified.
  2. A metric whose cross-model spread is one quantisation unit must be flagged.
  3. Under --rescore, uncertainty must describe the scored rows, not the frozen
     file on disk, and cannot be switched off.
  4. The pretty report must print the writing metrics it quotes denominators for,
     and the detector's measured recall belongs next to the scores it produced.
"""
import json
from pathlib import Path

import pytest

from config import (
    BENCHMARK_VERSION,
    ELABORATION_CONSTRAINTS,
    ELABORATION_PROMPTS,
    SYSTEM_PROMPT_ELABORATION,
)
from core._util import sha256_bytes
from core.artifacts import ArtifactValidationError, build_manifest, load_manifest, write_manifest
from core.generation_evaluator import load_constraint_map, evaluate_generation_output
from compare import (
    _detector_reliability_notes,
    _metric_separability,
    GENERATION_RANKING_KEYS,
    _json_comparison,
    _persist_rescored,
    _rescored_prompt_provenance,
    _separability_lines,
    _uncertainty_from_rows,
)
from compare import _refresh_criteria_records

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _clean_reply_for(task_id):
    """A reply that satisfies every criterion of `task_id` (content-free apart
    from them, so only the criteria themselves decide the score)."""
    constraints = load_constraint_map(str(ELABORATION_CONSTRAINTS))[task_id]
    clauses = []
    for group in ("required_actions", "required_facts"):
        for item in constraints.get(group, []) or []:
            pattern = item.get("pattern")
            if pattern:
                clauses.append(pattern)
    body = ". ".join(c.replace("\\s+", " ") for c in clauses[:3])
    return (
        "Exmos. Senhores,\n\n"
        f"{body}.\n\n"
        "Confirmamos a rececao e tratamos do assunto de imediato.\n\n"
        "Com os melhores cumprimentos,\nEquipa"
    )


def _rows(model, mutator=None):
    prompts = json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8"))["prompts"]
    rows = []
    for index, item in enumerate(prompts):
        task_id = str(item["id"])
        content = _clean_reply_for(task_id)
        if mutator:
            content = mutator(content)
        rows.append({
            "id": task_id,
            "model": model,
            "target_lang": "pt-pt",
            "status": "success",
            "content": content,
            "latency_ms": 1000 + index * 13,
            "prompt_tokens": 200,
            "completion_tokens": max(1, len(content.split())),
            "cost_usd": 0.002,
        })
    return rows


def _write_run(directory, model, *, prompt_sha=None, version=BENCHMARK_VERSION, mutator=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = _rows(model, mutator)
    payload = directory / f"{model.replace('/', '_')}_generation.jsonl"
    payload.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    parameters = {"system_prompt_sha256": prompt_sha} if prompt_sha is not None else {}
    manifest = build_manifest(
        payload,
        kind="generation",
        benchmark_version=version,
        model=model,
        input_hashes={"data/elaboration_prompts_pt_pt.json": "a" * 64},
        parameters=parameters,
    )
    write_manifest(payload, manifest)
    return payload


CURRENT_SHA = sha256_bytes(SYSTEM_PROMPT_ELABORATION.encode("utf-8"))


# --------------------------------------------------------------------------
# 1. prompt provenance on a re-derived artifact
# --------------------------------------------------------------------------

def test_rescored_manifest_inherits_the_run_prompt_hash_not_the_current_config():
    """The bug: `--rescore-output` hashed the prompt in config.py and stamped it
    as the run's input. A different prompt in config.py after the run would have
    been recorded as the one that produced the text."""
    other_prompt = SYSTEM_PROMPT_ELABORATION + " Extra instruction added after the run."
    other_sha = sha256_bytes(other_prompt.encode("utf-8"))
    assert other_sha != CURRENT_SHA, "fixture must actually differ from current config"

    source_manifest = {"parameters": {"system_prompt_sha256": other_sha}}
    provenance = _rescored_prompt_provenance(source_manifest)
    assert provenance["system_prompt_sha256"] == other_sha, "run-time hash must be inherited"
    assert provenance["system_prompt_provenance"] == "inherited-diverges-from-current-config"
    assert provenance["current_system_prompt_sha256"] == CURRENT_SHA, "current config recorded separately"


def test_rescored_manifest_records_unrecorded_provenance_instead_of_guessing():
    for manifest in (None, {"legacy": True}, {"parameters": {}}):
        provenance = _rescored_prompt_provenance(manifest)
        assert provenance["system_prompt_sha256"] is None, (
            f"a legacy artifact ({manifest}) must yield None, never a hash of the live prompt"
        )
        assert provenance["system_prompt_provenance"] == "unrecorded"


def test_matching_run_prompt_is_marked_inherited(tmp_path):
    payload = _write_run(tmp_path, "test/matched", prompt_sha=CURRENT_SHA)
    out_dir = tmp_path / "rederived"
    _persist_rescored(
        payload,
        _refresh_criteria_records(_rows("test/matched")),
        out_dir,
        source_manifest={"parameters": {"system_prompt_sha256": CURRENT_SHA}},
    )
    stamped = load_manifest(out_dir / payload.name)
    assert stamped["parameters"]["system_prompt_sha256"] == CURRENT_SHA
    assert stamped["parameters"]["system_prompt_provenance"] == "inherited-from-run"


def test_legacy_source_yields_unrecorded_prompt_in_persisted_manifest(tmp_path):
    rows = _rows("test/legacy")
    payload = tmp_path / "legacy.jsonl"
    payload.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    out_dir = tmp_path / "rederived"
    _persist_rescored(payload, _refresh_criteria_records(rows), out_dir, source_manifest={"legacy": True})
    stamped = load_manifest(out_dir / payload.name)
    assert stamped["parameters"]["system_prompt_sha256"] is None
    assert stamped["parameters"]["system_prompt_provenance"] == "unrecorded"


# --------------------------------------------------------------------------
# 2. quantisation guard
# --------------------------------------------------------------------------

def _summary(**values):
    n = values.pop("_n", 20)
    denominators = {key: n for key in values}
    return {**values, "denominators": denominators}


def test_per_email_binary_metrics_are_not_separable_below_one_email():  # 5.0 points = 1 of 20 emails
    """n=20 means 5.0 points is one email. Two artifacts one email apart must not
    read as a ranking."""
    left = _summary(ptpt_compliance_pct=90.0, _n=20)
    right = _summary(ptpt_compliance_pct=85.0, _n=20)
    report = _metric_separability([left, right], ("ptpt_compliance_pct",))
    item = report["metrics"]["ptpt_compliance_pct"]
    assert item["quantisation_unit"] == pytest.approx(5.0)
    assert item["separable"] is False
    assert item["min_gap"] == pytest.approx(5.0)


def test_larger_spread_is_admitted_as_separable():
    left = _summary(ptpt_compliance_pct=100.0, _n=20)
    right = _summary(ptpt_compliance_pct=80.0, _n=20)
    report = _metric_separability([left, right], ("ptpt_compliance_pct",))
    assert report["metrics"]["ptpt_compliance_pct"]["separable"] is True


def test_saturated_density_metrics_are_flagged():
    """Reproduces the reported shape: WF and graded compliance identical to one
    decimal across models. Identical means no information, not agreement."""
    artifacts = [
        _summary(wf_score=99.8, ptpt_compliance_graded_pct=99.9, _n=20)
        for _ in range(3)
    ]
    report = _metric_separability(artifacts, ("wf_score", "ptpt_compliance_graded_pct"))
    assert report["metrics"]["wf_score"]["separable"] is False
    assert report["metrics"]["ptpt_compliance_graded_pct"]["separable"] is False
    lines = _separability_lines(report)
    assert any("Word fidelity (density)" in line for line in lines)
    assert any("identical" in line for line in lines)


def test_guard_ignores_unreported_metrics_and_never_raises():
    report = _metric_separability(
        [{"denominators": {}}, {"denominators": {}}],
        ("ptpt_compliance_pct", "euptvid_probability"),
    )
    for item in report["metrics"].values():
        assert item["separable"] is False
        assert item["values"] == []


# --------------------------------------------------------------------------
# 3. uncertainty must follow the rescoring
# --------------------------------------------------------------------------

def test_uncertainty_uses_supplied_rows_not_the_file_on_disk(tmp_path):
    """The bug: --rescore recomputed the reported scores, then bootstrapped by
    re-reading the artifact, so the interval qualified frozen numbers while the
    report above printed re-derived ones."""
    rows = _rows("test/pair")
    for row in rows:
        row["instruction_adherence_pct"] = 0.0
    payload = tmp_path / "frozen.jsonl"
    payload.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    scored = [dict(r, instruction_adherence_pct=100.0) for r in rows]
    from core.statistics import paired_bootstrap

    # The artifact on disk holds the frozen 0.0 verdicts; the rescored rows hold
    # 100.0. A baseline run of the same content scored 50.0 per task, which is
    # what the old code would never have been able to contradict.
    baseline = [dict(r, instruction_adherence_pct=50.0) for r in _rows("test/pair")]
    on_disk = paired_bootstrap(baseline, rows, metrics=("instruction_adherence_pct",))
    reported = paired_bootstrap(baseline, scored, metrics=("instruction_adherence_pct",))
    assert on_disk["metrics"]["instruction_adherence_pct"]["mean_difference"] == pytest.approx(-50.0)
    assert reported["metrics"]["instruction_adherence_pct"]["mean_difference"] == pytest.approx(50.0)

    reports = _uncertainty_from_rows([baseline, scored], [payload, payload], repetitions=50)
    assert reports[0]["statistics"]["metrics"]["instruction_adherence_pct"]["mean_difference"] == pytest.approx(50.0), (
        "the interval must be computed from the scored rows"
    )


def test_json_comparison_uses_row_sets_for_uncertainty():
    summaries = [
        _summary(instruction_adherence_pct=60.0, ptpt_compliance_pct=90.0, _n=20),
        _summary(instruction_adherence_pct=80.0, ptpt_compliance_pct=90.0, _n=20),
    ]
    scored = [
        [dict(r, instruction_adherence_pct=0.0) for r in _rows("test/a")],
        [dict(r, instruction_adherence_pct=100.0) for r in _rows("test/b")],
    ]
    paths = [Path("a.jsonl"), Path("b.jsonl")]
    report = _json_comparison(paths, summaries, "generation", bootstrap_repetitions=50, row_sets=scored)
    metric = report["pairwise_uncertainty"][0]["statistics"]["metrics"]["instruction_adherence_pct"]
    assert metric["mean_difference"] == pytest.approx(100.0), "must reflect the supplied rows"
    assert "separability" in report


# --------------------------------------------------------------------------
# 4. report surface
# --------------------------------------------------------------------------

def test_detector_reliability_notes_carry_the_frozen_numbers():
    notes = _detector_reliability_notes()
    text = "\n".join(notes)
    assert "66.67%" in text, "detector recall from baselines/gold-agreement.json"
    assert "37.5%" in text, "precision without the suppression rule must be visible"
    assert "cannot be represented by a dictionary difference" in text.replace("\n", " ")
    assert text.count("Note:") == 1


def test_reading_notes_attach_detector_caveat_only_when_relevant():
    from compare import _generation_reading_notes

    with_detector = _generation_reading_notes({"ptbr_leakage_pct": 10.0, "wq_evaluator_version": "3.5.0"})
    without = _generation_reading_notes({"instruction_adherence_pct": 60.0, "wq_evaluator_version": "3.5.0"})
    assert any("capitalisation filter" in line for line in with_detector)
    assert not any("capitalisation filter" in line for line in without)


def test_writing_metrics_appear_in_the_pretty_report(tmp_path):
    """The denominator block listed 16 metrics; the report printed 12 of them."""
    from compare import _pretty_report

    summary = {
        "samples": 20, "successful_samples": 20, "failed_samples": 0,
        "avg_grammar_errors_per_email": 0.0, "avg_spelling_errors_per_email": 0.25,
        "structural_failures_pct": 10.0, "repetition_pct": 5.0,
        "ptpt_compliance_pct": None, "ptbr_leakage_pct": None,
        "wf_score": None, "ptpt_compliance_graded_pct": None,
        "local_writing_quality_defect_only": 82.9, "local_writing_quality": 77.0,
        "instruction_adherence_pct": 60.0, "semantic_preservation_pct": 90.0,
        "languagetool_available": True, "wq_evaluator_version": "3.5.0",
        "artifact_provenance": "manifest-backed", "denominators": {},
    }
    payload = tmp_path / "artifact.jsonl"
    payload.write_text(json.dumps({"model": "test/report"}) + "\n", encoding="utf-8")
    text = _pretty_report(payload, summary, "generation")
    assert "Spelling errors / email" in text and "0.25" in text
    assert "Structural failures" in text and "10.0%" in text
    assert "Grammar errors / email" in text



def test_quiet_report_hides_provenance_notes_and_denominators(tmp_path):
    """`--quiet` collapses the report to the metric table only.

    Everything the reliability work added to keep a number from being over-read
    (Provenance, denominators, the reading notes, the LanguageTool/Hunspell
    caveats) is hidden, while the metric sections and sample counts remain.
    """
    from compare import _pretty_report

    summary = {
        "samples": 20, "successful_samples": 20, "failed_samples": 0,
        "instruction_adherence_pct": 60.0, "semantic_preservation_pct": 90.0,
        "languagetool_available": False,
        "artifact_provenance": "re-derived from stored content",
        "denominators": {"instruction_adherence_pct": 20},
    }
    payload = tmp_path / "artifact.jsonl"
    payload.write_text(json.dumps({"model": "test/report"}) + "\n", encoding="utf-8")
    quiet = _pretty_report(payload, summary, "generation", quiet=True)
    loud = _pretty_report(payload, summary, "generation")

    assert "Instruction adherence" in quiet
    assert "Samples: 20" in quiet
    assert "Provenance" not in quiet
    assert "denominators" not in quiet
    assert "Note:" not in quiet
    assert "Provenance" in loud
    assert "Note:" in loud


def test_ranking_key_list_covers_every_printed_generation_metric():
    """The guard must watch the same metrics the report shows, or it is decoration."""
    from compare import SECTION_LABELS  # noqa: F401  (import smoke)
    printed = {
        "instruction_adherence_pct", "semantic_preservation_pct",
        "euptvid_probability", "ptpt_compliance_pct", "ptpt_compliance_graded_pct",
        "ptbr_leakage_pct", "avg_ptbr_violations_per_email",
        "wf_score", "avg_wf_penalty_per_email",
        "avg_grammar_errors_per_email", "avg_spelling_errors_per_email",
        "structural_failures_pct", "repetition_pct",
        "local_writing_quality_defect_only", "local_writing_quality",
    }
    assert printed == set(GENERATION_RANKING_KEYS)


def test_persist_still_refuses_to_overwrite(tmp_path):
    payload = _write_run(tmp_path, "test/imut", prompt_sha=CURRENT_SHA)
    out_dir = tmp_path / "rederived"
    _persist_rescored(payload, _refresh_criteria_records(_rows("test/imut")), out_dir, source_manifest={"parameters": {}})
    with pytest.raises(ArtifactValidationError):
        _persist_rescored(payload, _rows("test/imut"), out_dir, source_manifest={"parameters": {}})
