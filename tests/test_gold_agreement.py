"""Guards for tools/gold_agreement.py.

These tests exist to keep two promises the tool makes in its own docstring:
that it measures without changing anything on the scoring path, and that it
holds no vocabulary of its own. The second promise is what makes the first
worth trusting: a measurement harness that smuggles in the word lists it is
supposed to be grading would certify itself forever.
"""
from __future__ import annotations

import ast
import json
import re
import pytest
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "tools")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import gold_agreement  # noqa: E402

TOOL = ROOT / "tools" / "gold_agreement.py"
BASELINE = ROOT / "baselines" / "gold-agreement.json"

# Portuguese content words the detector could use as a hand-maintained
# marker lexicon. Function words and grammatical particles are excluded on
# purpose: the tool is allowed to know what a token is, not which dialect a
# word belongs to.
# Words that unambiguously mark a dialect (no English homographs, so ordinary
# prose in a note cannot trip this). If the tool ever embeds one as a literal
# string, it has started encoding the answer instead of measuring it.
PORTUGUESE_MARKERS = re.compile(
    r"\b(?:celular|equipe|usu[áa]rio|atenciosamente|[âa]tenciosamente|[ôo]nibus|voc[êe]s|"
    r"ger[êe]ncio|planejamento|boleto|motorista|fregu[êe]s|[cc]amarista)\b",
    re.IGNORECASE,
)


def test_measuring_changes_nothing_on_disk():
    """Run the measurement and prove the benchmark's own files are untouched.

    Real side-effect check rather than a grep: hash the tracked data and code
    before and after, because a tool that grades the detector must not be able
    to edit it.
    """
    watched = [ROOT / rel for rel in (
        "data/analysis_reference.jsonl",
        "data/elaboration_constraints.json",
        "core/pt_dialect.py",
        "baselines/smoke-baseline.json",
    )]
    before = {p: p.read_bytes() for p in watched}
    gold_agreement.build_report()
    assert {p: p.read_bytes() for p in watched} == before


def test_tool_contains_no_portuguese_marker_lexicon():
    """No docstring, no comment, no literal: the tool must not know which words are Brazilian.

    It classifies markers by asking the dictionaries it is given, so a future
    contributor cannot quietly hard-code the answer it is meant to measure.
    """
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # strip the marker values that legitimately come from gold data names
            for match in PORTUGUESE_MARKERS.finditer(node.value):
                offenders.append((node.lineno, match.group(0)))
    assert not offenders, f"hard-coded Portuguese markers in tool: {offenders}"


def test_report_is_deterministic():
    first = gold_agreement.build_report()
    second = gold_agreement.build_report()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_committed_baseline_matches_a_fresh_measurement():
    """CI runs this as --check-baseline; failing here means the detector drifted."""
    if not BASELINE.is_file():
        raise AssertionError("baselines/gold-agreement.json missing: regenerate with --output")
    committed = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert gold_agreement._comparable(gold_agreement.build_report()) == gold_agreement._comparable(committed), (
        "detector gold-agreement no longer matches the committed baseline; re-run "
        "`python tools/gold_agreement.py --output baselines/gold-agreement.json` and review the diff"
    )


def test_baseline_records_the_blindspot_as_a_measured_fact():
    """Pin the finding, not the fix: three rows are unrepresentable by a dictionary difference."""
    committed = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert committed["blindspot"]["missed_rows_wholly_invisible_to_mechanism"] == 3
    assert committed["counts"]["false_negative"] == 3
    assert committed["counts"]["false_positive"] == 0
    # Precision is carried by suppression, not by the detector being right.
    risk = committed["remediation_risk"]
    assert risk["nonleaking_rows_where_rule_fired"] == 10
    assert risk["nonleaking_rows_where_rule_scored"] == 0
    assert risk["precision_without_suppression_pct"] is not None
    assert risk["precision_without_suppression_pct"] < committed["detector"]["leakage_precision_pct"]


def test_missing_dictionaries_fail_loudly(monkeypatch):
    """The gate must never pass vacuously.

    With no dictionaries the rule finds nothing anywhere, which would otherwise
    look like "no blindspot, precision clean" and could match a baseline recorded
    on a real run. Absence of the managed resources is a setup error, not a result.
    """
    monkeypatch.setattr(gold_agreement, "get_ptpt_dictionary", lambda: None)
    monkeypatch.setattr(gold_agreement, "get_ptbr_dictionary", lambda: None)
    with pytest.raises(SystemExit) as exc:
        gold_agreement.build_report()
    assert "runner.py setup" in str(exc.value)


def test_missing_spacy_model_fails_loudly(monkeypatch):
    """A half-installed environment must not be reportable as a result.

    Without the pinned model the URL/email masking degrades and header
    local-parts are read as dialect evidence, which moves precision on this
    corpus from 100% to 58.33%. That is an environment state, not a finding
    about the rule, so it has to be an error rather than a caveat.
    """
    monkeypatch.setattr(gold_agreement, "_spacy_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        gold_agreement.build_report()
    assert gold_agreement.SPACY_MODEL_DISTRIBUTION in str(exc.value)


def test_baseline_binds_the_untracked_dictionaries():
    """The dictionaries decide every marker class but are not in git; their digests are.

    Without this, a Hunspell update that flips a both-dictionaries verdict would
    pass the gate as "same source, same numbers" while measuring something else.
    """
    committed = json.loads(BASELINE.read_text(encoding="utf-8"))
    bound = committed["managed_resource_sha256"]
    assert set(bound) == set(gold_agreement.MANAGED_RESOURCES)
    for rel, digest in bound.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), rel
        assert digest == gold_agreement._sha256(ROOT / rel), f"{rel} drifted from the baseline"
    assert gold_agreement._comparable(gold_agreement.build_report())


def test_gate_fails_when_the_detector_regresses(tmp_path):
    """A gate that cannot fail is decoration."""
    tampered = json.loads(BASELINE.read_text(encoding="utf-8"))
    tampered["detector"]["leakage_recall_pct"] = 11.0
    forged = tmp_path / "gold-agreement.json"
    forged.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(TOOL), "--check-baseline", str(forged)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 1
    assert "differs from the committed baseline" in proc.stderr


def test_relative_output_path_is_supported():
    """`--output baselines/...` is how CI invokes it; it must not crash on relative_to()."""
    out = ROOT / "baselines" / "_tmp_relative_check.json"
    try:
        proc = subprocess.run(
            [sys.executable, str(TOOL), "--output", "baselines/_tmp_relative_check.json"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        assert out.is_file()
    finally:
        out.unlink(missing_ok=True)


def test_tool_is_registered_in_the_source_inventory():
    """CI diffs MANIFEST.txt against git ls-files; an unlisted file breaks the build."""
    manifest = (ROOT / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    assert "tools/gold_agreement.py" in manifest
    assert "tests/test_gold_agreement.py" in manifest
    assert "baselines/gold-agreement.json" in manifest
    assert manifest == sorted(manifest, key=lambda line: line.encode("utf-8")), "MANIFEST.txt must stay byte-sorted"
