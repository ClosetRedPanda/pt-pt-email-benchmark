"""Small integrity suite: benchmark correctness, not application infrastructure."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from config import ANALYSIS_REFERENCE, ELABORATION_CONSTRAINTS, ELABORATION_PROMPTS, DATA_DIR
from core._util import sha256_file
from core.generation_evaluator import load_constraint_map
from core.schemas import validate_email_analysis
from core.pt_dialect import evaluate_pt_dialect
from core.wf_fidelity import compute_word_fidelity_from_dialect
from core.writing_quality import evaluate_writing_quality


def run_validation() -> None:
    root = Path(__file__).resolve().parent
    provenance = json.loads((DATA_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["dataset_type"] == "synthetic"
    assert provenance["llm_assisted"] is True
    for relative_path, expected in provenance["artifacts"].items():
        path = root / relative_path
        assert path.is_file(), f"provenance artifact missing: {relative_path}"
        assert sha256_file(path) == expected, f"update provenance digest for {relative_path}"

    rows = [json.loads(x) for x in ANALYSIS_REFERENCE.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 20
    assert len(rows) == len({r["id"] for r in rows}) and rows
    assert all(r.get("task_type") == "analysis" and "ground_truth" in r for r in rows)
    for row in rows:
        ok, err = validate_email_analysis(row["ground_truth"])
        assert ok, err
    constraints = load_constraint_map(str(ELABORATION_CONSTRAINTS))
    prompts = json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8"))
    assert isinstance(constraints, dict) and len(constraints) >= 10
    assert isinstance(prompts.get("prompts"), list) and prompts["prompts"]
    # A few representative deterministic smoke checks.
    pt = "Boa tarde, a nossa equipa está a enviar a fatura. Contactou-nos ontem."
    dialect = evaluate_pt_dialect(pt, use_languagetool=False)
    wf = compute_word_fidelity_from_dialect(pt, dialect)
    assert "ptpt_compliance_pct" in dialect and "wf_score" in wf
    wq = evaluate_writing_quality(pt, language="pt-PT", use_languagetool=False)
    assert "writing_quality_score" in wq
    # Agreement of the leakage detector with the labelled reference set. This is a
    # measurement of the rule, not of a submission: a mismatch means the detector
    # changed and its recorded accuracy/blindspot no longer describes it.
    tool = root / "tools" / "gold_agreement.py"
    assert tool.is_file(), "tools/gold_agreement.py is missing from the source inventory"
    if str(tool.parent) not in sys.path:
        sys.path.insert(0, str(tool.parent))
    import gold_agreement  # noqa: PLC0415

    baseline = root / "baselines" / "gold-agreement.json"
    assert baseline.is_file(), "baselines/gold-agreement.json is missing: regenerate with gold_agreement.py --output"
    expected = json.loads(baseline.read_text(encoding="utf-8"))
    assert gold_agreement._comparable(gold_agreement.build_report()) == gold_agreement._comparable(expected), (
        "detector no longer agrees with gold as recorded in baselines/gold-agreement.json"
    )
    print(f"PASS: {len(rows)} analysis references, {len(constraints)} generation constraint sets, deterministic core smoke tests")
    print("NOTE: formal multi-rater WQ validation is not claimed; see docs/VALIDATION.md")


if __name__ == "__main__":
    run_validation()
