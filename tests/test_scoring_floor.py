"""Scoring-scale validity: the metric must have a real zero and a reachable ceiling.

These tests guard the *instrument*, not model behaviour. A criterion that a
content-free boilerplate reply already satisfies inflates every model's score by
the same amount and compresses the band in which models actually differ, so it
is a defect of the scale rather than of the sample size. Nothing here needs more
tasks to be meaningful, and each test fails if the property regresses.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from core.generation_evaluator import (  # noqa: E402
    evaluate_generation_output,
    evaluate_instruction_adherence,
    load_constraint_map,
)
from run_smoke_baseline import CONSTANT_EMAIL  # noqa: E402
from tools.check_scoring_floor import check_floor  # noqa: E402

CONSTRAINTS_PATH = ROOT / "data" / "elaboration_constraints.json"
PROMPTS_PATH = ROOT / "data" / "elaboration_prompts_pt_pt.json"

# One genuinely compliant reply per retargeted task. These are ceiling probes:
# they prove the tightened criteria still describe achievable behaviour, so a
# pattern was never "fixed" by making it unsatisfiable.
IDEAL_REPLIES = {
    "elab_pt_02": (
        "Assunto: Segunda via da fatura FT-4832\n\n"
        "Exma. Senhora Marta Silva,\n\n"
        "Confirmamos que a fatura corrigida será enviada hoje para o seu email.\n\n"
        "Para qualquer esclarecimento adicional, o departamento financeiro está "
        "disponível em financeiro@exemplo.pt.\n\n"
        "Com os melhores cumprimentos,\nAna Ferreira"
    ),
    "elab_pt_08": (
        "Assunto: Cobrança duplicada de 89,90 €\n\nExmo. Senhor,\n\n"
        "Confirmamos a existência de uma situação anómala na cobrança do valor de "
        "89,90 €.\n\nFoi já solicitado o reembolso de uma das cobranças, que deverá "
        "estar refletido na conta até 3 dias úteis.\n\n"
        "Com os melhores cumprimentos,\nEquipa de Apoio a Clientes"
    ),
    "elab_pt_12": (
        "Assunto: Estado da sua candidatura\n\nExma. Senhora,\n\n"
        "Agradecemos o tempo que dedicou à sua candidatura.\n\n"
        "O processo continua em avaliação pela equipa de recrutamento. Daremos uma "
        "resposta concreta até 14 de novembro.\n\nAtentamente,\nMaria Lopes"
    ),
    "elab_pt_13": (
        "Assunto: Alteração da palavra-passe\n\nExmo. Senhor,\n\n"
        "Para alterar a palavra-passe, aceda a Definições > Segurança e escolha uma "
        "palavra-passe única, sem partilhar dados com terceiros.\n\n"
        "Caso persistam dificuldades, contacte o suporte em suporte@exemplo.pt.\n\n"
        "Com os melhores cumprimentos,\nEquipa Técnica"
    ),
    "elab_pt_14": (
        "Assunto: Confirmação de agenda\n\nExma. Senhora,\n\n"
        "Confirmamos a nossa chamada para sexta-feira às 11h, conforme proposto.\n\n"
        "Com os melhores cumprimentos,\nJoão Almeida"
    ),
    "elab_pt_20": (
        "Assunto: Número de processo para referência\n\nExmo. Senhor,\n\n"
        "O número do processo para referência futura é PT-2026-4471.\n\n"
        "Com os melhores cumprimentos,\nEquipa de Apoio"
    ),
}


def test_boilerplate_controls_are_the_traps_one_might_expect():
    """Guard the test's own premise: the control really is generic PT-PT prose.

    If someone "fixes" the floor by editing the control text into something no
    real email resembles, the gate becomes vacuous. Pinning the accidental
    vocabulary keeps that honest.
    """
    for trap in ("cumprimentos", "contacto", "pedido", "apoio"):
        assert trap in CONSTANT_EMAIL.lower()
    # "obrigad" and a concrete reference are absent: criteria may require them.
    assert "obrigad" not in CONSTANT_EMAIL.lower()
    assert not any(char.isdigit() for char in CONSTANT_EMAIL)


def test_scoring_floor_is_zero_on_every_task():
    report = check_floor(CONSTRAINTS_PATH, PROMPTS_PATH)
    assert report["tasks_checked"] == 20
    assert report["violations"] == [], json.dumps(report["violations"], indent=2, ensure_ascii=False)


def test_no_task_scores_above_zero_on_the_no_content_control():
    constraints = load_constraint_map(str(CONSTRAINTS_PATH))
    prompts = {item["id"]: item.get("prompt", "") for item in json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))["prompts"]}
    for task_id, spec in constraints.items():
        result = evaluate_generation_output(CONSTANT_EMAIL, spec, source_text=prompts.get(task_id, ""))
        assert result["instruction_adherence_score"] in (0.0, None), (task_id, result["adherence_details"]["passed_items"])
        assert result["semantic_preservation_score"] in (0.0, None), (task_id, result["semantic_details"])


@pytest.mark.parametrize("task_id", sorted(IDEAL_REPLIES))
def test_retargeted_criteria_remain_reachable(task_id):
    """Each tightened action still passes on a plausible compliant reply."""
    constraints = load_constraint_map(str(CONSTRAINTS_PATH))
    result = evaluate_instruction_adherence(IDEAL_REPLIES[task_id], constraints[task_id])
    assert result["adherence_score"] == 100.0, (task_id, result["failed_items"])


def test_courtesy_only_action_no_longer_counts_as_adherence():
    """Thanking + signing off is not task adherence; it is what any reply has."""
    constraints = load_constraint_map(str(CONSTRAINTS_PATH))
    courtesy = "Exma. Senhora,\n\nObrigado pelo seu contacto.\n\nCom os melhores cumprimentos,\nEquipa"
    for task_id in ("elab_pt_12", "elab_pt_13", "elab_pt_14", "elab_pt_20"):
        result = evaluate_instruction_adherence(courtesy, constraints[task_id])
        assert not result["passed_items"], (task_id, result["passed_items"])


def test_elaboration_task_dropped_a_non_discriminating_criterion():
    """elab_pt_20 'thank_contact' was removed, not silently retightened."""
    constraints = load_constraint_map(str(CONSTRAINTS_PATH))
    actions = [item["action_id"] for item in constraints["elab_pt_20"]["required_actions"]]
    assert "thank_contact" not in actions
    assert actions == ["case_number", "professional_close"] or "professional_close" in actions


def test_gate_fails_when_a_permissive_criterion_is_reintroduced(tmp_path):
    """The gate must be able to fail, or it is decoration."""
    constraints = json.loads(CONSTRAINTS_PATH.read_text(encoding="utf-8"))
    constraints["elab_pt_14"]["required_actions"].append(
        {"action_id": "courtesy_closer", "pattern": "cumprimentos|atentamente"}
    )
    mutated = tmp_path / "elaboration_constraints.json"
    mutated.write_text(json.dumps(constraints, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = check_floor(mutated, PROMPTS_PATH)
    offenders = {item["task_id"]: item["criteria_passed_by_boilerplate"] for item in report["violations"]}
    assert offenders.get("elab_pt_14") == ["courtesy_closer"]

    # And as CI runs it, via exit status rather than a log line.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_scoring_floor.py"),
         "--constraints", str(mutated), "--prompts", str(PROMPTS_PATH)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 1
    assert "courtesy_closer" in proc.stderr


def test_gate_passes_on_the_committed_constraint_file():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_scoring_floor.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout


def test_languagetool_identity_reports_absence_without_inventing_a_version(monkeypatch):
    """Manifest recording must degrade to empty, never to a plausible guess.

    Also pins the safety property that matters here: describing the tool must
    not construct a checker, since that would launch a JVM during a run that
    never asked for grammar checks.
    """
    import core.languagetool_local as ltp

    monkeypatch.setattr(ltp, "_CHECKERS", {})
    monkeypatch.setattr(ltp, "_CHECKER_LOCKS", {})
    monkeypatch.setattr(ltp, "_ENGINE_VERSIONS", {})
    monkeypatch.setattr(ltp, "language_tool_python", _Explode(), raising=True)

    described = ltp.describe_local_languagetool()
    assert described == {
        "mode": "local",
        "wrapper_version": None,
        "languages": [],
        "engine_versions": {},
        "engine_version_unrecorded": [],
    }


def test_engine_version_detection_reads_attributes_only():
    import core.languagetool_local as ltp

    class Named:
        ltp_version = "6.6"

    class FromPath:
        ltp_version = None
        install_path = "/tmp/.cache/language_tool_python/6.5"

    class Silent:
        pass

    assert ltp._detect_engine_version(Named()) == "6.6"
    assert ltp._detect_engine_version(FromPath()) == "6.5"
    assert ltp._detect_engine_version(Silent()) is None
    # A non-directory-shaped install path must not be reported as a version.
    class OddPath:
        install_path = "/tmp/.cache/language_tool_python/staging"

    assert ltp._detect_engine_version(OddPath()) is None


def _forbid_construction(*args, **kwargs):  # pragma: no cover - guards JVM launch
    raise AssertionError("describe_local_languagetool must not build a checker")


class _Explode:
    """Stand-in for the library: any attempt to build a checker is a failure.

    Replaces the module attribute rather than patching `LanguageTool` inside it,
    so the assertion holds even where `language_tool_python` is not installed
    (the adapter then holds `None`, which cannot carry attributes).
    """

    LanguageTool = staticmethod(_forbid_construction)


def test_constraints_are_hashed_into_the_scoring_definition():
    """A criterion retarget must be visible in every artifact's provenance."""
    source = (ROOT / "runner.py").read_text(encoding="utf-8")
    generation_block = source.split("system_prompt = SYSTEM_PROMPT_ELABORATION", 1)[0].rsplit("evaluator_paths = [", 1)[1]
    assert "ELABORATION_CONSTRAINTS" in generation_block


def test_smoke_baseline_and_floor_gate_share_one_control_text():
    """Prevents a control text tuned to satisfy the gate."""
    source = (ROOT / "tools" / "check_scoring_floor.py").read_text(encoding="utf-8")
    assert "from run_smoke_baseline import CONSTANT_EMAIL" in source
    baseline_source = (ROOT / "tools" / "run_smoke_baseline.py").read_text(encoding="utf-8")
    assert baseline_source.count('CONSTANT_EMAIL = """') == 1


def test_constraint_file_stays_well_formed():
    constraints = json.loads(CONSTRAINTS_PATH.read_text(encoding="utf-8"))
    assert len(constraints) == 20
    for task_id, spec in constraints.items():
        for group in ("required_facts", "required_actions", "forbidden_changes"):
            for item in spec.get(group, []) or []:
                # Every scored criterion must name itself, or failures become
                # un-debuggable and un-reportable.
                assert any(k in item for k in ("action_id", "fact_id", "forbidden_id")), (task_id, group, item)
                assert item.get("pattern"), (task_id, group, item)
