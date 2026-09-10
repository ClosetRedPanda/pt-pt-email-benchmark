#!/usr/bin/env python3
"""Publish a deterministic, non-LLM floor baseline for pipeline smoke testing.

This intentionally weak constant strategy is not ranking eligible. It makes no
network call and does not inspect test labels when choosing predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core._util import sha256_file
from core.generation_evaluator import evaluate_generation_output, load_constraint_map
from core.schemas import validate_email_analysis
BASELINE_ID = "deterministic-constant-smoke-v1"
CONSTANT_ANALYSIS = {
    "sender_name": None,
    "sender_email": None,
    "recipient_name": None,
    "recipient_email": None,
    "date_sent": None,
    "subject": None,
    "category": "other",
    "sub_category": None,
    "urgency": "low",
    "sentiment": "neutral",
    "action_required": False,
    "required_actions": [],
    "key_entities": [],
    "language_variant": "pt-pt",
    "summary": "Mensagem recebida para análise.",
}
# Shared no-content control. `tools/check_scoring_floor.py` imports this exact
# string to assert that no task criterion can be satisfied by boilerplate, so
# editing it changes the meaning of that gate as well as this baseline. Keeping
# a single copy is deliberate: a control text tuned to make the gate pass would
# be worth nothing.
CONSTANT_EMAIL = """Assunto: Resposta ao seu pedido

Exmo. Senhor,

Agradecemos o seu contacto. Registámos o pedido e responderemos assim que possível.

Com os melhores cumprimentos,
Equipa de Apoio"""


def pct(values: list[bool]) -> float | None:
    return round(100 * sum(values) / len(values), 2) if values else None


def entity_f1(truth: list[str], prediction: list[str]) -> float:
    norm = lambda value: " ".join(str(value).strip().lower().split())
    expected = {norm(value) for value in truth if value}
    actual = {norm(value) for value in prediction if value}
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision, recall = overlap / len(actual), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def build_report() -> dict[str, Any]:
    analysis_path = ROOT / "data/analysis_reference.jsonl"
    prompts_path = ROOT / "data/elaboration_prompts_pt_pt.json"
    constraints_path = ROOT / "data/elaboration_constraints.json"
    analysis_rows = [json.loads(line) for line in analysis_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid, error = validate_email_analysis(CONSTANT_ANALYSIS)
    if not valid:
        raise RuntimeError(f"constant analysis baseline is schema-invalid: {error}")

    keys = {
        "category_acc_pct": "category",
        "urgency_acc_pct": "urgency",
        "action_req_acc_pct": "action_required",
        "sentiment_acc_pct": "sentiment",
        "language_variant_acc_pct": "language_variant",
    }
    analysis_metrics: dict[str, Any] = {
        "samples": len(analysis_rows),
        "schema_validity_pct": 100.0,
    }
    for metric, field in keys.items():
        analysis_metrics[metric] = pct([
            row["ground_truth"].get(field) == CONSTANT_ANALYSIS[field]
            for row in analysis_rows
        ])
    analysis_metrics["entity_f1"] = round(sum(
        entity_f1(row["ground_truth"].get("key_entities", []), [])
        for row in analysis_rows
    ) / len(analysis_rows), 4)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))["prompts"]
    constraints = load_constraint_map(str(constraints_path))
    task_results = []
    for item in prompts:
        result = evaluate_generation_output(
            CONSTANT_EMAIL, constraints[item["id"]], source_text=item["prompt"]
        )
        task_results.append({
            "id": item["id"],
            "instruction_adherence_pct": result["instruction_adherence_score"],
            "semantic_preservation_pct": result["semantic_preservation_score"],
        })

    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in task_results if row[field] is not None]
        return round(sum(values) / len(values), 2) if values else None

    hashed_paths = [
        analysis_path, prompts_path, constraints_path,
        ROOT / "core/schemas.py", ROOT / "core/generation_evaluator.py", Path(__file__),
    ]
    return {
        "baseline_id": BASELINE_ID,
        "baseline_type": "deterministic non-LLM pipeline smoke/floor",
        "ranking_eligible": False,
        "uses_llm": False,
        "uses_network": False,
        "uses_test_labels_to_choose_predictions": False,
        "warning": "Not a candidate-model result and not evidence of model quality. Do not place it in model rankings.",
        "strategy": {
            "analysis": "One fixed schema-valid object for every sample.",
            "generation": "One fixed generic PT-PT email for every task.",
            "constant_generation_text": CONSTANT_EMAIL,
        },
        "analysis": analysis_metrics,
        "generation": {
            "samples": len(task_results),
            "instruction_adherence_pct": mean("instruction_adherence_pct"),
            "semantic_preservation_pct": mean("semantic_preservation_pct"),
            "task_results": task_results,
            "excluded_metrics": [
                "PT-PT fidelity", "PT-BR leakage", "Word Fidelity",
                "writing quality", "latency", "throughput", "cost",
            ],
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in hashed_paths},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baselines/smoke-baseline.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_report(), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
