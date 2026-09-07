"""Small, auditable runner: fixed inputs in, raw model outputs + diagnostics out."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import (
    ANALYSIS_REFERENCE, DEFAULT_CONCURRENCY, DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT, ELABORATION_CONSTRAINTS,
    ELABORATION_PROMPTS, RESULTS_DIR, SYSTEM_PROMPT_ANALYSIS,
    SYSTEM_PROMPT_ELABORATION, BENCHMARK_VERSION,
)
from core.api_client import OpenRouterClient
from core.artifacts import build_manifest, sha256_bytes, sha256_file, write_manifest
from core.generation_evaluator import evaluate_generation_output, load_constraint_map
from core.pt_dialect import evaluate_pt_dialect
from core.scorecard import build_elaboration_scorecard
from core.schemas import validate_email_analysis
from core.wf_fidelity import compute_word_fidelity_from_dialect
from core.writing_quality import evaluate_writing_quality


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _source_hashes(paths: Iterable[Path]) -> Dict[str, str]:
    return {str(path.relative_to(Path(__file__).resolve().parent)): sha256_file(path) for path in paths}


def _dependency_versions() -> Dict[str, str]:
    versions = {}
    for package in ("httpx", "jsonschema", "language-tool-python", "spylls"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "UNINSTALLED"
    return versions


def _write_run_manifest(out: Path, *, kind: str, model: str, parameters: Dict[str, Any]) -> None:
    root = Path(__file__).resolve().parent
    if kind == "analysis":
        input_paths = [ANALYSIS_REFERENCE]
        evaluator_paths = [root / "core" / "schemas.py", root / "runner.py"]
        system_prompt = SYSTEM_PROMPT_ANALYSIS
    else:
        input_paths = [ELABORATION_PROMPTS, ELABORATION_CONSTRAINTS]
        evaluator_paths = [
            root / "core" / "generation_evaluator.py",
            root / "core" / "pt_dialect.py",
            root / "core" / "wf_fidelity.py",
            root / "core" / "writing_quality.py",
            root / "core" / "scorecard.py",
            root / "runner.py",
        ]
        system_prompt = SYSTEM_PROMPT_ELABORATION
    resource_paths = [
        root / "docs" / name for name in ("pt_PT.dic", "pt_PT.aff", "pt_BR.dic", "pt_BR.aff")
    ] + [
        root / "models" / "model_quantized.ftz",
        root / "data" / "wq_length_neutral_calibration.json",
    ]
    manifest = build_manifest(
        out,
        kind=kind,
        benchmark_version=BENCHMARK_VERSION,
        model=model,
        input_hashes=_source_hashes(input_paths),
        evaluator_versions=_source_hashes(evaluator_paths),
        resource_hashes={
            str(path.relative_to(root)): sha256_file(path) if path.is_file() else "UNAVAILABLE"
            for path in resource_paths
        },
        dependency_versions=_dependency_versions(),
        parameters={
            **parameters,
            "system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
            "python_version": sys.version,
        },
        command=list(sys.argv),
    )
    write_manifest(out, manifest)


def _exact(a: Any, b: Any) -> bool:
    return a == b


def _entity_f1(truth: List[str], pred: List[str]) -> float:
    def norm(x: str) -> str:
        return " ".join(str(x).strip().lower().split())
    t = set(norm(x) for x in truth if x)
    p = set(norm(x) for x in pred if x)
    if not t and not p:
        return 1.0
    if not t or not p:
        return 0.0
    tp = len(t & p)
    prec = tp / len(p)
    rec = tp / len(t)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def score_analysis(truth_rows: List[Dict[str, Any]], result_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_ids = []
    by_id = {}
    for result in result_rows:
        if result.get("id") is None:
            continue
        result_id = str(result["id"])
        result_ids.append(result_id)
        by_id.setdefault(result_id, result)
    duplicate_ids = sorted({rid for rid in result_ids if result_ids.count(rid) > 1})
    truth_ids = {str(r["id"]) for r in truth_rows}
    unexpected_ids = sorted(set(result_ids) - truth_ids)
    metrics = {
        "schema_validity_pct": None,
        "category_acc_pct": None,
        "urgency_acc_pct": None,
        "action_req_acc_pct": None,
        "sentiment_acc_pct": None,
        "language_variant_acc_pct": None,
        "entity_f1": None,
        "samples": len(truth_rows),
        "missing_results": 0,
        "failed_results": 0,
        "duplicate_result_ids": duplicate_ids,
        "unexpected_result_ids": unexpected_ids,
        "requested_samples": len(truth_rows),
        "attempted_samples": len(set(result_ids)),
    }
    checks = {k: [] for k in ["schema", "category", "urgency", "action", "sentiment", "language"]}
    f1s: List[float] = []
    missing = 0
    for row in truth_rows:
        rid = str(row["id"])
        got = by_id.get(rid)
        if not got:
            missing += 1
            continue
        if got.get("status") == "error" or got.get("error"):
            missing += 1
            metrics["failed_results"] += 1
            continue
        parsed = got.get("parsed") or {}
        checks["schema"].append(bool(got.get("is_valid_schema")))
        gt = row.get("ground_truth", {})
        checks["category"].append(_exact(parsed.get("category"), gt.get("category")))
        checks["urgency"].append(_exact(parsed.get("urgency"), gt.get("urgency")))
        checks["action"].append(_exact(parsed.get("action_required"), gt.get("action_required")))
        checks["sentiment"].append(_exact(parsed.get("sentiment"), gt.get("sentiment")))
        checks["language"].append(_exact(parsed.get("language_variant"), gt.get("language_variant")))
        f1s.append(_entity_f1(gt.get("key_entities", []), parsed.get("key_entities", [])))
    def pct(xs: List[bool]):
        return round(100 * sum(xs) / len(xs), 2) if xs else None
    metrics.update({
        "schema_validity_pct": pct(checks["schema"]),
        "category_acc_pct": pct(checks["category"]),
        "urgency_acc_pct": pct(checks["urgency"]),
        "action_req_acc_pct": pct(checks["action"]),
        "sentiment_acc_pct": pct(checks["sentiment"]),
        "language_variant_acc_pct": pct(checks["language"]),
        "entity_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "missing_results": missing,
        "denominators": {
            "schema_validity_pct": len(checks["schema"]),
            "category_acc_pct": len(checks["category"]),
            "urgency_acc_pct": len(checks["urgency"]),
            "action_req_acc_pct": len(checks["action"]),
            "sentiment_acc_pct": len(checks["sentiment"]),
            "language_variant_acc_pct": len(checks["language"]),
            "entity_f1": len(f1s),
        },
    })
    return metrics


async def _run_analysis(model: str, rows: List[Dict[str, Any]], out: Path, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    client = OpenRouterClient()
    results: List[Dict[str, Any]] = []
    async def one(row: Dict[str, Any]) -> None:
        async with sem:
            started = time.time()
            try:
                res = await client.analyze_email_async(model, SYSTEM_PROMPT_ANALYSIS, row["email"], max_tokens=DEFAULT_MAX_TOKENS)
                results.append({"id": row["id"], "model": model, "status": "success", "finished_at": time.time(), "started_at": started, **res})
            except Exception as exc:
                results.append({"id": row["id"], "model": model, "status": "error", "finished_at": time.time(), "started_at": started, "error": str(exc)})
    await asyncio.gather(*(one(row) for row in rows))
    await client.aclose()
    write_jsonl(out, sorted(results, key=lambda r: str(r["id"])))
    _write_run_manifest(out, kind="analysis", model=model, parameters={"concurrency": concurrency})


async def _run_generation(model: str, prompts: Dict[str, Any], constraints: Dict[str, Any], out: Path, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    client = OpenRouterClient()
    results: List[Dict[str, Any]] = []
    async def one(item: Dict[str, Any]) -> None:
        async with sem:
            started = time.time()
            pid = item["id"]
            try:
                res = await client.elaborate_email_async(model, SYSTEM_PROMPT_ELABORATION, item["prompt"], max_tokens=DEFAULT_MAX_TOKENS)
                text = str(res.get("content") or "")
                ev = evaluate_generation_output(text, constraints.get(pid, {}), source_text=item["prompt"])
                target_lang = item.get("target_lang", "pt-pt")
                dialect = evaluate_pt_dialect(text, use_languagetool=True) if target_lang == "pt-pt" else {}
                wf = compute_word_fidelity_from_dialect(text, dialect) if target_lang == "pt-pt" else {"wf_score": None}
                wq = evaluate_writing_quality(text, language="pt-PT" if target_lang == "pt-pt" else "en-US", use_languagetool=True)
                results.append({
                    "id": pid, "model": model, "status": "success", "target_lang": target_lang,
                    "started_at": started, "finished_at": time.time(),
                    **res,
                    "instruction_adherence_pct": ev["instruction_adherence_score"],
                    "semantic_preservation_pct": ev["semantic_preservation_score"],
                    "pt_dialect_score": dialect.get("pt_dialect_score"),
                    "euptvid_probability": dialect.get("euptvid_prob"),
                    "ptpt_compliance_pct": dialect.get("ptpt_compliance_pct"),
                    "ptbr_leakage_detected": dialect.get("ptbr_leakage_detected"),
                    "wf_score": wf.get("wf_score"),
                    "writing_quality_score": wq.get("writing_quality_score"),
                    "writing_quality": wq,
                    "generation_evaluation": ev,
                    "dialect_evaluation": dialect,
                    "wf_evaluation": wf,
                })
            except Exception as exc:
                results.append({"id": pid, "model": model, "status": "error", "target_lang": item.get("target_lang", "pt-pt"), "error": str(exc), "started_at": started, "finished_at": time.time()})
    await asyncio.gather(*(one(item) for item in prompts.get("prompts", [])))
    await client.aclose()
    write_jsonl(out, sorted(results, key=lambda r: str(r["id"])))
    _write_run_manifest(out, kind="generation", model=model, parameters={"concurrency": concurrency})


def load_prompts() -> Dict[str, Any]:
    return json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8"))


def load_constraints() -> Dict[str, Any]:
    return load_constraint_map(str(ELABORATION_CONSTRAINTS))


def main() -> None:
    p = argparse.ArgumentParser(description="Lean PT-PT/English email benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("validate", help="run local benchmark integrity checks")
    s = sub.add_parser("analysis", help="run structured analysis against the fixed reference set")
    s.add_argument("--model", required=True)
    s.add_argument("--out", type=Path, default=RESULTS_DIR / "analysis.jsonl")
    s.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    s = sub.add_parser("generation", help="run fixed PT-PT generation prompts")
    s.add_argument("--model", required=True)
    s.add_argument("--out", type=Path, default=RESULTS_DIR / "generation.jsonl")
    s.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    s = sub.add_parser("score", help="score an existing generation result JSONL")
    s.add_argument("--results", type=Path, required=True)
    args = p.parse_args()

    if args.cmd == "validate":
        from validate import run_validation
        run_validation()
        return
    if args.cmd == "analysis":
        rows = read_jsonl(ANALYSIS_REFERENCE)
        asyncio.run(_run_analysis(args.model, rows, args.out, max(1, min(args.concurrency, 32))))
        print(json.dumps(score_analysis(rows, read_jsonl(args.out)), indent=2, ensure_ascii=False))
        return
    if args.cmd == "generation":
        asyncio.run(_run_generation(args.model, load_prompts(), load_constraints(), args.out, max(1, min(args.concurrency, 32))))
        print(json.dumps(build_elaboration_scorecard(read_jsonl(args.out)), indent=2, ensure_ascii=False))
        return
    if args.cmd == "score":
        print(json.dumps(build_elaboration_scorecard(read_jsonl(args.results)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
