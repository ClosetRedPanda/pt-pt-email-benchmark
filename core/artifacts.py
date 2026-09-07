"""Result-artifact validation and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ARTIFACT_SCHEMA_VERSION = "1.0"
MANIFEST_VERSION = "1.1"
_NUMERIC_FIELDS = {
    "latency_ms", "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "total_tokens", "cost_usd", "instruction_adherence_pct",
    "semantic_preservation_pct", "euptvid_probability", "ptpt_compliance_pct",
    "wf_score", "writing_quality_score",
}


class ArtifactValidationError(ValueError):
    """Raised when a result artifact cannot be safely scored."""


def manifest_path(result_path: Path) -> Path:
    """Return the sidecar path for a JSONL result artifact."""
    return result_path.with_suffix(".manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _number_is_valid(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _completion_errors(row: Dict[str, Any], row_id: str) -> List[str]:
    """Reject successful generation rows that are not complete model outputs."""
    if row.get("status") != "success":
        return []
    errors: List[str] = []
    if not str(row.get("content", "")).strip():
        errors.append(f"row ({row_id or '?' }): successful generation lacks content")
    raw_response = row.get("raw_response")
    if not isinstance(raw_response, dict):
        return errors
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        errors.append(f"row ({row_id or '?' }): successful generation lacks provider choices")
        return errors
    finish_reason = choices[0].get("finish_reason")
    if finish_reason not in {None, "stop"}:
        errors.append(
            f"row ({row_id or '?' }): unsuccessful finish_reason={finish_reason!r}"
        )
    message = choices[0].get("message")
    if isinstance(message, dict) and message.get("refusal"):
        errors.append(f"row ({row_id or '?' }): provider refusal in successful generation")
    if raw_response.get("error"):
        errors.append(f"row ({row_id or '?' }): provider error in successful generation")
    return errors


def validate_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    kind: str,
    expected_ids: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Validate row identity, status, and numeric provenance fields."""
    materialized = list(rows)
    errors: List[str] = []
    ids: List[str] = []
    for index, row in enumerate(materialized, 1):
        if not isinstance(row, dict):
            errors.append(f"row {index}: expected an object")
            continue
        raw_id = row.get("id")
        row_id = str(raw_id).strip() if raw_id is not None else ""
        if not row_id:
            errors.append(f"row {index}: missing id")
        else:
            ids.append(row_id)
        if strict and not str(row.get("model", "")).strip():
            errors.append(f"row {index} ({row_id or '?' }): missing model")
        status = row.get("status")
        if strict and status not in {"success", "error"}:
            errors.append(f"row {index} ({row_id or '?' }): status must be success or error")
        if status == "error" and not str(row.get("error", "")).strip():
            errors.append(f"row {index} ({row_id or '?' }): error status lacks error detail")
        if status == "success" and row.get("error"):
            errors.append(f"row {index} ({row_id or '?' }): success row contains error detail")
        if strict and kind == "generation":
            errors.extend(_completion_errors(row, row_id))
        for field in _NUMERIC_FIELDS:
            if field in row and row[field] is not None and not _number_is_valid(row[field]):
                errors.append(f"row {index} ({row_id or '?' }): {field} must be finite and non-negative")

    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    if duplicates:
        errors.append(f"duplicate ids: {duplicates}")
    if expected_ids is not None:
        expected = {str(value) for value in expected_ids}
        actual = set(ids)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            errors.append(f"missing expected ids: {missing}")
        if unexpected:
            errors.append(f"unexpected ids: {unexpected}")
    if kind not in {"analysis", "generation"}:
        errors.append(f"unsupported artifact kind: {kind}")
    if errors:
        raise ArtifactValidationError("; ".join(errors))
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": kind,
        "row_count": len(materialized),
        "ids": ids,
    }


def build_manifest(
    result_path: Path,
    *,
    kind: str,
    benchmark_version: str,
    model: str,
    input_hashes: Optional[Dict[str, str]] = None,
    evaluator_versions: Optional[Dict[str, str]] = None,
    resource_hashes: Optional[Dict[str, str]] = None,
    dependency_versions: Optional[Dict[str, str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    command: Optional[List[str]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a manifest whose result hash binds it to one JSONL artifact."""
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result_hash = sha256_file(result_path)
    return {
        "manifest_version": MANIFEST_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": kind,
        "artifact_id": result_hash,
        "result_path": result_path.name,
        "result_sha256": result_hash,
        "row_count": sum(1 for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "created_at_utc": created_at or datetime.now(timezone.utc).isoformat(),
        "benchmark_version": benchmark_version,
        "model": model,
        "input_hashes": input_hashes or {},
        "evaluator_versions": evaluator_versions or {},
        "resource_hashes": resource_hashes or {},
        "dependency_versions": dependency_versions or {},
        "parameters": parameters or {},
        "command": command or [],
    }


def write_manifest(result_path: Path, manifest: Dict[str, Any]) -> Path:
    """Write a deterministic, human-readable manifest beside a result file."""
    destination = manifest_path(result_path)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_manifest(result_path: Path) -> Dict[str, Any]:
    path = manifest_path(result_path)
    if not path.is_file():
        raise ArtifactValidationError(f"missing manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"invalid manifest JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"manifest must be an object: {path}")
    required = {
        "manifest_version", "artifact_schema_version", "artifact_kind",
        "result_sha256", "row_count", "dependency_versions",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ArtifactValidationError(f"manifest missing fields: {missing}")
    if value["result_sha256"] != sha256_file(result_path):
        raise ArtifactValidationError(f"manifest hash does not match result: {result_path}")
    return value