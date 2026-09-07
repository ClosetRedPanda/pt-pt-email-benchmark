#!/usr/bin/env python3
"""Persist an explicitly rescored generation artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare import _enrich_generation_records
from config import BENCHMARK_VERSION, ELABORATION_PROMPTS
from core.artifacts import (
    ArtifactValidationError,
    build_manifest,
    sha256_file,
    validate_rows,
    write_manifest,
)
from runner import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Persist an explicit exploratory generation rescore")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--full-dialect-checks", action="store_true")
    parser.add_argument("--force", action="store_true", help="allow replacing the destination artifact")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source does not exist: {args.source}")
    if args.destination.exists() and not args.force:
        parser.error(f"destination exists; choose a new path or pass --force: {args.destination}")
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must be different")

    rows = _enrich_generation_records(
        read_jsonl(args.source),
        use_languagetool=args.full_dialect_checks,
    )
    prompts = json.loads(ELABORATION_PROMPTS.read_text(encoding="utf-8"))
    expected_ids = [str(item["id"]) for item in prompts.get("prompts", [])]
    validate_rows(rows, kind="generation", expected_ids=expected_ids, strict=True)
    models = {str(row.get("model")) for row in rows}
    if len(models) != 1:
        raise ArtifactValidationError(f"rescore requires exactly one model, found {sorted(models)}")
    write_jsonl(args.destination, rows)
    manifest = build_manifest(
        args.destination,
        kind="generation",
        benchmark_version=BENCHMARK_VERSION,
        model=next(iter(models)),
        input_hashes={"original_result": sha256_file(args.source)},
        evaluator_versions={
            "compare.py": sha256_file(ROOT / "compare.py"),
            "generation_evaluator.py": sha256_file(ROOT / "core" / "generation_evaluator.py"),
            "pt_dialect.py": sha256_file(ROOT / "core" / "pt_dialect.py"),
            "wf_fidelity.py": sha256_file(ROOT / "core" / "wf_fidelity.py"),
        },
        parameters={
            "rescore": True,
            "source_artifact": str(args.source),
            "full_dialect_checks": args.full_dialect_checks,
        },
        command=sys.argv,
    )
    write_manifest(args.destination, manifest)
    print(json.dumps({"result": str(args.destination), "manifest": str(args.destination.with_suffix('.manifest.json'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())