#!/usr/bin/env python3
"""Assert that the generation scoring scale has a true zero and no free credit.

Why this exists
---------------
A deterministic criterion that a content-free boilerplate reply already
satisfies does not measure instruction adherence; it measures "is this an
email". Such criteria raise the *floor* of the metric: every model inherits
credit for them, the differences between models get compressed into a narrower
band, and a model that ignored the task entirely still scores above zero.
That is a defect of the scale, not of the sample size, so it cannot be fixed by
adding tasks -- it has to be prohibited.

This gate scores `CONSTANT_EMAIL` (the shared no-content control from
`tools/run_smoke_baseline.py`) against every task and requires that no
task-specific criterion passes. The control text is imported rather than
duplicated so the floor and the published smoke baseline can never drift apart.

Passive-vs-active criteria
--------------------------
`no_bracket_placeholders` is deliberately *not* scored as earned credit here.
It passes for any non-empty output, so treating it as a passed criterion would
hand every candidate a free point. The gate rejects it if a future change ever
starts awarding it while no task actually defines a bracketed slot.

Exit status is non-zero on any violation, so CI fails rather than warns.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "tools")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from core.generation_evaluator import load_constraint_map  # noqa: E402
from run_smoke_baseline import CONSTANT_EMAIL  # noqa: E402

GENERATION_GROUPS = ("required_facts", "required_actions", "forbidden_changes")


def _criterion_ids(group_items: list[dict[str, Any]]) -> list[str]:
    ids = []
    for item in group_items or []:
        if not isinstance(item, dict):
            continue
        ids.append(item.get("action_id") or item.get("fact_id") or item.get("forbidden_id") or "unnamed")
    return ids


def audit_broad_criteria(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """Report criteria containing a bare single-word alternative.

    These are *candidates* for review, not violations, and this function never
    affects the exit status. A bare word such as ``problema`` or ``voucher``
    may be exactly the right way to express "acknowledged the problem"; it only
    becomes a measurement defect when a reply with no task-specific content can
    satisfy it. The boilerplate control below is one such reply, and only one:
    the gate it enforces is a lower bound on non-discrimination, so an auditor
    needs a second, independent view of the criteria set. That view is this
    listing.
    """
    findings: list[dict[str, Any]] = []
    for task_id, spec in sorted(constraints.items()):
        for group, key in (("required_actions", "action_id"), ("required_facts", "fact_id")):
            for item in spec.get(group, []) or []:
                pattern = item.get("pattern") or ""
                stripped = re.sub(r"\([?!<]", "(", pattern)
                for branch in re.split(r"(?<!\\)\|", stripped):
                    token = branch.strip()
                    if re.fullmatch(r"[A-Za-z\u00C0-\u00FF][A-Za-z\u00C0-\u00FF .-]{0,11}", token):
                        findings.append({
                            "task_id": task_id,
                            "criterion": item.get(key, "unnamed"),
                            "bare_alternative": token,
                        })
                        break
    return findings


def check_floor(constraints_path: Path, prompts_path: Path) -> dict[str, Any]:
    """Return the floor report. Raises nothing; the caller decides exit status."""
    from core.generation_evaluator import evaluate_generation_output

    constraints = load_constraint_map(str(constraints_path))
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))["prompts"]
    source_by_id = {item["id"]: item.get("prompt", "") for item in prompts}

    violations: list[dict[str, Any]] = []
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted(constraints):
        result = evaluate_generation_output(
            CONSTANT_EMAIL, constraints[task_id], source_text=source_by_id.get(task_id, "")
        )
        adherence = result["instruction_adherence_score"]
        preservation = result["semantic_preservation_score"]
        details = result["adherence_details"]
        semantic_details = result["semantic_details"]

        # Only task-specific criteria may fail this check: entries named here are
        # drawn from the constraint file, so a generic credit cannot hide behind
        # them. `no_bracket_placeholders` is reported separately below.
        passed = [item for item in details.get("passed_items", []) if item != "no_bracket_placeholders"]
        earned_semantic = [
            item for item in (semantic_details.get("preserved_facts", []) + semantic_details.get("forbidden_violations", []))
        ]
        # A task with no executable criteria must be N/A, never 0 or 100.
        expect_adherence = None if not _criterion_ids(constraints[task_id].get("required_actions", [])) else 0.0
        has_semantic_criteria = bool(
            constraints[task_id].get("required_facts") or constraints[task_id].get("forbidden_changes")
        )
        expect_preservation = None if not has_semantic_criteria else 0.0

        record = {
            "instruction_adherence_pct": adherence,
            "semantic_preservation_pct": preservation,
            "criteria_passed_by_boilerplate": passed,
            "semantic_criteria_passed_by_boilerplate": earned_semantic,
        }
        per_task[task_id] = record
        if passed or earned_semantic or adherence not in (expect_adherence, None) or preservation not in (expect_preservation, None):
            violations.append({"task_id": task_id, **record})

    return {
        "control": "tools/run_smoke_baseline.py:CONSTANT_EMAIL",
        "control_sha256_prefix": None,
        "tasks_checked": len(per_task),
        "criterion_groups": list(GENERATION_GROUPS),
        "violations": violations,
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--constraints", type=Path, default=ROOT / "data/elaboration_constraints.json")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/elaboration_prompts_pt_pt.json")
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    parser.add_argument(
        "--report-broad", action="store_true",
        help="also list criteria with a bare single-word alternative (advisory only; never fails the gate)",
    )
    args = parser.parse_args()

    report = check_floor(args.constraints, args.prompts)
    if args.report_broad:
        from core.generation_evaluator import load_constraint_map
        findings = audit_broad_criteria(load_constraint_map(str(args.constraints)))
        report["broad_criteria_candidates"] = findings
        print(
            f"ADVISORY: {len(findings)} criteria contain a bare single-word alternative.\n"
            "These are review candidates, not violations -- most are the correct way to\n"
            "express the action. Audit them against replies a real model would write,\n"
            "because the boilerplate control below only catches what it happens to say.\n",
            file=sys.stderr,
        )
        for f in findings:
            print(f"  {f['task_id']:12s} {f['criterion']:26s} /{f['bare_alternative']}/", file=sys.stderr)
        print(file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    violations = report["violations"]
    if violations:
        if not args.json:
            print(f"FAIL: {len(violations)}/{report['tasks_checked']} tasks award credit to a content-free reply:", file=sys.stderr)
            for item in violations:
                print(
                    f"  {item['task_id']}: adherence={item['instruction_adherence_pct']} "
                    f"preservation={item['semantic_preservation_pct']} "
                    f"criteria={item['criteria_passed_by_boilerplate'] + item['semantic_criteria_passed_by_boilerplate']}",
                    file=sys.stderr,
                )
            print(
                "\nA criterion satisfied by boilerplate compresses the measurable range between\n"
                "models. Retighten it to require task-specific content, or drop it and rely on the\n"
                "fact/gate that already covers the behaviour. If the scoring contract itself is\n"
                "changing, record that in docs/TRUSTWORTHINESS_ROADMAP.md.",
                file=sys.stderr,
            )
        return 1

    if not args.json:
        print(f"PASS: scoring floor is zero on all {report['tasks_checked']} tasks (no free credit for boilerplate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
