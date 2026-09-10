#!/usr/bin/env python3
"""Score the dialect detector against the benchmark's own gold labels.

Why this exists
---------------
`data/analysis_reference.jsonl` already carries a hand-adjudicated answer key
(`reference_judgments.gold_ptbr_leakage`, `ptbr_leakage_markers`,
`gold_dialect`, `expected_euptvid_label`). Until now those fields were used for
nothing but row-count checks, so every claim about detector quality was an
argument instead of a measurement. This tool closes that loop: it runs the
current detector over the labelled rows and reports agreement.

What it deliberately is not
---------------------------
It adds no vocabulary, no keyword lists, and no thresholds of its own. The one
classifier threshold it applies (0.5 on the PT-PT probability) is stated
explicitly as `classifier_decision_threshold` in the report so it can be argued
about rather than buried.

It changes no score and nothing on the scoring path. It measures.

Overfitting warning
-------------------
The report includes `remediation_risk`. If a future change "fixes" a miss by
adding that miss's own gold markers to a list, this file becomes the test set
the detector was tuned on and stops being evidence. The mitigation is recorded
in the report and in docs/VALIDATION.md: grow a separate held-out set, and treat
these rows as development evidence.
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

from config import ANALYSIS_REFERENCE, BENCHMARK_VERSION  # noqa: E402
from core._util import sha256_file  # noqa: E402
from core.pt_dialect import (  # noqa: E402
    evaluate_pt_dialect,
    get_ptbr_dictionary,
    get_ptpt_dictionary,
)

BASELINE_VERSION = "gold-agreement-v1"
# Stated explicitly rather than hidden in a comparison: the classifier is
# converted to a leak decision at this probability of "not PT-PT".
CLASSIFIER_THRESHOLD = 0.5
DETECTOR_SOURCES = (
    "core/pt_dialect.py",
    "core/wf_fidelity.py",
    "data/analysis_reference.jsonl",
)

# The dictionaries decide every marker classification in this report but are not
# tracked by git, so their digests are recorded and compared explicitly: a
# Hunspell update that flips a both-dictionaries verdict must force a reviewed
# regeneration rather than slip through as "same source, same numbers".
MANAGED_RESOURCES = (
    "docs/pt_PT.dic",
    "docs/pt_BR.dic",
)


def _visible_to_difference(marker: str, ptpt: Any, ptbr: Any) -> dict[str, bool]:
    """Classify one gold marker against a set-difference detector.

    A word-list-free diagnostic: it asks whether the *mechanism* can represent
    the marker at all, using the dictionaries the detector already loads. It
    never asserts that the marker is or is not Brazilian.
    """
    out = {"all_tokens_in_both": True, "any_token_absent_from_pt_br": False}
    tokens = [t for t in marker.lower().replace("-", " ").split() if t]
    if not tokens:
        return {**out, "all_tokens_in_both": False}
    for token in tokens:
        in_ptpt = bool(ptpt.lookup(token))
        in_ptbr = bool(ptbr.lookup(token))
        if not (in_ptbr and not in_ptpt):
            out["all_tokens_in_both"] = out["all_tokens_in_both"] and (in_ptpt and in_ptbr)
        if not in_ptbr:
            out["any_token_absent_from_pt_br"] = True
    # A difference detector can only fire on a token that pt_BR has and pt_PT lacks.
    return out


def _euptvid_available() -> bool:
    """Whether the fastText classifier can actually load, not whether a file exists.

    Probing the loader rather than the path matters: a present-but-unloadable
    model (missing `fasttext`, wrong build) degrades the detector silently.
    """
    try:
        from core.pt_dialect import get_euptvid_model
        return get_euptvid_model() is not None
    except Exception:
        return False


def _spacy_available() -> bool:
    try:
        from core.pt_dialect import get_spacy_nlp
        return get_spacy_nlp() is not None
    except Exception:
        return False


def build_report() -> dict[str, Any]:
    rows = [json.loads(line) for line in ANALYSIS_REFERENCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    ptpt = get_ptpt_dictionary()
    ptbr = get_ptbr_dictionary()
    dictionaries_available = ptpt is not None and ptbr is not None
    engines = {
        "hunspell_dictionaries": bool(dictionaries_available),
        "euptvid_model": _euptvid_available(),
        "spacy_pipeline": _spacy_available(),
        # Recorded, not required: `runner.py` scores submissions with
        # use_languagetool=True while this gate measures with it False. Verified
        # equivalent while LT cannot start (Java < 17 degrades to the same
        # findings), but the baseline must not claim a configuration it did not run.
        "languagetool": False,
    }
    # The dictionaries are the mechanism under test: without them the lexical
    # contrast path never runs and the rule degrades to its regex checks, which is
    # a DIFFERENT detector. Measured, not assumed: with them absent this corpus
    # scores 66.67/100/85 and with them present 77.78/58.33/65. So absence is a
    # hard error, not a soft caveat -- and `--check-baseline` compares the full
    # engine state, so a baseline recorded on the degraded path can never pass in
    # a complete environment or vice versa.
    if not engines["hunspell_dictionaries"]:
        raise SystemExit(
            "gold_agreement: the PT-PT/PT-BR dictionaries are unavailable, so the lexical-contrast "
            "path did not run and the result would describe a different detector than the one being "
            "graded. `python runner.py setup` fetches the files; `spylls` (in requirements.lock) loads "
            "them. Install the hash-locked environment, then re-run."
        )

    per_row: list[dict[str, Any]] = []
    tp = fp = fn = tn = 0
    # Four counts over NON-leaking rows only. The gap between "fired" and
    # "fired and scored" is how much of the headline precision comes from the
    # suppression heuristic rather than from the detector being right.
    nonleak_rows = 0
    nonleak_rows_fired = 0
    nonleak_rows_scored = 0
    suppression_only_findings: list[str] = []
    euptvid_tp = euptvid_fp = euptvid_fn = euptvid_tn = 0
    missed: list[dict[str, Any]] = []

    for row in rows:
        gold = row.get("reference_judgments") or {}
        gold_leak = bool(gold.get("gold_ptbr_leakage"))
        result = evaluate_pt_dialect(row["email"], use_languagetool=False)
        detected = bool(result.get("ptbr_leakage_detected"))

        if gold_leak and detected:
            tp += 1
        elif gold_leak and not detected:
            fn += 1
        elif detected and not gold_leak:
            fp += 1
        else:
            tn += 1

        prob = result.get("euptvid_prob")
        if prob is None:
            classifier_pred = None
        else:
            classifier_pred = (1.0 - float(prob)) >= CLASSIFIER_THRESHOLD
        if classifier_pred is not None:
            if gold_leak and classifier_pred:
                euptvid_tp += 1
            elif gold_leak:
                euptvid_fn += 1
            elif classifier_pred:
                euptvid_fp += 1
            else:
                euptvid_tn += 1

        violations = result.get("violations") or []
        scored = [v for v in violations if bool(v.get("score_eligible", v.get("validated")))]
        if not gold_leak:
            nonleak_rows += 1
            if violations:
                nonleak_rows_fired += 1
            if scored:
                nonleak_rows_scored += 1
            else:
                suppression_only_findings.extend(str(v.get("context") or "")[:40] for v in violations)

        entry = {
            "id": row["id"],
            "gold_dialect": gold.get("gold_dialect"),
            "gold_ptbr_leakage": gold_leak,
            "detector_leakage_detected": detected,
            "agrees": detected == gold_leak,
            "ptpt_compliance_pct": result.get("ptpt_compliance_pct"),
            "ptpt_compliance_graded_pct": result.get("ptpt_compliance_graded_pct"),
            "euptvid_prob_ptpt": prob,
            "expected_euptvid_label": gold.get("expected_euptvid_label"),
            "euptvid_label": result.get("euptvid_label"),
        }

        if gold_leak and not detected:
            diagnostics = []
            for marker in gold.get("ptbr_leakage_markers") or []:
                cls = _visible_to_difference(marker, ptpt, ptbr) if dictionaries_available else {}
                diagnostics.append({
                    "marker": marker,
                    "representable_by_set_difference": bool(cls.get("all_tokens_in_both")) is False
                    and not cls.get("any_token_absent_from_pt_br", False)
                    if dictionaries_available else None,
                    "every_token_in_both_dictionaries": bool(cls.get("all_tokens_in_both", False))
                    if dictionaries_available else None,
                })
            invisible = sum(1 for d in diagnostics if d.get("every_token_in_both_dictionaries"))
            missed.append({
                "id": row["id"],
                "gold_markers": list(gold.get("ptbr_leakage_markers") or []),
                "marker_diagnostics": diagnostics,
                "markers_invisible_to_mechanism": invisible,
                "markers_total": len(diagnostics),
            })
        per_row.append(entry)

    n_positive = tp + fn
    n_predicted = tp + fp
    invisible_rows = sum(1 for m in missed if m["markers_total"] and m["markers_invisible_to_mechanism"] == m["markers_total"])
    gold_negative_rows_with_both_dict_markers = 0
    if dictionaries_available:
        for row in rows:
            gold = row.get("reference_judgments") or {}
            if gold.get("gold_ptbr_leakage"):
                continue
            for marker in gold.get("ptbr_leakage_markers") or []:
                if _visible_to_difference(marker, ptpt, ptbr).get("all_tokens_in_both"):
                    gold_negative_rows_with_both_dict_markers += 1
                    break

    return {
        "baseline_id": BASELINE_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "measurement_only": True,
        "changes_any_score": False,
        "classifier_decision_threshold": CLASSIFIER_THRESHOLD,
        "resource_note": "managed dictionaries loaded (gitignored; see runner.py setup)",
        "counts": {
            "rows": len(rows),
            "rows_with_gold": len(rows),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "nonleaking_rows": nonleak_rows,
        },
        "detector": {
            "leakage_recall_pct": round(100.0 * tp / n_positive, 2) if n_positive else None,
            "leakage_precision_pct": round(100.0 * tp / n_predicted, 2) if n_predicted else None,
            "agreement_pct": round(100.0 * sum(1 for e in per_row if e["agrees"]) / len(per_row), 2),
        },
        "classifier_reference": {
            "note": "EUPTVID is reported as a contrast, NOT as a drop-in replacement; see docs/TRUSTWORTHINESS_ROADMAP.md 'Never use EUPTVID probability as a substitute for compliance or WF'.",
            "leakage_recall_pct": round(100.0 * euptvid_tp / n_positive, 2) if n_positive else None,
            "leakage_precision_pct": round(100.0 * euptvid_tp / (euptvid_tp + euptvid_fp), 2) if (euptvid_tp + euptvid_fp) else None,
            "true_positive": euptvid_tp,
            "false_negative": euptvid_fn,
            "false_positive": euptvid_fp,
            "true_negative": euptvid_tn,
        },
        "suppression": {
            "discarded_finding_examples": sorted(set(suppression_only_findings))[:15],
            "discarded_finding_count": len(suppression_only_findings),
            "note": (
                "findings the rule produced on non-leaking text and then dropped because score_eligible was "
                "false. Inspect these before concluding the rule is conservative: they are what it "
                "actually matches, kept out of the score by one capitalization heuristic."
            ),
        },
        "misses": missed,
        "blindspot": {
            "missed_rows_wholly_invisible_to_mechanism": invisible_rows,
            "note": (
                "'invisible to mechanism' means every token of the gold marker exists in BOTH managed "
                "dictionaries, so a detector built on their difference cannot represent it at all. "
                "This is a property of the method, not a coverage gap that more words can close."
            ),
        },
        "remediation_risk": {
            "gold_rows_are_also_the_diagnostic_set": True,
            "note": (
                "Do not close a miss by adding that miss's own gold markers to a list: the detector would "
                "then be tuned on the rows used to grade it. Grow a held-out set instead "
                "(docs/TRUSTWORTHINESS_ROADMAP.md: 'Keep a frozen test set separate from development and "
                "calibration data')."
            ),
            "nonleaking_rows": nonleak_rows,
            "nonleaking_rows_where_rule_fired": nonleak_rows_fired,
            "nonleaking_rows_where_rule_scored": nonleak_rows_scored,
            "precision_without_suppression_pct": round(
                100.0 * tp / (tp + nonleak_rows_fired), 2
            ) if (tp + nonleak_rows_fired) else None,
            "adversarial_european_rows_counted": gold_negative_rows_with_both_dict_markers,
            "precision_warning": (
                "headline precision is carried by the suppression heuristic, not by the detector: the "
                f"rule fired on {nonleak_rows_fired}/{nonleak_rows} non-leaking rows and only the "
                "capital-first-letter rule kept them from becoming false alarms. Precision without that "
                "one rule is reported alongside so the dependency is visible."
            ),
        },
        "source_sha256": {p: sha256_file(ROOT / p) for p in DETECTOR_SOURCES},
        "managed_resource_sha256": {p: sha256_file(ROOT / p) for p in MANAGED_RESOURCES
                                      if (ROOT / p).is_file()},
        "detector_engines": engines,
        "per_row": per_row,
    }


def _comparable(report: dict[str, Any]) -> dict[str, Any]:
    """The part of a report that must not silently regress."""
    return {
        "baseline_id": report["baseline_id"],
        "benchmark_version": report["benchmark_version"],
        "counts": report["counts"],
        "detector": report["detector"],
        "classifier_reference": {k: v for k, v in report["classifier_reference"].items() if k != "note"},
        "blindspot": {k: v for k, v in report["blindspot"].items() if k != "note"},
        "remediation_risk": {k: v for k, v in report["remediation_risk"].items() if k not in {"note", "precision_warning"}},
        "source_sha256": report["source_sha256"],
        "managed_resource_sha256": report["managed_resource_sha256"],
        "detector_engines": report["detector_engines"],
    }


def _summarise(report: dict[str, Any]) -> str:
    d = report["detector"]
    c = report["counts"]
    lines = [
        f"gold-labelled rows: {c['rows']}   detector vs gold",
        f"  leakage recall    {d['leakage_recall_pct']}%  "
        f"({c['true_positive']}/{c['true_positive'] + c['false_negative']} gold-positive rows caught)",
        f"  leakage precision {d['leakage_precision_pct']}%  "
        f"({c['false_positive']} false alarm(s) over {c['false_positive'] + c['true_negative']} non-leaking rows)",
        f"  overall agreement {d['agreement_pct']}%",
    ]
    for m in report["misses"]:
        lines.append(f"  MISSED {m['id']}: {m['markers_invisible_to_mechanism']}/{m['markers_total']} "
                     f"marker(s) invisible to the mechanism -> {m['gold_markers']}")
    b = report["blindspot"]
    lines.append(
        f"blindspot: {b['missed_rows_wholly_invisible_to_mechanism']} missed row(s) are wholly unrepresentable "
        f"by a dictionary difference"
    )
    r = report["remediation_risk"]
    if r["nonleaking_rows_where_rule_fired"] and not r["nonleaking_rows_where_rule_scored"]:
        lines.append(
            f"precision provenance: the rule fired on {r['nonleaking_rows_where_rule_fired']}/"
            f"{r['nonleaking_rows']} non-leaking rows; all {report['suppression']['discarded_finding_count']} "
            f"findings were discarded by the capitalization rule. Precision without it: "
            f"{r['precision_without_suppression_pct']}% (vs {d['leakage_precision_pct']}% reported)"
        )
    if not report["detector_engines"]["spacy_pipeline"]:
        lines.append(
            "CAUTION: pinned spaCy model absent, so non-lexical masking is regex-only and header "
            "address local-parts can be read as dialect evidence. Numbers here are NOT comparable "
            "with a baseline recorded with the model installed."
        )
    if r["adversarial_european_rows_counted"] == 0:
        lines.append("WARNING: 0 non-leaking rows carry a both-dictionaries marker, so this set cannot "
                     "measure whether the rule misses real European false alarms at all")
    lines.append("NOTE: measurement only; no score on the ranking path was changed.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the report JSON to this path")
    parser.add_argument("--check-baseline", type=Path, help="compare against a committed report and fail on regression")
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args()
    # Resolve before use: a relative --output would otherwise be written against
    # the caller's cwd while being displayed relative to ROOT.
    if args.output is not None:
        args.output = Path(args.output).resolve()
    if args.check_baseline is not None:
        args.check_baseline = Path(args.check_baseline).resolve()

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(_summarise(report))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output.relative_to(ROOT) if str(args.output).startswith(str(ROOT)) else args.output}")

    if args.check_baseline:
        if not args.check_baseline.is_file():
            print(f"\nFAIL: no committed baseline at {args.check_baseline}. Generate one with --output.", file=sys.stderr)
            return 1
        committed = json.loads(args.check_baseline.read_text(encoding="utf-8"))
        mine, theirs = _comparable(report), _comparable(committed)
        if mine == theirs:
            print("\nPASS: detector agrees with gold exactly as recorded in the committed baseline.")
            return 0
        # A legitimate detector improvement changes source hashes and therefore this comparison.
        # Fail loudly, so the improvement is a reviewed regeneration rather than a silent drift.
        print("\nFAIL: detector gold-agreement differs from the committed baseline.", file=sys.stderr)
        for key in sorted(set(mine["detector"]) | set(theirs["detector"])):
            if mine["detector"].get(key) != theirs["detector"].get(key):
                print(f"  detector.{key}: baseline {theirs['detector'].get(key)} -> now {mine['detector'].get(key)}", file=sys.stderr)
        if mine["source_sha256"] != theirs["source_sha256"]:
            print("  (detector sources also changed: re-run --output, review the diff, commit it)", file=sys.stderr)
        elif mine["detector"]["leakage_recall_pct"] < theirs["detector"]["leakage_recall_pct"]:
            print("  recall regressed with unchanged sources -- investigate before committing", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
