"""
Deterministic Word Fidelity (WF) Evaluator for European Portuguese (PT-PT).
Implements tier-weighted dialect fidelity calculation:
- Tier 1 (weight 1.0): Brazilian gerund constructions (e.g. 'estou fazendo')
- Tier 2 (weight 1.75): Pronoun placement & clitic structure (e.g. 'me diga')
- Tier 3 (weight 2.5): Lexical mismatches & subtle PT-BR regionalisms
- Tier 4 (weight 3.0): Severe PT-BR vocabulary leaks identified by the existing dialect evaluator

Formula:
  weighted_leaks = sum(violation.weight for violation in violations)
  wf_score = max(0.0, min(100.0, 100.0 - (weighted_leaks / total_words) * 100.0))

WF is intentionally independent of EUPTVID probability. EUPTVID remains a
separate dialect-classification signal, while WF is derived from the
structured PT-BR fidelity violations supplied by the dialect evaluator.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from core.pt_dialect import evaluate_pt_dialect, is_predominantly_english

TIER_WEIGHTS = {
    1: 1.0,   # Gerund constructions
    2: 1.75,  # Proclisis / syntax
    3: 2.5,   # Dialect regionalisms
    4: 3.0,   # Hard lexical markers
}

WORD_SPLIT_REGEX = re.compile(r'\b[\wÀ-ÿ\-]+\b', re.UNICODE)


def assign_tier_for_violation(v: Dict[str, Any]) -> Tuple[int, float]:
    """Assigns fidelity tier (1 to 4) and corresponding penalty weight to a violation."""
    rule_id = str(v.get("rule_id", "")).upper()
    severity = str(v.get("severity", "medium")).lower()

    if "GERUND" in rule_id:
        tier = 1
    elif "PROCLISIS" in rule_id or "CLITIC" in rule_id or "TER_HAVER" in rule_id:
        tier = 2
    elif severity == "high" or "VOCAB" in rule_id or "LEAK" in rule_id or "REPLACEMENT" in rule_id:
        tier = 4
    else:
        tier = 3

    return tier, TIER_WEIGHTS[tier]



def compute_word_fidelity_from_dialect(
    text: str,
    dialect_res: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute WF using an already-computed PT dialect result.

    This avoids a second EUPTVID/LanguageTool pass when the batch runner has
    already evaluated the same text for PT-PT compliance.
    """
    if not text or not text.strip():
        return {
            "wf_score": None, "total_words": 0, "violation_count": 0,
            "weighted_leaks": 0.0, "violations_by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
            "violations": []
        }

    if dialect_res.get("pt_dialect_score") is None:
        return {
            "wf_score": None,
            "total_words": len(WORD_SPLIT_REGEX.findall(text)),
            "violation_count": 0,
            "weighted_leaks": 0.0,
            "violations_by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
            "violations": []
        }

    words = WORD_SPLIT_REGEX.findall(text)
    total_words = max(1, len(words))
    raw_violations = [
        v for v in (dialect_res.get("violations", []) or [])
        if bool(v.get("score_eligible", v.get("validated"))) and str(v.get("rule_id", "")).upper().startswith("PTBR_")
    ]
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    tier_violations = []
    total_weighted_penalty = 0.0

    for v in raw_violations:
        tier, weight = assign_tier_for_violation(v)
        tier_counts[tier] += 1
        total_weighted_penalty += weight
        tier_violations.append({
            "rule_id": v.get("rule_id"),
            "source": v.get("source"),
            "context": v.get("context"),
            "tier": tier,
            "weight": weight,
            "message": v.get("message"),
            "replacements": v.get("replacements", []),
        })

    # Dialect fidelity sensitivity:
    # Scale proportionally so that genuine dialect leaks noticeably register
    # rather than remaining flat at 99.7% on standard emails.
    # An email with 0 leaks achieves 100.0%.
    # 1 minor clitic placement on 150 words (weight 1.75) lowers the score to ~97.7%.
    # Frequent leaks (e.g. Llama with multiple gerunds and proclisis) scale down appropriately.
    scaled_leak_ratio = (total_weighted_penalty / total_words) * 200.0
    wf_score = max(0.0, min(100.0, 100.0 - scaled_leak_ratio))
    return {
        "wf_score": round(wf_score, 1),
        "total_words": total_words,
        "violation_count": len(tier_violations),
        "weighted_leaks": round(total_weighted_penalty, 2),
        "violations_by_tier": tier_counts,
        "violations": tier_violations,
    }

def compute_word_fidelity(
    text: str,
    use_languagetool: bool = True
) -> Dict[str, Any]:
    """Compute WF through the single canonical dialect-derived implementation.

    EUPTVID is intentionally never used as the numerical base for WF.
    """
    if not text or not text.strip():
        return {
            "wf_score": None,
            "total_words": 0,
            "violation_count": 0,
            "weighted_leaks": 0.0,
            "violations_by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
            "violations": []
        }

    if is_predominantly_english(text):
        return {
            "wf_score": 0.0,
            "total_words": len(WORD_SPLIT_REGEX.findall(text)),
            "violation_count": 0,
            "weighted_leaks": 0.0,
            "violations_by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
            "violations": [{
                "rule_id": "LANGUAGE_MISMATCH_ENGLISH",
                "tier": 4,
                "weight": 0.0,
                "message": "Text is in English instead of European Portuguese."
            }]
        }

    dialect_res = evaluate_pt_dialect(text, use_languagetool=use_languagetool)
    return compute_word_fidelity_from_dialect(text, dialect_res)

