"""
Deterministic Generation & Elaboration Evaluator for PT-PT Email LLM Benchmark.
Evaluates model-generated email responses across:
1. Instruction Adherence % (Required tasks, actions, tone, placeholders check)
2. Semantic Preservation % (Fact retention, dates, prices, quantities, forbidden hallucinations)
"""

import re
from typing import Dict, Any, List, Optional, Set, Tuple

# Template slots are inferred structurally without hardcoding word lists:
# 1. Verbatim slots reproduced from the prompt ([...])
# 2. Structural slot syntax: [X...], [___...], [...], [ALL_CAPS], or [Title Case Slot] (e.g. [Seu Nome], [Nome])
_SOURCE_SLOT_RE = re.compile(r"\[[^\]\n]{1,200}\]", re.UNICODE)
_STRUCTURAL_SLOT_RE = re.compile(
    r"\[(?:"
    r"\s*[_Xx.-]{2,}\s*"  # blank runs: [__...], [XX...], [X.......]
    r"|\.{3,}"  # ellipsis-style blank: [...]
    r"|_{2,}"  # underscore blank: [___]
    # Title-Case words, each capital-initial: [Seu Nome], [XXXX-XX-XX], [Empresa]
    r"|(?:[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ0-9_-]*\s*)+"
    # Title-Case slot with lowercase closed-class connectors between the
    # capitalised words: [Nome do Cliente], [Nome do Cliente e do Contato].
    # The connector is bounded structurally (1-3 lowercase letters), not by a
    # word list, so genuine slots like "[Nome do Cliente]" are detected while
    # all-lowercase prose ("[ver anexo]") is not.
    r"|[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ0-9_-]*(?:\s+(?:[a-zà-öø-ÿ]{1,3}\s+)*[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ0-9_-]*)+"
    r")\]",
    re.UNICODE,
)


def _bracketed_placeholders(text: str) -> List[str]:
    found = []
    seen = set()
    for match in _STRUCTURAL_SLOT_RE.finditer(text or ""):
        end = match.end()
        if end < len(text) and text[end] == "(":
            continue
        value = match.group(0)
        if value not in seen:
            found.append(value)
            seen.add(value)
    return found

def _clause_prefix(text: str, start: int) -> str:
    # Evaluate polarity within the local clause. A negation in an earlier
    # comma/semicolon-separated clause must not suppress a later affirmative
    # occurrence of the same required/forbidden pattern.
    sentence_start = max(
        text.rfind('.', 0, start), text.rfind('!', 0, start),
        text.rfind('?', 0, start), text.rfind(';', 0, start),
        text.rfind(',', 0, start), text.rfind(':', 0, start),
    ) + 1
    return re.sub(r"\s+", " ", text[sentence_start:start]).strip()


def _has_double_negation(prefix: str) -> bool:
    """Detect common constructions where a negation does not deny the target action/fact."""
    p = prefix.lower()
    return bool(
        re.search(r"\b(?:não|nao)\b.*\b(?:deixar|deixo|deixa|deixamos|deixam|deixou|deixaram|deixarei|deixará|deixaria|deixaríamos)\s+de\b", p, re.I)
        or re.search(r"\b(?:not|never|cannot|can't|cant|couldn't|couldn['’]t)\b.*\bfail\s+to\b", p, re.I)
        or re.search(r"\b(?:not|never|cannot|can't|cant)\b.*\bavoid\s+\w+ing\b", p, re.I)
        or re.search(r"\b(?:not|never)\b.*\bprevent\b.*\bfrom\b", p, re.I)
    )


def _is_direct_negation(prefix: str) -> bool:
    p = prefix.lower().strip()
    if not p:
        return False
    if _has_double_negation(p):
        return False
    # Explicitly negative directives that precede the matched proposition.
    # These are common ways to deny an action without putting "não" directly
    # in front of the matched words (e.g. "evite confirmar", "sem confirmar",
    # "é proibido confirmar", "do not confirm").
    if re.search(
        r"(?:^|\s)(?:sem|evite|evitar|imped[ae]|proibido|proibida|proibidos|proibidas|não é permitido|nao e permitido|não deve|nao deve|não pode|nao pode|não se deve|nao se deve|não faça|nao faca|do not|don't|dont|avoid|avoiding|forbidden|prohibited)\s*$",
        p, re.I,
    ):
        return True
    # Coordinating/contrastive constructions such as "não só confirmar" and
    # "não apenas confirmar" are affirmative, not negated actions.
    if re.search(r"(?:^|\s)(?:não|nao)\s+(?:só|so|apenas|somente)\b", p, re.I):
        return False
    # Direct action denial: "não confirmar ...", "not confirm ...".
    if re.search(r"(?:^|\s)(?:não|nao|nunca|jamais|not|never)\s+(?:\w|['’])", p, re.I):
        # Avoid treating "não posso deixar de ..." as a direct denial.
        if re.search(r"(?:não|nao)\s+(?:posso|pode|vou|vai|vamos)\b", p, re.I) and re.search(r"\bdeixar\s+de\b", p, re.I):
            return False
        return True
    # Predicate denial: "não é 100 euros", "is not 100 euros".
    if re.search(r"\b(?:é|e|são|sao|foi|foram|será|sera|seria|is|are|was|were|does|do|did|will|would|can|could)\s+(?:not|never)\s*$", p, re.I):
        return True
    if re.search(r"(?:^|\s)(?:não|nao)\s*$", p, re.I):
        return True
    if re.search(r"(?:^|\s)(?:not|never)\s*$", p, re.I):
        return True
    return False


def _is_negated_match(text: str, start: int) -> bool:
    """Return True only when the matched forbidden proposition is explicitly denied."""
    return _is_direct_negation(_clause_prefix(text, start))


def constraint_echo_vocabulary(constraints: Optional[Dict[str, Any]]) -> Set[str]:
    """Alphabetic vocabulary a task's own constraint patterns supply.

    The lexical-contrast dialect scorer must not penalise a model for echoing
    words the benchmark's own task definition used as its accepted answers.
    If a required action literally reads ``reembolso|estorno``, then ``estorno``
    is vocabulary the benchmark handed the model — flagging it as a PT-BR leak
    on the model's side would make the task un-winnable (the model is rewarded
    for matching the pattern and punished for the dialect colour of its words).
    Callers of the dialect evaluator that know the task (runner, rescorer)
    extract this set structurally from every pattern field and pass it as
    ``echo_vocab``; standalone evaluator calls without task context stay strict.
    """
    echo: Set[str] = set()
    if not isinstance(constraints, dict):
        return echo
    for group in ("required_facts", "required_actions", "forbidden_changes", "forbidden_facts"):
        for item in constraints.get(group, []) or []:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern")
            if not isinstance(pattern, str):
                continue
            for match in re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*", pattern):
                word = match.group(0).strip("-'’").lower()
                if len(word) >= 3:
                    echo.add(word)
    return echo


def _is_positive_match_negated(text: str, start: int) -> bool:
    """Return True when a positive fact/action match is explicitly negated."""
    return _is_direct_negation(_clause_prefix(text, start))


def detect_placeholders(generated_text: str, source_text: str = "") -> List[str]:
    """Return generated template slots, excluding Markdown link labels."""
    if not generated_text:
        return []
    placeholders = _bracketed_placeholders(generated_text)
    if source_text:
        for match in _SOURCE_SLOT_RE.finditer(source_text):
            slot = match.group(0)
            if slot in generated_text and slot not in placeholders:
                placeholders.append(slot)
    return placeholders


def evaluate_instruction_adherence(
    generated_text: str,
    constraints: Optional[Dict[str, Any]] = None,
    source_text: str = "",
) -> Dict[str, Any]:
    """
    Evaluates whether the candidate model followed the explicit instructions.
    Checks:
    - Required actions / directives
    - Absence of bracket placeholders
    - Non-empty output
    """
    if not generated_text or not generated_text.strip():
        return {
            "adherence_score": 0.0,
            "passed_count": 0,
            "total_count": 1,
            "passed_items": [],
            "failed_items": ["output_is_empty"],
            "placeholder_count": 0,
            "placeholders": []
        }

    constraints = constraints or {}
    required_actions = constraints.get("required_actions", [])
    
    passed_items = []
    failed_items = []

    # 1. Check anti-placeholder constraint on every generated response.
    placeholders = detect_placeholders(generated_text, source_text)
    source_has_slot = bool(source_text and _SOURCE_SLOT_RE.search(source_text))
    if placeholders:
        failed_items.append(f"bracket_placeholders_detected ({len(placeholders)})")
    elif source_has_slot:
        passed_items.append("no_bracket_placeholders")

    # 2. Check explicit required actions / directives
    for action in required_actions:
        action_id = action.get("action_id", "unnamed_action")
        pattern = action.get("pattern", "")
        if not pattern:
            continue

        matches = list(re.finditer(pattern, generated_text, re.IGNORECASE | re.UNICODE))
        # A positive requirement is satisfied when at least one occurrence is
        # affirmative. Inspect every occurrence so an earlier negated mention
        # cannot hide a later affirmative one.
        if any(not _is_positive_match_negated(generated_text, m.start()) for m in matches):
            passed_items.append(action_id)
        else:
            failed_items.append(action_id)

    total_checks = len(passed_items) + len(failed_items)
    score = (len(passed_items) / total_checks * 100.0) if total_checks > 0 else None

    return {
        "adherence_score": round(score, 1) if score is not None else None,
        "passed_count": len(passed_items),
        "total_count": total_checks,
        "passed_items": passed_items,
        "failed_items": failed_items,
        "placeholder_count": len(placeholders),
        "placeholders": placeholders
    }


def evaluate_semantic_preservation(
    generated_text: str,
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates whether the generated response preserved factual accuracy without
    hallucinating contradictory facts or violating explicit constraints.
    Checks:
    - Required facts (order numbers, dates, prices, quantities, names)
    - Forbidden changes / contradictions (wrong amounts, reversed terms, invented refunds)
    """
    if not generated_text or not generated_text.strip():
        return {
            "preservation_score": 0.0,
            "passed_count": 0,
            "total_count": 1,
            "preserved_facts": [],
            "missing_facts": ["output_is_empty"],
            "forbidden_violations": []
        }

    constraints = constraints or {}
    required_facts = constraints.get("required_facts", [])
    forbidden_changes = constraints.get("forbidden_changes", [])

    preserved_facts = []
    missing_facts = []
    forbidden_violations = []

    # 1. Required Facts check
    for fact in required_facts:
        fact_id = fact.get("fact_id", "unnamed_fact")
        pattern = fact.get("pattern", "")
        if not pattern:
            continue

        matches = list(re.finditer(pattern, generated_text, re.IGNORECASE | re.UNICODE))
        # A required fact is preserved when any occurrence affirms it. This
        # prevents a negated first mention from masking a later true mention.
        if any(not _is_positive_match_negated(generated_text, m.start()) for m in matches):
            preserved_facts.append(fact_id)
        else:
            missing_facts.append(fact_id)

    # 2. Forbidden Changes check
    for forbidden in forbidden_changes:
        forbidden_id = forbidden.get("forbidden_id", "unnamed_forbidden")
        pattern = forbidden.get("pattern", "")
        if not pattern:
            continue

        matches = list(re.finditer(pattern, generated_text, re.IGNORECASE | re.UNICODE))
        # Any affirmative occurrence is a violation. A negated first mention
        # must not suppress a later genuine forbidden change.
        for match in matches:
            if not _is_negated_match(generated_text, match.start()):
                forbidden_violations.append({
                    "forbidden_id": forbidden_id,
                    "matched_snippet": match.group(0)
                })
                break

    total_fact_checks = len(required_facts)
    total_forbidden_checks = len(forbidden_changes)
    total_criteria = total_fact_checks + total_forbidden_checks

    if total_criteria == 0:
        # No measurable semantic criteria is not the same thing as perfect
        # preservation. Keep this explicitly N/A so aggregates can ignore it.
        return {
            "preservation_score": None,
            "passed_count": 0,
            "total_count": 0,
            "preserved_facts": [],
            "missing_facts": [],
            "forbidden_violations": []
        }

    passed_count = len(preserved_facts) + (total_forbidden_checks - len(forbidden_violations))
    score = (passed_count / total_criteria * 100.0)

    return {
        "preservation_score": round(max(0.0, min(100.0, score)), 1),
        "passed_count": max(0, passed_count),
        "total_count": total_criteria,
        "preserved_facts": preserved_facts,
        "missing_facts": missing_facts,
        "forbidden_violations": forbidden_violations
    }


def evaluate_generation_output(
    generated_text: str,
    constraints: Optional[Dict[str, Any]] = None,
    source_text: str = "",
) -> Dict[str, Any]:
    """
    Combined generation evaluator returning both Instruction Adherence and Semantic Preservation.
    """
    adh = evaluate_instruction_adherence(generated_text, constraints, source_text)
    sem = evaluate_semantic_preservation(generated_text, constraints)

    return {
        "instruction_adherence_score": adh["adherence_score"],
        "semantic_preservation_score": sem["preservation_score"],
        "adherence_details": adh,
        "semantic_details": sem
    }


def _validate_constraint_patterns(data: Dict[str, Any]) -> None:
    """Validate every executable regex before benchmark scoring starts."""
    groups = ("required_actions", "required_facts", "forbidden_changes")
    for prompt_id, constraints in data.items():
        if not isinstance(constraints, dict):
            raise ValueError(f"Constraints for {prompt_id!r} must be an object")
        for group in groups:
            items = constraints.get(group, []) or []
            if not isinstance(items, list):
                raise ValueError(f"{prompt_id!r}.{group} must be a list")
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(f"{prompt_id!r}.{group}[{index}] must be an object")
                pattern = item.get("pattern", "")
                if not pattern:
                    continue
                try:
                    re.compile(pattern, re.IGNORECASE | re.UNICODE)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid regex in {prompt_id!r}.{group}[{index}]: {exc}"
                    ) from exc


def load_constraint_map(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load and validate the benchmark's structured elaboration constraints."""
    from pathlib import Path
    import json
    base = Path(__file__).parent.parent
    constraint_path = Path(path) if path else base / "data" / "elaboration_constraints.json"
    if not constraint_path.exists():
        return {}
    try:
        data = json.loads(constraint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid constraint-map file {constraint_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Constraint map must be a JSON object")
    _validate_constraint_patterns(data)
    return data
