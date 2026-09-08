"""
Local Deterministic Writing Quality (WQ) Engine — Task 6.

Provides a five-dimension linguistic quality assessment for model-generated
email text without using any external LLM judge or hardcoded word lists.

Dimensions:
  1. Grammar       — LanguageTool grammar rules (dialect-excluded)
  2. Spelling      — LanguageTool orthography rules (dialect-excluded)
  3. Repetition    — Consecutive-token repetition, repeated n-gram phrases,
                     duplicate sentences; all fully structural/statistical
  4. Structural    — Email structural completeness: greeting, non-empty body,
                     sign-off, sentence fragmentation, degenerate paragraphs;
                     all structural regex/heuristic, no word lists
  5. Style/Register— LanguageTool style/typography issues + local structural
                     checks (all-caps overuse, consecutive exclamations, run-ons)

Design constraints (from benchmark specification §16-17 and Task 7):
  - No local LLM judge, no external LLM judge.
  - No hardcoded word lists or lexicons.
  - Individual dimension counts reported separately from derived score.
  - Dialect/regionalism issues are explicitly EXCLUDED here; they are scored
    exclusively by pt_dialect.py (no double-penalisation).
  - Fixture RFC headers are stripped only when the caller marks the text as a
    calibration fixture; candidate model output is never globally stripped.
  - LanguageTool uses the sample's reference variety, not a universal pt-PT profile.
"""

# Bumped when WQ detection or aggregation changes in a way that can alter scores.
WQ_EVALUATOR_VERSION = "3.5.0"

# A defect-free deterministic evaluator cannot prove that prose is semantically
# excellent.  The score therefore reserves a small, explicit headroom rather
# than fabricating a positive-quality bonus from arbitrary text shape.
WQ_CLEAN_SCORE_CEILING = 99.9
WQ_EXACT_SCORE_CEILING = WQ_CLEAN_SCORE_CEILING

import ctypes
import ctypes.util
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
from typing import Any, Dict, List, Optional, Tuple

from core.languagetool_local import check_local_languagetool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LanguageTool issue type names that represent genuine writing quality issues.
# Anything not in this set (and not an empty typeName) is skipped.
_QUALITY_ISSUE_TYPES = frozenset({
    "grammar", "spelling", "style", "typographical",
    "punctuation", "misspelling", "uncategorized", "other",
})

# Per-dimension, per-severity penalty deductions used in _compute_score().
# Provisional until Task 7 validation against WQ-usable human labels is complete.
# One LanguageTool finding is a signal, not one unit of quality loss (§7).
_PENALTIES: Dict[Tuple[str, str], float] = {
    ("grammar",    "high"):    8.0,
    ("spelling",   "high"):    6.0,
    ("repetition", "high"):   14.0,
    ("repetition", "medium"):  8.0,
    ("structural", "high"):   12.0,
    ("structural", "medium"):  6.0,
    ("style",      "high"):    6.0,
    ("style",      "medium"):  2.0,
}
_DEFAULT_PENALTY = 3.0

# Task 7 length-neutral calibrated aggregation policy.
#
# Direct body-size/target-shape features are deliberately excluded from the
# headline score. The 3.3.x calibrator used body_token_count and length_quality
# together, which created an avoidable preference for longer emails. Those fields
# remain available in the quality profile as diagnostics only.
#
# `sentence_length_sd` is retained as a structural/rhythm feature because it
# captures sentence-shape quality rather than absolute email size. It is covered
# by explicit matched-verbosity regression tests in Task 7.
#
# This is a frozen one-time ordinal calibrator. It does not fit or train at
# runtime. The small tree ensemble is stored as data/wq_length_neutral_calibration.json
# and evaluated locally/deterministically.
WQ_CALIBRATION_VERSION = "task7-length-neutral-v1.1-school-scale"
_CALIBRATION_MODEL_PATH = BASE_DIR / "data" / "wq_length_neutral_calibration.json"
_CALIBRATION_FEATURES: Tuple[str, ...] = (
    "sentence_length_sd",
    "lexical_diversity",
    "register_quality",
    "rhythm_quality",
    "wq_defect_only_score_normalized",
)
_CALIBRATION_MODEL: Optional[Dict[str, Any]] = None
_CALIBRATION_TREES: List[List[Dict[str, Any]]] = []
_CALIBRATION_INIT = 0.0
_CALIBRATION_LEARNING_RATE = 0.0
_CALIBRATION_SCORE_BASE = 0.0
_CALIBRATION_SCORE_PER_LEVEL = 0.0

class WQCalibrationUnavailable(RuntimeError):
    """Raised when the frozen WQ calibration artifact cannot be loaded.

    FIX (P2.5): the loader previously let a bare FileNotFoundError escape, so a
    missing `wq_length_neutral_calibration.json` aborted the entire WQ pipeline
    with an opaque traceback.

    The calibration model is deliberately frozen and is the only thing that maps
    defect burden onto the published quality band, so it cannot be substituted
    with an uncalibrated stand-in: DESIGN.md requires "honest None/unavailable
    states instead of silently converting missing evidence to a score", and the
    roadmap's WQ section requires that a model must not look better merely
    because a detector class went missing. Callers therefore degrade to the
    existing `empty_wq_result(excluded=...)` contract, exactly as they already
    do when Hunspell dictionaries are absent.
    """


def _load_calibration_model() -> None:
    global _CALIBRATION_MODEL, _CALIBRATION_TREES, _CALIBRATION_INIT
    global _CALIBRATION_LEARNING_RATE, _CALIBRATION_SCORE_BASE
    global _CALIBRATION_SCORE_PER_LEVEL
    if _CALIBRATION_MODEL is not None:
        return
    try:
        with _CALIBRATION_MODEL_PATH.open("r", encoding="utf-8") as _fh:
            model = json.load(_fh)
    except FileNotFoundError as exc:
        raise WQCalibrationUnavailable(
            f"frozen WQ calibration artifact not found at {_CALIBRATION_MODEL_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise WQCalibrationUnavailable(
            f"frozen WQ calibration artifact at {_CALIBRATION_MODEL_PATH} is not valid JSON: {exc}"
        ) from exc
    if tuple(model.get("features", ())) != _CALIBRATION_FEATURES:
        raise WQCalibrationUnavailable("Frozen WQ calibration feature schema mismatch")
    for required_key in ("trees", "init_value", "learning_rate", "score_mapping"):
        if required_key not in model:
            raise WQCalibrationUnavailable(
                f"frozen WQ calibration artifact is missing required key {required_key!r}"
            )
    _CALIBRATION_MODEL = model
    _CALIBRATION_TREES = model["trees"]
    _CALIBRATION_INIT = float(model["init_value"])
    _CALIBRATION_LEARNING_RATE = float(model["learning_rate"])
    mapping = model["score_mapping"]
    _CALIBRATION_SCORE_BASE = float(mapping["base"])
    _CALIBRATION_SCORE_PER_LEVEL = float(mapping["per_level"])


_RULE_PENALTY_MULTIPLIERS: Dict[str, float] = {
    "MISSING_GREETING": 0.40,
    "MISSING_SIGNOFF": 0.35,
    # Repeated phrases are a genuine repetition defect, but the detector is
    # intentionally discounted versus hard grammatical/structural failures.
    "REPEATED_NGRAM_PHRASE": 0.50,
}

# Cap contribution per dimension so a swarm of low-severity LT nits cannot
# floor an otherwise coherent email.
_DIM_CAPS: Dict[str, float] = {
    "grammar": 40.0,
    "spelling": 30.0,
    "repetition": 40.0,
    "structural": 40.0,
    "style": 18.0,
}

# RFC 5322 structural header line (e.g. "Header-Field: value")
_FIXTURE_HEADER_LINE_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9-]{1,50}\s*:",
    re.IGNORECASE,
)

# Candidate emails may render headers in Markdown (e.g. **Subject:** ...).
_EMAIL_HEADER_LINE_RE = re.compile(
    r"^\s*(?:[*_~`]{1,3})?[A-Za-z][A-Za-z0-9-]{1,50}\s*:(?:[*_~`]{1,3})?\s*.*$",
    re.IGNORECASE,
)
_EMAIL_ADDRESS_RE = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_URL_RE = re.compile(r"(?i)\b(?:https?|ftp)://[^\s<>()]+|\b(?:www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s<>()]*)?")
_NUMERIC_ID_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .:/()\-]{2,}\d|[A-Z]{1,8}[-/]?\d{2,}[A-Z0-9/-]*|\d{4}[-/]\d{1,2}[-/]\d{1,4})(?!\w)")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _is_dialect_issue(msg: str, rule_id: str, rule_desc: str, category_id: str = "") -> bool:
    """
    Returns True if the LanguageTool match represents a dialect/regionalism
    difference (PT-BR vs PT-PT) rather than a genuine writing quality issue.

    Uses LanguageTool's native structural category/rule taxonomy.
    """
    del msg, rule_desc
    cid = str(category_id).upper()
    rid = str(rule_id).upper()
    return cid == "REGIONALISMS" or rid.startswith("PT_BR") or rid.startswith("PT_BRASIL")


def _protected_spans(text: str) -> List[Tuple[int, int]]:
    """Syntax-defined spans that are not ordinary prose."""
    spans = []
    for rx in (_EMAIL_ADDRESS_RE, _URL_RE, _NUMERIC_ID_RE):
        spans.extend((m.start(), m.end()) for m in rx.finditer(text))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _span_overlaps(spans: List[Tuple[int, int]], start: int, end: int) -> bool:
    return any(start < b and end > a for a, b in spans)


def _mask_protected_text(text: str) -> str:
    chars = list(text)
    for start, end in _protected_spans(text):
        for i in range(start, end):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _is_probable_structural_header(line: str) -> bool:
    return bool(_EMAIL_HEADER_LINE_RE.match(line.strip()))


def _is_probable_signoff_block(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or len(lines) > 8 or len(_tokenize(text)) > 35:
        return False
    compact = " ".join(lines)
    has_contact = bool(
        _EMAIL_ADDRESS_RE.search(compact) or
        _URL_RE.search(compact) or
        re.search(r"\+?\d[\d .:/()\-]{6,}", compact)
    )
    has_farewell_shape = len(lines) <= 3 and bool(re.search(r"[,;:]$", lines[0]))
    return has_contact or has_farewell_shape or (len(lines) <= 4 and len(_tokenize(text)) <= 18)


def _quality_profile(text: str) -> Dict[str, float]:
    """Compute transparent structural diagnostics for WQ.

    These measurements are intentionally *diagnostic* in the length-neutral WQ policy.  They are
    retained for auditability and future evidence-based ablation, but they do
    not award positive points merely because an email matches an arbitrary
    preferred length/sentence shape.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    profile_paragraphs = list(paragraphs)
    while profile_paragraphs:
        lines = [ln.strip() for ln in profile_paragraphs[0].splitlines() if ln.strip()]
        if lines and all(_is_probable_structural_header(ln) for ln in lines):
            profile_paragraphs.pop(0)
        else:
            break
    body_parts = profile_paragraphs[1:-1] if len(profile_paragraphs) > 2 else profile_paragraphs
    body_text = " ".join(body_parts).strip()
    tokens = _tokenize(body_text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", body_text) if s.strip()]
    sentence_lengths = [len(_tokenize(s)) for s in sentences]

    n_tokens = len(tokens)
    n_sentences = len(sentence_lengths)
    avg_sentence = (sum(sentence_lengths) / n_sentences) if n_sentences else 0.0
    if n_sentences:
        variance = sum((x - avg_sentence) ** 2 for x in sentence_lengths) / n_sentences
        sentence_sd = variance ** 0.5
    else:
        sentence_sd = 0.0
    ttr = (len(set(tokens)) / n_tokens) if n_tokens else 0.0

    greeting_line = paragraphs[0].splitlines()[0].strip() if paragraphs else ""
    greeting_tokens = len(_tokenize(greeting_line))

    # Structural register is retained as a diagnostic only.  Missing or broken
    # greetings/sign-offs are handled by detect_structural_issues().
    signoff_register_quality = 0.40
    if len(paragraphs) > 1:
        last_para = paragraphs[-1]
        last_lines = [ln.strip() for ln in last_para.splitlines() if ln.strip()]
        last_tokens = _tokenize(last_para)
        is_structural_signoff = (
            0 < len(last_lines) <= 5
            and len(last_tokens) <= 20
            and not _MID_SENTENCE_END_RE.search(last_para)
        )
        if is_structural_signoff:
            farewell_tokens = len(_tokenize(last_lines[0]))
            signoff_register_quality = (
                0.65 if farewell_tokens <= 1 else
                0.78 if farewell_tokens == 2 else
                0.90 if farewell_tokens <= 4 else
                0.96
            )

    greeting_register_quality = (
        0.55 if greeting_tokens <= 1 else
        0.72 if greeting_tokens == 2 else
        0.84 if greeting_tokens == 3 else
        0.94 if greeting_tokens <= 5 else
        1.0
    )
    register_quality = (greeting_register_quality + signoff_register_quality) / 2.0

    # Legacy-style normalized diagnostics are retained so existing consumers
    # remain compatible. They are NOT used as direct email-length bonuses in the frozen calibration.
    length_quality = max(0.0, min(1.0, n_tokens / 80.0)) if n_tokens else 0.0
    sentence_count_quality = min(1.0, n_sentences / 4.0) if n_sentences else 0.0
    sentence_length_quality = (
        max(0.0, min(1.0, 1.0 - abs(avg_sentence - 16.0) / 32.0))
        if avg_sentence else 0.0
    )
    diversity_quality = ttr if n_tokens else 0.0
    rhythm_quality = (
        0.65 if n_sentences <= 1 else
        0.82 if sentence_sd < 1.0 else
        1.0 if sentence_sd <= 7.0 else
        max(0.65, 1.0 - ((sentence_sd - 7.0) / 15.0))
    )

    return {
        "body_token_count": float(n_tokens),
        "sentence_count": float(n_sentences),
        "average_sentence_length": round(avg_sentence, 3),
        "lexical_diversity": round(ttr, 4),
        "sentence_length_sd": round(sentence_sd, 3),
        "greeting_token_count": float(greeting_tokens),
        "register_quality": round(register_quality, 4),
        "length_quality": round(length_quality, 4),
        "sentence_count_quality": round(sentence_count_quality, 4),
        "sentence_length_quality": round(sentence_length_quality, 4),
        "diversity_quality": round(diversity_quality, 4),
        "rhythm_quality": round(rhythm_quality, 4),
    }


def _compute_score(
    violations: List[Dict[str, Any]],
    dimension_caps: Optional[Dict[str, float]] = None,
    text: Optional[str] = None,
) -> float:
    """Compute the deterministic *defect-burden* score, 0–100.

    ``text`` is accepted for backward compatibility. The score itself is
    length-neutral: penalties are accumulated by dimension and capped, with no
    word-count multiplier. Positive-quality shape signals remain diagnostics
    until they independently survive the Task 7 ablation test.
    """
    caps = dimension_caps
    dim_penalty: Dict[str, float] = defaultdict(float)
    for v in violations:
        dim = str(v.get("dimension", "style") or "style")
        if dim == "warning":
            continue
        sev = str(v.get("severity", "medium") or "medium").lower()
        base = _PENALTIES.get((dim, sev), _DEFAULT_PENALTY)
        rule_id = str(v.get("rule_id", "") or "")
        dim_penalty[dim] += base * _RULE_PENALTY_MULTIPLIERS.get(rule_id, 1.0)

    raw_penalty = 0.0
    for dim, raw in dim_penalty.items():
        raw_penalty += raw if caps is None else min(raw, caps.get(dim, float("inf")))

    defect_score = max(0.0, 100.0 - raw_penalty)
    return round(defect_score, 1)


def _frozen_tree_predict(nodes: List[Dict[str, Any]], features: List[float]) -> float:
    idx = 0
    while True:
        node = nodes[idx]
        feature_idx = int(node["feature"])
        if feature_idx < 0:
            return float(node["value"])
        idx = int(node["left"] if features[feature_idx] <= float(node["threshold"]) else node["right"])


def _calibrated_quality_score(profile: Dict[str, float], defect_only_score: float) -> float:
    """Frozen Task 7 score using no direct body-length features."""
    _load_calibration_model()
    values = [
        float(profile["sentence_length_sd"]),
        float(profile["lexical_diversity"]),
        float(profile["register_quality"]),
        float(profile["rhythm_quality"]),
        float(defect_only_score) / 100.0,
    ]
    latent = _CALIBRATION_INIT + _CALIBRATION_LEARNING_RATE * sum(
        _frozen_tree_predict(tree, values) for tree in _CALIBRATION_TREES
    )
    latent = max(1.0, min(5.0, latent))
    raw_score = _CALIBRATION_SCORE_BASE + _CALIBRATION_SCORE_PER_LEVEL * latent
    return round(max(0.0, min(float(defect_only_score), raw_score, WQ_CLEAN_SCORE_CEILING)), 1)


def strip_fixture_headers(text: str) -> str:
    """Remove leading RFC-style metadata lines from a calibration fixture.

    Only consecutive leading header lines (and the blank lines that separate
    them from the body) are removed. This must not be used on candidate model
    output: if a model emits De:/From: lines, that remains observable.
    """
    if not text:
        return text
    lines = text.splitlines()
    idx = 0
    seen_header = False
    while idx < len(lines):
        stripped = lines[idx].strip()
        if _FIXTURE_HEADER_LINE_RE.match(stripped):
            seen_header = True
            idx += 1
            continue
        if stripped == "" and (seen_header or idx == 0):
            idx += 1
            if seen_header:
                while idx < len(lines) and lines[idx].strip() == "":
                    idx += 1
                break
            continue
        break
    return "\n".join(lines[idx:]).strip("\n")


def wq_language_profile(lang: str, gold_dialect: Optional[str] = None) -> Optional[str]:
    """BCP-47 LanguageTool profile for the sample's reference variety.

    WQ asks how well-written the text is in its relevant variety, not how
    closely it matches the benchmark's PT-PT target.
    """
    raw = (lang or "").strip().lower().replace("_", "-")
    dialect = (gold_dialect or "").strip().upper()

    if raw == "mixed":
        # A genuinely mixed/unknown reference variety must not silently
        # become PT-PT. If an explicit reference dialect is available, use it;
        # otherwise leave WQ unscored rather than inventing a ruleset.
        if dialect in ("PT-PT", "PTPT"):
            return "pt-PT"
        if dialect in ("PT-BR", "PTBR"):
            return "pt-BR"
        if dialect in ("EN", "EN-US", "EN-GB"):
            return "en-US"
        return None

    if raw in ("pt-br", "ptbr") or raw.startswith("pt-br"):
        return "pt-BR"
    if raw in ("pt-pt", "ptpt", "pt") or raw.startswith("pt"):
        return "pt-PT"
    if raw in ("en", "en-us") or raw.startswith("en-us"):
        return "en-US"
    if raw.startswith("en-gb"):
        return "en-GB"
    if raw.startswith("en"):
        return "en-US"
    if dialect in ("PT-BR", "PTBR"):
        return "pt-BR"
    if dialect in ("PT-PT", "PTPT"):
        return "pt-PT"
    if dialect.startswith("EN"):
        return "en-US"
    return None


def empty_wq_result(
    *,
    excluded: bool = False,
    excluded_reason: Optional[str] = None,
    fixture_headers_stripped: bool = False,
) -> Dict[str, Any]:
    """Structured empty / excluded WQ payload (no invented score)."""
    return {
        "writing_quality_score": None,
        "grammar_error_count": 0,
        "spelling_error_count": 0,
        "repetition_count": 0,
        "structural_issue_count": 0,
        "style_issue_count": 0,
        "wq_violation_count": 0,
        "wq_violations": [],
        "languagetool_api_used": False,
        "wq_evaluator_version": WQ_EVALUATOR_VERSION,
        "fixture_headers_stripped": fixture_headers_stripped,
        "wq_scored": False,
        "wq_excluded": excluded,
        "wq_excluded_reason": excluded_reason,
    }


# Token regex — shared by repetition and structural detectors.
_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)

# Intra-word apostrophe / quotation mark: a punctuation shape, not a lexicon.
# LanguageTool style rules cover language-specific register when local LT is on.
_INTERNAL_APOSTROPHE_RE = re.compile(r"\b[\wÀ-ÿ]+['’][\wÀ-ÿ]+\b")


def _tokenize(text: str) -> List[str]:
    """Lowercased word-token list from text, Unicode-aware."""
    return _TOKEN_RE.findall(text.lower())


_SPELLCHECKER_CACHE: Dict[str, Any] = {}
_SPELLING_FOLDED_CACHE: Dict[str, Dict[str, str]] = {}


def _hunspell_stem(language: str) -> Optional[Path]:
    lang = (language or "").strip().lower()
    if lang.startswith("pt-br"):
        return BASE_DIR / "docs" / "pt_BR"
    if lang.startswith("pt-pt") or lang == "pt":
        return BASE_DIR / "docs" / "pt_PT"
    return None


def _load_spellchecker(language: str) -> Any:
    """Return a conservative spell checker using bundled Hunspell data.

    Preferred backend is spylls (the project dependency). On Unix-like
    systems, a system libhunspell is accepted as a compatibility fallback so
    offline CI can still exercise the real dictionaries. No spelling word-list
    heuristics are used.
    """
    stem = _hunspell_stem(language)
    if stem is None:
        return None
    key = str(stem)
    if key in _SPELLCHECKER_CACHE:
        return _SPELLCHECKER_CACHE[key]

    dic = stem.with_suffix(".dic")
    aff = stem.with_suffix(".aff")
    if not dic.exists() or not aff.exists():
        _SPELLCHECKER_CACHE[key] = None
        return None

    try:
        from spylls.hunspell import Dictionary
        checker = ("spylls", Dictionary.from_files(str(stem)))
        _SPELLCHECKER_CACHE[key] = checker
        return checker
    except Exception:
        pass

    # System-library fallback. Attribute access is guarded so Windows without
    # a native Hunspell DLL simply falls back to LanguageTool/no local spelling.
    candidates = [ctypes.util.find_library("hunspell"), "libhunspell-1.7.so.0", "libhunspell-1.6.so.0"]
    for lib_name in candidates:
        if not lib_name:
            continue
        try:
            lib = ctypes.CDLL(lib_name)
            lib.Hunspell_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.Hunspell_create.restype = ctypes.c_void_p
            lib.Hunspell_spell.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.Hunspell_spell.restype = ctypes.c_int
            lib.Hunspell_destroy.argtypes = [ctypes.c_void_p]
            handle = lib.Hunspell_create(
                os.fsencode(str(aff)), os.fsencode(str(dic))
            )
            if not handle:
                continue
            checker = ("ctypes", lib, handle)
            _SPELLCHECKER_CACHE[key] = checker
            return checker
        except Exception:
            continue

    _SPELLCHECKER_CACHE[key] = None
    return None


def _spellchecker_lookup(checker: Any, word: str) -> bool:
    if not checker or not word:
        return True
    backend = checker[0]
    try:
        if backend == "spylls":
            dictionary = checker[1]
            return bool(
                dictionary.lookup(word)
                or dictionary.lookup(word.lower())
                or dictionary.lookup(word.capitalize())
            )
        _, lib, handle = checker
        encoded = word.encode("utf-8")
        return bool(
            lib.Hunspell_spell(handle, encoded)
            or (word.lower() != word and lib.Hunspell_spell(handle, word.lower().encode("utf-8")))
        )
    except Exception:
        return True


def _fold_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _load_dictionary_folded(language: str) -> Dict[str, str]:
    stem = _hunspell_stem(language)
    if stem is None:
        return {}
    key = str(stem)
    if key in _SPELLING_FOLDED_CACHE:
        return _SPELLING_FOLDED_CACHE[key]
    folded: Dict[str, str] = {}
    dic = stem.with_suffix(".dic")
    if dic.exists():
        try:
            with dic.open(encoding="utf-8", errors="ignore") as f:
                first = True
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    if first and raw.isdigit():
                        first = False
                        continue
                    first = False
                    word = raw.split("/", 1)[0]
                    if word and re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+", word):
                        folded.setdefault(_fold_diacritics(word), word)
        except Exception:
            folded = {}
    _SPELLING_FOLDED_CACHE[key] = folded
    return folded


def _spellchecker_suggestions(checker: Any, word: str) -> List[str]:
    """Return backend suggestions when safely available.

    The ctypes fallback deliberately does not call Hunspell_suggest: the native
    ABI varies across distributions and can crash the process. Diacritic-aware
    dictionary matching below supplies the offline correction signal without
    relying on that unsafe optional API.
    """
    try:
        if checker[0] == "spylls":
            return list(checker[1].suggest(word) or [])[:5]
    except Exception:
        pass
    return []


def detect_local_spelling_issues(text: str, language: str) -> List[Dict[str, Any]]:
    checker = _load_spellchecker(language)
    if not checker:
        return []
    issues: List[Dict[str, Any]] = []
    seen: set[str] = set()
    protected = _protected_spans(text)
    folded_dict = _load_dictionary_folded(language)
    for match in re.finditer(r"\b[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]{2,}\b", text):
        if _span_overlaps(protected, match.start(), match.end()):
            continue
        token = match.group(0)
        if len(token) < 4 or not token.islower() or token.lower() in seen:
            continue
        if _spellchecker_lookup(checker, token):
            continue
        suggestions = _spellchecker_suggestions(checker, token)
        folded_match = folded_dict.get(_fold_diacritics(token))
        likely_diacritic_error = bool(folded_match and folded_match.lower() != token.lower())
        if not suggestions and not likely_diacritic_error:
            continue
        if likely_diacritic_error and folded_match not in suggestions:
            suggestions.insert(0, folded_match)
        seen.add(token.lower())
        issues.append({
            "source": f"Hunspell fallback ({language})",
            "rule_id": "SPELLING_ERROR",
            "message": f"Token '{token}' is not recognised by the bundled Hunspell dictionary; suggested correction: {suggestions[0] if suggestions else folded_match}.",
            "context": text[max(0, match.start() - 30):match.end() + 30].strip(),
            "replacements": suggestions[:5],
            "severity": "high",
            "issue_type": "spelling",
            "dimension": "spelling",
        })
    return issues


# ---------------------------------------------------------------------------
# Dimension 1 & 2 & 5 (LT portion): LanguageTool (local)
# ---------------------------------------------------------------------------

def check_languagetool_quality(
    text: str,
    language: str,
    timeout: int = 10,
) -> Tuple[bool, List[Dict[str, Any]], bool]:
    """Run LanguageTool locally via language_tool_python.

    ``timeout`` is retained for API compatibility; the local wrapper manages
    its own server communication. No remote endpoint is accepted or used.
    """
    del timeout
    return check_local_languagetool(
        text,
        language,
        dialect_filter=_is_dialect_issue,
    )


# ---------------------------------------------------------------------------
# Dimension 3: Repetition detection (structural/statistical, no word lists)
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def detect_repetition(text: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    masked = _mask_protected_text(text)
    tokens = _tokenize(masked)
    if not tokens:
        return issues

    i = 0
    while i < len(tokens):
        run = 1
        while i + run < len(tokens) and tokens[i + run] == tokens[i]:
            run += 1
        if run >= 2:
            issues.append({
                "source": "Repetition Detector",
                "rule_id": "CONSECUTIVE_TOKEN_REPEAT",
                "message": f"Token '{tokens[i]}' repeated {run} consecutive times.",
                "context": " ".join(tokens[max(0, i - 1):i + run + 2]),
                "replacements": [],
                "severity": "high" if run >= 3 else "medium",
                "issue_type": "repetition",
                "dimension": "repetition",
            })
        i += run

    # Repeated multi-token prose is a weaker signal. Report at most one longest
    # non-overlapping phrase, reducing false positives from normal email terminology.
    # Build counts and the first few positions in one pass. The previous version
    # rescanned the entire token list for every candidate gram, which could become
    # quadratic on long documents and caused corpus-level timeouts.
    for n in (7, 6, 5):
        if len(tokens) < n:
            continue

        gram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        gram_positions: Dict[Tuple[str, ...], List[int]] = defaultdict(list)

        for j in range(len(tokens) - n + 1):
            gram = tuple(tokens[j:j + n])
            gram_counts[gram] += 1
            if len(gram_positions[gram]) < 8:
                gram_positions[gram].append(j)

        found = None
        for gram, count in gram_counts.items():
            if count < 2:
                continue
            positions = gram_positions[gram]
            if any((b - a) >= n for a, b in zip(positions, positions[1:])):
                found = (gram, count)
                break

        if found:
            gram, count = found
            phrase = " ".join(gram)
            issues.append({
                "source": "Repetition Detector",
                "rule_id": "REPEATED_NGRAM_PHRASE",
                "message": f"Phrase '{phrase}' appears {count} times in the text.",
                "context": phrase,
                "replacements": [],
                "severity": "medium",
                "issue_type": "repetition",
                "dimension": "repetition",
            })
            break

    raw_sentences = [s.strip() for s in re.split(r"[.!?]+|\n+", masked) if len(s.strip()) > 12]
    normalised = [re.sub(r"\s+", " ", s.lower().strip()) for s in raw_sentences]
    seen_sents = Counter(normalised)
    for norm_sent, count in seen_sents.items():
        if count >= 2:
            issues.append({
                "source": "Repetition Detector",
                "rule_id": "DUPLICATE_SENTENCE",
                "message": f"Sentence appears {count} times: '{norm_sent[:60]}...'",
                "context": norm_sent[:80],
                "replacements": [],
                "severity": "high",
                "issue_type": "repetition",
                "dimension": "repetition",
            })
            break
    return issues


# ---------------------------------------------------------------------------
# Dimension 4: Structural completeness (email-specific, structural regex only)
# ---------------------------------------------------------------------------

# A greeting line is structurally: a line in the opening paragraph (text
# before the first blank line) that starts with an uppercase letter
# (including accented: À–Ü) and ends with a comma or colon — i.e. the
# canonical "<Salutation> <Name>," pattern.
# We do NOT enumerate salutation words; the structural delimiter is enough.
# Searching only the first paragraph prevents a short email's sign-off
# ("Com os melhores cumprimentos,") from being mistaken for a greeting.
_GREETING_LINE_RE = re.compile(
    r"^\s*[A-ZÀ-Ü][^\n]{2,70}[,:][ \t]*$",
    re.MULTILINE,
)

# A sign-off is structurally the last paragraph of the email if:
#   - it has ≤5 lines and ≤20 total word-tokens
#   - it contains no mid-block sentence terminator (period/!/? followed by
#     a space then a word), which distinguishes a farewell + name block
#     from a genuine paragraph that simply ends with punctuation.
_MID_SENTENCE_END_RE = re.compile(r"[.!?]\s+\w")


def detect_structural_issues(text: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    stripped = text.strip()
    if not stripped:
        return issues
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", stripped) if p.strip()]

    header_count = 0
    while header_count < len(paragraphs):
        lines = [ln.strip() for ln in paragraphs[header_count].splitlines() if ln.strip()]
        if lines and all(_is_probable_structural_header(ln) for ln in lines):
            header_count += 1
        else:
            break
    content_paragraphs = paragraphs[header_count:]
    if not content_paragraphs:
        return issues

    first_para = content_paragraphs[0]
    opening_lines = [ln.strip() for ln in first_para.splitlines() if ln.strip()][:4]
    if not any(_GREETING_LINE_RE.match(ln) for ln in opening_lines) and len(content_paragraphs) >= 2:
        issues.append({
            "source": "Structure Detector",
            "rule_id": "MISSING_GREETING",
            "message": "No recognisable greeting line found at the opening of the email.",
            "context": "\n".join(opening_lines)[:100],
            "replacements": [],
            "severity": "high",
            "issue_type": "structural",
            "dimension": "structural",
        })

    body_paras = content_paragraphs[1:-1] if len(content_paragraphs) > 2 else content_paragraphs
    body_text = " ".join(body_paras)
    body_tokens = _tokenize(body_text)
    if len(body_tokens) < 5:
        issues.append({
            "source": "Structure Detector",
            "rule_id": "EMPTY_BODY",
            "message": f"Email body contains only {len(body_tokens)} word tokens. Expected substantive email content.",
            "context": body_text[:100],
            "replacements": [],
            "severity": "high",
            "issue_type": "structural",
            "dimension": "structural",
        })

    last_para = content_paragraphs[-1]
    if len(content_paragraphs) >= 2 and not _is_probable_signoff_block(last_para):
        issues.append({
            "source": "Structure Detector",
            "rule_id": "MISSING_SIGNOFF",
            "message": "No recognisable sign-off/signature block detected at the end of the email.",
            "context": last_para[:100],
            "replacements": [],
            "severity": "medium",
            "issue_type": "structural",
            "dimension": "structural",
        })

    body_sentences = [s for s in _SENTENCE_END_RE.split(body_text) if s.strip()]
    if len(body_sentences) > 3:
        fragments = [s for s in body_sentences if len(_tokenize(s)) < 4]
        if fragments and len(fragments) / len(body_sentences) > 0.40:
            issues.append({
                "source": "Structure Detector",
                "rule_id": "SENTENCE_FRAGMENTATION",
                "message": f"{len(fragments)}/{len(body_sentences)} body sentences have fewer than 4 words, suggesting fragmented or incomplete model output.",
                "context": body_text[:120],
                "replacements": [],
                "severity": "medium",
                "issue_type": "structural",
                "dimension": "structural",
            })

    for idx, para in enumerate(paragraphs, start=1):
        if len(_tokenize(para)) > 200:
            issues.append({
                "source": "Structure Detector",
                "rule_id": "DEGENERATE_PARAGRAPH",
                "message": f"Paragraph {idx} contains more than 200 words without a paragraph break.",
                "context": para[:80] + "...",
                "replacements": [],
                "severity": "medium",
                "issue_type": "structural",
                "dimension": "structural",
            })

    return issues


# ---------------------------------------------------------------------------
# Dimension 5 (local): Style / Register checks
# ---------------------------------------------------------------------------

# All-caps words of ≥ 5 characters (short acronyms like "SaaS" or "PT" are
# excluded by the length threshold; standard acronyms are ≤4 chars).
_ALLCAPS_LONG_RE = re.compile(r"\b[A-ZÀÁÂÃÄÅÆÇ-ÖØ-Ý]{5,}\b")

# Two or more consecutive exclamation or question marks, or ≥ 3 isolated
# exclamation marks anywhere in the text (e.g. "Great! Ok! Sure!").
_MULTI_EXCLAMATION_RE = re.compile(r"!{2,}|\?{2,}|(?:!\s*){3,}")

# Text blocks longer than 300 characters with no sentence-ending punctuation,
# newline, or comma — indicates a run-on output with no sentence structure.
_RUN_ON_BLOCK_RE = re.compile(r"[^.!?,\n]{300,}")


def detect_style_issues(text: str) -> List[Dict[str, Any]]:
    """
    Detects register and formatting anomalies using structural regex patterns.
    No hardcoded word lists.

    Checks:
      1. Excessive all-caps long words (≥ 5 chars) — more than 2 in a business email
      2. Consecutive exclamation / question marks — informal and unprofessional
      3. Run-on blocks — > 300 chars of continuous text without sentence boundary
      4. Intra-word apostrophes — punctuation-shape contraction/elision, not a word list
    """
    issues: List[Dict[str, Any]] = []

    # --- 1. Excessive all-caps words ---
    caps_words = _ALLCAPS_LONG_RE.findall(text)
    if len(caps_words) > 2:
        issues.append({
            "source": "Style Detector",
            "rule_id": "EXCESSIVE_ALLCAPS",
            "message": (
                f"{len(caps_words)} all-caps words (≥5 chars) found. "
                "Unusual in formal business correspondence."
            ),
            "context": ", ".join(caps_words[:6]),
            "replacements": [],
            "severity": "medium",
            "issue_type": "style",
            "dimension": "style",
        })

    # --- 2. Consecutive exclamation / question marks ---
    exc_matches = _MULTI_EXCLAMATION_RE.findall(text)
    if exc_matches:
        issues.append({
            "source": "Style Detector",
            "rule_id": "EXCESSIVE_EXCLAMATION",
            "message": (
                f"Consecutive or repeated exclamation/question marks found "
                f"({len(exc_matches)} occurrence(s)). Informal register."
            ),
            "context": exc_matches[0][:40],
            "replacements": [],
            "severity": "medium",
            "issue_type": "style",
            "dimension": "style",
        })

    # --- 3. Run-on blocks ---
    run_on = _RUN_ON_BLOCK_RE.findall(text)
    if run_on:
        issues.append({
            "source": "Style Detector",
            "rule_id": "RUN_ON_BLOCK",
            "message": (
                f"{len(run_on)} text block(s) of > 300 characters with no sentence boundary. "
                "Possible run-on generation."
            ),
            "context": run_on[0][:80] + "...",
            "replacements": [],
            "severity": "medium",
            "issue_type": "style",
            "dimension": "style",
        })

    # --- 4. Intra-word apostrophes (register / contraction shape) ---
    apostrophe_tokens = _INTERNAL_APOSTROPHE_RE.findall(text)
    # Standard contraction morphological suffix pattern (e.g. word'm, word're, word've, word'll, word'd, word't, word's)
    # This checks linguistic structural formation without enumerating a dictionary of words.
    standard_contraction_shape = re.compile(r"^[A-Za-z]+['’](?:m|re|ve|ll|d|t|s)$", re.IGNORECASE)
    unusual = [t for t in apostrophe_tokens if not standard_contraction_shape.match(t)]
    if len(unusual) > 1:
        issues.append({
            "source": "Style Detector",
            "rule_id": "INFORMAL_CONTRACTION",
            "message": (
                f"{len(unusual)} unusual intra-word apostrophe token(s) found. "
                "Possible informal contraction or elision in a business email."
            ),
            "context": ", ".join(unusual[:6]),
            "replacements": [],
            "severity": "medium",
            "issue_type": "style",
            "dimension": "style",
        })

    # --- 5. Basic sentence/typography anomalies ---
    bad_spacing = re.findall(r"\s+[,:;.!?]", text)
    lowercase_after_end = re.findall(r"[.!?]\s+[a-zà-öø-ÿ]", text)
    if bad_spacing or lowercase_after_end:
        issues.append({
            "source": "Style Detector",
            "rule_id": "PUNCTUATION_ANOMALY",
            "message": (
                f"Found {len(bad_spacing)} punctuation-spacing anomaly/anomalies "
                f"and {len(lowercase_after_end)} probable sentence-start anomaly/anomalies."
            ),
            "context": ((bad_spacing + lowercase_after_end)[:4]),
            "replacements": [],
            "severity": "medium",
            "issue_type": "punctuation",
            "dimension": "style",
        })

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_writing_quality(
    text: str,
    language: str,
    use_languagetool: bool = True,
    strip_fixture_headers_flag: bool = False,
) -> Dict[str, Any]:
    """
    Full five-dimension writing quality evaluation.

    Dimensions:
      1. Grammar       (LanguageTool grammar rules, dialect-excluded)
      2. Spelling      (LanguageTool orthography rules, dialect-excluded)
      3. Repetition    (consecutive-token runs, n-gram phrase repeats, duplicate sentences)
      4. Structural    (greeting, body, sign-off, fragmentation, degenerate paragraphs)
      5. Style/Register(LanguageTool style/typo + local caps/exclamation/run-on checks)

    Each dimension count is reported independently. The derived
    writing_quality_score [0–100] is a penalty-based composite.

    Args:
      text:             Generated email text to evaluate.
      language:         BCP-47 language code for LanguageTool (e.g. "pt-PT", "en-US").
      use_languagetool: Call LanguageTool (local) (set False for offline unit tests).
      strip_fixture_headers_flag: If True, strip leading De:/From: metadata
                        used by calibration fixtures. Must stay False for
                        candidate model output.

    Returns a dict with keys:
      writing_quality_score   — float [0–100] or None when text is excluded
      wq_defect_only_score    — objective defect-burden score before clean ceiling
      wq_baseline_polish_score — supported positive-quality component retained for compatibility/auditability
      wq_quality_profile      — transparent component measurements
      grammar_error_count     — int
      spelling_error_count    — int
      repetition_count        — int
      structural_issue_count  — int
      style_issue_count       — int
      wq_violation_count      — int (total across all dimensions)
      wq_violations           — list of structured violation dicts
      languagetool_api_used   — bool (legacy field name; True means local LanguageTool was used)
      wq_evaluator_version    — str
      fixture_headers_stripped — bool
      wq_scored / wq_excluded / wq_excluded_reason
    """
    headers_stripped = False
    if strip_fixture_headers_flag and text:
        stripped = strip_fixture_headers(text)
        headers_stripped = stripped != text
        text = stripped

    # --- Empty text ---
    if not text or not text.strip():
        return empty_wq_result(
            excluded=True,
            excluded_reason="empty_text",
            fixture_headers_stripped=headers_stripped,
        )

    # RFC-style Subject/From/To metadata is observable output, but it is not
    # linguistic writing-quality evidence. Exclude leading metadata blocks from
    # WQ scoring/detection without removing them from the caller-visible text.
    scoring_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    header_count = 0
    while header_count < len(scoring_paragraphs):
        lines = [ln.strip() for ln in scoring_paragraphs[header_count].splitlines() if ln.strip()]
        if lines and all(_is_probable_structural_header(ln) for ln in lines):
            header_count += 1
            continue
        break
    scoring_text = "\n\n".join(scoring_paragraphs[header_count:]).strip()
    if not scoring_text:
        scoring_text = text.strip()

    # Language/task compliance is deliberately not converted into a WQ score.
    # The caller's language/profile selection belongs to benchmark compliance;
    # WQ itself measures the quality signals within that selected variety.

    # --- Dimension 1, 2, and 5 (LT portion) ---
    all_violations: List[Dict[str, Any]] = []
    api_ok = False

    if use_languagetool:
        api_ok, lt_issues, _ = check_languagetool_quality(scoring_text, language)
        # Strip out API warning pseudo-issues from the violation list
        all_violations.extend(v for v in lt_issues if v.get("severity") != "warning")

    # --- Dimension 2 offline fallback: spelling/diacritics ---
    # Offline spelling is a fallback when LanguageTool is unavailable. It is not
    # additive when the API succeeds, so the two sources do not double-count.
    if not api_ok:
        all_violations.extend(detect_local_spelling_issues(scoring_text, language))

    # --- Dimension 3: Repetition ---
    all_violations.extend(detect_repetition(scoring_text))

    # --- Dimension 4: Structural completeness ---
    all_violations.extend(detect_structural_issues(text))

    # --- Dimension 5 (local): Style/Register ---
    all_violations.extend(detect_style_issues(scoring_text))

    # --- Dimension-level counts ---
    grammar_violations  = [v for v in all_violations if v.get("dimension") == "grammar"]
    spelling_violations = [v for v in all_violations if v.get("dimension") == "spelling"]
    repetition_items    = [v for v in all_violations if v.get("dimension") == "repetition"]
    structural_items    = [v for v in all_violations if v.get("dimension") == "structural"]
    style_items         = [v for v in all_violations if v.get("dimension") == "style"]

    profile = _quality_profile(scoring_text)
    defect_only_score = _compute_score(all_violations, _DIM_CAPS, text=scoring_text)
    # Task 7: apply the frozen deterministic ordinal calibration. All profile
    # features remain transparent diagnostics; the fitted combination exists
    # only because the Task 7 reference set demonstrated that clean adjacent
    # quality bands cannot be recovered from defect counts alone.
    try:
        score = _calibrated_quality_score(profile, defect_only_score)
    except WQCalibrationUnavailable as exc:
        # P2.5: no calibration artifact means no publishable WQ score. Report
        # the metric as explicitly excluded rather than crashing the run or
        # inventing an uncalibrated number.
        return empty_wq_result(
            excluded=True,
            excluded_reason=f"WQ calibration unavailable: {exc}",
            fixture_headers_stripped=headers_stripped,
        )
    polish_score = round(
        (0.60 * profile["diversity_quality"] + 0.40 * profile["rhythm_quality"]) * 100.0,
        1,
    )
    # 100.0 is reserved as an unreachable exact ceiling so a clean sample is
    # not indistinguishable from a formally perfect score.
    if score >= WQ_CLEAN_SCORE_CEILING:
        score = WQ_CLEAN_SCORE_CEILING

    return {
        "writing_quality_score": round(score, 1),
        "wq_defect_only_score": round(defect_only_score, 1),
        "wq_baseline_polish_score": polish_score,
        "wq_quality_profile": profile,
        "wq_calibration_version": WQ_CALIBRATION_VERSION,
        "grammar_error_count":    len(grammar_violations),
        "spelling_error_count":   len(spelling_violations),
        "repetition_count":       len(repetition_items),
        "structural_issue_count": len(structural_items),
        "style_issue_count":      len(style_items),
        "wq_violation_count":     len(all_violations),
        "wq_violations":          all_violations,
        "languagetool_api_used":  api_ok,
        "wq_evaluator_version":   WQ_EVALUATOR_VERSION,
        "fixture_headers_stripped": headers_stripped,
        "wq_scored": True,
        "wq_excluded": False,
        "wq_excluded_reason": None,
    }