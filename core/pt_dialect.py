"""
European Portuguese (PT-PT) vs Brazilian Portuguese (PT-BR) Dialect Validator & EUPTVID Classifier.

Multi-layered dialect evaluation conforming to PT_PT_EMAIL_LLM_BENCHMARK_FINAL_SPECIFICATION:
1. Independent EUPTVID local classifier signal (models/model_quantized.ftz)
2. LanguageTool (local) (pt-PT regionalism ruleset)
3. Lexicon markers & contrasts (equipa vs equipe, telemóvel vs celular, etc.)
4. Grammatical & syntactic constructions (gerund overuse, sentence-initial proclisis, ter/haver)
"""

import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
from typing import Dict, Any, List, Optional, Tuple

from config import LANGUAGETOOL_LANG
from core.languagetool_local import check_local_languagetool


# --------------------------------------------------------------------------- #
# EUPTVID Local Classifier (fastText Singleton)
# --------------------------------------------------------------------------- #

_EUPTVID_MODEL = None
_EUPTVID_MODEL_PATH = BASE_DIR / "models" / "model_quantized.ftz"


def get_euptvid_model():
    """Singleton loader for fastText EUPTVID dialect classifier."""
    global _EUPTVID_MODEL
    if _EUPTVID_MODEL is None:
        if _EUPTVID_MODEL_PATH.exists():
            try:
                import fasttext
                # Suppress fasttext warning banner
                fasttext.FastText.eprint = lambda x: None
                _EUPTVID_MODEL = fasttext.load_model(str(_EUPTVID_MODEL_PATH))
            except Exception as e:
                print(f"[warn] Failed to load EUPTVID model from {_EUPTVID_MODEL_PATH}: {e}", file=sys.stderr)
                _EUPTVID_MODEL = False
        else:
            _EUPTVID_MODEL = False
    return _EUPTVID_MODEL if _EUPTVID_MODEL is not False else None


def evaluate_euptvid_signal(text: str) -> Dict[str, Any]:
    """
    Evaluates text with EUPTVID model to return raw fastText classification signal.
    Returns:
      label: "PT-PT" | "PT-BR" | "OTHER"
      ptpt_prob: probability of PT-PT [0.0 - 1.0]
      confidence: top predicted label probability [0.0 - 1.0]
    """
    model = get_euptvid_model()
    if model is None or not text or not text.strip():
        return {
            "label": "UNKNOWN",
            "ptpt_prob": None,
            "confidence": 0.0,
            "available": False
        }

    clean_text = text.replace("\n", " ").strip()
    try:
        # Ask for the full label distribution so PT-PT probability is the
        # actual class probability rather than an invalid complement of PT-BR
        # when an OTHER class exists. FastText accepts k=-1 for all labels.
        try:
            labels, probs = model.predict(clean_text, k=-1)
        except (TypeError, ValueError):
            labels, probs = model.predict(clean_text, k=100)
        # probs may be a numpy ndarray; avoid ambiguous truth-value checks
        if labels is None or len(labels) == 0 or probs is None or len(probs) == 0:
            raise ValueError("EUPTVID returned no labels/probabilities")
        top_label = labels[0].replace("__label__", "").replace("_", "-").upper()
        confidence = float(probs[0])

        ptpt_prob = 0.0
        for lbl, prob in zip(labels, probs):
            norm_lbl = lbl.replace("__label__", "").replace("_", "-").upper()
            if norm_lbl in ("PT-PT", "PTPT", "EU-PT"):
                ptpt_prob = float(prob)
                break

        return {
            "label": top_label,
            "ptpt_prob": round(ptpt_prob, 4),
            "confidence": round(confidence, 4),
            "available": True
        }
    except Exception as e:
        return {
            "label": "ERROR",
            "ptpt_prob": None,
            "confidence": 0.0,
            "available": False,
            "error": str(e)
        }


# --------------------------------------------------------------------------- #
# Hunspell Dictionary Singleton & Grammatical Patterns
# --------------------------------------------------------------------------- #

_HUNSPELL_PTPT = None
_HUNSPELL_PTPT_PATH = BASE_DIR / "docs" / "pt_PT"
_HUNSPELL_PTBR = None
_HUNSPELL_PTBR_PATH = BASE_DIR / "docs" / "pt_BR"


def get_ptpt_dictionary():
    """Singleton loader for official Hunspell pt_PT dictionary."""
    global _HUNSPELL_PTPT
    if _HUNSPELL_PTPT is None:
        if not _HUNSPELL_PTPT_PATH.with_suffix(".dic").exists():
            _HUNSPELL_PTPT = False
        else:
            try:
                from spylls.hunspell import Dictionary
                _HUNSPELL_PTPT = Dictionary.from_files(str(_HUNSPELL_PTPT_PATH))
            except Exception as e:
                print(f"[warn] Failed to load Hunspell pt_PT dictionary from {_HUNSPELL_PTPT_PATH}: {e}", file=sys.stderr)
                _HUNSPELL_PTPT = False
    return _HUNSPELL_PTPT if _HUNSPELL_PTPT is not False else None


def get_ptbr_dictionary():
    """Load the repository's official Brazilian Portuguese Hunspell dictionary."""
    global _HUNSPELL_PTBR
    if _HUNSPELL_PTBR is None:
        if not _HUNSPELL_PTBR_PATH.with_suffix(".dic").exists():
            _HUNSPELL_PTBR = False
        else:
            try:
                from spylls.hunspell import Dictionary
                _HUNSPELL_PTBR = Dictionary.from_files(str(_HUNSPELL_PTBR_PATH))
            except Exception as e:
                print(f"[warn] Failed to load Hunspell pt_BR dictionary from {_HUNSPELL_PTBR_PATH}: {e}", file=sys.stderr)
                _HUNSPELL_PTBR = False
    return _HUNSPELL_PTBR if _HUNSPELL_PTBR is not False else None


# --------------------------------------------------------------------------- #
# NLP & Linguistic Dialect Analyzers (spaCy pt_core_news_sm)
# Zero hardcoded dictionaries, word arrays, or keyword lists.
# --------------------------------------------------------------------------- #

_SPACY_NLP = None

def get_spacy_nlp():
    """Lazy-load the spaCy Portuguese linguistic pipeline (singleton)."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        # Only tagger, morphologizer, and parser are needed; disable NER for speed
        _SPACY_NLP = spacy.load("pt_core_news_sm", disable=["ner"])
    return _SPACY_NLP


# --------------------------------------------------------------------------- #
# LanguageTool & Offline Rule Checkers
# --------------------------------------------------------------------------- #

def check_languagetool_api(text: str, timeout: int = 10) -> Tuple[bool, List[Dict[str, Any]], bool]:
    """Compatibility wrapper for the local LanguageTool dialect signal.

    The historical function name is retained so persisted manifests and callers
    remain compatible, but it no longer performs any HTTP request itself.
    Uses LanguageTool's native structural category taxonomy (no keyword scanning).
    """
    del timeout
    ok, raw_issues, used = check_local_languagetool(text, LANGUAGETOOL_LANG)
    issues: List[Dict[str, Any]] = []
    for issue in raw_issues:
        if issue.get("severity") == "warning":
            issues.append({
                "source": "LanguageTool (local)",
                "rule_id": "LOCAL_WARNING",
                "message": issue.get("message", "Local LanguageTool unavailable"),
                "context": "", "replacements": [], "severity": "warning"
            })
            continue
        msg = issue.get("message", "")
        rule_id = issue.get("rule_id", "")
        category_id = str(issue.get("category_id", "")).upper()
        
        # Native LanguageTool taxonomy for dialect/regionalism issues
        if category_id == "REGIONALISMS" or rule_id.startswith("PT_BR") or rule_id.startswith("PT_BRASIL"):
            issues.append({
                "source": "LanguageTool (local, pt-PT ruleset)",
                "rule_id": rule_id,
                "message": msg,
                "context": issue.get("context", ""),
                "replacements": issue.get("replacements", []),
                "severity": "high",
                "evidence_type": "languagetool_regionalism",
                "validated": True,
                "score_eligible": True,
            })
    return ok, issues, used


def check_lexical_contrasts(text: str) -> List[Dict[str, Any]]:
    """Find lexical items recognized by pt_BR but not pt_PT using bundled dictionaries.

    This is deliberately resource-driven: the benchmark does not maintain a second
    hand-authored Portuguese word list. A token is reported only when the existing
    Brazilian dictionary accepts it and the existing European dictionary does not.
    """
    ptpt = get_ptpt_dictionary()
    ptbr = get_ptbr_dictionary()
    if ptpt is None or ptbr is None or not text or not text.strip():
        return []

    issues: List[Dict[str, Any]] = []
    seen = set()
    for token in re.findall(r"\b[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*\b", text, re.UNICODE):
        normalized = token.strip("-'’").lower()
        if len(normalized) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        if ptbr.lookup(normalized) and not ptpt.lookup(normalized):
            score_eligible = token[:1].islower()
            issues.append({
                "source": "Hunspell PT-BR/PT-PT lexical contrast",
                "rule_id": "PTBR_LEXICAL_CONTRAST",
                "message": f"Token '{token}' is recognized by the repository PT-BR dictionary but not by the PT-PT dictionary.",
                "context": token,
                "replacements": [],
                "severity": "medium",
                "validated": score_eligible,
                "score_eligible": score_eligible,
                "evidence_type": "lexical_contrast" if score_eligible else "dictionary_coverage_gap",
            })
    return issues


def check_lexicon_and_rules(text: str) -> List[Dict[str, Any]]:
    """
    Linguistic grammar checker for Brazilian grammar patterns in European Portuguese context.
    Uses spaCy POS tagging, morphological features, and dependency parsing.
    Zero hardcoded dictionaries or word lists.
    """
    if not text or not text.strip():
        return []

    nlp = get_spacy_nlp()
    doc = nlp(text)
    issues = []

    # 1. Continuous Gerund syntax check (Auxiliary/Verb + Gerund vs PT-PT "a + infinitive")
    for i, token in enumerate(doc):
        if "VerbForm=Ger" in str(token.morph):
            # Check if preceded by an auxiliary or verbal head in the same clause
            for prev in reversed(list(doc[:i])):
                if prev.is_punct or prev.text in (".", "!", "?", ";", "\n"):
                    break
                if prev.pos_ in ("VERB", "AUX"):
                    context = f"{prev.text} {token.text}"
                    issues.append({
                        "source": "Linguistic Grammar Ruleset (spaCy)",
                        "rule_id": "PTBR_GERUND_OVERUSE",
                        "message": f"Detected Brazilian continuous gerund construction '{context}'. In European Portuguese (PT-PT), preference is 'a + infinitive' (e.g., '{prev.text} a ...').",
                        "context": context,
                        "replacements": [f"{prev.text} a ..."],
                        "severity": "medium",
                        "evidence_type": "grammar_rule",
                        "validated": True,
                    })
                    break

    # 2. Brazilian Proclisis at sentence/clause start (clitic pronoun preceding verb)
    for sent in doc.sents:
        # Find first non-punctuation token
        first_token = None
        for t in sent:
            if not t.is_punct and not t.is_space:
                first_token = t
                break
        if first_token and "-" not in first_token.text:
            # Clitic pronouns in Portuguese cannot serve as nominative subjects (Case=Nom).
            # When an oblique clitic pronoun (Case=Acc/Dat or dep in obj/iobj/expl) precedes
            # a verb at sentence start, it is a proclisis construction.
            is_clitic = (
                first_token.dep_ in ("obj", "iobj", "expl") or
                any(c in first_token.morph.get("Case") for c in ("Acc", "Dat"))
            ) and "Nom" not in first_token.morph.get("Case")

            if is_clitic and first_token.i + 1 < len(doc):
                next_t = doc[first_token.i + 1]
                if next_t.pos_ in ("VERB", "AUX"):
                    context = f"{first_token.text} {next_t.text}"
                    issues.append({
                        "source": "Linguistic Grammar Ruleset (spaCy)",
                        "rule_id": "PTBR_PROCLISIS_START",
                        "message": f"Detected proclisis pronoun placement at start of sentence '{context}'. European Portuguese prefers enclisis (e.g., '{next_t.text}-{first_token.text.lower()}').",
                        "context": context,
                        "replacements": [f"{next_t.text}-{first_token.text.lower()}"],
                        "severity": "medium",
                        "evidence_type": "grammar_rule",
                        "validated": True,
                    })

    # 3. Brazilian "Ter" used for existential "Haver" (Subjectless transitive 'ter')
    for token in doc:
        if token.lemma_ == "ter" and token.pos_ in ("VERB", "AUX"):
            has_explicit_nsubj = any(c.dep_ in ("nsubj", "nsubj:pass") for c in token.children)
            has_obj = any(c.dep_ in ("obj", "obl") for c in token.children)
            if not has_explicit_nsubj and has_obj:
                context = token.text
                issues.append({
                    "source": "Linguistic Grammar Ruleset (spaCy)",
                    "rule_id": "PTBR_TER_HAVER",
                    "message": f"Detected existential 'ter' in '{context}'. In European Portuguese (PT-PT), use 'haver' (e.g., 'Há...').",
                    "context": context,
                    "replacements": ["Há ..."],
                    "severity": "medium",
                    "evidence_type": "grammar_rule",
                    "validated": True,
                })

    # 4. Brazilian "Para mim" + Infinitive verb
    for i, token in enumerate(doc):
        if token.lemma_ == "mim" and token.pos_ == "PRON":
            prev_t = doc[i - 1] if i > 0 else None
            next_t = doc[i + 1] if i + 1 < len(doc) else None
            if prev_t and prev_t.lemma_ == "para" and next_t and "VerbForm=Inf" in str(next_t.morph):
                context = f"{prev_t.text} {token.text} {next_t.text}"
                issues.append({
                    "source": "Linguistic Grammar Ruleset (spaCy)",
                    "rule_id": "PTBR_PARA_MIM_VERBO",
                    "message": f"Detected '{context}'. In Portuguese, personal pronouns as subject of infinitive take 'eu' ('para eu {next_t.text}').",
                    "context": context,
                    "replacements": [f"para eu {next_t.text}"],
                    "severity": "high",
                    "evidence_type": "grammar_rule",
                    "validated": True,
                })

    return issues


# --------------------------------------------------------------------------- #
# Language Identification Check (Hunspell Dictionary-Based)
# --------------------------------------------------------------------------- #

def is_predominantly_english(text: str) -> bool:
    """
    Detects if text is non-Portuguese/English by checking word recognition against
    the official European Portuguese Hunspell dictionary (dics/pt_PT.dic).
    """
    if not text or not text.strip():
        return False
    words = re.findall(r'\b[A-Za-zÀ-ÿ]+\b', text)

    hdict = get_ptpt_dictionary()
    if hdict is None:
        return False

    if not words:
        return False

    valid_count = sum(1 for w in words if hdict.lookup(w) or hdict.lookup(w.lower()))
    # A single token is insufficient evidence for reliable language ID. The
    # caller handles an unrecognised one-word output as "not measurable" rather
    # than inventing either an English mismatch or a perfect PT-PT result.
    if len(words) == 1:
        return False

    match_ratio = valid_count / len(words)
    return match_ratio < 0.35


# --------------------------------------------------------------------------- #
# Complete Multi-dimensional PT Dialect Evaluation
# --------------------------------------------------------------------------- #

def evaluate_pt_dialect(text: str, use_languagetool: bool = True) -> Dict[str, Any]:
    """
    Evaluates Portuguese text for PT-PT fidelity and detects PT-BR dialect leaks.
    Returns:
      euptvid_prob: fastText classifier PT-PT probability [0.0 - 1.0]
      euptvid_label: predicted dialect label
      is_clean_ptpt: True if 0 PT-BR violations detected
      ptpt_compliance_pct: 100.0 if clean, 0.0 if violations present
      ptbr_leakage_detected: True if >= 1 PT-BR violations
      violation_count: score-eligible PT-BR violation count
      ptbr_candidate_count: all PT-BR-looking candidates, including diagnostic-only findings
      violations: list of structured violation items
      pt_dialect_score: calibrated dialect score [0.0 - 100.0]
    """
    if not text or not text.strip():
        return {
            "euptvid_prob": None,
            "euptvid_label": None,
            "is_clean_ptpt": None,
            "ptpt_compliance_pct": None,
            "ptbr_leakage_detected": None,
            "pt_dialect_score": None,
            "violation_count": 0,
            "violations": [],
            "languagetool_api_used": False
        }

    # 0. Check language mismatch. This is a heuristic gate, not the EUPTVID
    # classifier, so an English mismatch must not fabricate an EUPTVID probability.
    ptpt_dict = get_ptpt_dictionary()
    ptpt_dict_available = ptpt_dict is not None
    if ptpt_dict_available:
        language_words = re.findall(r"\b[A-Za-zÀ-ÿ]+\b", text)
        if len(language_words) == 1:
            word = language_words[0]
            recognized = bool(ptpt_dict.lookup(word) or ptpt_dict.lookup(word.lower()))
            if not recognized:
                return {
                    "euptvid_prob": None,
                    "euptvid_label": "UNKNOWN",
                    "is_clean_ptpt": None,
                    "ptpt_compliance_pct": None,
                    "ptbr_leakage_detected": None,
                    "pt_dialect_score": None,
                    "violation_count": 0,
                    "ptbr_violation_count": 0,
                    "ptbr_candidate_count": 0,
                    "violations": [],
                    "languagetool_api_used": False,
                    "is_language_mismatch": None,
                    "language_adherence_available": False,
                    "language_adherence_status": "DEGRADED: insufficient text for reliable language identification",
                }
    if ptpt_dict_available and is_predominantly_english(text):
        return {
            "euptvid_prob": None,
            "euptvid_label": "EN",
            "is_clean_ptpt": False,
            "ptpt_compliance_pct": 0.0,
            "ptbr_leakage_detected": False,
            "pt_dialect_score": 0.0,
            "violation_count": 1,
            "violations": [{
                "source": "Language Verification",
                "rule_id": "PT_INPUT_ENGLISH_OUTPUT",
                "message": "Output was generated in English instead of European Portuguese (PT-PT).",
                "context": text[:80] + "..." if len(text) > 80 else text,
                "replacements": [],
                "severity": "high"
            }],
            "languagetool_api_used": False,
            "is_language_mismatch": True,
            "language_adherence_available": True,
            "language_adherence_status": "OK"
        }

    # 1. EUPTVID Classifier Signal
    euptvid_res = evaluate_euptvid_signal(text)

    # 2. LanguageTool (local)
    all_violations = []
    api_ok = False
    if use_languagetool:
        api_ok, api_issues, _ = check_languagetool_api(text)
        all_violations.extend([i for i in api_issues if i.get("severity") != "warning"])

    # 3. Rule & Lexicon inspection
    rule_issues = check_lexicon_and_rules(text) + check_lexical_contrasts(text)
    for ri in rule_issues:
        # Deduplicate overlapping matches
        if not any(ri["context"].lower() in ai.get("context", "").lower() for ai in all_violations):
            all_violations.append(ri)

    # 4. Compute independent PT-BR leakage and PT-PT compliance.
    # LanguageTool, lexical-dictionary contrasts, and EUPTVID are signals;
    # only evidence explicitly marked as validated is allowed to drive the
    # headline PT-BR leakage metric. Generic linguistic errors must not be
    # relabeled as Brazilian Portuguese.
    validated_ptbr_violations = [
        v for v in all_violations
        if bool(v.get("score_eligible", v.get("validated"))) and str(v.get("rule_id", "")).upper().startswith("PTBR_")
    ]
    ptbr_candidate_violations = [
        v for v in all_violations
        if str(v.get("rule_id", "")).upper().startswith("PTBR_")
    ]

    # Without the Hunspell language gate, a response with no independently
    # validated PT-BR evidence cannot honestly be called clean PT-PT. Preserve
    # any real PT-BR findings, but otherwise expose the result as no-data.
    if not ptpt_dict_available and not validated_ptbr_violations:
        return {
            "euptvid_prob": euptvid_res.get("ptpt_prob"),
            "euptvid_label": euptvid_res.get("label", "UNKNOWN"),
            "is_clean_ptpt": None,
            "ptpt_compliance_pct": None,
            "ptbr_leakage_detected": None,
            "pt_dialect_score": None,
            "violation_count": 0,
            "ptbr_violation_count": 0,
            "ptbr_candidate_count": len(ptbr_candidate_violations),
            "violations": all_violations,
            "languagetool_api_used": api_ok,
            "is_language_mismatch": None,
            "language_adherence_available": False,
            "language_adherence_status": "DEGRADED: pt_PT Hunspell dictionary unavailable",
        }

    ptbr_leakage_detected = bool(validated_ptbr_violations)
    is_language_mismatch = False  # already passed the early-return mismatch gate
    is_clean_ptpt = not is_language_mismatch and not ptbr_leakage_detected
    ptpt_compliance_pct = 100.0 if is_clean_ptpt else 0.0

    # Use a word-normalized penalty so this headline signal does not flatten
    # short emails or scale purely with message length. This mirrors the
    # canonical tiered fidelity normalization used by WF while retaining
    # severity-based weights for the dialect headline.
    words = max(1, len(re.findall(r"\b[\wÀ-ÿ\-]+\b", text, re.UNICODE)))
    severity_weights = {"low": 0.75, "medium": 1.5, "high": 3.0}
    weighted_penalty = sum(
        severity_weights.get(str(v.get("severity", "medium")).lower(), 1.5)
        for v in validated_ptbr_violations
    )
    score = max(0.0, min(100.0, 100.0 - (weighted_penalty / words) * 100.0))

    return {
        "euptvid_prob": euptvid_res.get("ptpt_prob"),
        "euptvid_label": euptvid_res.get("label"),
        "is_clean_ptpt": is_clean_ptpt,
        "ptpt_compliance_pct": ptpt_compliance_pct,
        "ptbr_leakage_detected": ptbr_leakage_detected,
        "pt_dialect_score": round(score, 1),
        "violation_count": len(validated_ptbr_violations),
        "ptbr_violation_count": len(validated_ptbr_violations),
        "ptbr_candidate_count": len(ptbr_candidate_violations),
        "violations": all_violations,
        "languagetool_api_used": api_ok,
        "is_language_mismatch": is_language_mismatch,
        "language_adherence_available": True,
        "language_adherence_status": "OK",
    }