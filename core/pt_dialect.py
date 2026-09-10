"""
European Portuguese (PT-PT) vs Brazilian Portuguese (PT-BR) Dialect Validator & EUPTVID Classifier.

Multi-layered dialect evaluation conforming to PT_PT_EMAIL_LLM_BENCHMARK_FINAL_SPECIFICATION:
1. Independent EUPTVID local classifier signal (models/model_quantized.ftz)
2. LanguageTool (local) (pt-PT regionalism ruleset)
3. Lexicon markers & contrasts (equipa vs equipe, telemóvel vs celular, etc.)
4. Grammatical & syntactic constructions (gerund overuse, sentence-initial proclisis, ter/haver)
"""

import re
import sys
from importlib import metadata
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


def _patch_fasttext_numpy2() -> None:
    """Make fastText's predict() work under NumPy >= 2.

    fastText 0.9.x calls ``np.array(probs, copy=False)``. NumPy 2 turned that
    into a hard error ("Unable to avoid copy while creating an array as
    requested") instead of a silent copy, so *every* prediction raises. The
    upstream package is effectively unmaintained, so the call is adapted here:
    ``copy=None`` is the NumPy 2 spelling of "copy only if required", which is
    exactly what ``copy=False`` meant in NumPy 1.
    """
    import numpy as np

    if getattr(np, "_ptpt_fasttext_patched", False):
        return
    if int(np.__version__.split(".", 1)[0]) < 2:
        return

    import fasttext.FastText as _ft

    original_predict = _ft._FastText.predict

    def predict(self, *args, **kwargs):
        real_array = np.array

        def tolerant_array(obj, *a, **k):
            if k.get("copy") is False:
                k["copy"] = None
            return real_array(obj, *a, **k)

        np.array = tolerant_array
        try:
            return original_predict(self, *args, **kwargs)
        finally:
            np.array = real_array

    _ft._FastText.predict = predict
    np._ptpt_fasttext_patched = True


def get_euptvid_model():
    """Singleton loader for the fastText EUPTVID dialect classifier.

    The model file is a managed resource (see ``core/resources.py``): it is too
    large to commit, is pinned to an exact upstream revision, and is verified by
    SHA-256 before use. ``python runner.py setup`` fetches it.

    A missing model still yields an unavailable signal rather than a fabricated
    one, matching the Hunspell behaviour, but the warning now says how to fix it
    instead of failing silently.
    """
    global _EUPTVID_MODEL
    if _EUPTVID_MODEL is None:
        from core.resources import EUPTVID, verify

        problem = verify(EUPTVID)
        if problem is not None:
            print(
                f"[warn] EUPTVID unavailable ({problem}). "
                f"Run `python runner.py setup` to fetch it; "
                f"euptvid_probability will be reported as unavailable.",
                file=sys.stderr,
            )
            _EUPTVID_MODEL = False
        else:
            try:
                import fasttext
                # Suppress fasttext warning banner
                fasttext.FastText.eprint = lambda x: None
                _patch_fasttext_numpy2()
                _EUPTVID_MODEL = fasttext.load_model(str(EUPTVID.path))
            except Exception as e:
                print(f"[warn] Failed to load EUPTVID model from {EUPTVID.path}: {e}", file=sys.stderr)
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


def _managed_hunspell_ready(resources: Tuple[Any, Any], label: str) -> bool:
    """Verify both managed dictionary files before any parser sees them."""
    from core.resources import verify

    problems = [problem for resource in resources if (problem := verify(resource))]
    if problems:
        print(
            f"[warn] Hunspell {label} unavailable ({'; '.join(problems)}). "
            "Run `python runner.py setup` to fetch verified dictionaries.",
            file=sys.stderr,
        )
        return False
    return True


def get_ptpt_dictionary():
    """Singleton loader for the checksum-verified Hunspell pt_PT dictionary."""
    global _HUNSPELL_PTPT
    if _HUNSPELL_PTPT is None:
        from core.resources import HUNSPELL_PT_PT_AFF, HUNSPELL_PT_PT_DIC

        resources = (HUNSPELL_PT_PT_AFF, HUNSPELL_PT_PT_DIC)
        if not _managed_hunspell_ready(resources, "pt_PT"):
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
    """Load the checksum-verified Brazilian Portuguese Hunspell dictionary."""
    global _HUNSPELL_PTBR
    if _HUNSPELL_PTBR is None:
        from core.resources import HUNSPELL_PT_BR_AFF, HUNSPELL_PT_BR_DIC

        resources = (HUNSPELL_PT_BR_AFF, HUNSPELL_PT_BR_DIC)
        if not _managed_hunspell_ready(resources, "pt_BR"):
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
SPACY_MODEL_DISTRIBUTION = "pt-core-news-sm"
SPACY_MODEL_VERSION = "3.8.0"

def get_spacy_nlp():
    """Lazy-load the spaCy Portuguese linguistic pipeline (singleton).

    Mirrors the other optional-resource loaders: when the model is not
    installed the loader records a permanent miss and returns ``None`` so
    callers can degrade gracefully instead of crashing. Grammar-dependent
    checks read ``None`` and skip their evidence (see README external
    requirements).
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = False
        try:
            installed = metadata.version(SPACY_MODEL_DISTRIBUTION)
            if installed != SPACY_MODEL_VERSION:
                raise RuntimeError(
                    f"expected {SPACY_MODEL_DISTRIBUTION}=={SPACY_MODEL_VERSION}, "
                    f"found {installed}; install requirements.lock"
                )
            import spacy
            # Only tagger, morphologizer, and parser are needed; disable NER for speed
            _SPACY_NLP = spacy.load("pt_core_news_sm", disable=["ner"])
        except Exception as exc:  # model missing, wrong version, or not loadable
            print(f"[warn] Failed to load pinned spaCy pt_core_news_sm: {exc}", file=sys.stderr)
    return _SPACY_NLP if _SPACY_NLP is not False else None


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


_URI_SCHEME_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*(?=://)", re.UNICODE)

def _mask_nonlexical_spans(text: str) -> str:
    """Mask syntax-identifiable non-prose spans before lexical lookup.

    This helper deliberately contains no vocabulary, acronym list, or
    word-level exception list. Non-lexical spans are identified by syntax
    (spaCy URL/email token attributes and generic URI-scheme syntax).
    """
    spans: List[Tuple[int, int]] = []
    nlp = get_spacy_nlp()
    if nlp is not None:  # spaCy absent -> fall back to regex-only URI masking
        doc = nlp(text)
        spans = [
            (token.idx, token.idx + len(token.text))
            for token in doc
            if token.like_url or token.like_email
        ]
    spans.extend((match.start(), match.end()) for match in _URI_SCHEME_RE.finditer(text))

    if not spans:
        return text

    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    chars = list(text)
    for start, end in merged:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)

# URI scheme names (RFC 3986 §3.1 scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )).
# A scheme name is a *protocol label* (http, https, ftp, wss, mailto, ...), not a
# word of any human language: it is written identically in PT-PT, PT-BR, and every
# other language. The managed PT-BR dictionary happens to list a handful of these
# labels (http, https, ftp, jar, pop, iris, ...) while the PT-PT one does not, so a
# bare protocol mention such as "via https" would otherwise be reported as a PT-BR
# leak purely because of that dictionary-coverage artifact. Because the tokens here
# are protocol vocabulary — derived structurally from the IANA scheme registry, not
# hand-picked Portuguese words — the lexical-contrast rule treats the whole class as
# dialect-neutral. Ordinary words that coincide with a scheme name (e.g. data, pop)
# lose nothing real: neither dialect spells them differently, so they carry no
# dialect signal, and genuine PT-BR markers are unaffected.
_URI_SCHEME_TOKENS = frozenset("""
    about acap acct acd acr activity afp ahad ans apacheapi app apt ar ark attachment
    aw aviator bb bbi bzr c2pa call cap chrome chrome-extension chrome-untrusted cid
    civta clio cmis coffee content crid crs cvs data dav dab dc dct dns dnt doi dpp
    drm dtn dvb dwl eid elsi beispiel enc esperanto eth ethereum evergreen facetime
    fax feed file finger fish ftp ft fid geo gg git gopher graph gs gtalk h323 ham
    http https hxt iax icap icon im imap info iot ipn ipp ips irc irc6 ircs iris
    isostore itms jabber jar jms keyparc lastfm ldap ldaps leapinfo lore magnet
    mailserver mailto maps market marker matrix mcpe me mid mms modem moz message
    mongodb mtqp mumble mupdate mvn news nfs ni nih node nntp notes oauth ocf odds
    oid onion openpgp4fpr opr otpauth pack palm paparazzi payto pkcs11 platform pop
    pres prospero proxy psyc pwid pvp qb query1 r3d raidpn reload res resource rmip
    rsync rtmfp rtmp rtsp rutracker s3 sar secret shaw shell sieve simplenote sips
    skype smb smp sms smtp snews snmp soap soldat spotify ssh steam stun stuns
    submit svn swh swid swr t agrt tel teliae telnet tftp things threema tip tis
    tn3270 tool tv udp unreal upt urn ut2004 v-event vevent vemmi ventrilo ves
    videotex view-source vnc wais webcal wsps wss wtai wyciwyg wys xcon xcon-userid
    xfire xmlrpc xmpp xri ymsgr z39.50r z39.50s
""".split())


def check_lexical_contrasts(
    text: str,
    echo_vocab: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Find lexical items recognized by pt_BR but not pt_PT using managed dictionaries.

    This is deliberately resource-driven: the benchmark does not maintain a second
    hand-authored Portuguese word list. A token is reported only when the existing
    Brazilian dictionary accepts it and the existing European dictionary does not.

    ``echo_vocab`` (optional, lowercase alphabetic tokens) exempts task-echoed
    vocabulary from the lexical contrast: when the benchmark's own task
    constraints use a word as an accepted answer (e.g. ``estorno`` in a required
    action pattern), the model is *expected* to write it, so flagging it as a
    PT-BR leak would make the task un-winnable. The exemption is structural —
    the caller derives the set from the task's own pattern fields — and applies
    only when the caller supplies it: standalone use without task context keeps
    the strict dictionary contrast. Grammar rules (spaCy) are unaffected.
    """
    ptpt = get_ptpt_dictionary()
    ptbr = get_ptbr_dictionary()
    if ptpt is None or ptbr is None or not text or not text.strip():
        return []

    issues: List[Dict[str, Any]] = []
    seen = set()
    lexical_text = _mask_nonlexical_spans(text)
    for token in re.findall(r"\b[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*\b", lexical_text, re.UNICODE):
        normalized = token.strip("-'’").lower()
        if len(normalized) < 3 or normalized in seen:
            continue
        if normalized in _URI_SCHEME_TOKENS:
            # Protocol labels are dialect-neutral vocabulary (see _URI_SCHEME_TOKENS).
            continue
        if echo_vocab and normalized in echo_vocab:
            # Task-echoed vocabulary: the benchmark's own task patterns supplied
            # this word as an accepted answer, so its dialect colour is not a
            # model-initiated leak (see docstring).
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
    if nlp is None:  # spaCy model unavailable -> no grammar evidence (see README)
        return []
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

    # 3. Brazilian "Ter" used for existential "Haver" (subjectless 3rd-person 'ter')
    # Tightened: only flag clear existential candidates. Ordinary possession,
    # obligation, and 1st/2nd-person forms of "ter" are valid in PT-PT and must
    # not zero compliance. No vocabulary list is used — only morphological and
    # dependency features from spaCy.
    #
    # Known limitation (documented, not chased): plural-quantifier objects are
    # not detected. `pt_core_news_sm` tags "vários" as NOUN/amod (not DET), so
    # "Tem vários erros no relatório." — the same existential shape with a
    # plural quantifier — never reaches the indefinite-determiner scan below and
    # is a deterministic false negative. The rule's conditions are otherwise
    # consistent; any extension must avoid new false positives on pro-drop
    # possession ("tem bons resultados" == "[ele] tem bons resultados").
    for token in doc:
        if token.lemma_ != "ter" or token.pos_ not in ("VERB", "AUX"):
            continue
        # Existential BR "ter" is characteristically 3rd person.
        person = token.morph.get("Person")
        if person and "3" not in person:
            continue
        # Further restrict to finite indicative. Subjunctive / future-subjunctive
        # forms (tenha, tiver, …) are ordinary PT-PT and were false positives.
        mood = token.morph.get("Mood")
        if mood and "Ind" not in mood:
            continue
        verb_form = token.morph.get("VerbForm")
        if verb_form and "Fin" not in verb_form:
            continue
        has_explicit_nsubj = any(
            c.dep_ in ("nsubj", "nsubj:pass") for c in token.children
        )
        # Also treat a preceding nominal subject attached to this verb as explicit.
        if not has_explicit_nsubj:
            for prev in reversed(list(doc[: token.i])):
                if prev.is_punct or prev.text in (".", "!", "?", ";", "\n"):
                    break
                if prev.dep_ in ("nsubj", "nsubj:pass") and prev.head == token:
                    has_explicit_nsubj = True
                    break
                if prev.pos_ in ("NOUN", "PROPN", "PRON") and prev.head == token:
                    has_explicit_nsubj = True
                    break
        # FIX (P0.3): Portuguese is pro-drop, so an absent `nsubj` does NOT
        # imply an existential reading. "Tem razão." (= "[você] tem razão",
        # perfectly good PT-PT) has no subject child and a direct object, so
        # the previous test flagged it and — because ptpt_compliance_pct is
        # binary — drove a clean email straight to 0% compliance.
        #
        # Existential BR "ter" introduces a *new, indefinite* entity into the
        # discourse ("Tem um problema no sistema", "Tem vários erros"), whereas
        # the pro-drop possessive/light-verb uses take a bare or definite
        # object ("tem razão", "tem tempo", "tem a certeza", "tem medo").
        # Requiring an overt indefinite determiner on the object separates the
        # two without any hand-maintained vocabulary list.
        existential_obj = None
        for child in token.children:
            if child.dep_ != "obj":
                continue
            for det in child.children:
                if det.dep_ not in ("det", "nummod"):
                    continue
                det_morph = str(det.morph)
                # Indefiniteness is read purely from morphology, never from a
                # word list. `pt_core_news_sm` omits the Definite feature on
                # plural "uns"/"umas", so an article that is *not* explicitly
                # Definite=Def is treated as indefinite -- this generalises to
                # the whole determiner system rather than enumerating members.
                is_indef_article = (
                    "Definite=Ind" in det_morph
                    or "PronType=Ind" in det_morph
                    or ("PronType=Art" in det_morph and "Definite=Def" not in det_morph)
                )
                if is_indef_article:
                    existential_obj = child
                    break
            if existential_obj is not None:
                break

        # Bare cardinals are deliberately NOT treated as existential evidence.
        # A counted object is systematically ambiguous between an existential
        # ("tem duas reuniões na agenda") and an ordinary possessive/temporal
        # reading ("tem duas semanas para responder"), and no morphological
        # feature separates the two. Flagging them would trade the pro-drop
        # false positives for a new, equally arbitrary class of them, so this
        # detector abstains: under a binary compliance metric a false positive
        # is far more damaging than a miss.

        if not has_explicit_nsubj and existential_obj is not None:
            context = token.text
            issues.append({
                "source": "Linguistic Grammar Ruleset (spaCy)",
                "rule_id": "PTBR_TER_HAVER",
                "message": (
                    f"Detected existential 'ter' in '{context}'. "
                    "In European Portuguese (PT-PT), use 'haver' (e.g., 'Há...')."
                ),
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
    the official European Portuguese Hunspell dictionary (docs/pt_PT.dic).
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

def evaluate_pt_dialect(
    text: str,
    use_languagetool: bool = True,
    echo_vocab: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Evaluates Portuguese text for PT-PT fidelity and detects PT-BR dialect leaks.
    Returns:
      euptvid_prob: fastText classifier PT-PT probability [0.0 - 1.0]
      euptvid_label: predicted dialect label
      is_clean_ptpt: True if 0 PT-BR violations detected
      ptpt_compliance_pct: 100.0 if clean, 0.0 if violations present
      ptpt_compliance_graded_pct: violation-density compliance [0.0 - 100.0],
        a lower-variance companion to the binary metric above
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
            "ptpt_compliance_graded_pct": None,
            "ptbr_leakage_detected": None,
            "pt_dialect_score": None,
            "violation_count": 0,
            "ptbr_violation_count": 0,
            "ptbr_candidate_count": 0,
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
                    "ptpt_compliance_graded_pct": None,
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
            "ptpt_compliance_graded_pct": 0.0,
            "ptbr_leakage_detected": False,
            "pt_dialect_score": 0.0,
            "violation_count": 1,
            # P3.1: every branch reports the same keys. The single violation
            # here is a language-adherence failure, not PT-BR leakage, so the
            # PT-BR-specific counters are explicitly zero rather than absent.
            "ptbr_violation_count": 0,
            "ptbr_candidate_count": 0,
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
    rule_issues = check_lexicon_and_rules(text) + check_lexical_contrasts(text, echo_vocab=echo_vocab)
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
            "ptpt_compliance_graded_pct": None,
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

    # FIX (P0.3c): `ptpt_compliance_pct` is deliberately binary, which means a
    # single violation in a long, otherwise-perfect email reads exactly the
    # same as an email that is PT-BR throughout. That is very high variance for
    # a headline metric feeding paired bootstrap comparisons. Expose a graded
    # companion based on weighted violation density so consumers can choose a
    # lower-variance signal; the binary metric is unchanged for continuity.
    ptpt_compliance_graded_pct = round(score, 1)

    return {
        "euptvid_prob": euptvid_res.get("ptpt_prob"),
        "euptvid_label": euptvid_res.get("label"),
        "is_clean_ptpt": is_clean_ptpt,
        "ptpt_compliance_pct": ptpt_compliance_pct,
        "ptpt_compliance_graded_pct": ptpt_compliance_graded_pct,
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