"""Local-only LanguageTool adapter.

The benchmark deliberately uses the local ``language_tool_python`` Java server
and never configures a remote LanguageTool endpoint.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import language_tool_python  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    language_tool_python = None  # type: ignore

_CHECKERS: Dict[str, Any] = {}
_CHECKER_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()
# Engine identity, recorded when a checker is first built. The Java engine and
# its rule sets are downloaded by `language_tool_python`, not by this
# repository, so the wrapper version alone does not identify the ruleset that
# produced a grammar score. Surfacing it lets a run manifest bind results to
# the engine that actually ran.
_ENGINE_VERSIONS: Dict[str, str] = {}


def _detect_engine_version(checker: Any) -> Optional[str]:
    """Best-effort LanguageTool engine version, from the wrapper's own state.

    Reads attributes only. It must never construct a server, make a network
    call, or raise: an absent or unexpected engine stays absent in the manifest
    rather than becoming a fabricated version string or a hard failure.
    """
    for attribute in ("ltp_version", "language_tool_version", "engine_version"):
        value = getattr(checker, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    install_path = getattr(checker, "install_path", None)
    if install_path:
        # The extracted directory name carries the engine version.
        try:
            name = Path(str(install_path)).name
        except OSError:
            return None
        if re.match(r"^[0-9]+(\.[0-9]+)+$", name):
            return name
    return None


def describe_local_languagetool() -> Dict[str, Any]:
    """Return locally observed LanguageTool identity for a run manifest.

    Empty when no local checker has been built, so callers recording this in a
    manifest get an explicit absence rather than an inferred version.
    """
    with _CACHE_LOCK:
        languages = sorted(_CHECKERS)
        versions = dict(_ENGINE_VERSIONS)
    return {
        "mode": "local",
        "wrapper_version": versions.get("__wrapper__"),
        "languages": languages,
        "engine_versions": {language: versions[language] for language in languages if language in versions},
        "engine_version_unrecorded": [language for language in languages if language not in versions],
    }


def _get_checker(language: str) -> Tuple[Any, threading.Lock]:
    """Return a reusable local checker and its serialization lock."""
    with _CACHE_LOCK:
        checker = _CHECKERS.get(language)
        if checker is not None:
            return checker, _CHECKER_LOCKS[language]

        if language_tool_python is None:
            raise RuntimeError(
                "language_tool_python is unavailable; LanguageTool checks are degraded locally."
            )

        # Do not pass remote_server, proxies, or any externally supplied URL.
        checker = language_tool_python.LanguageTool(language)
        # Defensive invariant: this benchmark must never switch to a remote LT mode.
        if bool(getattr(checker, "is_remote", False)):
            try:
                checker.close()
            except Exception:
                pass
            raise RuntimeError("LanguageTool was initialized in remote mode; local-only execution is required.")
        lock = threading.Lock()
        _CHECKERS[language] = checker
        _CHECKER_LOCKS[language] = lock
        version = _detect_engine_version(checker)
        if version:
            _ENGINE_VERSIONS[language] = version
        wrapper = getattr(language_tool_python, "__version__", None)
        if isinstance(wrapper, str) and wrapper.strip():
            _ENGINE_VERSIONS["__wrapper__"] = wrapper.strip()
        return checker, lock


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _replacements(match: Any) -> List[str]:
    raw = _value(match, "replacements", default=[]) or []
    out: List[str] = []
    for item in raw:
        value = _value(item, "value", default=None)
        if value:
            out.append(str(value))
    return out[:5]


def check_local_languagetool(
    text: str,
    language: str,
    *,
    dialect_filter=None,
) -> Tuple[bool, List[Dict[str, Any]], bool]:
    """Run a local LanguageTool check and normalize matches.

    Returns ``(success, issues, tool_was_used)``. The third value preserves the
    existing persisted field semantics while making clear that usage is local.
    """
    if not text or not text.strip():
        return True, [], False

    try:
        checker, lock = _get_checker(language)
        with lock:
            matches = checker.check(text)
    except Exception as exc:
        return False, [{
            "source": "LanguageTool (local)",
            "rule_id": "LOCAL_WARNING",
            "message": f"Local LanguageTool unavailable: {exc}",
            "context": "",
            "replacements": [],
            "severity": "warning",
            "issue_type": "warning",
            "dimension": "warning",
        }], False

    issues: List[Dict[str, Any]] = []
    for match in matches:
        rule_id = str(_value(match, "ruleId", "rule_id", default="") or "")
        message = str(_value(match, "message", default="") or "")
        category_obj = _value(match, "category", default=None)
        category_id = str(_value(category_obj, "id", default="") or "")
        issue_type = str(_value(match, "ruleIssueType", "issueType", "issue_type", default="") or "").lower()
        rule_desc = str(_value(match, "ruleDescription", "description", default="") or "")
        context = str(_value(match, "context", default="") or "").strip()

        # Some language_tool_python versions expose category/rule metadata via
        # nested objects/dicts; keep the filter compatible with both forms.
        if dialect_filter and dialect_filter(message, rule_id, rule_desc, category_id):
            continue

        if issue_type in ("grammar",):
            dimension, severity = "grammar", "high"
        elif issue_type in ("misspelling", "spelling", "typographical"):
            dimension, severity = "spelling", "high"
        else:
            dimension, severity = "style", "medium"

        issues.append({
            "source": f"LanguageTool (local, {language})",
            "rule_id": rule_id,
            "message": message,
            "context": context,
            "replacements": _replacements(match),
            "severity": severity,
            "issue_type": issue_type or "uncategorized",
            "dimension": dimension,
            "category_id": category_id,
        })

    return True, issues, True
