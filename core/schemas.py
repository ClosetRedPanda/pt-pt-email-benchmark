"""
Core Data Schemas and JSON Specifications for benchmark.

FIXED: EMAIL_ANALYSIS_SCHEMA set "strict": True (in api_client.py's response_format)
but "required" only listed 11 of 15 properties. OpenAI-compatible strict JSON-schema
mode requires EVERY key in "properties" to also appear in "required" — optional
fields are expressed via nullable types (["string", "null"]), not by omission from
required. A schema that violates this contract gets handled inconsistently across
providers: some silently fall back to non-strict best-effort JSON, which can return
subtly wrong types (e.g. "action_required" as the string "true" instead of a real
boolean) that fail local validate_email_analysis() even though the content is
otherwise usable. This is why schema_compliance_pct could read 0% while
category_acc_pct/urgency_acc_pct read >0% in the same run: two independent checks,
only one of them tripped by the malformed-but-readable output.
"""

from typing import Dict, Any, List, Optional
import json
import re

# JSON Schema for Email Analysis (for OpenRouter response_format & jsonschema validation)
EMAIL_ANALYSIS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "EmailAnalysis",
    "type": "object",
    "properties": {
        "sender_name": {
            "type": ["string", "null"],
            "description": "Full name of sender if identifiable, else null"
        },
        "sender_email": {
            "type": ["string", "null"],
            "description": "Email address of sender if present, else null"
        },
        "recipient_name": {
            "type": ["string", "null"],
            "description": "Recipient name if mentioned, else null"
        },
        "recipient_email": {
            "type": ["string", "null"],
            "description": "Recipient email address if mentioned, else null"
        },
        "date_sent": {
            "type": ["string", "null"],
            "description": "Date/timestamp sent or referenced, else null"
        },
        "subject": {
            "type": ["string", "null"],
            "description": "Subject line of the email"
        },
        "category": {
            "type": "string",
            "enum": ["complaint", "support", "sales", "billing", "internal", "inquiry", "other"],
            "description": "Primary high-level email classification category"
        },
        "sub_category": {
            "type": ["string", "null"],
            "description": "Granular classification (e.g., refund_request, tech_issue, quote_request)"
        },
        "urgency": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Urgency priority level"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
            "description": "Overall tone/sentiment of the sender"
        },
        "action_required": {
            "type": "boolean",
            "description": "True if the email explicitly requests follow-up action or response"
        },
        "required_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific action items requested"
        },
        "key_entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key entities extracted (product names, invoice numbers, locations, dates)"
        },
        "language_variant": {
            "type": "string",
            "enum": ["pt-pt", "pt-br", "en", "other"],
            "description": "Detected language dialect variant: pt-pt (European Portuguese), pt-br (Brazilian Portuguese), en (English), other"
        },
        "summary": {
            "type": "string",
            "description": "Concise 1-2 sentence executive summary of the email"
        }
    },
    # FIX: every key above now appears here — was missing recipient_name,
    # recipient_email, date_sent, sub_category, which broke the strict-mode contract.
    "required": [
        "sender_name",
        "sender_email",
        "recipient_name",
        "recipient_email",
        "date_sent",
        "subject",
        "category",
        "sub_category",
        "urgency",
        "sentiment",
        "action_required",
        "required_actions",
        "key_entities",
        "language_variant",
        "summary"
    ],
    "additionalProperties": False
}


def validate_email_analysis(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate an analysis object against the benchmark schema.

    Missing validator dependencies are evaluator-environment failures, not
    model failures, so they are raised rather than converted into ``False``.
    """
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "The jsonschema dependency is required for benchmark scoring; "
            "install the benchmark requirements before running evaluation."
        ) from exc
    try:
        jsonschema.validate(instance=data, schema=EMAIL_ANALYSIS_SCHEMA)
        return True, None
    except jsonschema.ValidationError as exc:
        return False, str(exc)
    except jsonschema.SchemaError as exc:
        raise RuntimeError(f"Benchmark schema is invalid: {exc}") from exc


def _strip_json_fence(content: str) -> str:
    """Strip one conventional Markdown fence, tolerating BOM/case/spacing."""
    text = content.lstrip("\ufeff\u200b").strip()
    match = re.match(r"^```\s*json\s*\n?(.*?)\n?```\s*$", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.match(r"^```\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _extract_json_object_conservatively(content: str) -> Optional[Dict[str, Any]]:
    """Extract an unambiguous standalone JSON object from surrounding text.

    If more than one valid top-level JSON object exists, return None rather
    than guessing which object is the model's intended structured response.
    """
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    index = 0
    while index < len(content):
        if content[index] != "{":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(value, dict):
            candidates.append(value)
            if len(candidates) > 1:
                return None
            index += end
        else:
            index += 1
    return candidates[0] if len(candidates) == 1 else None


def repair_json_content(content: str) -> Optional[Dict[str, Any]]:
    """Conservatively repair common JSON formatting problems."""
    if not isinstance(content, str) or not content.strip():
        return None
    normalized = _strip_json_fence(content)
    try:
        value = json.loads(normalized)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return _extract_json_object_conservatively(normalized)
