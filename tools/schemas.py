"""Lightweight JSON-schema validation for day-12 tools (stdlib only)."""

from __future__ import annotations

from typing import Any

RenewalRequest = {
    "type": "object",
    "required": ["expiry_date"],
    "properties": {
        "expiry_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "as_of": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
    },
    "additionalProperties": False,
}

RenewalResponse = {
    "type": "object",
    "required": [
        "tool",
        "expiry_date",
        "apply_by",
        "days_before_expiry",
        "days_until_apply_by",
        "status",
        "channel",
        "source_id",
        "source_url",
        "disclaimer",
    ],
    "properties": {
        "tool": {"const": "permit_renewal"},
        "expiry_date": {"type": "string"},
        "apply_by": {"type": "string"},
        "days_before_expiry": {"type": "integer"},
        "days_until_apply_by": {"type": "integer"},
        "status": {"enum": ["ok", "due_soon", "overdue"]},
        "channel": {"type": "string"},
        "source_id": {"type": "string"},
        "source_url": {"type": "string"},
        "disclaimer": {"type": "string"},
    },
}

ChecklistRequest = {
    "type": "object",
    "required": ["kind"],
    "properties": {
        "kind": {
            "enum": ["study_permit_extension", "study_permit_first_issue"],
        }
    },
    "additionalProperties": False,
}

ChecklistResponse = {
    "type": "object",
    "required": ["tool", "kind", "items", "disclaimer"],
    "properties": {
        "tool": {"const": "document_checklist"},
        "kind": {"type": "string"},
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["item", "source_id", "source_url"],
                "properties": {
                    "item": {"type": "string"},
                    "source_id": {"type": "string"},
                    "source_url": {"type": "string"},
                },
            },
        },
        "disclaimer": {"type": "string"},
    },
}

OfficeRequest = {
    "type": "object",
    "required": ["office_id"],
    "properties": {
        "office_id": {"type": "string", "minLength": 2},
    },
    "additionalProperties": False,
}

OfficeResponse = {
    "type": "object",
    "required": [
        "tool",
        "office_id",
        "name",
        "role",
        "channel",
        "url",
        "source_id",
        "source_url",
        "disclaimer",
    ],
    "properties": {
        "tool": {"const": "office_lookup"},
        "office_id": {"type": "string"},
        "name": {"type": "string"},
        "role": {"type": "string"},
        "channel": {"type": "string"},
        "url": {"type": "string"},
        "source_id": {"type": "string"},
        "source_url": {"type": "string"},
        "disclaimer": {"type": "string"},
    },
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    required = schema.get("required", [])
    for key in required:
        if key not in payload:
            raise ValueError(f"missing required field: {key}")
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = set(payload) - set(props)
        if extra:
            raise ValueError(f"unexpected fields: {sorted(extra)}")
    for key, value in payload.items():
        if key not in props:
            continue
        rule = props[key]
        if "const" in rule and value != rule["const"]:
            raise ValueError(f"{key} must be {rule['const']!r}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{key} must be one of {rule['enum']}")
        expected = rule.get("type")
        if expected and not _type_ok(value, expected):
            raise ValueError(f"{key} must be type {expected}")
        if expected == "string":
            pattern = rule.get("pattern")
            if pattern:
                import re

                if not re.match(pattern, value):
                    raise ValueError(f"{key} must match {pattern}")
            if "minLength" in rule and len(value) < rule["minLength"]:
                raise ValueError(f"{key} too short")
        if expected == "array":
            if "minItems" in rule and len(value) < rule["minItems"]:
                raise ValueError(f"{key} needs at least {rule['minItems']} items")
            item_schema = rule.get("items")
            if item_schema:
                for item in value:
                    validate_payload(item, item_schema)
    return payload
