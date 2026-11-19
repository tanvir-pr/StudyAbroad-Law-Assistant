"""Day-12 tools: renewal deadline, document checklist, office lookup."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from tools.schemas import (
    ChecklistRequest,
    ChecklistResponse,
    OfficeRequest,
    OfficeResponse,
    RenewalRequest,
    RenewalResponse,
    validate_payload,
)

# Official-rule constants mirrored from allow-list facts (golden g04 / OIF extension).
STUDY_PERMIT_APPLY_DAYS_BEFORE = 30
SOURCE_EXTENSION = {
    "source_id": "oif-study-permit-extension",
    "source_url": (
        "https://www.oif.gov.hu/factsheets/"
        "how-to-extend-your-residence-permit-for-the-purpose-of-studies-know-how"
    ),
}
SOURCE_HEI = {
    "source_id": "oif-higher-education-institutions",
    "source_url": "https://oif.gov.hu/factsheets/for-higher-education-institutions",
}

CHECKLISTS: dict[str, list[dict[str, str]]] = {
    "study_permit_extension": [
        {
            "item": "Submit extension on Enter Hungary at least 30 days before expiry",
            "source_id": SOURCE_EXTENSION["source_id"],
            "source_url": SOURCE_EXTENSION["source_url"],
        },
        {
            "item": "Upload an ID photo (JPEG, ≤3 MB, not older than 3 months)",
            "source_id": SOURCE_EXTENSION["source_id"],
            "source_url": SOURCE_EXTENSION["source_url"],
        },
        {
            "item": "Provide biometrics at the regional client service within 15 days of submission",
            "source_id": SOURCE_EXTENSION["source_id"],
            "source_url": SOURCE_EXTENSION["source_url"],
        },
        {
            "item": "Attach the educational institution certificate of academic progress",
            "source_id": "njt-decree-35-2024",
            "source_url": "https://njt.jog.gov.hu/jogszabaly/2024-35-20-22",
        },
    ],
    "study_permit_first_issue": [
        {
            "item": "Apply through Enter Hungary with a valid travel document",
            "source_id": SOURCE_HEI["source_id"],
            "source_url": SOURCE_HEI["source_url"],
        },
        {
            "item": "Prove admission / enrolment at a Hungarian higher-education institution",
            "source_id": SOURCE_HEI["source_id"],
            "source_url": SOURCE_HEI["source_url"],
        },
        {
            "item": "Permit length is at least 1 year and at most 3 years (or programme length if shorter)",
            "source_id": SOURCE_HEI["source_id"],
            "source_url": SOURCE_HEI["source_url"],
        },
    ],
}

OFFICES: dict[str, dict[str, Any]] = {
    "oif": {
        "name": "National Directorate-General for Aliens Policing (OIF)",
        "role": "Student residence permits and extensions",
        "channel": "Enter Hungary online; biometrics at regional client service",
        "url": "https://oif.gov.hu/",
        "source_id": "oif-study-permit-extension",
        "source_url": SOURCE_EXTENSION["source_url"],
    },
    "tempus": {
        "name": "Tempus Public Foundation",
        "role": "Stipendium Hungaricum programme administration",
        "channel": "stipendiumhungaricum.hu",
        "url": "https://stipendiumhungaricum.hu/",
        "source_id": "sh-operational-regulations",
        "source_url": "https://stipendiumhungaricum.hu/documents/stipendium_hungaricum/OR/SH_OR.pdf",
    },
    "elte": {
        "name": "Eötvös Loránd University (ELTE)",
        "role": "University study rules and international student guidance",
        "channel": "ELTE international / registrar channels",
        "url": "https://www.elte.hu/en",
        "source_id": "elte-international-guide-2026-27",
        "source_url": "https://www.elte.hu/en/dstore/document/335230/guide_2026.pdf",
    },
    "bme": {
        "name": "Budapest University of Technology and Economics (BME)",
        "role": "Code of Studies and faculty rules",
        "channel": "BME faculty / registrar",
        "url": "https://www.bme.hu/",
        "source_id": "bme-code-of-studies-en",
        "source_url": "https://vik.bme.hu/document/6459/original/BME_Code_of_Studies_2016_mod(202402)_EN.pdf",
    },
    "nje": {
        "name": "John von Neumann University (NJE)",
        "role": "University regulations and documents",
        "channel": "nje.hu regulations pages",
        "url": "https://nje.hu/en/regulation-and-documents",
        "source_id": "nje-regulations-index",
        "source_url": "https://nje.hu/en/regulation-and-documents",
    },
}


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def renewal_from_expiry(payload: dict[str, Any]) -> dict[str, Any]:
    data = validate_payload(payload, RenewalRequest)
    expiry = _parse_iso_date(data["expiry_date"])
    apply_by = expiry - timedelta(days=STUDY_PERMIT_APPLY_DAYS_BEFORE)
    today = _parse_iso_date(data["as_of"]) if data.get("as_of") else date.today()
    days_left = (apply_by - today).days
    response = {
        "tool": "permit_renewal",
        "expiry_date": expiry.isoformat(),
        "apply_by": apply_by.isoformat(),
        "days_before_expiry": STUDY_PERMIT_APPLY_DAYS_BEFORE,
        "days_until_apply_by": days_left,
        "status": "overdue" if days_left < 0 else ("due_soon" if days_left <= 14 else "ok"),
        "channel": "Enter Hungary",
        "disclaimer": "Not legal advice. Confirm on the cited OIF page before you act.",
        **SOURCE_EXTENSION,
    }
    return validate_payload(response, RenewalResponse)


def document_checklist(payload: dict[str, Any]) -> dict[str, Any]:
    data = validate_payload(payload, ChecklistRequest)
    kind = data["kind"]
    items = CHECKLISTS.get(kind)
    if not items:
        raise ValueError(f"Unknown checklist kind: {kind}")
    response = {
        "tool": "document_checklist",
        "kind": kind,
        "items": items,
        "disclaimer": "Not legal advice. Lists mirror public allow-list pages only.",
    }
    return validate_payload(response, ChecklistResponse)


def office_lookup(payload: dict[str, Any]) -> dict[str, Any]:
    data = validate_payload(payload, OfficeRequest)
    key = data["office_id"].lower().strip()
    office = OFFICES.get(key)
    if not office:
        known = ", ".join(sorted(OFFICES))
        raise ValueError(f"Unknown office_id={key!r}. Known: {known}")
    response = {
        "tool": "office_lookup",
        "office_id": key,
        **office,
        "disclaimer": "Not legal advice. Public directory pointers only.",
    }
    return validate_payload(response, OfficeResponse)


TOOLS = {
    "permit_renewal": renewal_from_expiry,
    "document_checklist": document_checklist,
    "office_lookup": office_lookup,
}


def run_tool(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}. Known: {', '.join(sorted(TOOLS))}")
    return TOOLS[name](payload)
