from tools.core import document_checklist, office_lookup, renewal_from_expiry, run_tool
from tools.schemas import ChecklistRequest, validate_payload


def test_renewal_from_expiry_applies_30_day_rule():
    out = renewal_from_expiry({"expiry_date": "2026-12-31", "as_of": "2026-11-01"})
    assert out["apply_by"] == "2026-12-01"
    assert out["days_before_expiry"] == 30
    assert out["status"] == "ok"
    assert out["source_id"] == "oif-study-permit-extension"
    assert out["channel"] == "Enter Hungary"


def test_renewal_overdue_and_due_soon():
    overdue = renewal_from_expiry({"expiry_date": "2026-11-10", "as_of": "2026-11-01"})
    assert overdue["apply_by"] == "2026-10-11"
    assert overdue["status"] == "overdue"
    due = renewal_from_expiry({"expiry_date": "2026-12-10", "as_of": "2026-11-01"})
    assert due["apply_by"] == "2026-11-10"
    assert due["status"] == "due_soon"


def test_document_checklist_extension_has_cited_items():
    out = document_checklist({"kind": "study_permit_extension"})
    assert out["tool"] == "document_checklist"
    assert len(out["items"]) >= 3
    assert all(item["source_url"].startswith("https://") for item in out["items"])


def test_office_lookup_oif():
    out = office_lookup({"office_id": "OIF"})
    assert out["office_id"] == "oif"
    assert "Aliens Policing" in out["name"]
    assert out["url"].startswith("https://")


def test_run_tool_and_schema_reject_bad_payload():
    out = run_tool("office_lookup", {"office_id": "elte"})
    assert out["tool"] == "office_lookup"
    try:
        validate_payload({"kind": "not-a-real-kind"}, ChecklistRequest)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        renewal_from_expiry({"expiry_date": "31-12-2026"})
        assert False, "expected ValueError"
    except ValueError:
        pass
