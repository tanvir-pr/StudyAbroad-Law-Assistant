"""Metadata written beside every fetched or skipped allow-list page."""

from datetime import datetime, timezone

METADATA_FIELDS = (
    "source_id",
    "title",
    "country",
    "institution",
    "doc_type",
    "language",
    "source_url",
    "last_updated",
    "retrieved_at",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_metadata(source: dict, retrieved_at: datetime, **extra) -> dict:
    record = {
        "source_id": source["id"],
        "title": source["title"],
        "country": source["country"],
        "institution": source["institution"],
        "doc_type": source["doc_type"],
        "language": source["language"],
        "topics": list(source["topics"]),
        "source_url": source["url"],
        "last_updated": source.get("last_updated"),
        "retrieved_at": retrieved_at.isoformat(),
    }
    record.update(extra)
    missing = [field for field in METADATA_FIELDS if field not in record]
    if missing:
        raise ValueError(f"metadata missing {missing}")
    return record
