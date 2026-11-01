"""Read configs/sources.yaml. Day 2's crawler may fetch only these URLs."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "configs" / "sources.yaml"

REQUIRED_FIELDS = (
    "id",
    "title",
    "url",
    "publisher",
    "country",
    "doc_type",
    "language",
    "topics",
    "format",
    "locked_on",
)


def load_allowlist(path: Path = SOURCES_PATH) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "sources" not in data:
        raise ValueError(f"{path} must contain a sources list")
    return data


def iter_sources(path: Path = SOURCES_PATH) -> list[dict]:
    sources = load_allowlist(path)["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    return sources
