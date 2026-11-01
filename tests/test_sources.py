from urllib.parse import urlparse

from ingestion.sources import REQUIRED_FIELDS, iter_sources, load_allowlist

ALLOWED_TOPICS = {
    "student_residence_permit",
    "work_during_studies",
    "stipendium_hungaricum",
    "university_regulations",
    "eu_student_mobility",
}


def test_allowlist_is_locked_and_fetch_uses_primary_urls_only():
    data = load_allowlist()
    assert data["locked_on"] == "2026-11-01"
    assert data["crawl"]["allow_only_listed_urls"] is True
    assert data["crawl"]["respect_robots_txt"] is True
    assert data["crawl"]["fetch_alternate_urls"] is False


def test_every_source_has_a_https_official_url_and_unique_id():
    ids = []
    for source in iter_sources():
        missing = [field for field in REQUIRED_FIELDS if field not in source]
        assert missing == [], source.get("id")
        ids.append(source["id"])
        _assert_https(source["url"])
        for alternate in source.get("alternate_urls") or []:
            _assert_https(alternate)
            assert alternate != source["url"]
        assert source["country"] in {"HU", "EU"}
        assert source["format"] in {"html", "pdf"}
        assert source["language"] in {"en", "hu"}
        assert source["topics"], source["id"]
        assert set(source["topics"]) <= ALLOWED_TOPICS
        assert source["locked_on"] == "2026-11-01"
    assert len(ids) == len(set(ids))
    assert len(ids) >= 20


def _assert_https(url: str) -> None:
    parsed = urlparse(url)
    assert parsed.scheme == "https", url
    assert parsed.netloc, url
