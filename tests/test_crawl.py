import json
from datetime import datetime, timezone

import pytest

from ingestion.crawl import crawl
from ingestion.metadata import METADATA_FIELDS
from ingestion.parse_html import html_to_text
from ingestion.robots import robots_allows
from ingestion.sources import iter_sources

WHEN = datetime(2026, 11, 4, 19, 0, tzinfo=timezone.utc)


class FakeFetcher:
    def __init__(self, robots: dict[str, tuple[int, str]], pages: dict[str, tuple[int, str, str, bytes]]):
        self.robots = robots
        self.pages = pages
        self.fetched: list[str] = []

    def fetch_robots(self, url: str) -> tuple[int, str]:
        return self.robots[url]

    def fetch(self, url: str):
        self.fetched.append(url)
        return self.pages[url]


def test_robots_404_allows_and_disallow_blocks():
    assert robots_allows(None, 404, "StudyAbroadLawAssistant/0.1", "https://example.com/rules")
    rules = "User-agent: *\nDisallow: /private\n"
    agent = "StudyAbroadLawAssistant/0.1"
    assert robots_allows(rules, 200, agent, "https://example.com/rules")
    assert not robots_allows(rules, 200, agent, "https://example.com/private/page")
    assert not robots_allows("ignored", 503, agent, "https://example.com/rules")
    assert robots_allows("", 202, agent, "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016L0801")


def test_html_parser_drops_script_and_keeps_the_rule():
    html = "<html><style>body{}</style><script>secret()</script><h1>Permit</h1><p>30 hours a week</p></html>"
    text = html_to_text(html)
    assert "30 hours a week" in text
    assert "secret" not in text
    assert "body{}" not in text


def test_crawler_refuses_a_url_that_is_not_on_the_allowlist(tmp_path):
    source = dict(iter_sources()[0])
    source["url"] = "https://example.com/not-official"
    fetcher = FakeFetcher({}, {})
    with pytest.raises(ValueError, match="not on the allow-list"):
        crawl(tmp_path, fetcher=fetcher, sources=[source], retrieved_at=WHEN, delay_seconds=0)
    assert fetcher.fetched == []


def test_crawler_skips_when_robots_disallows_and_saves_html_when_allowed(tmp_path):
    by_source_id = {source["id"]: source for source in iter_sources()}
    allowed = by_source_id["oif-student-residence-en"]
    blocked = by_source_id["elte-regulations-index"]
    allowed_robots = _robots_url(allowed["url"])
    blocked_robots = _robots_url(blocked["url"])
    html = b"<html><h1>Student permit</h1><script>track()</script><p>Cite this page.</p></html>"
    fetcher = FakeFetcher(
        robots={
            allowed_robots: (200, "User-agent: *\nDisallow:\n"),
            blocked_robots: (200, "User-agent: *\nDisallow: /\n"),
        },
        pages={
            allowed["url"]: (200, "text/html; charset=utf-8", allowed["url"], html),
        },
    )
    records = crawl(
        tmp_path,
        fetcher=fetcher,
        sources=[allowed, blocked],
        retrieved_at=WHEN,
        delay_seconds=0,
    )
    by_id = {record["source_id"]: record for record in records}
    saved = by_id[allowed["id"]]
    for field in METADATA_FIELDS:
        assert field in saved
    assert saved["source_url"] == allowed["url"]
    assert saved["last_updated"] == allowed.get("last_updated")
    assert saved["retrieved_at"] == "2026-11-04T19:00:00+00:00"
    assert saved["robots_allowed"] is True
    assert saved["raw_path"].endswith(".html")
    assert "Cite this page." in (tmp_path / f"{allowed['id']}.txt").read_text(encoding="utf-8")
    assert "track" not in (tmp_path / f"{allowed['id']}.txt").read_text(encoding="utf-8")

    skipped = by_id[blocked["id"]]
    assert skipped["robots_allowed"] is False
    assert skipped["raw_path"] is None
    assert skipped["skip_reason"]
    assert blocked["url"] not in fetcher.fetched
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {row["source_id"] for row in manifest} == {allowed["id"], blocked["id"]}


def _robots_url(page_url: str) -> str:
    from ingestion.robots import robots_url_for

    return robots_url_for(page_url)
