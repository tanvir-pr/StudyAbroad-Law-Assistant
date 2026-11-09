"""Download the locked allow-list. Primary URLs only, and only when robots.txt allows."""

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

from ingestion.metadata import build_metadata, utc_now
from ingestion.parse_html import decode_html, html_to_text
from ingestion.robots import robots_allows, robots_url_for
from ingestion.sources import ROOT, load_allowlist

# The ELTE 2026/27 international guide PDF is about 51 MB.
MAX_BYTES = 80 * 1024 * 1024
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


class HttpFetcher:
    def __init__(self, user_agent: str, timeout: float = 60.0):
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def fetch_robots(self, url: str) -> tuple[int, str]:
        response = self._client.get(url)
        return response.status_code, response.text

    def fetch(self, url: str) -> tuple[int, str, str, bytes]:
        response = self._client.get(url)
        content_type = response.headers.get("content-type", "")
        return response.status_code, content_type, str(response.url), response.content


def crawl(
    raw_dir: Path,
    *,
    fetcher,
    sources: list[dict] | None = None,
    retrieved_at: datetime | None = None,
    delay_seconds: float = 1.0,
) -> list[dict]:
    allowlist = load_allowlist()
    crawl_cfg = allowlist["crawl"]
    if not crawl_cfg["respect_robots_txt"]:
        raise RuntimeError("Refusing to crawl while robots.txt checks are off.")
    if crawl_cfg["fetch_alternate_urls"]:
        raise RuntimeError("Refusing to fetch alternate URLs.")
    if not crawl_cfg["allow_only_listed_urls"]:
        raise RuntimeError("Refusing to crawl without the allow-list lock.")

    listed = {item["url"] for item in allowlist["sources"]}
    chosen = allowlist["sources"] if sources is None else sources
    user_agent = crawl_cfg["user_agent"]
    moment = retrieved_at or utc_now()
    raw_dir.mkdir(parents=True, exist_ok=True)

    robots_cache: dict[str, tuple[int, str]] = {}
    records: list[dict] = []
    for index, source in enumerate(chosen):
        if index and delay_seconds:
            time.sleep(delay_seconds)
        records.append(
            _fetch_one(
                source,
                listed=listed,
                user_agent=user_agent,
                fetcher=fetcher,
                robots_cache=robots_cache,
                raw_dir=raw_dir,
                retrieved_at=moment,
            )
        )

    manifest = raw_dir / "manifest.json"
    _merge_manifest(manifest, records)
    return records


def _fetch_one(
    source: dict,
    *,
    listed: set[str],
    user_agent: str,
    fetcher,
    robots_cache: dict[str, tuple[int, str]],
    raw_dir: Path,
    retrieved_at: datetime,
) -> dict:
    url = source["url"]
    if url not in listed:
        raise ValueError(f"URL is not on the allow-list: {url}")
    if not ID_RE.fullmatch(source["id"]):
        raise ValueError(f"Unsafe source id: {source['id']}")

    robots_target = robots_url_for(url)
    if robots_target not in robots_cache:
        try:
            robots_cache[robots_target] = fetcher.fetch_robots(robots_target)
        except httpx.HTTPError as exc:
            robots_cache[robots_target] = (0, str(exc))
    status, body = robots_cache[robots_target]
    if not robots_allows(body, status, user_agent, url):
        if status == 0:
            reason = f"robots.txt request failed: {body}"
        elif status not in (200, 404):
            reason = f"robots.txt returned HTTP {status}"
        else:
            reason = "robots.txt disallows this user-agent"
        record = _skipped(source, retrieved_at, reason)
        _write_meta(raw_dir, source["id"], record)
        return record

    try:
        page_status, content_type, final_url, payload = fetcher.fetch(url)
    except httpx.HTTPError as exc:
        record = _skipped(source, retrieved_at, f"fetch failed: {exc}", robots_allowed=True)
        _write_meta(raw_dir, source["id"], record)
        return record

    if page_status != 200:
        record = _skipped(
            source,
            retrieved_at,
            f"HTTP {page_status}",
            robots_allowed=True,
            http_status=page_status,
            final_url=final_url,
        )
        _write_meta(raw_dir, source["id"], record)
        return record
    if len(payload) > MAX_BYTES:
        record = _skipped(
            source,
            retrieved_at,
            f"response is {len(payload)} bytes, over the {MAX_BYTES} byte cap",
            robots_allowed=True,
            http_status=page_status,
            final_url=final_url,
        )
        _write_meta(raw_dir, source["id"], record)
        return record

    extension = "pdf" if source["format"] == "pdf" or "pdf" in content_type.lower() else "html"
    raw_name = f"{source['id']}.{extension}"
    raw_path = raw_dir / raw_name
    raw_path.write_bytes(payload)

    text_path = None
    text_chars = None
    if extension == "html":
        text = html_to_text(decode_html(payload, content_type))
        text_file = raw_dir / f"{source['id']}.txt"
        text_file.write_text(text, encoding="utf-8")
        text_path = _relative(text_file)
        text_chars = len(text)

    record = build_metadata(
        source,
        retrieved_at,
        final_url=final_url,
        http_status=page_status,
        content_type=content_type.split(";", 1)[0].strip(),
        robots_allowed=True,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_length=len(payload),
        raw_path=_relative(raw_path),
        text_path=text_path,
        text_chars=text_chars,
        skip_reason=None,
    )
    _write_meta(raw_dir, source["id"], record)
    return record


def _merge_manifest(manifest: Path, records: list[dict]) -> list[dict]:
    existing: list[dict] = []
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
    by_id = {row["source_id"]: row for row in existing}
    order = [row["source_id"] for row in existing]
    for record in records:
        if record["source_id"] not in by_id:
            order.append(record["source_id"])
        by_id[record["source_id"]] = record
    merged = [by_id[source_id] for source_id in order]
    manifest.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged


def _skipped(source: dict, retrieved_at: datetime, reason: str, **extra) -> dict:
    record = build_metadata(
        source,
        retrieved_at,
        robots_allowed=extra.pop("robots_allowed", False),
        http_status=extra.pop("http_status", None),
        final_url=extra.pop("final_url", None),
        content_type=None,
        sha256=None,
        byte_length=None,
        raw_path=None,
        text_path=None,
        text_chars=None,
        skip_reason=reason,
    )
    return record


def _write_meta(raw_dir: Path, source_id: str, record: dict) -> None:
    path = raw_dir / f"{source_id}.meta.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    allowlist = load_allowlist()
    fetcher = HttpFetcher(allowlist["crawl"]["user_agent"])
    try:
        records = crawl(ROOT / "data" / "raw", fetcher=fetcher)
    finally:
        fetcher.close()
    saved = [record for record in records if record["raw_path"]]
    skipped = [record for record in records if not record["raw_path"]]
    print(f"saved {len(saved)}  skipped {len(skipped)}")
    for record in skipped:
        print(f"skip {record['source_id']}: {record['skip_reason']}")


if __name__ == "__main__":
    main()
