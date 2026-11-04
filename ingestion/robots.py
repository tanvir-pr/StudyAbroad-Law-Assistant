"""Decide whether a URL may be fetched, using that host's robots.txt."""

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser


def robots_url_for(page_url: str) -> str:
    parts = urlsplit(page_url)
    return f"{parts.scheme}://{parts.netloc}/robots.txt"


def robots_allows(robots_txt: str | None, status: int, user_agent: str, page_url: str) -> bool:
    """A missing robots.txt (HTTP 404) allows the fetch. Any other failure blocks it."""
    if status == 404:
        return True
    if status != 200 or not robots_txt:
        return False
    parser = RobotFileParser()
    parser.parse(robots_txt.splitlines())
    return parser.can_fetch(user_agent, page_url)
