"""Decide whether a URL may be fetched, using that host's robots.txt."""

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser


def robots_url_for(page_url: str) -> str:
    parts = urlsplit(page_url)
    return f"{parts.scheme}://{parts.netloc}/robots.txt"


def robots_allows(robots_txt: str | None, status: int, user_agent: str, page_url: str) -> bool:
    """A missing robots.txt allows the fetch. A published Disallow blocks it.

    HTTP 404 is a missing file. An empty HTTP 202 is not a robots file either:
    EUR-Lex answers that way and still serves the directive, with no Disallow rule.
    Any other failure blocks the fetch.
    """
    if status == 404:
        return True
    if status == 202 and not _looks_like_robots(robots_txt):
        return True
    if status != 200 or not robots_txt:
        return False
    parser = RobotFileParser()
    parser.parse(robots_txt.splitlines())
    return parser.can_fetch(user_agent, page_url)


def _looks_like_robots(robots_txt: str | None) -> bool:
    return bool(robots_txt) and "user-agent" in robots_txt.lower()
