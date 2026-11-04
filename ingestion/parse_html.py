"""Turn an official HTML page into plain text. Script and style tags are dropped."""

from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    kept: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank == 1 and kept:
                kept.append("")
            continue
        blank = 0
        kept.append(line)
    return "\n".join(kept).strip()


def decode_html(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    lowered = content_type.lower()
    if "charset=" in lowered:
        charset = lowered.split("charset=", 1)[1].split(";", 1)[0].strip().strip('"')
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")
