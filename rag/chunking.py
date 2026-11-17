"""Chunking strategies compared on day 6: fixed, recursive, section-aware."""

from __future__ import annotations

import re

PAGE_RE = re.compile(r"(?m)^---\s*page\s+\d+\s*---\s*$", re.IGNORECASE)
HEADING_RE = re.compile(
    r"(?m)^(?:"
    r"#{1,6}\s+\S.*|"  # markdown heading
    r"[A-Z][A-Z0-9][A-Z0-9\s,/&\-]{6,}|"  # ALL CAPS line
    r"\d+(?:\.\d+)*\.?\s+\S.|"  # numbered section
    r"Article\s+\d+|"
    r"[§]\s*\d+"
    r")\s*$"
)


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")
    body = text.strip()
    if not body:
        return []
    step_floor = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + size)
        if end < len(body):
            snap = body.rfind(" ", start + step_floor, end)
            if snap > start:
                end = snap
        piece = body[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(body):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + step_floor
        start = next_start
    return chunks


def _pack_parts(parts: list[str], size: int, overlap: int) -> list[str]:
    """Merge small parts into windows of about `size`, then apply fixed overlap cuts."""
    merged: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if len(piece) > size:
            if buf:
                merged.append("\n\n".join(buf))
                buf, buf_len = [], 0
            merged.extend(fixed_chunks(piece, size, overlap))
            continue
        extra = len(piece) + (2 if buf else 0)
        if buf and buf_len + extra > size:
            merged.append("\n\n".join(buf))
            buf, buf_len = [piece], len(piece)
        else:
            buf.append(piece)
            buf_len += extra
    if buf:
        merged.append("\n\n".join(buf))
    # Re-cut any oversized merges with fixed windows so overlap stays consistent.
    out: list[str] = []
    for block in merged:
        if len(block) <= size:
            out.append(block)
        else:
            out.extend(fixed_chunks(block, size, overlap))
    return out


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split on coarse separators first, then pack to the target size."""
    body = text.strip()
    if not body:
        return []
    if len(body) <= size:
        return [body]
    separators = ["\n\n", "\n", ". ", " "]
    parts = [body]
    for sep in separators:
        next_parts: list[str] = []
        for part in parts:
            if len(part) <= size:
                next_parts.append(part)
            else:
                bits = [bit for bit in part.split(sep) if bit.strip()]
                if len(bits) == 1:
                    next_parts.append(part)
                else:
                    glue = sep
                    for i, bit in enumerate(bits):
                        next_parts.append(bit if i == 0 else glue + bit)
        parts = next_parts
        if all(len(part) <= size for part in parts):
            break
    return _pack_parts(parts, size, overlap)


def _section_splits(text: str) -> list[str]:
    markers: list[int] = [0]
    for regex in (PAGE_RE, HEADING_RE):
        for match in regex.finditer(text):
            if match.start() > 0:
                markers.append(match.start())
    markers = sorted(set(markers))
    markers.append(len(text))
    sections: list[str] = []
    for i in range(len(markers) - 1):
        piece = text[markers[i] : markers[i + 1]].strip()
        if piece:
            sections.append(piece)
    return sections or [text]


def section_aware_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Prefer page markers and headings as boundaries; do not merge across them."""
    body = text.strip()
    if not body:
        return []
    chunks: list[str] = []
    for section in _section_splits(body):
        if len(section) <= size:
            chunks.append(section)
        else:
            chunks.extend(fixed_chunks(section, size, overlap))
    return chunks


STRATEGIES = {
    "fixed": fixed_chunks,
    "recursive": recursive_chunks,
    "section_aware": section_aware_chunks,
}


def chunk_text(text: str, strategy: str, size: int, overlap: int) -> list[str]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown chunk strategy: {strategy}") from exc
    return fn(text, size, overlap)
