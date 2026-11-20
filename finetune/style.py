"""Day-13 answer style helpers: short steps + safe fallback."""

from __future__ import annotations

import re

from safety.guards import REFUSE_TEXT

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9-]{0,80})\]\((https://[^)\s]+)\)")
STEP_HINT = re.compile(
    r"\b(submit|apply|upload|visit|attach|provide|must|shall|through|via|within|before)\b",
    re.I,
)


def safe_fallback() -> str:
    return REFUSE_TEXT


def split_clauses(text: str) -> list[str]:
    parts = re.split(r"(?<=[.:;])\s+|\n+", text.strip())
    return [part.strip(" .-") for part in parts if len(part.strip(" .-")) > 20]


def format_short_steps(answer_body: str, source_id: str, source_url: str, max_steps: int = 3) -> str:
    """Turn a procedural span into numbered steps; otherwise one cited sentence."""
    body = " ".join(answer_body.split()).strip()
    if not body:
        return safe_fallback()
    cite = f"[{source_id}]({source_url})"
    clauses = split_clauses(body)
    procedural = bool(STEP_HINT.search(body)) and len(clauses) >= 2
    if procedural:
        steps = []
        for clause in clauses[:max_steps]:
            steps.append(f"{len(steps) + 1}. {clause.rstrip('.')} {cite}.")
        return "\n".join(steps) + "\nNot legal advice."
    sentence = body.rstrip(".")
    return f"{sentence} {cite}.\nNot legal advice."


def looks_like_refusal(text: str) -> bool:
    return text.strip().startswith("REFUSE:")
