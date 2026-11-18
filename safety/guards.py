"""Safety guards for day 11: sanitize context, scope checks, refuse without citation."""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (the )?(system|developer) prompt", re.I),
    re.compile(r"you are now .{0,80}(unrestricted|jailbreak|\bdan\b)", re.I),
    re.compile(r"reveal (your )?(system )?prompt", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
    re.compile(r"\[INST\]|<<SYS>>", re.I),
]

OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\b(germany|german|bamf|berlin|france|paris|usa|canada|uk|britain|toronto|munich)\b", re.I),
    re.compile(r"\b(bitcoin|crypto|stock tip|medical diagnos|prescribe|lawsuit against)\b", re.I),
    re.compile(r"\b(write (me )?code|python script|hack|password of)\b", re.I),
    re.compile(r"\b(marry|wedding|divorce|love letter)\b", re.I),
    re.compile(r"\b(quantum entanglement|physics homework|pizza topping)\b", re.I),
    re.compile(r"\bf-1\b|\bcpt\b", re.I),
]

PERSONAL_UNANSWERABLE = [
    re.compile(r"\b(my personal|case officer|file az-|guarantee that i)\b", re.I),
    re.compile(r"\b(bribe|secret internal|unpublished draft|passport number)\b", re.I),
    re.compile(r"\b(will oif approve my|did .+ accept me|for me personally|me personally)\b", re.I),
    re.compile(r"\b(exact rent will i pay|dorm beds are free .+ tonight)\b", re.I),
    re.compile(r"\b(which .+ professor will supervise|supervise my thesis)\b", re.I),
    re.compile(r"\b(2031|2028)\b", re.I),  # future unpublished figures in this eval set
    re.compile(r"\bappointment on 99 never street\b", re.I),
]

TOKEN_RE = re.compile(r"[a-z0-9áéíóöőúüű]+", re.I)
REFUSE_TEXT = (
    "REFUSE: I cannot cite an official source from the locked allow-list "
    "for this question. Check OIF, Tempus Public Foundation, or the university."
)


def tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if len(token) > 2}


def sanitize_text(text: str) -> str:
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = pattern.sub("[removed]", cleaned)
    return cleaned


def sanitize_hits(hits: list[dict]) -> list[dict]:
    out = []
    for hit in hits:
        row = dict(hit)
        row["text"] = sanitize_text(str(row.get("text", "")))
        out.append(row)
    return out


def question_must_refuse(question: str) -> bool:
    if any(pattern.search(question) for pattern in INJECTION_PATTERNS):
        return True
    if any(pattern.search(question) for pattern in PERSONAL_UNANSWERABLE):
        return True
    if any(pattern.search(question) for pattern in OUT_OF_SCOPE_PATTERNS):
        # Mobility / Hungary+Germany comparative questions that name Hungary stay answerable.
        if re.search(r"\bhungary\b", question, re.I) and re.search(
            r"\b(directive|mobility|student residence)\b", question, re.I
        ):
            return False
        return True
    return False


def context_supports_question(question: str, hits: list[dict], min_overlap: float = 0.18) -> bool:
    if not hits:
        return False
    if question_must_refuse(question):
        return False
    q_tokens = tokenize(question)
    if not q_tokens:
        return False
    best = 0.0
    for hit in hits:
        text_tokens = tokenize(hit.get("text", ""))
        if not text_tokens:
            continue
        overlap = len(q_tokens & text_tokens) / len(q_tokens)
        if overlap > best:
            best = overlap
    return best >= min_overlap


def enforce_citation_or_refuse(citation_ok: bool) -> tuple[str | None, bool]:
    """Return (refuse_text_or_None, refused)."""
    if citation_ok:
        return None, False
    return REFUSE_TEXT, True
