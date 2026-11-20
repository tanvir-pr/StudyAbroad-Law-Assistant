"""Draft Q&A pairs from indexed allow-list chunks (day 13). Golden questions are excluded later."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from finetune.style import format_short_steps
from ingestion.sources import ROOT
from rag.bm25 import load_chunk_rows
from rag.index_faiss import load_rag_config

SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
NOISE_RE = re.compile(
    r"^(cookie|privacy|copyright|all rights|click here|home\b|menu\b|skip to)",
    re.I,
)
FACT_RE = re.compile(
    r"\b(\d+|must|may|shall|permit|residence|student|scholarship|credit|semester|"
    r"hour|fee|apply|extend|submit|within|before|after|university|hungary)\b",
    re.I,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def pick_sentences(text: str) -> list[str]:
    out = []
    for raw in SENT_RE.split(text):
        sentence = " ".join(raw.split()).strip()
        if len(sentence) < 40 or len(sentence) > 320:
            continue
        if NOISE_RE.search(sentence):
            continue
        if not FACT_RE.search(sentence):
            continue
        out.append(sentence)
    return out


def _topic_phrase(sentence: str) -> str:
    lower = sentence.lower()
    if "stipendium" in lower or "scholarship" in lower:
        return "Stipendium Hungaricum or scholarship rules"
    if "extend" in lower or "extension" in lower:
        return "extending a study residence permit"
    if "work" in lower and "hour" in lower:
        return "student work hours during studies"
    if "credit" in lower or "semester" in lower:
        return "credits or semesters"
    if "biometric" in lower:
        return "biometrics after submitting an application"
    if "enter hungary" in lower:
        return "Enter Hungary filing"
    if "residence" in lower or "permit" in lower:
        return "student residence permits"
    if "dorm" in lower or "accommodation" in lower:
        return "accommodation or dormitory rules"
    return "studying in Hungary under the official rules"


def draft_question(sentence: str, title: str) -> str:
    lower = sentence.lower()
    topic = _topic_phrase(sentence)
    source_hint = title.strip() or "the official source"
    focus_words = [
        word
        for word in re.findall(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű0-9]+", sentence)
        if len(word) > 3
    ][:7]
    focus = " ".join(focus_words) if focus_words else topic
    if re.search(r"\b(\d+\s*(hour|day|month|year|credit|huf)|how many|number of)\b", lower):
        return f"What limit or amount does {source_hint} set about {focus}?"
    if re.search(r"\b(must|shall|required|have to)\b", lower):
        return f"What must a student do under {source_hint} regarding {focus}?"
    if re.search(r"\b(may|can|allowed|entitle)\b", lower):
        return f"What is allowed according to {source_hint} about {focus}?"
    if re.search(r"\b(within|before|after|deadline|days?)\b", lower):
        return f"What timing or deadline does {source_hint} give for {focus}?"
    return f"What does {source_hint} say about {focus}?"


def draft_from_chunk(chunk: dict, per_chunk: int = 2) -> list[dict]:
    sentences = pick_sentences(chunk.get("text", ""))
    drafts = []
    for sentence in sentences[:per_chunk]:
        answer = format_short_steps(sentence, chunk["source_id"], chunk["source_url"])
        drafts.append(
            {
                "draft_id": hashlib.sha1(
                    f"{chunk.get('chunk_id')}|{sentence}".encode("utf-8")
                ).hexdigest()[:12],
                "chunk_id": chunk.get("chunk_id"),
                "source_id": chunk["source_id"],
                "source_url": chunk["source_url"],
                "title": chunk.get("title"),
                "language": chunk.get("language"),
                "institution": chunk.get("institution"),
                "country": chunk.get("country"),
                "question": draft_question(sentence, str(chunk.get("title") or chunk["source_id"])),
                "answer": answer,
                "answer_span": sentence,
                "chunk_text": chunk["text"],
                "status": "draft",
            }
        )
    return drafts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--per-chunk", type=int, default=1)
    parser.add_argument("--limit-chunks", type=int, default=0, help="0 = all chunks")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "finetune" / "qa_drafts.jsonl",
    )
    args = parser.parse_args()

    cfg = load_rag_config()
    index_dir = args.index_dir or (ROOT / cfg["index_dir"])
    chunks = load_chunk_rows(index_dir)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            for draft in draft_from_chunk(chunk, per_chunk=args.per_chunk):
                handle.write(json.dumps(draft, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"drafts": n, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
