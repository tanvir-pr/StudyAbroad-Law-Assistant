"""BM25 sparse index over saved FAISS chunk metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9áéíóöőúüű]+", re.IGNORECASE)
META_NAME = "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_chunk_rows(index_dir: Path) -> list[dict]:
    path = index_dir / META_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Bm25Index:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self._tokens = [tokenize(row["text"]) for row in rows]
        self._bm25 = BM25Okapi(self._tokens) if rows else None

    @classmethod
    def from_index_dir(cls, index_dir: Path) -> "Bm25Index":
        return cls(load_chunk_rows(index_dir))

    def search(self, query: str, k: int = 20, allowed_ids: set[int] | None = None) -> list[dict]:
        if not self.rows or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[dict] = []
        for row_id in ranked:
            if allowed_ids is not None and row_id not in allowed_ids:
                continue
            if scores[row_id] <= 0:
                break
            hit = dict(self.rows[row_id])
            hit["bm25_score"] = float(scores[row_id])
            hit["score"] = float(scores[row_id])
            hits.append(hit)
            if len(hits) >= k:
                break
        return hits
