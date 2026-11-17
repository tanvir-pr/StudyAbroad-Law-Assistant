"""Hybrid retrieval: dense FAISS + BM25 with optional country/institution filters."""

from __future__ import annotations

from pathlib import Path

from rag.bm25 import Bm25Index, load_chunk_rows
from rag.index_faiss import search as dense_search


def apply_metadata_filters(
    rows: list[dict],
    *,
    country: str | None = None,
    institution: str | None = None,
) -> list[dict]:
    filtered = rows
    if country:
        needle = country.strip().upper()
        filtered = [row for row in filtered if str(row.get("country", "")).upper() == needle]
    if institution:
        needle = institution.strip().upper()
        filtered = [
            row for row in filtered if str(row.get("institution", "")).upper() == needle
        ]
    return filtered


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    *,
    rrf_k: int = 60,
    weights: list[float] | None = None,
    limit: int = 5,
) -> list[dict]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[int, float] = {}
    payloads: dict[int, dict] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, hit in enumerate(ranked):
            chunk_id = int(hit["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (1.0 / (rrf_k + rank + 1))
            payloads[chunk_id] = hit
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[dict] = []
    for chunk_id, score in ordered[:limit]:
        hit = dict(payloads[chunk_id])
        hit["score"] = float(score)
        hit["rrf_score"] = float(score)
        fused.append(hit)
    return fused


def retrieve(
    query: str,
    embedder,
    index_dir: Path,
    *,
    k: int = 5,
    candidate_k: int = 30,
    rrf_k: int = 60,
    mode: str = "hybrid",
    country: str | None = None,
    institution: str | None = None,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    bm25_index: Bm25Index | None = None,
    reranker=None,
    rerank_candidates: int | None = None,
) -> list[dict]:
    rows = load_chunk_rows(index_dir)
    filtered = apply_metadata_filters(rows, country=country, institution=institution)
    if not filtered:
        return []
    allowed_ids = {int(row["chunk_id"]) for row in filtered}
    pool = max(candidate_k, k)
    rerank_pool = max(rerank_candidates or pool, k)

    dense_hits: list[dict] = []
    if mode in {"dense", "hybrid"}:
        # Pull a wider pool, then keep only rows that survive metadata filters.
        raw_dense = dense_search(query, embedder, index_dir, k=min(len(rows), max(pool * 5, pool)))
        dense_hits = [hit for hit in raw_dense if int(hit["chunk_id"]) in allowed_ids][:pool]

    sparse_hits: list[dict] = []
    if mode in {"bm25", "hybrid"}:
        index = bm25_index or Bm25Index(rows)
        sparse_hits = index.search(query, k=pool, allowed_ids=allowed_ids)

    if mode == "dense":
        fused = dense_hits[:rerank_pool]
    elif mode == "bm25":
        fused = sparse_hits[:rerank_pool]
    elif mode == "hybrid":
        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            rrf_k=rrf_k,
            weights=[dense_weight, bm25_weight],
            limit=rerank_pool,
        )
    else:
        raise ValueError(f"unknown retrieve mode: {mode}")

    if reranker is None:
        return fused[:k]
    return reranker.rerank(query, fused, top_k=k)
