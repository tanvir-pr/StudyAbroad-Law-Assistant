"""Cross-encoder reranking. Real model: BAAI/bge-reranker-v2-m3."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]: ...


class BgeReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits:
            return []
        pairs = [(query, hit["text"]) for hit in hits]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(hits, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        out: list[dict] = []
        for hit, score in ranked[:top_k]:
            row = dict(hit)
            row["rerank_score"] = float(score)
            row["score"] = float(score)
            out.append(row)
        return out


class OverlapReranker:
    """Test stand-in: prefer chunks that share more query tokens."""

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        tokens = [token for token in query.lower().split() if len(token) > 2]

        def score(hit: dict) -> float:
            text = hit["text"].lower()
            if not tokens:
                return float(hit.get("score", 0))
            return sum(1.0 for token in tokens if token in text) / len(tokens)

        ranked = sorted(hits, key=score, reverse=True)
        out: list[dict] = []
        for hit in ranked[:top_k]:
            row = dict(hit)
            row["rerank_score"] = score(hit)
            row["score"] = row["rerank_score"]
            out.append(row)
        return out
