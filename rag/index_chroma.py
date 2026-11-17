"""Chroma dense index built from an existing FAISS index (same vectors, no re-embed)."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from ingestion.sources import ROOT
from rag.bm25 import load_chunk_rows
from rag.index_faiss import INDEX_NAME, load_rag_config

COLLECTION = "chunks"
CHROMA_DIRNAME = "chroma"


def chroma_dir(index_dir: Path | None = None) -> Path:
    config = load_rag_config()
    base = Path(index_dir) if index_dir else (ROOT / config["index_dir"])
    return base / CHROMA_DIRNAME


def build_chroma_from_faiss(index_dir: Path, chroma_path: Path | None = None) -> int:
    import chromadb

    chroma_path = chroma_path or (index_dir / CHROMA_DIRNAME)
    rows = load_chunk_rows(index_dir)
    index = faiss.read_index(str(index_dir / INDEX_NAME))
    if index.ntotal != len(rows):
        raise ValueError(f"FAISS has {index.ntotal} vectors but chunks.jsonl has {len(rows)} rows")
    if chroma_path.exists():
        for path in chroma_path.rglob("*"):
            if path.is_file():
                path.unlink()
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "ip"})
    batch = 256
    for start in range(0, len(rows), batch):
        end = min(len(rows), start + batch)
        ids = [str(rows[i]["chunk_id"]) for i in range(start, end)]
        documents = [rows[i]["text"] for i in range(start, end)]
        metadatas = [
            {
                "source_id": rows[i]["source_id"],
                "source_url": rows[i]["source_url"],
                "title": rows[i].get("title") or "",
                "country": rows[i].get("country") or "",
                "institution": rows[i].get("institution") or "",
                "language": rows[i].get("language") or "",
                "last_updated": rows[i].get("last_updated") or "",
                "chunk_id": int(rows[i]["chunk_id"]),
            }
            for i in range(start, end)
        ]
        embeddings = [index.reconstruct(i).astype(float).tolist() for i in range(start, end)]
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(rows)


def search_chroma(query: str, embedder, chroma_path: Path, k: int = 5) -> list[dict]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(COLLECTION)
    vector = np.asarray(embedder.embed_query(query), dtype=np.float32).reshape(-1)
    result = collection.query(query_embeddings=[vector.tolist()], n_results=min(k, collection.count()))
    hits: list[dict] = []
    if not result or not result.get("ids") or not result["ids"][0]:
        return hits
    for i, chunk_id in enumerate(result["ids"][0]):
        meta = result["metadatas"][0][i] if result.get("metadatas") else {}
        doc = result["documents"][0][i] if result.get("documents") else ""
        distance = result["distances"][0][i] if result.get("distances") else None
        # With inner-product space Chroma returns distances as 1 - score-ish depending on version;
        # keep the raw value and also expose a descending score for ranking displays.
        score = float(-distance) if distance is not None else 0.0
        hits.append(
            {
                "chunk_id": int(meta.get("chunk_id", chunk_id)),
                "source_id": meta.get("source_id"),
                "source_url": meta.get("source_url"),
                "title": meta.get("title"),
                "country": meta.get("country"),
                "institution": meta.get("institution"),
                "language": meta.get("language"),
                "last_updated": meta.get("last_updated") or None,
                "text": doc,
                "score": score,
                "chroma_distance": float(distance) if distance is not None else None,
            }
        )
    return hits


def main() -> None:
    config = load_rag_config()
    index_dir = ROOT / config["index_dir"]
    path = chroma_dir(index_dir)
    count = build_chroma_from_faiss(index_dir, path)
    print(f"copied {count} FAISS vectors into Chroma at {path}")


if __name__ == "__main__":
    main()
