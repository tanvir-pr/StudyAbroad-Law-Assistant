"""Build and search a FAISS inner-product index over fixed chunks."""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import yaml

from ingestion.sources import ROOT
from rag.chunking import fixed_chunks

INDEX_NAME = "chunks.faiss"
META_NAME = "chunks.jsonl"


def load_rag_config(path: Path | None = None) -> dict:
    path = path or (ROOT / "configs" / "rag.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_corpus(raw_dir: Path | None = None) -> list[dict]:
    raw_dir = raw_dir or (ROOT / "data" / "raw")
    records = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    documents = []
    for record in records:
        text_path = record.get("text_path")
        if not text_path or not record.get("text_chars"):
            continue
        path = Path(text_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            path = raw_dir / Path(text_path).name
        documents.append(
            {
                "source_id": record["source_id"],
                "source_url": record["source_url"],
                "title": record["title"],
                "country": record["country"],
                "institution": record["institution"],
                "language": record["language"],
                "last_updated": record.get("last_updated"),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return documents


def build_index(documents: list[dict], embedder, index_dir: Path, size: int, overlap: int) -> int:
    rows: list[dict] = []
    texts: list[str] = []
    for document in documents:
        for piece in fixed_chunks(document["text"], size, overlap):
            rows.append(
                {
                    "source_id": document["source_id"],
                    "source_url": document["source_url"],
                    "title": document["title"],
                    "country": document["country"],
                    "institution": document["institution"],
                    "language": document["language"],
                    "last_updated": document["last_updated"],
                    "text": piece,
                }
            )
            texts.append(piece)
    if not texts:
        raise ValueError("No chunks to index")
    vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    if vectors.shape != (len(texts), embedder.dim):
        raise ValueError(f"Expected embeddings {(len(texts), embedder.dim)}, got {vectors.shape}")
    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_NAME))
    meta_path = index_dir / META_NAME
    with meta_path.open("w", encoding="utf-8") as handle:
        for row_id, row in enumerate(rows):
            row["chunk_id"] = row_id
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def search(query: str, embedder, index_dir: Path, k: int = 5) -> list[dict]:
    index = faiss.read_index(str(index_dir / INDEX_NAME))
    rows = [
        json.loads(line)
        for line in (index_dir / META_NAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vector = np.asarray(embedder.embed_query(query), dtype=np.float32).reshape(1, -1)
    scores, ids = index.search(vector, min(k, len(rows)))
    hits = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0:
            continue
        hit = dict(rows[int(row_id)])
        hit["score"] = float(score)
        hits.append(hit)
    return hits


class HashEmbedder:
    """Bag-of-tokens embedder for tests. Shared words score higher."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or search the FAISS chunk index.")
    parser.add_argument("--query", help="Search an existing index instead of rebuilding it.")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    config = load_rag_config()
    from rag.embed import BgeM3Embedder

    embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
    index_dir = ROOT / config["index_dir"]
    if args.query:
        hits = search(args.query, embedder, index_dir, k=args.k or int(config["top_k"]))
        for hit in hits:
            print(f"{hit['score']:.3f}\t{hit['source_id']}\t{hit['source_url']}")
            print(hit["text"][:500].replace("\n", " "))
            print()
        return
    documents = load_corpus()
    count = build_index(
        documents,
        embedder,
        index_dir,
        size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"],
    )
    print(f"indexed {count} chunks from {len(documents)} documents into {index_dir}")


if __name__ == "__main__":
    main()
