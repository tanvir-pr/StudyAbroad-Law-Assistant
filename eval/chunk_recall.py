"""Compare chunk strategies with span coverage and Recall@k on the golden set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ingestion.sources import ROOT
from rag.chunking import STRATEGIES, chunk_text
from rag.index_faiss import HashEmbedder, build_index, load_corpus, load_rag_config, search

GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
TOKEN_RE = re.compile(r"[a-z0-9áéíóöőúüű]+", re.IGNORECASE)


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def span_hit(reference_answer: str, chunks: list[str], min_overlap: float = 0.45) -> bool:
    ref_tokens = normalize(reference_answer).split()
    if not ref_tokens:
        return False
    # Prefer a long contiguous phrase when the answer has one.
    phrase = " ".join(ref_tokens[:12])
    for chunk in chunks:
        norm = normalize(chunk)
        if phrase and phrase in norm:
            return True
        chunk_tokens = set(norm.split())
        overlap = sum(1 for token in ref_tokens if token in chunk_tokens) / len(ref_tokens)
        if overlap >= min_overlap:
            return True
    return False


def recall_at_k(hits: list[dict], source_id: str, k: int) -> bool:
    return any(hit["source_id"] == source_id for hit in hits[:k])


def evaluate_strategy(
    strategy: str,
    documents: list[dict],
    golden: list[dict],
    embedder,
    *,
    size: int,
    overlap: int,
    ks: list[int],
    work_dir: Path,
) -> dict:
    docs_by_id = {doc["source_id"]: doc for doc in documents}
    span_hits = 0
    chunk_total = 0
    for row in golden:
        doc = docs_by_id.get(row["source_id"])
        if not doc:
            continue
        chunks = chunk_text(doc["text"], strategy, size, overlap)
        chunk_total += len(chunks)
        if span_hit(row["reference_answer"], chunks):
            span_hits += 1

    index_dir = work_dir / strategy
    if index_dir.exists():
        for path in index_dir.glob("*"):
            path.unlink()
    n_chunks = build_index(documents, embedder, index_dir, size, overlap, strategy=strategy)
    max_k = max(ks)
    recalls = {f"recall@{k}": 0 for k in ks}
    examples = []
    for row in golden:
        hits = search(row["question"], embedder, index_dir, k=max_k)
        for k in ks:
            if recall_at_k(hits, row["source_id"], k):
                recalls[f"recall@{k}"] += 1
        examples.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "hit_at_5": recall_at_k(hits, row["source_id"], 5),
                "top_source_ids": [hit["source_id"] for hit in hits[:5]],
            }
        )

    n = len(golden) or 1
    return {
        "strategy": strategy,
        "chunk_count_indexed": n_chunks,
        "avg_chunks_per_gold_source_pass": round(chunk_total / max(len(golden), 1), 2),
        "span_coverage": round(span_hits / n, 4),
        "span_hits": span_hits,
        "n_questions": len(golden),
        **{key: round(value / n, 4) for key, value in recalls.items()},
        **{f"{key}_count": value for key, value in recalls.items()},
        "examples": examples[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunk strategies on the golden set.")
    parser.add_argument(
        "--embedder",
        choices=["hash", "bge-m3"],
        default="hash",
        help="hash is fast for day-6 comparison; bge-m3 rebuilds dense indexes (slow on CPU).",
    )
    parser.add_argument("--ks", default="1,3,5")
    args = parser.parse_args()
    config = load_rag_config()
    size = int(config["chunking"]["size"])
    overlap = int(config["chunking"]["overlap"])
    ks = [int(part) for part in args.ks.split(",") if part.strip()]
    documents = load_corpus()
    golden = load_golden()
    if args.embedder == "bge-m3":
        from rag.embed import BgeM3Embedder

        embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
        embedder_name = config["embedding_model"]
    else:
        embedder = HashEmbedder(dim=64)
        embedder_name = "HashEmbedder(dim=64)"

    work_dir = ROOT / "data" / "index" / "compare"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for strategy in STRATEGIES:
        print(f"evaluating {strategy} ...")
        results.append(
            evaluate_strategy(
                strategy,
                documents,
                golden,
                embedder,
                size=size,
                overlap=overlap,
                ks=ks,
                work_dir=work_dir,
            )
        )

    payload = {
        "embedder": embedder_name,
        "chunk_size": size,
        "overlap": overlap,
        "ks": ks,
        "strategies": [{k: v for k, v in row.items() if k != "examples"} for row in results],
        "example_rows": {row["strategy"]: row["examples"] for row in results},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "chunking_recall.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Chunking comparison (day 6)",
        "",
        f"Embedder: `{embedder_name}`. Size {size}, overlap {overlap}.",
        "",
        "| Strategy | Chunks | Span coverage | Recall@1 | Recall@3 | Recall@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['strategy']} | {row['chunk_count_indexed']} | {row['span_coverage']:.3f} | "
            f"{row['recall@1']:.3f} | {row['recall@3']:.3f} | {row['recall@5']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Span coverage = share of golden answers whose reference text is largely present "
        "inside some chunk of the cited source. Recall@k = cited `source_id` appears in the top-k hits."
    )
    out_md = RESULTS_DIR / "chunking_recall.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
