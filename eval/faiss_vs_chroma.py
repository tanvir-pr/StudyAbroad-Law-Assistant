"""Compare FAISS vs Chroma dense recall on the golden set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.chunk_recall import load_golden, recall_at_k
from ingestion.sources import ROOT
from rag.embed import BgeM3Embedder
from rag.index_chroma import build_chroma_from_faiss, chroma_dir, search_chroma
from rag.index_faiss import HashEmbedder, load_rag_config, search as faiss_search

RESULTS_DIR = ROOT / "eval" / "results"


def score_backend(name: str, search_fn, golden: list[dict], ks=(1, 3, 5)) -> dict:
    counts = {f"recall@{k}": 0 for k in ks}
    for row in golden:
        hits = search_fn(row["question"], k=max(ks))
        for k in ks:
            if recall_at_k(hits, row["source_id"], k):
                counts[f"recall@{k}"] += 1
    n = len(golden) or 1
    return {
        "backend": name,
        "n_questions": len(golden),
        **{key: round(value / n, 4) for key, value in counts.items()},
        **{f"{key}_count": value for key, value in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS vs Chroma recall on golden questions.")
    parser.add_argument("--embedder", choices=["bge-m3", "hash"], default="bge-m3")
    parser.add_argument("--rebuild-chroma", action="store_true")
    args = parser.parse_args()
    config = load_rag_config()
    index_dir = ROOT / config["index_dir"]
    chroma_path = chroma_dir(index_dir)
    if args.rebuild_chroma or not chroma_path.exists() or not any(chroma_path.iterdir()):
        print(f"building Chroma from FAISS into {chroma_path} ...")
        build_chroma_from_faiss(index_dir, chroma_path)
    if args.embedder == "hash":
        embedder = HashEmbedder(dim=64)
        embedder_name = "HashEmbedder(dim=64)"
    else:
        embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
        embedder_name = config["embedding_model"]
    golden = load_golden()

    def faiss_fn(question: str, k: int):
        return faiss_search(question, embedder, index_dir, k=k)

    def chroma_fn(question: str, k: int):
        return search_chroma(question, embedder, chroma_path, k=k)

    rows = [
        score_backend("faiss", faiss_fn, golden),
        score_backend("chroma", chroma_fn, golden),
    ]
    payload = {
        "embedder": embedder_name,
        "index_dir": str(index_dir),
        "chroma_dir": str(chroma_path),
        "note": "Chroma vectors are copied from the FAISS IndexFlatIP (same embeddings).",
        "backends": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "faiss_vs_chroma.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# FAISS vs Chroma (day 8)",
        "",
        f"Embedder: `{embedder_name}`. Same vectors; Chroma is a copy of the FAISS index.",
        "",
        "| Backend | Recall@1 | Recall@3 | Recall@5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['recall@1']:.3f} | {row['recall@3']:.3f} | {row['recall@5']:.3f} |"
        )
    lines.append("")
    out_md = RESULTS_DIR / "faiss_vs_chroma.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
