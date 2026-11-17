"""Compare hybrid retrieval with and without the cross-encoder reranker."""

from __future__ import annotations

import argparse
import json

from eval.chunk_recall import load_golden, recall_at_k
from ingestion.sources import ROOT
from rag.bm25 import Bm25Index
from rag.embed import BgeM3Embedder
from rag.index_faiss import load_rag_config
from rag.rerank import BgeReranker, OverlapReranker
from rag.retrieve import retrieve

RESULTS_DIR = ROOT / "eval" / "results"


def score(mode_name: str, golden, embedder, index_dir, config, bm25, reranker) -> dict:
    hybrid = config.get("hybrid", {})
    rerank_cfg = config.get("rerank", {})
    counts = {"recall@1": 0, "recall@3": 0, "recall@5": 0}
    for row in golden:
        hits = retrieve(
            row["question"],
            embedder,
            index_dir,
            k=5,
            candidate_k=int(hybrid.get("candidate_k", 30)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            mode="hybrid",
            dense_weight=float(hybrid.get("dense_weight", 1.0)),
            bm25_weight=float(hybrid.get("bm25_weight", 1.0)),
            bm25_index=bm25,
            reranker=reranker,
            rerank_candidates=int(rerank_cfg.get("candidates", 20)),
        )
        for k in (1, 3, 5):
            if recall_at_k(hits, row["source_id"], k):
                counts[f"recall@{k}"] += 1
    n = len(golden) or 1
    return {
        "mode": mode_name,
        "n_questions": len(golden),
        **{key: round(value / n, 4) for key, value in counts.items()},
        **{f"{key}_count": value for key, value in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid vs hybrid+rerank recall.")
    parser.add_argument("--reranker", choices=["bge", "overlap"], default="bge")
    args = parser.parse_args()
    config = load_rag_config()
    index_dir = ROOT / config["index_dir"]
    embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
    bm25 = Bm25Index.from_index_dir(index_dir)
    golden = load_golden()
    if args.reranker == "bge":
        reranker = BgeReranker(config.get("rerank", {}).get("model", "BAAI/bge-reranker-v2-m3"))
        reranker_name = reranker.model_name
    else:
        reranker = OverlapReranker()
        reranker_name = "OverlapReranker"
    rows = [
        score("hybrid", golden, embedder, index_dir, config, bm25, None),
        score("hybrid+rerank", golden, embedder, index_dir, config, bm25, reranker),
    ]
    payload = {
        "embedder": config["embedding_model"],
        "reranker": reranker_name,
        "modes": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "rerank_recall.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Reranker comparison (day 8)",
        "",
        f"Reranker: `{reranker_name}`.",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['recall@1']:.3f} | {row['recall@3']:.3f} | {row['recall@5']:.3f} |"
        )
    lines.append("")
    out_md = RESULTS_DIR / "rerank_recall.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
