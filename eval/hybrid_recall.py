"""Compare dense vs BM25 vs hybrid Recall@k on the golden set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.chunk_recall import load_golden, recall_at_k
from ingestion.sources import ROOT
from rag.bm25 import Bm25Index
from rag.embed import BgeM3Embedder
from rag.index_faiss import HashEmbedder, load_rag_config
from rag.retrieve import retrieve

RESULTS_DIR = ROOT / "eval" / "results"


def evaluate_mode(mode: str, golden: list[dict], embedder, index_dir: Path, config: dict, bm25: Bm25Index) -> dict:
    ks = [1, 3, 5]
    counts = {f"recall@{k}": 0 for k in ks}
    hybrid = config.get("hybrid", {})
    for row in golden:
        hits = retrieve(
            row["question"],
            embedder,
            index_dir,
            k=5,
            candidate_k=int(hybrid.get("candidate_k", 30)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            mode=mode,
            dense_weight=float(hybrid.get("dense_weight", 1.0)),
            bm25_weight=float(hybrid.get("bm25_weight", 1.0)),
            bm25_index=bm25,
        )
        for k in ks:
            if recall_at_k(hits, row["source_id"], k):
                counts[f"recall@{k}"] += 1
    n = len(golden) or 1
    return {
        "mode": mode,
        "n_questions": len(golden),
        **{key: round(value / n, 4) for key, value in counts.items()},
        **{f"{key}_count": value for key, value in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense vs BM25 vs hybrid recall on golden questions.")
    parser.add_argument("--embedder", choices=["hash", "bge-m3"], default="bge-m3")
    args = parser.parse_args()
    config = load_rag_config()
    index_dir = ROOT / config["index_dir"]
    golden = load_golden()
    if args.embedder == "hash":
        # Hash over the real chunk texts still needs the FAISS index built with HashEmbedder.
        # For day-7 default we score the existing bge-m3 index; hash mode is for tiny fixtures.
        embedder = HashEmbedder(dim=64)
        embedder_name = "HashEmbedder(dim=64)"
    else:
        embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
        embedder_name = config["embedding_model"]
    bm25 = Bm25Index.from_index_dir(index_dir)
    modes = ["dense", "bm25", "hybrid"]
    rows = [evaluate_mode(mode, golden, embedder, index_dir, config, bm25) for mode in modes]
    payload = {
        "embedder": embedder_name,
        "index_dir": str(index_dir),
        "chunking": config["chunking"],
        "hybrid": config.get("hybrid", {}),
        "modes": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "hybrid_recall.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Hybrid retrieval comparison (day 7)",
        "",
        f"Embedder/index: `{embedder_name}` over `{index_dir}`.",
        f"Chunking locked: `{config['chunking'].get('strategy')}` "
        f"{config['chunking'].get('size')}/{config['chunking'].get('overlap')}.",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['recall@1']:.3f} | {row['recall@3']:.3f} | {row['recall@5']:.3f} |"
        )
    lines.append("")
    out_md = RESULTS_DIR / "hybrid_recall.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
