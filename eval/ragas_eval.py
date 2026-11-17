"""RAGAS-aligned local metrics (no paid LLM judge).

Metrics mirror the RAGAS scorecard used in the plan:
- context_precision / context_recall
- faithfulness
- answer_relevancy

Day-9 default uses the project's retrieval stack plus extractive/echo answers so the
table can be produced on CPU. Optional --generator qwen uses the local instruct model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval.chunk_recall import load_golden
from ingestion.sources import ROOT
from rag.ask import ask
from rag.bm25 import Bm25Index
from rag.embed import BgeM3Embedder
from rag.generate import ExtractiveGenerator, GenerationConfig, QwenGenerator
from rag.index_faiss import load_rag_config
from rag.rerank import BgeReranker, OverlapReranker

RESULTS_DIR = ROOT / "eval" / "results"
TOKEN_RE = re.compile(r"[a-z0-9áéíóöőúüű]+", re.IGNORECASE)
CITATION_STRIP = re.compile(r"\[[^\]]+\]\([^)]+\)")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def strip_citations(answer: str) -> str:
    return CITATION_STRIP.sub(" ", answer)


def context_precision(hits: list[dict], gold_source_id: str) -> float:
    if not hits:
        return 0.0
    return sum(1 for hit in hits if hit.get("source_id") == gold_source_id) / len(hits)


def context_recall(hits: list[dict], gold_source_id: str) -> float:
    return 1.0 if any(hit.get("source_id") == gold_source_id for hit in hits) else 0.0


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer content tokens that appear in the retrieved contexts."""
    answer_tokens = tokenize(strip_citations(answer))
    # Drop boilerplate that is always appended.
    stop = {"not", "legal", "advice", "refuse", "see", "the", "retrieved", "official", "passage"}
    answer_tokens = [token for token in answer_tokens if token not in stop and len(token) > 2]
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize("\n".join(contexts)))
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / len(answer_tokens)


def answer_relevancy(question: str, answer: str, embedder) -> float:
    q = np.asarray(embedder.embed_query(question), dtype=np.float32)
    a = np.asarray(embedder.embed_query(strip_citations(answer)), dtype=np.float32)
    denom = float(np.linalg.norm(q) * np.linalg.norm(a))
    if denom == 0:
        return 0.0
    return float(np.dot(q, a) / denom)


def evaluate_rows(
    golden: list[dict],
    *,
    embedder,
    generator,
    index_dir: Path,
    config: dict,
    bm25: Bm25Index,
    reranker,
    limit: int | None = None,
) -> list[dict]:
    hybrid = config.get("hybrid", {})
    rerank_cfg = config.get("rerank", {})
    rows = golden if limit is None else golden[:limit]
    scored = []
    for row in rows:
        result = ask(
            row["question"],
            embedder,
            generator,
            index_dir,
            top_k=int(config["top_k"]),
            prompt_version=config["generation"]["prompt_version"],
            retrieve_mode=hybrid.get("mode", "hybrid"),
            candidate_k=int(hybrid.get("candidate_k", 30)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            dense_weight=float(hybrid.get("dense_weight", 1.0)),
            bm25_weight=float(hybrid.get("bm25_weight", 1.0)),
            bm25_index=bm25,
            reranker=reranker,
            rerank_candidates=int(rerank_cfg.get("candidates", 20)),
            model_name=getattr(generator, "config", GenerationConfig()).model_name
            if hasattr(generator, "config")
            else "echo",
            generation_mode=getattr(generator, "mode", "echo"),
        )
        contexts = [hit["text"] for hit in result.hits]
        scored.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "refused": result.refused,
                "context_precision": round(context_precision(result.hits, row["source_id"]), 4),
                "context_recall": round(context_recall(result.hits, row["source_id"]), 4),
                "faithfulness": round(faithfulness(result.answer, contexts), 4),
                "answer_relevancy": round(
                    answer_relevancy(row["question"], result.answer, embedder), 4
                ),
                "answer": result.answer,
                "retrieved_source_ids": [hit["source_id"] for hit in result.hits],
            }
        )
    return scored


def summarize(scored: list[dict]) -> dict:
    keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    summary = {key: round(sum(row[key] for row in scored) / max(len(scored), 1), 4) for key in keys}
    summary["n_questions"] = len(scored)
    summary["refuse_rate"] = round(sum(1 for row in scored if row["refused"]) / max(len(scored), 1), 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-9 RAGAS-aligned score table.")
    parser.add_argument("--generator", choices=["extractive", "qwen"], default="extractive")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N golden rows.")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--tag", default="day9", help="Output tag, e.g. day9 or v0.2")
    args = parser.parse_args()

    config = load_rag_config()
    index_dir = ROOT / config["index_dir"]
    golden = load_golden()
    embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
    bm25 = Bm25Index.from_index_dir(index_dir)
    if args.no_rerank:
        reranker = None
        reranker_name = "none"
    elif args.generator == "extractive":
        reranker = OverlapReranker()
        reranker_name = "OverlapReranker"
    else:
        reranker = BgeReranker(config.get("rerank", {}).get("model", "BAAI/bge-reranker-v2-m3"))
        reranker_name = config.get("rerank", {}).get("model", "BAAI/bge-reranker-v2-m3")

    if args.generator == "extractive":
        generator = ExtractiveGenerator()
        generator_name = "extractive-quote"
    else:
        gen_cfg = GenerationConfig(
            model_name=config["generation"]["model"],
            load_in_4bit=bool(config["generation"].get("load_in_4bit", True)),
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            temperature=float(config["generation"]["temperature"]),
        )
        generator = QwenGenerator(gen_cfg)
        generator_name = f"{gen_cfg.model_name} ({generator.mode})"

    scored = evaluate_rows(
        golden,
        embedder=embedder,
        generator=generator,
        index_dir=index_dir,
        config=config,
        bm25=bm25,
        reranker=reranker,
        limit=args.limit,
    )
    summary = summarize(scored)
    payload = {
        "framework": "RAGAS-aligned local metrics",
        "generator": generator_name,
        "reranker": reranker_name,
        "embedder": config["embedding_model"],
        "summary": summary,
        "rows": scored,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"ragas_{args.tag}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    title = "RAGAS-aligned scores (v0.2 freeze)" if args.tag == "v0.2" else f"RAGAS-aligned scores ({args.tag})"
    lines = [
        f"# {title}",
        "",
        f"Generator: `{generator_name}`. Reranker: `{reranker_name}`.",
        "Local judges (no paid API): context precision/recall use gold `source_id`; "
        "faithfulness is context-token support; answer relevancy is bge-m3 cosine.",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| context_precision | {summary['context_precision']:.3f} |",
        f"| context_recall | {summary['context_recall']:.3f} |",
        f"| faithfulness | {summary['faithfulness']:.3f} |",
        f"| answer_relevancy | {summary['answer_relevancy']:.3f} |",
        f"| refuse_rate | {summary['refuse_rate']:.3f} |",
        "",
        f"Questions scored: {summary['n_questions']}.",
        "",
    ]
    out_md = RESULTS_DIR / f"ragas_{args.tag}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
