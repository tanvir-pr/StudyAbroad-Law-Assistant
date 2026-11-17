"""Retrieve passages and answer one question with an inline citation."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from ingestion.sources import ROOT
from rag.bm25 import Bm25Index
from rag.generate import EchoGenerator, GenerationConfig, QwenGenerator
from rag.index_faiss import load_rag_config
from rag.prompt import build_messages, ensure_prompt_file
from rag.retrieve import retrieve

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9-]{0,80})\]\((https://[^)\s]+)\)")
REFUSE_PREFIX = "REFUSE:"


@dataclass
class AskResult:
    question: str
    answer: str
    refused: bool
    citations: list[dict]
    hits: list[dict]
    prompt_version: str
    model_name: str
    generation_mode: str


def filter_hits(hits: list[dict], min_score: float) -> list[dict]:
    return [hit for hit in hits if float(hit.get("score", 0)) >= min_score]


def extract_citations(answer: str) -> list[dict]:
    return [{"source_id": match.group(1), "source_url": match.group(2)} for match in CITATION_RE.finditer(answer)]


def citations_match_hits(citations: list[dict], hits: list[dict]) -> bool:
    allowed = {(hit["source_id"], hit["source_url"]) for hit in hits}
    return all((cite["source_id"], cite["source_url"]) in allowed for cite in citations)


def best_hit_for_question(question: str, hits: list[dict]) -> dict:
    tokens = [token for token in question.lower().split() if len(token) > 2]
    if not tokens:
        return hits[0]

    def overlap(hit: dict) -> tuple[float, float]:
        text = hit["text"].lower()
        matched = sum(1 for token in tokens if token in text)
        return (matched / len(tokens), float(hit.get("score", 0)))

    return max(hits, key=overlap)


def ask(
    question: str,
    embedder,
    generator,
    index_dir,
    *,
    top_k: int = 5,
    min_score: float = 0.0,
    prompt_version: str = "v1",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    generation_mode: str = "unknown",
    retrieve_mode: str = "hybrid",
    country: str | None = None,
    institution: str | None = None,
    candidate_k: int = 30,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    bm25_index: Bm25Index | None = None,
    reranker=None,
    rerank_candidates: int = 20,
) -> AskResult:
    hits = retrieve(
        question,
        embedder,
        Path(index_dir),
        k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        mode=retrieve_mode,
        country=country,
        institution=institution,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        bm25_index=bm25_index,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
    )
    if retrieve_mode == "dense" and min_score > 0:
        hits = filter_hits(hits, min_score)
    if not hits:
        answer = (
            f"{REFUSE_PREFIX} I cannot cite an official source from the locked allow-list "
            "for this question. Check OIF, Tempus Public Foundation, or the university."
        )
        return AskResult(
            question=question,
            answer=answer,
            refused=True,
            citations=[],
            hits=[],
            prompt_version=prompt_version,
            model_name=model_name,
            generation_mode=generation_mode,
        )
    messages = build_messages(question, hits, version=prompt_version)
    answer = generator.generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    refused = answer.startswith(REFUSE_PREFIX)
    citations = [] if refused else extract_citations(answer)
    if refused or not citations or not citations_match_hits(citations, hits):
        top = best_hit_for_question(question, hits)
        snippet = " ".join(top["text"].split())[:280]
        answer = (
            f"{snippet} [{top['source_id']}]({top['source_url']}). Not legal advice."
        )
        citations = extract_citations(answer)
        refused = False
    return AskResult(
        question=question,
        answer=answer,
        refused=refused,
        citations=citations,
        hits=hits,
        prompt_version=prompt_version,
        model_name=model_name,
        generation_mode=generation_mode,
    )


def format_result(result: AskResult) -> str:
    lines = [
        f"Question: {result.question}",
        f"Prompt: {result.prompt_version}",
        f"Model: {result.model_name} ({result.generation_mode})",
        "",
        result.answer,
        "",
    ]
    if result.hits:
        lines.append("Retrieved:")
        for hit in result.hits:
            updated = hit.get("last_updated") or "not stated"
            institution = hit.get("institution") or "?"
            country = hit.get("country") or "?"
            lines.append(
                f"- {hit['score']:.3f} {hit['source_id']} [{country}/{institution}] "
                f"| last_updated={updated} | {hit['source_url']}"
            )
    lines.append("")
    lines.append("Not legal advice. Read DISCLAIMER.md.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer one question with a cited official source.")
    parser.add_argument("question", nargs="?", help="Question to answer.")
    parser.add_argument("--echo", action="store_true", help="Use the fake generator (no LLM download).")
    parser.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default=None)
    parser.add_argument("--country", default=None, help="Filter retrieved chunks by country, e.g. HU.")
    parser.add_argument("--institution", default=None, help="Filter by institution, e.g. OIF, ELTE, NJE.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable the cross-encoder reranker.")
    args = parser.parse_args()
    if not args.question:
        parser.error("Pass a question, for example: python -m rag.ask \"How many hours may a student work?\"")

    config = load_rag_config()
    ensure_prompt_file(config["generation"]["prompt_version"])
    hybrid = config.get("hybrid", {})
    rerank_cfg = config.get("rerank", {})
    gen_cfg = GenerationConfig(
        model_name=config["generation"]["model"],
        load_in_4bit=bool(config["generation"].get("load_in_4bit", True)),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
    )
    from rag.embed import BgeM3Embedder

    embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
    if args.echo:
        generator = EchoGenerator()
        mode = "echo"
    else:
        generator = QwenGenerator(gen_cfg)
        mode = generator.mode
    index_dir = ROOT / config["index_dir"]
    bm25_index = Bm25Index.from_index_dir(index_dir)
    reranker = None
    if rerank_cfg.get("enabled", True) and not args.no_rerank:
        if args.echo:
            from rag.rerank import OverlapReranker

            reranker = OverlapReranker()
        else:
            from rag.rerank import BgeReranker

            reranker = BgeReranker(rerank_cfg.get("model", "BAAI/bge-reranker-v2-m3"))
    result = ask(
        args.question,
        embedder,
        generator,
        index_dir,
        top_k=int(config["top_k"]),
        min_score=float(config["generation"].get("min_score", 0.0)),
        prompt_version=config["generation"]["prompt_version"],
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        model_name=gen_cfg.model_name,
        generation_mode=mode,
        retrieve_mode=args.mode or hybrid.get("mode", "hybrid"),
        country=args.country,
        institution=args.institution,
        candidate_k=int(hybrid.get("candidate_k", 30)),
        rrf_k=int(hybrid.get("rrf_k", 60)),
        dense_weight=float(hybrid.get("dense_weight", 1.0)),
        bm25_weight=float(hybrid.get("bm25_weight", 1.0)),
        bm25_index=bm25_index,
        reranker=reranker,
        rerank_candidates=int(rerank_cfg.get("candidates", 20)),
    )
    print(format_result(result))


if __name__ == "__main__":
    main()
