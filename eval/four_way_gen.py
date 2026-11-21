"""Day 18: start 4-way generation — Base, Base+LoRA, RAG, RAG+LoRA."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from finetune.generators import load_generator
from ingestion.sources import ROOT
from rag.ask import ask, extract_citations
from rag.bm25 import Bm25Index
from rag.index_faiss import load_rag_config
from rag.prompt import build_messages
from rag.rerank import OverlapReranker
from rag.retrieve import retrieve

ARMS = ("base", "base_lora", "rag", "rag_lora")
CITATION_RE = re.compile(r"\[[a-z0-9][a-z0-9-]{0,80}\]\(https://[^)\s]+\)")


def load_eval_questions(path: Path, limit: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def generate_no_rag(question: str, generator, max_new_tokens: int, temperature: float) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are StudyAbroad Law Assistant for Hungary study rules. "
                "Not legal advice. If you cannot cite an official source, refuse."
            ),
        },
        {"role": "user", "content": f"QUESTION: {question}\n\nWrite the answer now."},
    ]
    return generator.generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)


def run_arm(
    arm: str,
    question: str,
    *,
    embedder,
    generators: dict,
    index_dir: Path,
    config: dict,
    bm25,
    reranker,
) -> dict:
    t0 = time.perf_counter()
    hybrid = config.get("hybrid", {})
    gen_cfg = config.get("generation", {})
    max_new = int(gen_cfg.get("max_new_tokens", 256))
    temperature = float(gen_cfg.get("temperature", 0.1))

    if arm in {"base", "base_lora"}:
        generator = generators[arm]
        answer = generate_no_rag(question, generator, max_new, temperature)
        refused = "REFUSE:" in answer
        citations = [] if refused else extract_citations(answer)
        hits = []
        model_name = getattr(generator, "tag", arm)
        mode = getattr(generator, "mode", arm)
    else:
        use_lora = arm == "rag_lora"
        generator = generators["rag_lora" if use_lora else "rag"]
        result = ask(
            question,
            embedder,
            generator,
            index_dir,
            top_k=int(config.get("top_k", 5)),
            min_score=float(gen_cfg.get("min_score", 0.0)),
            prompt_version=str(gen_cfg.get("prompt_version", "v1")),
            model_name=getattr(generator, "mode", arm),
            generation_mode=getattr(generator, "mode", arm),
            retrieve_mode=str(hybrid.get("mode", "hybrid")),
            candidate_k=int(hybrid.get("candidate_k", 30)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            dense_weight=float(hybrid.get("dense_weight", 1.0)),
            bm25_weight=float(hybrid.get("bm25_weight", 1.0)),
            bm25_index=bm25,
            reranker=reranker,
            rerank_candidates=int(config.get("rerank", {}).get("candidates", 20)),
            max_new_tokens=max_new,
            temperature=temperature,
        )
        answer = result.answer
        refused = result.refused
        citations = result.citations
        hits = [{"source_id": h.get("source_id"), "score": h.get("score")} for h in result.hits]
        model_name = result.model_name
        mode = result.generation_mode

    ms = int((time.perf_counter() - t0) * 1000)
    return {
        "arm": arm,
        "answer": answer,
        "refused": refused,
        "has_citation": bool(citations) or bool(CITATION_RE.search(answer)),
        "citations": citations,
        "hits": hits,
        "latency_ms": ms,
        "model_name": model_name,
        "generation_mode": mode,
    }


def summarize(rows: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for row in rows:
        for arm, payload in row["arms"].items():
            by_arm[arm].append(payload)
    summary = {"n": len(rows), "arms": {}}
    for arm, items in by_arm.items():
        n = len(items) or 1
        latencies = sorted(item["latency_ms"] for item in items)
        summary["arms"][arm] = {
            "refuse_rate": round(sum(1 for i in items if i["refused"]) / n, 3),
            "citation_rate": round(sum(1 for i in items if i["has_citation"]) / n, 3),
            "p50_ms": latencies[len(latencies) // 2] if latencies else 0,
            "p95_ms": latencies[max(int(len(latencies) * 0.95) - 1, 0)] if latencies else 0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "eval" / "golden_set.jsonl",
        help="Eval questions (golden is OK for generation eval, not for training).",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--echo", action="store_true", default=True)
    parser.add_argument("--no-echo", action="store_true", help="Use real Qwen / LoRA adapters.")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "finetune" / "runs" / "qlora_r16_seed42" / "adapter",
    )
    parser.add_argument("--tag", default="day18")
    args = parser.parse_args()
    echo = not args.no_echo

    config = load_rag_config()
    index_dir = ROOT / config["index_dir"]
    questions = load_eval_questions(args.questions, args.limit)

    if echo:
        from rag.index_faiss import HashEmbedder

        # Prefer real BGE when index exists so RAG arms retrieve correctly.
        try:
            from rag.embed import BgeM3Embedder

            embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))
        except Exception:
            embedder = HashEmbedder(dim=32)
    else:
        from rag.embed import BgeM3Embedder

        embedder = BgeM3Embedder(config["embedding_model"], batch_size=int(config.get("batch_size", 8)))

    generators = {
        "base": load_generator(use_lora=False, echo=echo, adapter_dir=None)[0],
        "base_lora": load_generator(use_lora=True, echo=echo, adapter_dir=args.adapter)[0],
        "rag": load_generator(use_lora=False, echo=echo, adapter_dir=None)[0],
        "rag_lora": load_generator(use_lora=True, echo=echo, adapter_dir=args.adapter)[0],
    }
    bm25 = Bm25Index.from_index_dir(index_dir) if (index_dir / "chunks.jsonl").exists() else None
    reranker = OverlapReranker() if config.get("rerank", {}).get("enabled", True) else None

    rows = []
    for row in questions:
        q = row["question"]
        arms = {}
        for arm in ARMS:
            arms[arm] = run_arm(
                arm,
                q,
                embedder=embedder,
                generators=generators,
                index_dir=index_dir,
                config=config,
                bm25=bm25,
                reranker=reranker,
            )
        rows.append(
            {
                "id": row.get("id"),
                "question": q,
                "source_id": row.get("source_id"),
                "arms": arms,
            }
        )

    summary = summarize(rows)
    out = {
        "tag": args.tag,
        "echo": echo,
        "n": len(rows),
        "summary": summary,
        "rows": rows,
    }
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"four_way_{args.tag}.json"
    md_path = out_dir / f"four_way_{args.tag}.md"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# 4-way generation — `{args.tag}`",
        "",
        f"- Questions: **{len(rows)}**",
        f"- Echo stand-in: **{echo}**",
        "",
        "| Arm | refuse_rate | citation_rate | p50_ms | p95_ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        s = summary["arms"][arm]
        lines.append(
            f"| {arm} | {s['refuse_rate']:.3f} | {s['citation_rate']:.3f} | {s['p50_ms']} | {s['p95_ms']} |"
        )
    lines.extend(
        [
            "",
            "Arms: `base` (no RAG), `base_lora` (no RAG + LoRA), "
            "`rag` (hybrid+rerank), `rag_lora` (hybrid+rerank + LoRA).",
            "",
            "Day 19 finishes scoring (faithfulness, correctness, style, human 30).",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tag": args.tag, "n": len(rows), "summary": summary}, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
