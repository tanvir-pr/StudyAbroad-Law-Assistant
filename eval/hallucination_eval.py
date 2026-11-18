"""Day 11: refusal / hallucination rate on unanswerable + out-of-scope questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.sources import ROOT
from rag.ask import ask
from rag.bm25 import Bm25Index
from rag.embed import BgeM3Embedder
from rag.generate import EchoGenerator, GenerationConfig, QwenGenerator
from rag.index_faiss import load_rag_config
from rag.rerank import OverlapReranker


def load_cases(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_generator(mode: str, model_name: str, config: dict):
    if mode == "echo":
        return EchoGenerator(), "echo", "echo"
    gen_cfg = GenerationConfig(
        model_name=model_name or config["generation"]["model"],
        load_in_4bit=bool(config["generation"].get("load_in_4bit", True)),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
    )
    gen = QwenGenerator(gen_cfg)
    return gen, gen_cfg.model_name, gen.mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "eval" / "unanswerable.jsonl",
    )
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--tag", default="day11")
    parser.add_argument("--mode", choices=["echo", "qwen"], default="echo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = parser.parse_args()

    cfg = load_rag_config()
    index_dir = Path(args.index_dir) if args.index_dir else ROOT / cfg["index_dir"]
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    embedder = BgeM3Embedder(cfg["embedding_model"], batch_size=int(cfg.get("batch_size", 8)))
    bm25 = Bm25Index.from_index_dir(index_dir)
    reranker = OverlapReranker() if cfg.get("rerank", {}).get("enabled") else None
    generator, model_name, generation_mode = build_generator(
        args.mode, args.model or cfg["generation"]["model"], cfg
    )
    hybrid = cfg.get("hybrid", {})
    rerank_cfg = cfg.get("rerank", {})

    rows = []
    refused_n = 0
    hallucinated_n = 0
    for case in cases:
        result = ask(
            case["question"],
            embedder,
            generator,
            index_dir,
            top_k=int(cfg.get("top_k", 5)),
            min_score=float(cfg.get("generation", {}).get("min_score", 0.0)),
            prompt_version=str(cfg.get("generation", {}).get("prompt_version", "v1")),
            model_name=model_name,
            generation_mode=generation_mode,
            retrieve_mode=str(hybrid.get("mode", "hybrid")),
            candidate_k=int(hybrid.get("candidate_k", 30)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            dense_weight=float(hybrid.get("dense_weight", 1.0)),
            bm25_weight=float(hybrid.get("bm25_weight", 1.0)),
            bm25_index=bm25,
            reranker=reranker,
            rerank_candidates=int(rerank_cfg.get("candidates", 20)),
        )
        expected_refuse = case.get("expected", "refuse") == "refuse"
        hallucinated = expected_refuse and not result.refused
        if result.refused:
            refused_n += 1
        if hallucinated:
            hallucinated_n += 1
        rows.append(
            {
                "id": case["id"],
                "kind": case.get("kind"),
                "question": case["question"],
                "expected": case.get("expected"),
                "refused": result.refused,
                "hallucinated": hallucinated,
                "answer": result.answer,
                "citations": result.citations,
            }
        )

    n = len(rows) or 1
    refuse_rate = refused_n / n
    hallucination_rate = hallucinated_n / n
    summary = {
        "tag": args.tag,
        "n": len(rows),
        "refused": refused_n,
        "hallucinated": hallucinated_n,
        "refuse_rate": round(refuse_rate, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "mode": generation_mode,
        "model": model_name,
        "rows": rows,
    }

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"hallucination_{args.tag}.json"
    md_path = out_dir / f"hallucination_{args.tag}.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Hallucination / refusal — `{args.tag}`",
        "",
        f"- Cases: **{len(rows)}** (unanswerable / out-of-scope)",
        f"- Refuse rate: **{refuse_rate:.3f}** ({refused_n}/{len(rows)})",
        f"- Hallucination rate (answered when should refuse): **{hallucination_rate:.3f}** ({hallucinated_n}/{len(rows)})",
        f"- Generator: `{model_name}` ({generation_mode})",
        "",
        "| id | kind | refused | hallucinated |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['kind']} | {row['refused']} | {row['hallucinated']} |"
        )
    if hallucinated_n:
        lines.extend(["", "## Hallucinated answers", ""])
        for row in rows:
            if row["hallucinated"]:
                lines.append(f"- **{row['id']}**: {row['question']}")
                lines.append(f"  - Answer: {row['answer'][:240]}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tag", "n", "refuse_rate", "hallucination_rate")}, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
