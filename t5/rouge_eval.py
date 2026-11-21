"""Day 20: ROUGE between model/extractive summaries and 20 human 5-line references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ingestion.sources import ROOT

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def load_refs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extractive_summary(text: str, n_lines: int = 5) -> str:
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 20]
    if not sentences:
        sentences = [text.strip()]
    picked = sentences[:n_lines]
    while len(picked) < n_lines:
        picked.append(picked[-1])
    return "\n".join(f"{i + 1}. {sent}" for i, sent in enumerate(picked[:n_lines]))


def mt5_summarize(text: str, model_dir: Path | None) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    name = str(model_dir) if model_dir and model_dir.exists() else "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def rouge_scores(pred: str, ref: str) -> dict:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(ref, pred)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except Exception:
        # Tiny fallback if rouge-score is not installed.
        def f1(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            overlap = len(a & b)
            p = overlap / len(a)
            r = overlap / len(b)
            return 0.0 if p + r == 0 else 2 * p * r / (p + r)

        pred_t = set(pred.lower().split())
        ref_t = set(ref.lower().split())
        return {"rouge1": round(f1(pred_t, ref_t), 4), "rouge2": 0.0, "rougeL": round(f1(pred_t, ref_t), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refs",
        type=Path,
        default=ROOT / "t5" / "human_refs_20.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["extractive", "mt5"],
        default="extractive",
        help="extractive is CPU-fast; mt5 needs model download/warmup weights",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "t5" / "runs" / "mt5_small_warmup" / "model",
    )
    parser.add_argument("--tag", default="day20")
    args = parser.parse_args()

    refs = load_refs(args.refs)
    rows = []
    for row in refs:
        if args.mode == "extractive":
            pred = extractive_summary(row["document_excerpt"], n_lines=5)
        else:
            pred = mt5_summarize(row["document_excerpt"], args.model_dir if args.model_dir.exists() else None)
        scores = rouge_scores(pred, row["reference_summary"])
        rows.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "language": row.get("language"),
                "prediction": pred,
                "reference": row["reference_summary"],
                **scores,
            }
        )

    n = len(rows) or 1
    summary = {
        "tag": args.tag,
        "mode": args.mode,
        "n": len(rows),
        "rouge1": round(sum(r["rouge1"] for r in rows) / n, 4),
        "rouge2": round(sum(r["rouge2"] for r in rows) / n, 4),
        "rougeL": round(sum(r["rougeL"] for r in rows) / n, 4),
        "rows": rows,
    }
    out_dir = ROOT / "t5" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"rouge_{args.tag}.json"
    md_path = out_dir / f"rouge_{args.tag}.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# ROUGE — `{args.tag}`",
        "",
        f"- References: **{len(rows)}** five-line human summaries (`t5/human_refs_20.jsonl`)",
        f"- Mode: **{args.mode}**",
        f"- ROUGE-1: **{summary['rouge1']:.4f}**",
        f"- ROUGE-2: **{summary['rouge2']:.4f}**",
        f"- ROUGE-L: **{summary['rougeL']:.4f}**",
        "",
        "| id | source_id | R1 | R2 | RL |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['source_id']} | {row['rouge1']:.3f} | {row['rouge2']:.3f} | {row['rougeL']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tag", "mode", "n", "rouge1", "rouge2", "rougeL")}, indent=2))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
