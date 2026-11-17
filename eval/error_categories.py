"""Categorize weak/wrong day-10 retrieval-answer examples from a RAGAS JSON run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.sources import ROOT

RESULTS = ROOT / "eval" / "results"

CATEGORIES = {
    "wrong_source_top": "Gold source missing from top-k (context_recall = 0)",
    "diluted_context": "Gold source present but context_precision < 0.4",
    "low_relevancy": "Answer relevancy < 0.35 while context_recall = 1",
    "low_faithfulness": "Faithfulness < 0.7",
}


def categorize(row: dict) -> str | None:
    if row["context_recall"] < 1:
        return "wrong_source_top"
    if row["context_precision"] < 0.4:
        return "diluted_context"
    if row["faithfulness"] < 0.7:
        return "low_faithfulness"
    if row["answer_relevancy"] < 0.35:
        return "low_relevancy"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS / "ragas_v0.2.json"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    weak = []
    for row in payload["rows"]:
        label = categorize(row)
        if label:
            weak.append({**row, "error_category": label})
    # Take a diverse sample across categories, then fill remaining slots by severity.
    order = {"wrong_source_top": 0, "diluted_context": 1, "low_faithfulness": 2, "low_relevancy": 3}
    by_cat: dict[str, list[dict]] = {key: [] for key in order}
    for row in weak:
        by_cat[row["error_category"]].append(row)
    for key in by_cat:
        by_cat[key].sort(key=lambda row: (row["context_precision"], row["answer_relevancy"], row["faithfulness"]))
    chosen = []
    per_cat = max(1, args.limit // len(order))
    for key in order:
        chosen.extend(by_cat[key][:per_cat])
    if len(chosen) < args.limit:
        rest = [row for row in weak if row not in chosen]
        rest.sort(key=lambda row: (order[row["error_category"]], row["context_precision"], row["answer_relevancy"]))
        chosen.extend(rest[: args.limit - len(chosen)])
    chosen = chosen[: args.limit]
    counts: dict[str, int] = {}
    for row in chosen:
        counts[row["error_category"]] = counts.get(row["error_category"], 0) + 1
    out = {
        "source_run": args.input,
        "n_examples": len(chosen),
        "category_counts": counts,
        "category_definitions": CATEGORIES,
        "examples": [
            {
                "id": row["id"],
                "error_category": row["error_category"],
                "gold_source_id": row["source_id"],
                "retrieved_source_ids": row["retrieved_source_ids"],
                "context_precision": row["context_precision"],
                "context_recall": row["context_recall"],
                "faithfulness": row["faithfulness"],
                "answer_relevancy": row["answer_relevancy"],
                "answer": row["answer"][:300],
            }
            for row in chosen
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "error_categories_v0.2.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Error categories — v0.2 (30 examples)",
        "",
        f"Source run: `{args.input}`.",
        "",
        "| Category | Count | Meaning |",
        "| --- | ---: | --- |",
    ]
    for key, meaning in CATEGORIES.items():
        lines.append(f"| {key} | {counts.get(key, 0)} | {meaning} |")
    lines.extend(["", "## Examples", ""])
    for row in out["examples"]:
        lines.append(
            f"- **{row['id']}** `{row['error_category']}` gold=`{row['gold_source_id']}` "
            f"retrieved={row['retrieved_source_ids'][:3]} "
            f"(P={row['context_precision']:.2f}, R={row['context_recall']:.2f}, "
            f"F={row['faithfulness']:.2f}, Rel={row['answer_relevancy']:.2f})"
        )
    lines.append("")
    out_md = RESULTS / "error_categories_v0.2.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
