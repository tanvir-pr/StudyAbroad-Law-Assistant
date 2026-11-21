"""Split reviewed Q&A into topic-stratified train/val/test (day 16)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from finetune.review_qa import golden_question_norms, load_jsonl, normalize
from ingestion.sources import ROOT

TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("student_residence_permit", ("oif-student-residence", "oif-study-permit", "oif-enter-hungary", "oif-higher-education", "oif-employer", "njt-")),
    ("stipendium_hungaricum", ("sh-",)),
    ("university_regulations", ("elte-", "bme-", "nje-")),
    ("eu_student_mobility", ("eurlex-", "eu-portal-")),
]


def infer_topic(source_id: str) -> str:
    sid = source_id.lower()
    for topic, prefixes in TOPIC_RULES:
        if any(sid.startswith(prefix) or prefix in sid for prefix in prefixes):
            return topic
    return "other"


def assert_no_golden_leak(rows: list[dict], golden_path: Path) -> None:
    golden = golden_question_norms(golden_path)
    leaks = [row["id"] for row in rows if normalize(row["question"]) in golden]
    if leaks:
        raise SystemExit(f"Golden leak in fine-tune rows: {leaks[:10]}")


def stratified_split(
    rows: list[dict],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        topic = infer_topic(row["source_id"])
        enriched = dict(row)
        enriched["topic"] = topic
        by_topic[topic].append(enriched)

    splits = {"train": [], "validation": [], "test": []}
    for topic, items in by_topic.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Ensure tiny topics still get a test row when possible.
        if n >= 3 and n_train + n_val >= n:
            n_val = max(1, n // 10)
            n_train = max(1, n - n_val - 1)
        train = items[:n_train]
        val = items[n_train : n_train + n_val]
        test = items[n_train + n_val :]
        splits["train"].extend(train)
        splits["validation"].extend(val)
        splits["test"].extend(test)

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "finetune" / "qa_reviewed.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "finetune" / "splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "eval" / "golden_set.jsonl",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    assert_no_golden_leak(rows, args.golden)
    splits = stratified_split(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    summary = {"n_total": len(rows), "seed": args.seed, "splits": {}, "topics": {}}
    topic_counts: dict[str, int] = defaultdict(int)
    for name, items in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", items)
        summary["splits"][name] = len(items)
        for row in items:
            topic_counts[row["topic"]] += 1
    summary["topics"] = dict(sorted(topic_counts.items()))

    summary_path = args.out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote splits under {args.out_dir}")


if __name__ == "__main__":
    main()
