"""Review drafted Q&A: keep source-matched pairs, never golden questions (day 13)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from finetune.style import looks_like_refusal
from ingestion.sources import ROOT

GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"
CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9-]{0,80})\]\((https://[^)\s]+)\)")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def golden_question_norms(path: Path = GOLDEN_PATH) -> set[str]:
    return {normalize(row["question"]) for row in load_jsonl(path)}


def span_in_chunk(span: str, chunk_text: str, min_ratio: float = 0.92) -> bool:
    needle = normalize(span)
    hay = normalize(chunk_text)
    if not needle or not hay:
        return False
    if needle in hay:
        return True
    # Allow light punctuation drift: shared token ratio on a window.
    n_tokens = needle.split()
    if len(n_tokens) < 6:
        return False
    h_tokens = hay.split()
    need = set(n_tokens)
    window = len(n_tokens)
    for i in range(0, max(len(h_tokens) - window + 1, 1)):
        chunk = h_tokens[i : i + window + 3]
        overlap = len(need & set(chunk)) / len(need)
        if overlap >= min_ratio:
            return True
    return False


def citations_match(row: dict) -> bool:
    # Exact marker — URLs may contain ')' (e.g. BME PDF paths), so avoid naive regex cuts.
    marker = f"[{row['source_id']}]({row['source_url']})"
    return marker in row.get("answer", "")


def review_row(row: dict, golden_qs: set[str]) -> tuple[bool, str]:
    if looks_like_refusal(row.get("answer", "")):
        return False, "refusal"
    if normalize(row.get("question", "")) in golden_qs:
        return False, "golden_leak"
    if not span_in_chunk(row.get("answer_span", ""), row.get("chunk_text", "")):
        return False, "span_mismatch"
    if not citations_match(row):
        return False, "bad_citation"
    if len(row.get("question", "").split()) < 5:
        return False, "short_question"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drafts",
        type=Path,
        default=ROOT / "finetune" / "qa_drafts.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "finetune" / "qa_reviewed_day13.jsonl",
    )
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--report", type=Path, default=ROOT / "finetune" / "REVIEW_DAY13.md")
    parser.add_argument("--label", default="Q&A review", help="Report title label, e.g. Day 14")
    args = parser.parse_args()

    golden_qs = golden_question_norms()
    drafts = load_jsonl(args.drafts)
    accepted: list[dict] = []
    seen_q: set[str] = set()
    source_counts: dict[str, int] = {}
    reject_counts: dict[str, int] = {}

    # Prefer diversity: round-robin by source_id after a first pass order.
    by_source: dict[str, list[dict]] = {}
    for row in drafts:
        by_source.setdefault(row["source_id"], []).append(row)

    # Interleave sources
    queues = {sid: list(rows) for sid, rows in by_source.items()}
    while queues and len(accepted) < args.target:
        progress = False
        for sid in list(queues.keys()):
            if not queues[sid]:
                queues.pop(sid, None)
                continue
            row = queues[sid].pop(0)
            ok, reason = review_row(row, golden_qs)
            if not ok:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                progress = True
                continue
            qn = normalize(row["question"])
            if qn in seen_q:
                reject_counts["duplicate_question"] = reject_counts.get("duplicate_question", 0) + 1
                progress = True
                continue
            # Cap per source so one PDF cannot fill all 200.
            if source_counts.get(sid, 0) >= max(20, args.target // 5):
                reject_counts["source_cap"] = reject_counts.get("source_cap", 0) + 1
                progress = True
                continue
            kept = {
                "id": f"ft{len(accepted) + 1:04d}",
                "question": row["question"],
                "answer": row["answer"],
                "answer_span": row["answer_span"],
                "source_id": row["source_id"],
                "source_url": row["source_url"],
                "chunk_id": row.get("chunk_id"),
                "language": row.get("language"),
                "institution": row.get("institution"),
                "country": row.get("country"),
                "review_status": "accepted",
                "review_note": "span+citation matched; not in golden set",
            }
            accepted.append(kept)
            seen_q.add(qn)
            source_counts[sid] = source_counts.get(sid, 0) + 1
            progress = True
            if len(accepted) >= args.target:
                break
        if not progress:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        f"# {args.label}",
        "",
        f"- Drafts read: **{len(drafts)}**",
        f"- Accepted: **{len(accepted)}** (target {args.target})",
        f"- Output: `{args.out.as_posix()}`",
        f"- Golden questions blocked: **{len(golden_qs)}**",
        "",
        "## Rejects",
        "",
    ]
    for reason, count in sorted(reject_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Per-source accepted", ""])
    for sid, count in sorted(source_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{sid}`: {count}")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "target": args.target,
                "out": str(args.out),
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
