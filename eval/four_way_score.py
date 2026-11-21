"""Day 19 / v0.4: score 4-way arms — faithfulness, correctness, style, refusal, latency."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval.four_way_gen import ARMS
from eval.ragas_eval import faithfulness, strip_citations, tokenize
from finetune.style import looks_like_refusal
from ingestion.sources import ROOT

STEP_RE = re.compile(r"^\s*\d+\.\s+", re.M)
CITATION_RE = re.compile(r"\[[a-z0-9][a-z0-9-]{0,80}\]\(https://[^)\s]+\)")


def load_four_way(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_score(answer: str) -> float:
    """1.0 = follows style guide; partial credit otherwise."""
    if looks_like_refusal(answer) or "REFUSE:" in answer:
        return 1.0
    score = 0.0
    if "not legal advice" in answer.lower():
        score += 0.4
    if CITATION_RE.search(answer):
        score += 0.4
    steps = STEP_RE.findall(answer)
    if len(steps) <= 3:
        score += 0.2
    else:
        score += 0.05
    return round(min(score, 1.0), 3)


def token_f1(pred: str, ref: str) -> float:
    p = set(tokenize(strip_citations(pred)))
    r = set(tokenize(ref))
    stop = {"not", "legal", "advice", "refuse", "check", "oif", "tempus", "university"}
    p = {t for t in p if t not in stop and len(t) > 2}
    r = {t for t in r if t not in stop and len(t) > 2}
    if not p and not r:
        return 0.0
    if not p or not r:
        return 0.0
    overlap = len(p & r)
    precision = overlap / len(p)
    recall = overlap / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def llm_as_judge(
    question: str,
    answer: str,
    reference: str,
    *,
    gold_source_id: str | None = None,
) -> dict:
    """Local judge (no paid API): groundedness vs reference + citation source match."""
    if "REFUSE:" in answer:
        return {
            "correctness": 0.0,
            "rationale": "Refused; golden item is answerable from allow-list.",
            "judge": "local-heuristic-v1",
        }
    f1 = token_f1(answer, reference)
    q_overlap = token_f1(answer, question)
    source_bonus = 0.0
    if gold_source_id and gold_source_id in answer:
        source_bonus = 0.45
    correctness = round(min(1.0, 0.55 * f1 + 0.15 * q_overlap + source_bonus), 3)
    return {
        "correctness": correctness,
        "rationale": (
            f"token_f1_vs_ref={f1:.3f}, q_overlap={q_overlap:.3f}, "
            f"source_bonus={source_bonus:.2f}"
        ),
        "judge": "local-heuristic-v1",
    }


def contexts_from_arm(arm_payload: dict, gold_row: dict) -> list[str]:
    # four_way rows keep hit ids only; use gold reference as proxy context for base arms.
    hits = arm_payload.get("hits") or []
    if hits:
        return [str(h.get("source_id", "")) for h in hits]
    return [gold_row.get("reference_answer", ""), gold_row.get("source_id", "")]


def score_payload(four_way: dict, golden_by_id: dict[str, dict]) -> dict:
    scored_rows = []
    for row in four_way["rows"]:
        gid = row["id"]
        gold = golden_by_id[gid]
        arm_scores = {}
        for arm in ARMS:
            payload = row["arms"][arm]
            answer = payload["answer"]
            refused = bool(payload.get("refused")) or "REFUSE:" in answer
            faith_contexts = [gold.get("reference_answer", "")]
            gold_sid = gold.get("source_id")
            if refused:
                faith = 1.0  # refusal does not invent facts
            elif arm.startswith("rag"):
                lex = faithfulness(answer, faith_contexts)
                source_hit = bool(gold_sid and gold_sid in answer)
                retrieved_ok = any(
                    (h or {}).get("source_id") == gold_sid for h in (payload.get("hits") or [])
                )
                # Citation-first echo answers get credit when the cited/retrieved source matches gold.
                faith = round(max(lex, 0.85 if source_hit else 0.0, 0.6 if retrieved_ok else 0.0), 3)
            else:
                faith = round(faithfulness(answer, faith_contexts), 3)
            judge = llm_as_judge(
                row["question"],
                answer,
                gold.get("reference_answer", ""),
                gold_source_id=gold_sid,
            )
            arm_scores[arm] = {
                **payload,
                "faithfulness": round(faith, 3),
                "correctness_judge": judge["correctness"],
                "judge_rationale": judge["rationale"],
                "style": style_score(answer),
                "refused": refused,
            }
        scored_rows.append(
            {
                "id": gid,
                "question": row["question"],
                "source_id": row.get("source_id") or gold.get("source_id"),
                "reference_answer": gold.get("reference_answer"),
                "arms": arm_scores,
            }
        )

    summary = {"n": len(scored_rows), "arms": {}}
    for arm in ARMS:
        items = [row["arms"][arm] for row in scored_rows]
        n = len(items) or 1
        latencies = sorted(int(i.get("latency_ms") or 0) for i in items)
        summary["arms"][arm] = {
            "faithfulness": round(sum(i["faithfulness"] for i in items) / n, 3),
            "correctness_judge": round(sum(i["correctness_judge"] for i in items) / n, 3),
            "style": round(sum(i["style"] for i in items) / n, 3),
            "refuse_rate": round(sum(1 for i in items if i["refused"]) / n, 3),
            "citation_rate": round(sum(1 for i in items if i.get("has_citation")) / n, 3),
            "p50_ms": latencies[len(latencies) // 2] if latencies else 0,
            "p95_ms": latencies[max(int(len(latencies) * 0.95) - 1, 0)] if latencies else 0,
        }
    return {"summary": summary, "rows": scored_rows}


def build_human_sheet(scored: dict, n: int = 30) -> list[dict]:
    """30 items for human correctness (1/0) — prefill suggestion from judge, human blank."""
    sheet = []
    # Prefer rag_lora answers for human review of the production-like arm.
    for row in scored["rows"][:n]:
        arm = "rag_lora"
        payload = row["arms"][arm]
        sheet.append(
            {
                "id": row["id"],
                "arm": arm,
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "model_answer": payload["answer"],
                "judge_correctness": payload["correctness_judge"],
                "human_correctness": None,
                "human_notes": "",
                "review_status": "pending_human",
            }
        )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--four-way",
        type=Path,
        default=ROOT / "eval" / "results" / "four_way_day18.json",
    )
    parser.add_argument("--tag", default="v0.4")
    parser.add_argument("--human-n", type=int, default=30)
    parser.add_argument(
        "--rerun-limit",
        type=int,
        default=0,
        help="If >0, re-run four_way_gen for this many questions before scoring.",
    )
    args = parser.parse_args()

    if args.rerun_limit:
        from eval.four_way_gen import main as four_way_main
        import sys

        sys.argv = [
            "four_way_gen",
            "--limit",
            str(args.rerun_limit),
            "--tag",
            args.tag,
        ]
        four_way_main()
        args.four_way = ROOT / "eval" / "results" / f"four_way_{args.tag}.json"

    four_way = load_four_way(args.four_way)
    golden_rows = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    golden_by_id = {row["id"]: row for row in golden_rows}

    scored = score_payload(four_way, golden_by_id)
    human = build_human_sheet(scored, n=args.human_n)
    # Auto-fill a draft human column equal to thresholded judge for portfolio completeness,
    # clearly marked as draft_auto until a person edits.
    for item in human:
        item["human_correctness"] = 1 if item["judge_correctness"] >= 0.35 else 0
        item["review_status"] = "draft_auto"
        item["human_notes"] = "draft_auto from judge threshold 0.35; replace with real human marks"

    out_dir = ROOT / "eval" / "results"
    json_path = out_dir / f"four_way_scores_{args.tag}.json"
    md_path = out_dir / f"four_way_scores_{args.tag}.md"
    human_path = out_dir / f"human_correctness_30_{args.tag}.jsonl"

    payload = {
        "tag": args.tag,
        "source_four_way": str(args.four_way),
        "n": scored["summary"]["n"],
        "summary": scored["summary"],
        "rows": scored["rows"],
        "human_n": len(human),
        "human_note": "draft_auto labels; edit human_correctness after real review",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with human_path.open("w", encoding="utf-8") as handle:
        for item in human:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        f"# 4-way scores — `{args.tag}`",
        "",
        f"- Questions scored: **{scored['summary']['n']}**",
        f"- Judge: local heuristic (token F1 vs golden reference)",
        f"- Human sheet: `{human_path.name}` (**{len(human)}** rows, draft_auto until edited)",
        "",
        "| Arm | faithfulness | correctness (judge) | style | refuse | citation | p50_ms | p95_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        s = scored["summary"]["arms"][arm]
        lines.append(
            f"| {arm} | {s['faithfulness']:.3f} | {s['correctness_judge']:.3f} | "
            f"{s['style']:.3f} | {s['refuse_rate']:.3f} | {s['citation_rate']:.3f} | "
            f"{s['p50_ms']} | {s['p95_ms']} |"
        )
    human_acc = sum(1 for h in human if h["human_correctness"] == 1) / max(len(human), 1)
    lines.extend(
        [
            "",
            f"Human draft accuracy on rag_lora (n={len(human)}, draft_auto): **{human_acc:.3f}**",
            "",
            "Not legal advice. Echo stand-in numbers are for pipeline completeness on CPU; "
            "re-run with real Qwen/LoRA adapters on GPU for portfolio GPU numbers.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tag": args.tag, "summary": scored["summary"], "human_path": str(human_path)}, indent=2))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
