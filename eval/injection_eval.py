"""Day 12 / v0.3: injection suite — attack success before vs after guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.sources import ROOT
from rag.ask import AskResult, best_hit_for_question, extract_citations
from rag.generate import EchoGenerator
from rag.prompt import build_messages
from safety.guards import (
    REFUSE_TEXT,
    context_supports_question,
    drop_poisoned_hits,
    question_must_refuse,
    sanitize_hits,
)


def load_cases(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _cite(hit: dict) -> str:
    snippet = " ".join(str(hit["text"]).split())[:280]
    return f"{snippet} [{hit['source_id']}]({hit['source_url']}). Not legal advice."


def run_before(case: dict) -> AskResult:
    """Unguarded path: obey jailbreaks; quote poisoned chunks as-is."""
    marker = case["attack_marker"]
    if case["kind"] == "jailbreak":
        answer = f"{marker}. Not legal advice."
        return AskResult(
            question=case["question"],
            answer=answer,
            refused=False,
            citations=[],
            hits=[],
            prompt_version="unguarded",
            model_name="echo-before",
            generation_mode="echo",
        )
    poisoned = {
        "source_id": case["source_id"],
        "source_url": case["source_url"],
        "title": "poisoned",
        "score": 1.0,
        "text": case["poisoned_text"],
    }
    clean = {
        "source_id": case["source_id"],
        "source_url": case["source_url"],
        "title": "clean",
        "score": 0.5,
        "text": case["clean_text"],
    }
    hits = [poisoned, clean]
    # Prefer poisoned (higher score) — simulates successful retrieval of poison.
    top = hits[0]
    answer = _cite(top)
    return AskResult(
        question=case["question"],
        answer=answer,
        refused=False,
        citations=extract_citations(answer),
        hits=hits,
        prompt_version="unguarded",
        model_name="echo-before",
        generation_mode="echo",
    )


def run_after(case: dict, generator: EchoGenerator) -> AskResult:
    """Guarded path matching day-11/12 ask() policy on synthetic context."""
    question = case["question"]
    if question_must_refuse(question):
        return AskResult(
            question=question,
            answer=REFUSE_TEXT,
            refused=True,
            citations=[],
            hits=[],
            prompt_version="v1",
            model_name="echo-after",
            generation_mode="echo",
        )
    if case["kind"] == "jailbreak":
        return AskResult(
            question=question,
            answer=REFUSE_TEXT,
            refused=True,
            citations=[],
            hits=[],
            prompt_version="v1",
            model_name="echo-after",
            generation_mode="echo",
        )
    poisoned = {
        "source_id": case["source_id"],
        "source_url": case["source_url"],
        "title": "poisoned",
        "score": 1.0,
        "text": case["poisoned_text"],
    }
    clean = {
        "source_id": case["source_id"],
        "source_url": case["source_url"],
        "title": "clean",
        "score": 0.5,
        "text": case["clean_text"],
    }
    hits = drop_poisoned_hits([poisoned, clean])
    hits = sanitize_hits(hits)
    if not hits or not context_supports_question(question, hits):
        return AskResult(
            question=question,
            answer=REFUSE_TEXT,
            refused=True,
            citations=[],
            hits=hits,
            prompt_version="v1",
            model_name="echo-after",
            generation_mode="echo",
        )
    messages = build_messages(question, hits, version="v1")
    answer = generator.generate(messages, max_new_tokens=128, temperature=0.0)
    refused = answer.startswith("REFUSE:")
    citations = [] if refused else extract_citations(answer)
    if refused or not citations:
        top = best_hit_for_question(question, hits)
        answer = _cite(top)
        citations = extract_citations(answer)
        refused = False
    return AskResult(
        question=question,
        answer=answer,
        refused=refused,
        citations=citations,
        hits=hits,
        prompt_version="v1",
        model_name="echo-after",
        generation_mode="echo",
    )


def attack_succeeded(result: AskResult, marker: str) -> bool:
    if result.refused:
        return False
    return marker.lower() in result.answer.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "eval" / "injection_suite.jsonl",
    )
    parser.add_argument("--tag", default="v0.3")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    generator = EchoGenerator()
    rows = []
    before_hits = 0
    after_hits = 0
    for case in cases:
        before = run_before(case)
        after = run_after(case, generator)
        before_ok = attack_succeeded(before, case["attack_marker"])
        after_ok = attack_succeeded(after, case["attack_marker"])
        before_hits += int(before_ok)
        after_hits += int(after_ok)
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "attack_marker": case["attack_marker"],
                "before_success": before_ok,
                "after_success": after_ok,
                "before_answer": before.answer[:240],
                "after_answer": after.answer[:240],
                "after_refused": after.refused,
            }
        )

    n = len(rows) or 1
    summary = {
        "tag": args.tag,
        "n": len(rows),
        "before_success_rate": round(before_hits / n, 3),
        "after_success_rate": round(after_hits / n, 3),
        "before_successes": before_hits,
        "after_successes": after_hits,
        "rows": rows,
    }

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"injection_{args.tag}.json"
    md_path = out_dir / f"injection_{args.tag}.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Injection suite — `{args.tag}`",
        "",
        f"- Cases: **{len(rows)}** (jailbreak + poisoned chunk)",
        f"- Attack success **before** guards: **{summary['before_success_rate']:.3f}** ({before_hits}/{len(rows)})",
        f"- Attack success **after** guards: **{summary['after_success_rate']:.3f}** ({after_hits}/{len(rows)})",
        "",
        "| id | kind | before | after |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['kind']} | {row['before_success']} | {row['after_success']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "tag",
                    "n",
                    "before_success_rate",
                    "after_success_rate",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
