"""Day 21: en/hu → Bangla gloss translate + chrF against human bn refs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ingestion.sources import ROOT

# Small domain glossary for CPU-safe demo translations (not a full MT system).
GLOSSARY = [
    (r"\bstudy residence permit\b", "অধ্যয়নের উদ্দেশ্যে বসবাসের অনুমতি"),
    (r"\bresidence permit\b", "বসবাসের অনুমতি"),
    (r"\bEnter Hungary\b", "Enter Hungary"),
    (r"\bStipendium Hungaricum\b", "স্টিপেন্ডিয়াম হাঙ্গারিকাম"),
    (r"\bOIF\b", "OIF"),
    (r"\bbiometrics\b", "বায়োমেট্রিক্স"),
    (r"\bstudent\b", "শিক্ষার্থী"),
    (r"\bstudents\b", "শিক্ষার্থীরা"),
    (r"\buniversity\b", "বিশ্ববিদ্যালয়"),
    (r"\bscholarship\b", "বৃত্তি"),
    (r"\bextension\b", "মেয়াদ বৃদ্ধি"),
    (r"\bextend\b", "মেয়াদ বাড়ান"),
    (r"\bHungary\b", "হাঙ্গেরি"),
    (r"\bHungarian\b", "হাঙ্গেরীয়"),
    (r"\bwork\b", "কাজ"),
    (r"\bhours\b", "ঘণ্টা"),
    (r"\bdays\b", "দিন"),
    (r"\bbefore\b", "আগে"),
    (r"\bmust\b", "অবশ্যই"),
    (r"\bmay\b", "পারে"),
    (r"\bNot legal advice\b", "এটি আইনি পরামর্শ নয়"),
    (r"\bhallgató\b", "শিক্ষার্থী"),
    (r"\btartózkodási\b", "বসবাসের"),
    (r"\bengedély\b", "অনুমতি"),
]


def gloss_translate(text: str) -> str:
    out = text
    for pattern, bn in GLOSSARY:
        out = re.sub(pattern, bn, out, flags=re.IGNORECASE)
    return out


def chrf(pred: str, ref: str, n: int = 3) -> float:
    try:
        from sacrebleu.metrics import CHRF

        metric = CHRF(word_order=0)
        return float(metric.sentence_score(pred, [ref]).score) / 100.0
    except Exception:
        def ngrams(s: str) -> list[str]:
            s = re.sub(r"\s+", " ", s.lower()).strip()
            chars = list(s.replace(" ", "▁"))
            return ["".join(chars[i : i + n]) for i in range(max(len(chars) - n + 1, 0))]

        p = ngrams(pred)
        r = ngrams(ref)
        if not p or not r:
            return 0.0
        from collections import Counter

        pc, rc = Counter(p), Counter(r)
        overlap = sum((pc & rc).values())
        precision = overlap / max(sum(pc.values()), 1)
        recall = overlap / max(sum(rc.values()), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def load_bn_refs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refs",
        type=Path,
        default=ROOT / "t5" / "bn_refs.jsonl",
    )
    parser.add_argument("--tag", default="day21")
    args = parser.parse_args()

    rows_in = load_bn_refs(args.refs)
    scored = []
    for row in rows_in:
        pred = gloss_translate(row["source_text"])
        score = chrf(pred, row["bn_reference"])
        scored.append(
            {
                "id": row["id"],
                "source_lang": row.get("source_lang", "en"),
                "chrf": round(score, 4),
                "prediction": pred,
                "bn_reference": row["bn_reference"],
            }
        )
    n = len(scored) or 1
    summary = {
        "tag": args.tag,
        "n": len(scored),
        "chrf_mean": round(sum(r["chrf"] for r in scored) / n, 4),
        "rows": scored,
    }
    out_dir = ROOT / "t5" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"chrf_{args.tag}.json"
    md_path = out_dir / f"chrf_{args.tag}.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# chrF en/hu→bn — `{args.tag}`",
        "",
        f"- Pairs: **{len(scored)}**",
        f"- Mean chrF: **{summary['chrf_mean']:.4f}**",
        "",
        "| id | lang | chrF |",
        "|---|---|---:|",
    ]
    for row in scored:
        lines.append(f"| {row['id']} | {row.get('source_lang', 'en')} | {row['chrf']:.3f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tag", "n", "chrf_mean")}, indent=2))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
