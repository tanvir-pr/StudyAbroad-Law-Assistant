"""Compare two document versions and list plain-language changes (day 21)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ingestion.sources import ROOT

TOKEN_RE = re.compile(r"[a-z0-9áéíóöőúüű]+", re.I)


def tokenize(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if len(t) > 2}


def diff_changes(old_text: str, new_text: str, max_items: int = 5) -> list[str]:
    old_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", old_text) if s.strip()]
    new_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", new_text) if s.strip()]
    old_norm = {s.lower() for s in old_sents}
    added = [s for s in new_sents if s.lower() not in old_norm]
    removed = [s for s in old_sents if s.lower() not in {x.lower() for x in new_sents}]
    changes: list[str] = []
    for sent in added[:max_items]:
        changes.append(f"Added/emphasized: {sent}")
    for sent in removed[: max(0, max_items - len(changes))]:
        changes.append(f"Removed/softened: {sent}")
    if not changes:
        only_new = tokenize(new_text) - tokenize(old_text)
        if only_new:
            changes.append("New key terms: " + ", ".join(sorted(only_new)[:12]))
        else:
            changes.append("No major sentence-level change detected.")
    return changes


def load_pairs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["pairs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "t5" / "what_changed_pairs.json",
    )
    parser.add_argument("--tag", default="day21")
    args = parser.parse_args()
    pairs = load_pairs(args.pairs)
    rows = []
    for pair in pairs:
        predicted = diff_changes(pair["old_text"], pair["new_text"])
        rows.append(
            {
                "id": pair["id"],
                "title": pair["title"],
                "source_id": pair["source_id"],
                "source_url": pair["source_url"],
                "predicted_changes": predicted,
                "expected_changes": pair.get("expected_changes", []),
            }
        )
    out_dir = ROOT / "t5" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"what_changed_{args.tag}.json"
    path.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    md = out_dir / f"what_changed_{args.tag}.md"
    lines = [f"# What-changed — `{args.tag}`", "", f"- Pairs: **{len(rows)}**", ""]
    for row in rows:
        lines.append(f"## {row['id']}: {row['title']}")
        lines.append(f"- Source: `{row['source_id']}`")
        for change in row["predicted_changes"]:
            lines.append(f"- {change}")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(rows), "out": str(md)}, indent=2))


if __name__ == "__main__":
    main()
