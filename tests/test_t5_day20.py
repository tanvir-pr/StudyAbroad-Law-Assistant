import json
from pathlib import Path

from t5.rouge_eval import extractive_summary, load_refs, rouge_scores
from t5.warmup_mt5 import dry_run_loss, load_xlsum_slice


ROOT = Path(__file__).resolve().parents[1]


def test_human_refs_are_twenty_five_line_summaries():
    rows = load_refs(ROOT / "t5" / "human_refs_20.jsonl")
    assert len(rows) == 20
    for row in rows:
        lines = [line for line in row["reference_summary"].splitlines() if line.strip()]
        assert len(lines) == 5
        assert row["source_id"]
        assert row["document_excerpt"]


def test_extractive_and_rouge_helpers():
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta four. Epsilon five. Zeta six."
    pred = extractive_summary(text, n_lines=5)
    assert pred.startswith("1.")
    assert "5." in pred
    scores = rouge_scores(pred, pred)
    assert scores["rouge1"] >= 0.99


def test_warmup_dry_run_and_slice(tmp_path: Path):
    rows = load_xlsum_slice(n=20, seed=20)
    assert len(rows) >= 8
    path = dry_run_loss(tmp_path, n=len(rows), steps=5, seed=20)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert payload["history"]
