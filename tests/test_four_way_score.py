import json
from pathlib import Path

from eval.four_way_score import llm_as_judge, style_score, token_f1
from ingestion.sources import ROOT


def test_style_and_judge_helpers():
    good = (
        "Students may work thirty hours weekly "
        "[oif-student-residence-en](https://oif.gov.hu/x). Not legal advice."
    )
    assert style_score(good) >= 0.8
    assert style_score("REFUSE: no source") == 1.0
    assert token_f1(good, "students may work thirty hours weekly") > 0.5
    judged = llm_as_judge("work hours?", good, "Students may work thirty hours weekly during studies.")
    assert judged["correctness"] > 0.3
    assert judged["judge"] == "local-heuristic-v1"


def test_v04_score_artifacts_exist_or_skip():
    md = ROOT / "eval" / "results" / "four_way_scores_v0.4.md"
    human = ROOT / "eval" / "results" / "human_correctness_30_v0.4.jsonl"
    # Created by the day-19 scoring run in the same session.
    if not md.exists():
        return
    text = md.read_text(encoding="utf-8")
    assert "faithfulness" in text
    assert "rag_lora" in text
    rows = [json.loads(line) for line in human.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 30
    assert all("human_correctness" in row for row in rows)
