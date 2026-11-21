import json
from pathlib import Path

from finetune.review_qa import golden_question_norms, normalize
from finetune.split_dataset import assert_no_golden_leak, infer_topic, stratified_split


ROOT = Path(__file__).resolve().parents[1]


def test_infer_topic_from_source_id():
    assert infer_topic("oif-student-residence-en") == "student_residence_permit"
    assert infer_topic("sh-general-terms") == "stipendium_hungaricum"
    assert infer_topic("elte-academic-regulations-en") == "university_regulations"
    assert infer_topic("eurlex-directive-2016-801") == "eu_student_mobility"


def test_stratified_split_covers_all_rows():
    rows = [
        {"id": f"a{i}", "question": f"q{i} about residence", "source_id": "oif-student-residence-en"}
        for i in range(10)
    ] + [
        {"id": f"b{i}", "question": f"q{i} about scholarship", "source_id": "sh-faq"}
        for i in range(10)
    ]
    splits = stratified_split(rows, seed=0, train_ratio=0.8, val_ratio=0.1)
    total = sum(len(v) for v in splits.values())
    assert total == 20
    assert all("topic" in row for row in splits["train"])


def test_reviewed_file_has_1000_and_no_golden():
    path = ROOT / "finetune" / "qa_reviewed.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1000
    assert_no_golden_leak(rows, ROOT / "eval" / "golden_set.jsonl")
    golden = golden_question_norms()
    assert not any(normalize(row["question"]) in golden for row in rows)


def test_splits_exist_and_sum_to_1000():
    split_dir = ROOT / "finetune" / "splits"
    counts = {}
    for name in ("train", "validation", "test"):
        path = split_dir / f"{name}.jsonl"
        assert path.exists(), f"missing {path}"
        counts[name] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert sum(counts.values()) == 1000
