from finetune.draft_qa import draft_from_chunk
from finetune.review_qa import golden_question_norms, review_row, span_in_chunk
from finetune.style import format_short_steps, safe_fallback


def test_format_short_steps_single_rule():
    text = format_short_steps(
        "Students may work thirty hours weekly during the study period",
        "oif-student-residence-en",
        "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
    )
    assert "thirty hours" in text
    assert "[oif-student-residence-en](https://oif.gov.hu/factsheets/residence-of-the-student-pupil)" in text
    assert text.strip().endswith("Not legal advice.")


def test_format_short_steps_procedure():
    text = format_short_steps(
        "Submit the extension through Enter Hungary. Visit the client service within 15 days. Attach the progress certificate.",
        "oif-study-permit-extension",
        "https://www.oif.gov.hu/factsheets/how-to-extend-your-residence-permit-for-the-purpose-of-studies-know-how",
    )
    assert text.startswith("1.")
    assert "2." in text
    assert "Not legal advice." in text


def test_safe_fallback_is_refuse():
    assert safe_fallback().startswith("REFUSE:")


def test_draft_and_review_accepts_span_matched_pair():
    chunk = {
        "chunk_id": 1,
        "source_id": "oif-student-residence-en",
        "source_url": "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
        "title": "Residence of the student",
        "language": "en",
        "institution": "OIF",
        "country": "HU",
        "text": (
            "Students may work thirty hours weekly during the study period. "
            "Other unrelated navigation text appears here for padding."
        ),
    }
    drafts = draft_from_chunk(chunk, per_chunk=1)
    assert drafts
    ok, reason = review_row(drafts[0], golden_qs=set())
    assert ok, reason
    assert span_in_chunk(drafts[0]["answer_span"], chunk["text"])


def test_review_blocks_golden_questions():
    golden = golden_question_norms()
    assert len(golden) == 150
    row = {
        "question": next(iter(golden)),  # normalized - need original
        "answer": "x [oif-student-residence-en](https://oif.gov.hu/x). Not legal advice.",
        "answer_span": "Students may work thirty hours weekly",
        "chunk_text": "Students may work thirty hours weekly during the study period.",
        "source_id": "oif-student-residence-en",
        "source_url": "https://oif.gov.hu/x",
    }
    # Use a real golden question string from file via reverse lookup pattern
    from pathlib import Path
    import json

    path = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["question"] = first["question"]
    ok, reason = review_row(row, golden)
    assert not ok
    assert reason == "golden_leak"
