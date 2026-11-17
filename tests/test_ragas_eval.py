from eval.ragas_eval import answer_relevancy, context_precision, context_recall, faithfulness, summarize
from rag.index_faiss import HashEmbedder


def test_ragas_metric_helpers():
    hits = [
        {"source_id": "oif-student-residence-en", "text": "thirty hours weekly"},
        {"source_id": "other", "text": "dormitory keys"},
    ]
    assert context_precision(hits, "oif-student-residence-en") == 0.5
    assert context_recall(hits, "oif-student-residence-en") == 1.0
    assert context_recall(hits, "missing") == 0.0
    faith = faithfulness(
        "Students may work thirty hours weekly. Not legal advice.",
        ["Students may work thirty hours weekly during the study period."],
    )
    assert faith > 0.5
    embedder = HashEmbedder(dim=32)
    rel = answer_relevancy("thirty hours work", "thirty hours weekly work limit", embedder)
    assert 0.0 <= rel <= 1.0
    summary = summarize(
        [
            {
                "context_precision": 1.0,
                "context_recall": 1.0,
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "refused": False,
            }
        ]
    )
    assert summary["context_precision"] == 1.0
    assert summary["n_questions"] == 1
