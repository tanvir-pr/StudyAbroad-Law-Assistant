from eval.error_categories import categorize


def test_error_category_rules():
    assert categorize({"context_recall": 0, "context_precision": 0, "faithfulness": 1, "answer_relevancy": 1}) == "wrong_source_top"
    assert categorize({"context_recall": 1, "context_precision": 0.2, "faithfulness": 1, "answer_relevancy": 1}) == "diluted_context"
    assert categorize({"context_recall": 1, "context_precision": 1, "faithfulness": 0.2, "answer_relevancy": 1}) == "low_faithfulness"
    assert categorize({"context_recall": 1, "context_precision": 1, "faithfulness": 1, "answer_relevancy": 0.1}) == "low_relevancy"
    assert categorize({"context_recall": 1, "context_precision": 1, "faithfulness": 1, "answer_relevancy": 0.9}) is None
