import json
from pathlib import Path

from eval.chunk_recall import recall_at_k, span_hit
from rag.chunking import chunk_text
from rag.index_faiss import HashEmbedder, build_index, search


def test_span_hit_and_recall_helpers():
    assert span_hit("thirty hours a week", ["Students may work thirty hours a week during term."])
    assert not span_hit("thirty hours a week", ["Dormitory keys and laundry hours."])
    assert recall_at_k([{"source_id": "a"}, {"source_id": "b"}], "b", 2)
    assert not recall_at_k([{"source_id": "a"}], "b", 1)


def test_chunk_strategies_can_be_indexed_and_searched(tmp_path: Path):
    documents = [
        {
            "source_id": "oif-student-residence-en",
            "source_url": "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
            "title": "Work",
            "country": "HU",
            "institution": "OIF",
            "language": "en",
            "last_updated": None,
            "text": "Students may work thirty hours weekly during the study period.\n\n--- page 2 ---\n\nOutside the study period full-time work is at most ninety days.",
        }
    ]
    embedder = HashEmbedder()
    for strategy in ("fixed", "recursive", "section_aware"):
        out = tmp_path / strategy
        count = build_index(documents, embedder, out, size=120, overlap=20, strategy=strategy)
        assert count >= 1
        assert chunk_text(documents[0]["text"], strategy, 120, 20)
        hits = search("thirty hours weekly", embedder, out, k=1)
        assert hits[0]["source_id"] == "oif-student-residence-en"
