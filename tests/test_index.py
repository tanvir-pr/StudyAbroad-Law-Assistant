from rag.chunking import fixed_chunks
from rag.index_faiss import HashEmbedder, build_index, search


def test_fixed_chunks_cover_the_text_and_overlap():
    text = "alpha " * 400
    chunks = fixed_chunks(text, size=100, overlap=20)
    assert len(chunks) > 2
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[0][:5] == "alpha"
    assert "alpha" in chunks[-1]


def test_fixed_chunks_reject_bad_sizes_and_empty_text():
    assert fixed_chunks("   ", 100, 10) == []
    try:
        fixed_chunks("hello", 10, 10)
    except ValueError:
        return
    raise AssertionError("overlap equal to size should fail")


def test_faiss_search_returns_the_chunk_that_shares_the_query_words(tmp_path):
    documents = [
        {
            "source_id": "work-rule",
            "source_url": "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
            "title": "Work",
            "country": "HU",
            "institution": "OIF",
            "language": "en",
            "last_updated": None,
            "text": "Students may work thirty hours weekly during the study period.",
        },
        {
            "source_id": "holiday-rule",
            "source_url": "https://nje.hu/en/regulation-and-documents",
            "title": "Calendar",
            "country": "HU",
            "institution": "NJE",
            "language": "en",
            "last_updated": "2026-06-01",
            "text": "The combined length of the winter spring and summer holidays is at least six weeks.",
        },
    ]
    embedder = HashEmbedder()
    count = build_index(documents, embedder, tmp_path, size=200, overlap=20)
    assert count == 2
    hits = search("thirty hours weekly", embedder, tmp_path, k=1)
    assert hits[0]["source_id"] == "work-rule"
    assert hits[0]["source_url"].startswith("https://oif.gov.hu/")
    assert "thirty hours" in hits[0]["text"]
