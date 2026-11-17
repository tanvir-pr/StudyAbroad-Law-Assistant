from rag.index_chroma import build_chroma_from_faiss, search_chroma
from rag.index_faiss import HashEmbedder, build_index, search as faiss_search
from rag.rerank import OverlapReranker
from rag.retrieve import retrieve


def test_chroma_matches_faiss_top_source(tmp_path):
    documents = [
        {
            "source_id": "oif-student-residence-en",
            "source_url": "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
            "title": "Work",
            "country": "HU",
            "institution": "OIF",
            "language": "en",
            "last_updated": None,
            "text": "Students may work thirty hours weekly during the study period.",
        },
        {
            "source_id": "nje-regulations-index",
            "source_url": "https://nje.hu/en/regulation-and-documents",
            "title": "Index",
            "country": "HU",
            "institution": "NJE",
            "language": "en",
            "last_updated": None,
            "text": "Holiday schedule and dormitory places only.",
        },
    ]
    embedder = HashEmbedder()
    build_index(documents, embedder, tmp_path, size=200, overlap=20, strategy="fixed")
    chroma_path = tmp_path / "chroma"
    assert build_chroma_from_faiss(tmp_path, chroma_path) >= 2
    query = "thirty hours weekly"
    faiss_hits = faiss_search(query, embedder, tmp_path, k=1)
    chroma_hits = search_chroma(query, embedder, chroma_path, k=1)
    assert faiss_hits[0]["source_id"] == "oif-student-residence-en"
    assert chroma_hits[0]["source_id"] == "oif-student-residence-en"


def test_reranker_moves_overlapping_chunk_first(tmp_path):
    documents = [
        {
            "source_id": "a",
            "source_url": "https://example.com/a",
            "title": "A",
            "country": "HU",
            "institution": "OIF",
            "language": "en",
            "last_updated": None,
            "text": "Appeal within eight days of delivery.",
        },
        {
            "source_id": "b",
            "source_url": "https://example.com/b",
            "title": "B",
            "country": "HU",
            "institution": "OIF",
            "language": "en",
            "last_updated": None,
            "text": "Students may work thirty hours weekly during studies.",
        },
    ]
    embedder = HashEmbedder()
    build_index(documents, embedder, tmp_path, size=200, overlap=20, strategy="fixed")
    hits = retrieve(
        "thirty hours weekly",
        embedder,
        tmp_path,
        k=2,
        mode="hybrid",
        reranker=OverlapReranker(),
        rerank_candidates=2,
    )
    assert hits
    assert hits[0]["source_id"] == "b"
