from rag.bm25 import Bm25Index, tokenize
from rag.index_faiss import HashEmbedder, build_index
from rag.retrieve import apply_metadata_filters, reciprocal_rank_fusion, retrieve


def test_tokenize_and_filters(tmp_path):
    documents = [
        {
            "source_id": "oif-student-residence-en",
            "source_url": "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
            "title": "Work",
            "country": "HU",
            "institution": "OIF",
            "language": "en",
            "last_updated": None,
            "text": "Students may work thirty hours weekly during the study period in Hungary.",
        },
        {
            "source_id": "eurlex-directive-2016-801",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016L0801",
            "title": "Directive",
            "country": "EU",
            "institution": "EU",
            "language": "en",
            "last_updated": "2016-05-11",
            "text": "Member States shall not set a maximum below fifteen hours per week for student work.",
        },
    ]
    embedder = HashEmbedder()
    build_index(documents, embedder, tmp_path, size=200, overlap=20, strategy="fixed")
    bm25 = Bm25Index.from_index_dir(tmp_path)
    assert tokenize("Thirty HOURS") == ["thirty", "hours"]

    hybrid = retrieve(
        "thirty hours weekly",
        embedder,
        tmp_path,
        k=2,
        mode="hybrid",
        bm25_index=bm25,
    )
    assert hybrid
    assert hybrid[0]["source_id"] == "oif-student-residence-en"

    oif_only = retrieve(
        "hours per week",
        embedder,
        tmp_path,
        k=5,
        mode="hybrid",
        institution="OIF",
        country="HU",
        bm25_index=bm25,
    )
    assert oif_only
    assert all(hit["institution"] == "OIF" for hit in oif_only)
    assert apply_metadata_filters(oif_only, country="EU") == []


def test_rrf_prefers_shared_high_ranks():
    left = [
        {"chunk_id": 1, "source_id": "a", "text": "a"},
        {"chunk_id": 2, "source_id": "b", "text": "b"},
    ]
    right = [
        {"chunk_id": 2, "source_id": "b", "text": "b"},
        {"chunk_id": 3, "source_id": "c", "text": "c"},
    ]
    fused = reciprocal_rank_fusion([left, right], limit=2)
    assert fused[0]["chunk_id"] == 2
