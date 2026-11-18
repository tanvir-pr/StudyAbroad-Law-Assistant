from pathlib import Path

from rag.ask import ask, best_hit_for_question, extract_citations, filter_hits
from rag.generate import EchoGenerator
from rag.index_faiss import HashEmbedder, build_index
from rag.prompt import build_messages, ensure_prompt_file, format_context, load_prompt


def test_prompt_v1_is_versioned_on_disk():
    path = ensure_prompt_file("v1")
    assert path.exists()
    text = load_prompt("v1")
    assert "{context}" in text
    assert "{question}" in text
    assert "REFUSE:" in text
    assert "Not legal advice" in text
    assert "Instruction hierarchy" in text


def test_format_context_lists_source_url_and_last_updated():
    hits = [
        {
            "source_id": "oif-student-residence-en",
            "source_url": "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
            "title": "Residence",
            "last_updated": None,
            "score": 0.9,
            "text": "Students may work thirty hours weekly.",
        }
    ]
    block = format_context(hits)
    assert "source_id=oif-student-residence-en" in block
    assert "https://oif.gov.hu/factsheets/residence-of-the-student-pupil" in block
    assert "last_updated=not stated on the source metadata" in block


def test_ask_cites_a_retrieved_source(tmp_path: Path):
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
        }
    ]
    embedder = HashEmbedder()
    build_index(documents, embedder, tmp_path, size=200, overlap=20)
    result = ask(
        "thirty hours weekly",
        embedder,
        EchoGenerator(),
        tmp_path,
        top_k=1,
        min_score=0.01,
        prompt_version="v1",
        model_name="echo",
        generation_mode="echo",
    )
    assert not result.refused
    assert result.citations
    assert result.citations[0]["source_id"] == "oif-student-residence-en"
    assert result.citations[0]["source_url"].startswith("https://oif.gov.hu/")


def test_ask_refuses_when_nothing_is_retrieved(tmp_path: Path):
    documents = [
        {
            "source_id": "nje-regulations-index",
            "source_url": "https://nje.hu/en/regulation-and-documents",
            "title": "Index",
            "country": "HU",
            "institution": "NJE",
            "language": "en",
            "last_updated": None,
            "text": "Holiday schedule and dormitory places only.",
        }
    ]
    embedder = HashEmbedder()
    build_index(documents, embedder, tmp_path, size=200, overlap=20)
    result = ask(
        "quantum cryptography patent fees",
        embedder,
        EchoGenerator(),
        tmp_path,
        top_k=1,
        min_score=0.99,
        prompt_version="v1",
        model_name="echo",
        generation_mode="echo",
        retrieve_mode="dense",
        country="DE",
    )
    assert result.refused
    assert result.answer.startswith("REFUSE:")
    assert result.citations == []


def test_ask_refuses_out_of_scope_germany(tmp_path: Path):
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
        }
    ]
    embedder = HashEmbedder()
    build_index(documents, embedder, tmp_path, size=200, overlap=20)
    result = ask(
        "How do I get a German student visa for TU Munich?",
        embedder,
        EchoGenerator(),
        tmp_path,
        top_k=1,
        prompt_version="v1",
        model_name="echo",
        generation_mode="echo",
    )
    assert result.refused
    assert result.citations == []


def test_extract_citations_and_low_score_filter():
    text = "Work limit is thirty hours [oif-student-residence-en](https://oif.gov.hu/factsheets/residence-of-the-student-pupil)."
    cites = extract_citations(text)
    assert cites == [
        {
            "source_id": "oif-student-residence-en",
            "source_url": "https://oif.gov.hu/factsheets/residence-of-the-student-pupil",
        }
    ]
    assert filter_hits([{"score": 0.1}, {"score": 0.5}], 0.25) == [{"score": 0.5}]
    messages = build_messages("hours?", [{"source_id": "a", "source_url": "https://example.com", "title": "t", "last_updated": "2026-01-01", "score": 0.4, "text": "x"}], "v1")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "hours?" in messages[1]["content"]
    chosen = best_hit_for_question(
        "thirty hours weekly work",
        [
            {"source_id": "a", "source_url": "https://example.com/a", "score": 0.9, "text": "appeal within eight days"},
            {"source_id": "b", "source_url": "https://example.com/b", "score": 0.7, "text": "may work thirty hours weekly"},
        ],
    )
    assert chosen["source_id"] == "b"
