from rag.chunking import chunk_text, fixed_chunks, recursive_chunks, section_aware_chunks


def test_recursive_and_section_strategies_return_chunks():
    text = "Intro paragraph.\n\n## Work rules\nStudents may work thirty hours weekly.\n\n--- page 2 ---\n\nOutside term they may work ninety days."
    fixed = fixed_chunks(text, size=80, overlap=20)
    recursive = recursive_chunks(text, size=80, overlap=20)
    section = section_aware_chunks(text, size=80, overlap=20)
    assert fixed and recursive and section
    assert any("thirty hours" in chunk for chunk in recursive)
    assert any("thirty hours" in chunk for chunk in section)
    assert chunk_text(text, "fixed", 80, 20) == fixed


def test_section_aware_keeps_page_boundaries_separate_when_small():
    text = "--- page 1 ---\nAlpha rule one.\n\n--- page 2 ---\nBeta rule two."
    chunks = section_aware_chunks(text, size=200, overlap=20)
    assert len(chunks) >= 2
    assert any("Alpha" in chunk for chunk in chunks)
    assert any("Beta" in chunk for chunk in chunks)


def test_unknown_strategy_raises():
    try:
        chunk_text("hello", "magic", 50, 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
