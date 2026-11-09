import json

from ingestion.sources import iter_sources

GOLDEN = __import__("pathlib").Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"


def test_golden_set_has_forty_questions_cited_to_the_allowlist():
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 40
    urls = {source["url"] for source in iter_sources()}
    ids = {source["id"] for source in iter_sources()}
    seen = set()
    for row in rows:
        assert row["id"] not in seen
        seen.add(row["id"])
        assert row["question"].strip()
        assert row["reference_answer"].strip()
        assert row["source_id"] in ids
        assert row["source_url"] in urls
        match = next(source for source in iter_sources() if source["id"] == row["source_id"])
        assert row["source_url"] == match["url"]
