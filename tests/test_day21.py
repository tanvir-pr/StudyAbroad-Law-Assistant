from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from t5.translate_bn import chrf, gloss_translate
from t5.what_changed import diff_changes, load_pairs
from ingestion.sources import ROOT


def test_human_refs_total_forty():
    path = ROOT / "t5" / "human_refs_40.jsonl"
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert n == 40


def test_gloss_translate_and_chrf():
    text = "Students may extend a study residence permit on Enter Hungary. Not legal advice."
    bn = gloss_translate(text)
    assert "Enter Hungary" in bn or "অধ্যয়ন" in bn or "শিক্ষার্থী" in bn
    assert chrf(bn, bn) > 0.9


def test_what_changed_two_pairs():
    pairs = load_pairs(ROOT / "t5" / "what_changed_pairs.json")
    assert len(pairs) == 2
    changes = diff_changes(pairs[0]["old_text"], pairs[0]["new_text"])
    assert changes


def test_api_summarize_translate_changes(monkeypatch):
    monkeypatch.setenv("ASK_ECHO", "1")
    with TestClient(app) as client:
        s = client.post(
            "/summarize",
            json={
                "text": "Students may work thirty hours weekly during the study period. "
                "Extension is filed on Enter Hungary. Biometrics follow. Not legal advice. Extra padding here.",
                "source_id": "oif-student-residence-en",
                "source_url": "https://oif.gov.hu/x",
            },
        )
        assert s.status_code == 200
        assert "summary" in s.json()
        assert s.json()["disclaimer"]

        t = client.post("/translate", json={"text": "Not legal advice.", "source_lang": "en"})
        assert t.status_code == 200
        assert t.json()["target_lang"] == "bn"

        c = client.post(
            "/changes",
            json={"old_text": "Old rule A. Keep B.", "new_text": "New rule C. Keep B.", "source_id": "demo"},
        )
        assert c.status_code == 200
        assert c.json()["changes"]
