import json
from pathlib import Path

from ingestion.parse_pdf import extract_saved_pdfs, pdf_to_text

def _tiny_pdf(sentence: str) -> bytes:
    safe = sentence.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({safe}) Tj ET".encode("ascii")
    parts = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        b"4 0 obj\n<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    blob = b"%PDF-1.4\n"
    offsets = [0]
    for part in parts:
        offsets.append(len(blob))
        blob += part
    xref_pos = len(blob)
    xref = [b"xref\n", f"0 {len(offsets)}\n".encode(), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode())
    blob += b"".join(xref)
    blob += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return blob


def test_pdf_to_text_keeps_the_page_sentence():
    text, pages = pdf_to_text(_tiny_pdf("Thirty hours a week"))
    assert pages == 1
    assert "Thirty hours a week" in text
    assert "--- page 1 ---" in text


def test_extract_saved_pdfs_writes_text_and_updates_manifest(tmp_path: Path):
    pdf_path = tmp_path / "sample-rule.pdf"
    pdf_path.write_bytes(_tiny_pdf("Thirty hours a week"))
    record = {
        "source_id": "sample-rule",
        "raw_path": "sample-rule.pdf",
        "text_path": None,
        "text_chars": None,
    }
    (tmp_path / "manifest.json").write_text(json.dumps([record]), encoding="utf-8")
    (tmp_path / "sample-rule.meta.json").write_text("{}", encoding="utf-8")
    extract_saved_pdfs(tmp_path)
    written = (tmp_path / "sample-rule.txt").read_text(encoding="utf-8")
    assert "Thirty hours a week" in written
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["text_chars"] == len(written)
    assert manifest[0]["page_count"] == 1
    assert manifest[0]["extraction_error"] is None
    meta = json.loads((tmp_path / "sample-rule.meta.json").read_text(encoding="utf-8"))
    assert meta["page_count"] == 1
