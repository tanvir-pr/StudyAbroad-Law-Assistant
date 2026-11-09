"""Extract text from allow-list PDFs already saved under data/raw."""

import io
import json
from pathlib import Path

from pypdf import PdfReader

from ingestion.sources import ROOT


def pdf_to_text(payload: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(payload))
    if reader.is_encrypted:
        reader.decrypt("")
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        pages.append(f"--- page {index} ---\n{extracted.strip()}")
    return "\n\n".join(pages).strip(), len(reader.pages)


def extract_saved_pdfs(raw_dir: Path | None = None) -> list[dict]:
    raw_dir = raw_dir or (ROOT / "data" / "raw")
    manifest_path = raw_dir / "manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in records:
        raw_rel = record.get("raw_path")
        if not raw_rel or not str(raw_rel).endswith(".pdf"):
            continue
        pdf_path = ROOT / raw_rel if not Path(raw_rel).is_absolute() else Path(raw_rel)
        if not pdf_path.exists():
            pdf_path = raw_dir / Path(raw_rel).name
        try:
            text, page_count = pdf_to_text(pdf_path.read_bytes())
        except Exception as exc:
            record["text_path"] = None
            record["text_chars"] = 0
            record["page_count"] = None
            record["extraction_error"] = f"{type(exc).__name__}: {exc}"
            _write_meta(raw_dir, record)
            continue
        text_path = raw_dir / f"{record['source_id']}.txt"
        text_path.write_text(text, encoding="utf-8")
        record["text_path"] = _relative(text_path)
        record["text_chars"] = len(text)
        record["page_count"] = page_count
        record["extraction_error"] = None
        _write_meta(raw_dir, record)
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return records


def _write_meta(raw_dir: Path, record: dict) -> None:
    path = raw_dir / f"{record['source_id']}.meta.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    records = extract_saved_pdfs()
    pdfs = [record for record in records if str(record.get("raw_path") or "").endswith(".pdf")]
    for record in pdfs:
        if record.get("extraction_error"):
            print(f"fail {record['source_id']}: {record['extraction_error']}")
        else:
            print(f"ok {record['source_id']}: {record['page_count']} pages, {record['text_chars']} chars")


if __name__ == "__main__":
    main()
