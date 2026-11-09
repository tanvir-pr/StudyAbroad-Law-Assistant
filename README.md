# StudyAbroad Law Assistant

Plain-language answers about studying in Hungary, with a citation to an official page. This is not legal advice. Read [DISCLAIMER.md](DISCLAIMER.md) before using any answer.

Day 4 of 22 splits the saved texts into fixed chunks, embeds them with BAAI/bge-m3, and stores a FAISS index. The golden set is 40 questions. The assistant does not write an answer yet.

## Scope locked on 1 November 2026

Hungary only: student residence permits, work during studies, Stipendium Hungaricum, plus the public study rules of ELTE, BME, and John von Neumann University (Kecskemét). EU context is Directive (EU) 2016/801 and the EU Immigration Portal page for students in Hungary.

The allow-list is [configs/sources.yaml](configs/sources.yaml). Twenty-seven public URLs. The crawler fetches the primary URL of each entry and must respect `robots.txt`. Alternate URLs are the same text on another official host and are not fetched.

Pages left out on purpose:

- The 2020 Stipendium Hungaricum rules PDF. The Operational Regulations effective 31 August 2026 replace it.
- The Enter Hungary login site. It is where you submit a form, not the rule text.
- Germany and BAMF. Those come after day 22.

## Run the day-1 check

```bash
pip install -r requirements.txt
pytest
```

`pytest` checks that every source has a unique id, a required field set, and an `https` URL.

## Download the allow-list

```bash
python -m ingestion.crawl
```

Raw files land in `data/raw/`, which stays out of git. Each saved page gets a `.meta.json` with country, institution, doc type, language, source URL, last-updated, and retrieved-at. HTML is parsed during the download. PDFs are parsed afterwards:

```bash
python -m ingestion.parse_pdf
```

The evaluation set is [eval/golden_set.jsonl](eval/golden_set.jsonl): 40 questions with reference answers and source URLs. It is described in [eval/DATA_CARD.md](eval/DATA_CARD.md). Do not add those questions to the fine-tuning set. A URL that `robots.txt` disallows, or whose certificate does not verify, is recorded in `manifest.json` and not downloaded.

## Build the FAISS index

```bash
python -m rag.index_faiss
```

Chunk size is 1,200 characters with a 200-character overlap (`configs/rag.yaml`). Embeddings are `BAAI/bge-m3`, L2-normalised, searched with inner product. The index is written to `data/index/` and is not committed. The golden questions are not indexed.

Search the saved index without rebuilding it:

```bash
python -m rag.index_faiss --query "how many hours a week may a student work"
```
