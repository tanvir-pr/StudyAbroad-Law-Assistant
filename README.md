# StudyAbroad Law Assistant

Plain-language answers about studying in Hungary, with a citation to an official page. This is not legal advice. Read [DISCLAIMER.md](DISCLAIMER.md) before using any answer.

Day 11 adds 30 unanswerable/out-of-scope questions, refusal/hallucination rates, and guardrails (instruction hierarchy, context sanitize, refuse without a matching citation).

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

The evaluation set is [eval/golden_set.jsonl](eval/golden_set.jsonl): **150 frozen** questions (see [eval/GOLDEN_FREEZE.md](eval/GOLDEN_FREEZE.md)). Described in [eval/DATA_CARD.md](eval/DATA_CARD.md). Do not add those questions to the fine-tuning set. A URL that `robots.txt` disallows, or whose certificate does not verify, is recorded in `manifest.json` and not downloaded.

## Build the FAISS index

```bash
python -m rag.index_faiss
```

Chunk size is 1,200 characters with a 200-character overlap (`configs/rag.yaml`). Embeddings are `BAAI/bge-m3`, L2-normalised, searched with inner product. The index is written to `data/index/` and is not committed. The golden questions are not indexed.

Search the saved index without rebuilding it:

```bash
python -m rag.index_faiss --query "how many hours a week may a student work"
```

## Ask one question

```bash
python -m rag.ask "How many hours a week may a student work during the study period?"
```

Uses prompt `rag/prompts/v1.txt`, retrieves top-5 chunks, and answers with `Qwen/Qwen2.5-1.5B-Instruct`. On CUDA the model loads in 4-bit; on CPU it loads full weights. A reply without a citation that matches a retrieved source is turned into a refuse. For a smoke test without downloading Qwen:

```bash
python -m rag.ask --echo "How many hours a week may a student work during the study period?"
```

## Compare chunk strategies

```bash
python -m eval.chunk_recall
```

Builds temporary indexes for `fixed`, `recursive`, and `section_aware`, then writes span coverage and Recall@1/3/5 to [eval/results/chunking_recall.md](eval/results/chunking_recall.md). Default embedder is a fast hash embedder so the bake-off finishes on CPU. For the same comparison with `BAAI/bge-m3`:

```bash
python -m eval.chunk_recall --embedder bge-m3
```

## Hybrid retrieve

`python -m rag.ask` now defaults to hybrid (FAISS + BM25, RRF). Optional filters:

```bash
python -m rag.ask --echo --institution OIF --country HU "How many hours may a student work?"
python -m rag.ask --echo --mode dense "How many hours may a student work?"
```

Compare dense / BM25 / hybrid on the golden set:

```bash
python -m eval.hybrid_recall
```

Scores land in [eval/results/hybrid_recall.md](eval/results/hybrid_recall.md).

## FAISS vs Chroma and reranking

Copy the FAISS vectors into a Chroma store (no re-embedding):

```bash
python -m rag.index_chroma
python -m eval.faiss_vs_chroma
```

Hybrid retrieve now reranks the top candidates with `BAAI/bge-reranker-v2-m3` by default:

```bash
python -m eval.rerank_recall
python -m rag.ask --echo --no-rerank "How many hours may a student work?"
```

## RAGAS-aligned scores

```bash
python -m eval.ragas_eval
python -m eval.ragas_eval --generator qwen
```

Default table uses extractive quoted answers on CPU:

```bash
python -m eval.ragas_eval --tag v0.2
python -m eval.error_categories --input eval/results/ragas_v0.2.json
```

Chart-ready numbers from days 6–9 live in [eval/results/charts_v0.2.md](eval/results/charts_v0.2.md).

## Unanswerable / hallucination (day 11)

```bash
python -m eval.hallucination_eval
python -m eval.hallucination_eval --mode qwen
```

Cases: [eval/unanswerable.jsonl](eval/unanswerable.jsonl) (30). Results: [eval/results/hallucination_day11.md](eval/results/hallucination_day11.md). Guards live in `safety/guards.py` and are wired into `rag.ask`.
