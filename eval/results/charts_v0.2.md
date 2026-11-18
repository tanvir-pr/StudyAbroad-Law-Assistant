# Retrieval comparison numbers for charts (day 10 / v0.2)

These figures are copied from the day-6 to day-9 result files so README graphs and slides can quote one place.

## Chunking (day 6, HashEmbedder bake-off)

Source: `chunking_recall.md`

| Strategy | Chunks | Span coverage | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 3920 | 0.933 | 0.183 | 0.317 | 0.367 |
| recursive | 3487 | 0.933 | 0.133 | 0.267 | 0.317 |
| section_aware | 4627 | 0.933 | 0.133 | 0.283 | 0.317 |

Locked choice: **fixed 1200/200**.

## Dense vs BM25 vs hybrid (day 7, bge-m3, golden 60)

Source: `hybrid_recall.md`

| Mode | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| dense | 0.483 | 0.667 | 0.717 |
| bm25 | 0.483 | 0.700 | 0.800 |
| hybrid | 0.567 | 0.717 | 0.750 |

## FAISS vs Chroma (day 8, same vectors, golden 90)

Source: `faiss_vs_chroma.md`

| Backend | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| faiss | 0.478 | 0.656 | 0.733 |
| chroma | 0.478 | 0.656 | 0.733 |

## Reranker (day 8, golden 90)

Source: `rerank_recall.md`

| Mode | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| hybrid | 0.522 | 0.722 | 0.756 |
| hybrid+rerank | 0.644 | 0.833 | 0.889 |

## RAGAS-aligned quality

| Metric | Day 9 (n=120) | v0.2 freeze (n=150) |
| --- | ---: | ---: |
| context_precision | 0.370 | 0.365 |
| context_recall | 0.783 | 0.793 |
| faithfulness | 0.981 | 0.978 |
| answer_relevancy | 0.484 | 0.474 |

Sources: `ragas_day9.md`, `ragas_v0.2.md`.

## Error categories (v0.2, 30 examples)

Source: `error_categories_v0.2.md` — main failure mode is retrieving a related but wrong allow-list source (for example OIF EN vs HU, or FAQ vs call PDF).

## Unanswerable / hallucination (day 11)

Source: `hallucination_day11.md` (echo generator)

| Metric | Value |
| --- | ---: |
| Cases | 30 |
| Refuse rate | 1.000 |
| Hallucination rate | 0.000 |
