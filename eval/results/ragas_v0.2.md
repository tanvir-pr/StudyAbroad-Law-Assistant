# RAGAS-aligned scores (v0.2 freeze)

Generator: `extractive-quote`. Reranker: `OverlapReranker`.
Local judges (no paid API): context precision/recall use gold `source_id`; faithfulness is context-token support; answer relevancy is bge-m3 cosine.

| Metric | Score |
| --- | ---: |
| context_precision | 0.365 |
| context_recall | 0.793 |
| faithfulness | 0.978 |
| answer_relevancy | 0.474 |
| refuse_rate | 0.000 |

Questions scored: 150.
