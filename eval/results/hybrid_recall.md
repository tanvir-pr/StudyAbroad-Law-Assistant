# Hybrid retrieval comparison (day 7)

Embedder/index: `BAAI/bge-m3` over `E:\Ai ml all project\Phase 4 Sequence, NLP and LLM\Project2\data\index`.
Chunking locked: `fixed` 1200/200.

| Mode | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| dense | 0.483 | 0.667 | 0.717 |
| bm25 | 0.483 | 0.700 | 0.800 |
| hybrid | 0.567 | 0.717 | 0.750 |

