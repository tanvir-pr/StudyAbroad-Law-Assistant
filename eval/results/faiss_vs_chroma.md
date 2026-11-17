# FAISS vs Chroma (day 8)

Embedder: `BAAI/bge-m3`. Same vectors; Chroma is a copy of the FAISS index.

| Backend | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| faiss | 0.478 | 0.656 | 0.733 |
| chroma | 0.478 | 0.656 | 0.733 |

