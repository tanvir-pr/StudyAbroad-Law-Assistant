# Chunking comparison (day 6)

Embedder: `HashEmbedder(dim=64)`. Size 1200, overlap 200.

| Strategy | Chunks | Span coverage | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 3920 | 0.933 | 0.183 | 0.317 | 0.367 |
| recursive | 3487 | 0.933 | 0.133 | 0.267 | 0.317 |
| section_aware | 4627 | 0.933 | 0.133 | 0.283 | 0.317 |

Span coverage = share of golden answers whose reference text is largely present inside some chunk of the cited source. Recall@k = cited `source_id` appears in the top-k hits.
