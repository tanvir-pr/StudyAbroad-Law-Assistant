# 4-way generation — `v0.4`

- Questions: **30**
- Echo stand-in: **True**

| Arm | refuse_rate | citation_rate | p50_ms | p95_ms |
|---|---:|---:|---:|---:|
| base | 0.000 | 0.000 | 0 | 0 |
| base_lora | 1.000 | 0.000 | 0 | 0 |
| rag | 0.000 | 1.000 | 277 | 357 |
| rag_lora | 0.000 | 1.000 | 277 | 324 |

Arms: `base` (no RAG), `base_lora` (no RAG + LoRA), `rag` (hybrid+rerank), `rag_lora` (hybrid+rerank + LoRA).

Day 19 finishes scoring (faithfulness, correctness, style, human 30).
