# 4-way generation — `day18`

- Questions: **20**
- Echo stand-in: **True**

| Arm | refuse_rate | citation_rate | p50_ms | p95_ms |
|---|---:|---:|---:|---:|
| base | 0.000 | 0.000 | 0 | 0 |
| base_lora | 1.000 | 0.000 | 0 | 0 |
| rag | 0.000 | 1.000 | 371 | 522 |
| rag_lora | 0.000 | 1.000 | 345 | 496 |

Arms: `base` (no RAG), `base_lora` (no RAG + LoRA), `rag` (hybrid+rerank), `rag_lora` (hybrid+rerank + LoRA).

Day 19 finishes scoring (faithfulness, correctness, style, human 30).
