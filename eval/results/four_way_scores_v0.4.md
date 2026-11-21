# 4-way scores — `v0.4`

- Questions scored: **30**
- Judge: local heuristic (token F1 vs golden reference)
- Human sheet: `human_correctness_30_v0.4.jsonl` (**30** rows, draft_auto until edited)

| Arm | faithfulness | correctness (judge) | style | refuse | citation | p50_ms | p95_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.072 | 0.024 | 0.600 | 0.000 | 0.000 | 0 | 0 |
| base_lora | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0 | 0 |
| rag | 0.628 | 0.245 | 1.000 | 0.000 | 1.000 | 277 | 357 |
| rag_lora | 0.628 | 0.245 | 1.000 | 0.000 | 1.000 | 277 | 324 |

Human draft accuracy on rag_lora (n=30, draft_auto): **0.433**

Not legal advice. Echo stand-in numbers are for pipeline completeness on CPU; re-run with real Qwen/LoRA adapters on GPU for portfolio GPU numbers.
