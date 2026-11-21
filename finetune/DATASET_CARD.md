---
license: other
task_categories:
  - question-answering
language:
  - en
  - hu
tags:
  - hungary
  - study-abroad
  - residence-permit
  - stipendium-hungaricum
  - not-legal-advice
pretty_name: StudyAbroad Law Q&A (allow-list)
size_categories:
  - 1K<n<10K
---

# StudyAbroad Law Q&A — dataset card (day 16)

**Not legal advice.** Answers are plain-language paraphrases of public official pages on a locked allow-list. Do not use this as counsel for a real immigration or university decision.

## Summary

| Item | Value |
| --- | --- |
| Rows | 1000 reviewed Q&A pairs |
| Split seed | 42 |
| Split ratios | train 80% / validation 10% / test 10% (topic-stratified) |
| Source | Fixed-chunk spans from `configs/sources.yaml` allow-list |
| Golden eval set | **Excluded** (`eval/golden_set.jsonl` never copied here) |
| Style | Short steps + safe fallback (`finetune/STYLE_GUIDE.md`) |

## Topics

Assigned from `source_id` prefixes:

- `student_residence_permit` — OIF / NJT residence and extension pages
- `stipendium_hungaricum` — Tempus / SH calls and regulations
- `university_regulations` — ELTE, BME, NJE
- `eu_student_mobility` — Directive (EU) 2016/801 and EU Immigration Portal

Exact counts live in `split_summary.json` after `python -m finetune.split_dataset`.

## Fields

Each JSONL row includes: `id`, `question`, `answer`, `answer_span`, `source_id`, `source_url`, `chunk_id`, `language`, `institution`, `country`, `topic`, `review_status`.

## How it was built

1. Draft from FAISS chunk metadata (`python -m finetune.draft_qa`).
2. Accept only span+citation matches; block golden questions (`python -m finetune.review_qa`).
3. Topic-stratified split (`python -m finetune.split_dataset`).

## What this is not

- Not a substitute for OIF, Tempus, or university offices.
- Not the golden evaluation set (150 frozen questions).
- Not scraped beyond the robots-respecting allow-list crawl.

## Upload (token never committed)

```bash
set HF_TOKEN=hf_...   # PowerShell: $env:HF_TOKEN="hf_..."
python -m finetune.upload_hf --repo-id YOUR_USER/studyabroad-law-qa-v0
```

Do not commit `.hf_token`, `hf_token.txt`, or any `HF_TOKEN` value.
