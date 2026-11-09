# Data card — Hungary student-rules corpus

Draft for day 4. This is not legal advice. The files under `data/raw/` and `data/index/` are local copies and a local FAISS index. They are not committed.

## What this is

Plain text and PDFs collected so a later retrieval system can answer questions about studying in Hungary and point at the page the sentence came from. The golden questions are an evaluation set. They must not be copied into the fine-tuning data.

## Scope

Locked on 1 November 2026 in `configs/sources.yaml`. Retrieved on 4 November 2026, with the EUR-Lex HTML copy added on 9 November 2026.

Hungary only: student residence permits, work during studies, Stipendium Hungaricum, and the public study rules of ELTE, BME, and John von Neumann University. EU context is Directive (EU) 2016/801 and the EU Immigration Portal page for students in Hungary.

## Size

| Item | Count |
| --- | --- |
| Allow-list URLs | 27 |
| Saved | 26 |
| Not saved | 1 |
| HTML pages with extracted text | 15 |
| PDFs with extracted text | 11 |
| Golden questions | 40 |
| Chunk size / overlap | 1,200 / 200 characters |
| Embedding model | BAAI/bge-m3 |
| Dense index | FAISS IndexFlatIP, under `data/index/` |

The one page not saved is the BME English rules index at `etk.bme.hu`. Its TLS certificate did not verify, so the crawler did not download it. The English Code of Studies (from `vik.bme.hu`) and the Hungarian TVSZ are in the corpus. If those two disagree, the English PDF says the Hungarian text prevails.

## Where the text came from

- OIF factsheets on student residence, extension, and employer notification.
- Nemzeti Jogszabálytár: Act XC of 2023 and Government Decree 35/2024. The act text is the in-force consolidation, not the January–September 2026 time slice.
- Tempus Public Foundation: Operational Regulations effective 31 August 2026, General Terms effective 1 September 2026, the 2026/27 calls and application guide, and the scholarship-holder FAQ. The 2020 rules PDF is not included.
- ELTE Academic Regulations for Students, amended version effective 1 September 2026, and the 2026/27 international student guide.
- BME Code of Studies (English translation consolidated through February 2024) and the Hungarian code effective 17 July 2025.
- John von Neumann University Academic and Examination Regulations, 6th version, in force 1 June 2026, and the 2026/27 study guide.
- Directive (EU) 2016/801 as the EUR-Lex HTML for CELEX 32016L0801. The ELI URL returned HTTP 202 with an empty body on 4 November 2026. The HTML copy was fetched on 9 November 2026. `robots.txt` on that host also returned an empty HTTP 202, which is not a Disallow rule.
- EU Immigration Portal, "Student in Hungary", page dated 18 September 2025. It is a summary. For a permit inside Hungary, cite OIF or the statute.

## Terms

Only public official URLs, fetched with `robots.txt` respected. Downloaded files stay in `data/raw/` and out of git. Do not republish the full regulations. Each later answer has to cite a saved source or refuse.

## Known gaps

- BME's English index page is missing because the certificate did not verify.
- The John von Neumann University visa HTML page is mostly navigation. The study guide PDF is the page to quote for permit collection.
- Jogtár and university sites can change a paragraph without changing the URL. `retrieved_at` on each `.meta.json` is the fetch time. `last_updated` is set only when the allow-list recorded a date from the document itself.
