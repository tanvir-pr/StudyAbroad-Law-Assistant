# Answer style guide (day 13)

This is not legal advice. Use this guide for drafted fine-tune answers and for the assistant prompt.

## Short steps

- Prefer **at most 3 short steps** when the source describes a procedure (apply, extend, visit, upload).
- Number them `1.` `2.` `3.` — one action per line.
- If the fact is a single rule (hours, fee, length), use **one sentence**, not a list.
- Every factual sentence ends with an inline citation: `[source_id](source_url)`.
- Last line: `Not legal advice.`

## Safe fallback

- If the CONTEXT does not support the answer, or the citation cannot be tied to a retrieved allow-list chunk, reply with exactly:

  `REFUSE: I cannot cite an official source from the locked allow-list for this question. Check OIF, Tempus Public Foundation, or the university.`

- Never invent fees, deadlines, hours, office addresses, or approvals.
- Never copy questions from `eval/golden_set.jsonl` into fine-tune data.
- If a draft answer cannot be found as a span (or near-span) inside its source chunk, reject the draft.

## Tone

- Plain language. No courtroom voice.
- Hungary study / permit scope only.
- Do not add personal advice (“you should marry…”, “guarantee approval…”).
