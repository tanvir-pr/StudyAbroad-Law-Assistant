# StudyAbroad Law Assistant

Plain-language answers about studying in Hungary, with a citation to an official page. This is not legal advice. Read [DISCLAIMER.md](DISCLAIMER.md) before using any answer.

Day 1 of 22 is the locked source list. Nothing is downloaded yet, and the assistant does not answer questions yet.

## Scope locked on 1 November 2026

Hungary only: student residence permits, work during studies, Stipendium Hungaricum, plus the public study rules of ELTE, BME, and John von Neumann University (Kecskemét). EU context is Directive (EU) 2016/801 and the EU Immigration Portal page for students in Hungary.

The allow-list is [configs/sources.yaml](configs/sources.yaml). Twenty-seven public URLs. The crawler, when it is written on day 2, may fetch the primary URL of each entry and must respect `robots.txt`. Alternate URLs are the same text on another official host and are not fetched.

Pages left out on purpose:

- The 2020 Stipendium Hungaricum rules PDF. The Operational Regulations effective 31 August 2026 replace it.
- The Enter Hungary login site. It is where you submit a form, not the rule text.
- Germany and BAMF. Those come after day 22.

## Run the day-1 check

```bash
pip install -r requirements.txt
pytest
```

`pytest` checks that every source has a unique id, a required field set, and an `https` URL.

## What day 2 adds

A crawler that downloads only this list into `data/raw/`, which stays out of git.
