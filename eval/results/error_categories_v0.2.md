# Error categories — v0.2 (30 examples)

Source run: `eval/results/ragas_v0.2.json`.

| Category | Count | Meaning |
| --- | ---: | --- |
| wrong_source_top | 15 | Gold source missing from top-k (context_recall = 0) |
| diluted_context | 7 | Gold source present but context_precision < 0.4 |
| low_relevancy | 7 | Answer relevancy < 0.35 while context_recall = 1 |
| low_faithfulness | 1 | Faithfulness < 0.7 |

## Examples

- **g62** `wrong_source_top` gold=`sh-regulations-index` retrieved=['sh-call-phd-2026-27', 'sh-call-ba-ma-2026-27', 'elte-academic-regulations-en'] (P=0.00, R=0.00, F=1.00, Rel=0.28)
- **g106** `wrong_source_top` gold=`bme-tvsz-hu` retrieved=['elte-academic-regulations-en', 'bme-code-of-studies-en', 'bme-code-of-studies-en'] (P=0.00, R=0.00, F=1.00, Rel=0.28)
- **g67** `wrong_source_top` gold=`bme-tvsz-hu` retrieved=['sh-application-guide-2026-27', 'elte-regulations-index', 'elte-regulations-index'] (P=0.00, R=0.00, F=1.00, Rel=0.29)
- **g86** `wrong_source_top` gold=`nje-study-guide-2026-27` retrieved=['elte-international-guide-2026-27', 'oif-study-permit-extension', 'elte-international-guide-2026-27'] (P=0.00, R=0.00, F=1.00, Rel=0.30)
- **g41** `wrong_source_top` gold=`oif-student-residence-hu` retrieved=['oif-higher-education-institutions', 'sh-regulations-index', 'sh-regulations-index'] (P=0.00, R=0.00, F=1.00, Rel=0.32)
- **g42** `wrong_source_top` gold=`nje-study-guide-2026-27` retrieved=['elte-international-guide-2026-27', 'elte-international-guide-2026-27', 'elte-international-guide-2026-27'] (P=0.00, R=0.00, F=1.00, Rel=0.32)
- **g66** `wrong_source_top` gold=`elte-regulations-index` retrieved=['elte-academic-regulations-en', 'elte-academic-regulations-en', 'elte-academic-regulations-en'] (P=0.00, R=0.00, F=1.00, Rel=0.35)
- **g92** `diluted_context` gold=`oif-student-residence-en` retrieved=['oif-student-residence-en', 'elte-international-guide-2026-27', 'elte-international-guide-2026-27'] (P=0.20, R=1.00, F=1.00, Rel=0.29)
- **g139** `diluted_context` gold=`nje-study-guide-2026-27` retrieved=['nje-study-guide-2026-27', 'elte-international-guide-2026-27', 'elte-international-guide-2026-27'] (P=0.20, R=1.00, F=1.00, Rel=0.29)
- **g113** `diluted_context` gold=`nje-academic-regulations-en` retrieved=['nje-academic-regulations-en', 'elte-academic-regulations-en', 'elte-academic-regulations-en'] (P=0.20, R=1.00, F=0.00, Rel=0.33)
- **g126** `diluted_context` gold=`sh-call-ba-ma-2026-27` retrieved=['sh-call-phd-2026-27', 'sh-operational-regulations', 'sh-regulations-index'] (P=0.20, R=1.00, F=1.00, Rel=0.33)
- **g141** `diluted_context` gold=`sh-application-guide-2026-27` retrieved=['sh-application-guide-2026-27', 'sh-call-ba-ma-2026-27', 'sh-call-phd-2026-27'] (P=0.20, R=1.00, F=1.00, Rel=0.34)
- **g80** `diluted_context` gold=`sh-general-terms` retrieved=['elte-international-guide-2026-27', 'sh-call-ba-ma-2026-27', 'sh-regulations-index'] (P=0.20, R=1.00, F=1.00, Rel=0.36)
- **g124** `diluted_context` gold=`sh-call-ba-ma-2026-27` retrieved=['sh-application-guide-2026-27', 'sh-call-phd-2026-27', 'sh-call-phd-2026-27'] (P=0.20, R=1.00, F=1.00, Rel=0.37)
- **g122** `low_faithfulness` gold=`oif-study-permit-extension` retrieved=['oif-student-residence-en', 'oif-study-permit-extension', 'oif-study-permit-extension'] (P=0.40, R=1.00, F=0.00, Rel=0.30)
- **g114** `low_relevancy` gold=`nje-study-guide-2026-27` retrieved=['nje-study-guide-2026-27', 'elte-international-guide-2026-27', 'nje-study-guide-2026-27'] (P=0.40, R=1.00, F=1.00, Rel=0.27)
- **g32** `low_relevancy` gold=`nje-academic-regulations-en` retrieved=['nje-study-guide-2026-27', 'nje-academic-regulations-en', 'nje-academic-regulations-en'] (P=0.40, R=1.00, F=1.00, Rel=0.28)
- **g36** `low_relevancy` gold=`oif-enter-hungary-student-manual` retrieved=['oif-enter-hungary-student-manual', 'oif-enter-hungary-student-manual', 'sh-call-ba-ma-2026-27'] (P=0.40, R=1.00, F=1.00, Rel=0.28)
- **g135** `low_relevancy` gold=`sh-general-terms` retrieved=['sh-general-terms', 'sh-general-terms', 'sh-operational-regulations'] (P=0.40, R=1.00, F=1.00, Rel=0.30)
- **g117** `low_relevancy` gold=`oif-enter-hungary-student-manual` retrieved=['oif-enter-hungary-student-manual', 'oif-enter-hungary-student-manual', 'elte-international-guide-2026-27'] (P=0.40, R=1.00, F=1.00, Rel=0.32)
- **g146** `low_relevancy` gold=`sh-operational-regulations` retrieved=['sh-operational-regulations', 'sh-operational-regulations', 'elte-international-guide-2026-27'] (P=0.40, R=1.00, F=1.00, Rel=0.33)
- **g28** `low_relevancy` gold=`sh-general-terms` retrieved=['sh-general-terms', 'sh-general-terms', 'sh-general-terms'] (P=1.00, R=1.00, F=1.00, Rel=0.29)
- **g148** `wrong_source_top` gold=`bme-tvsz-hu` retrieved=['sh-operational-regulations', 'sh-application-guide-2026-27', 'bme-code-of-studies-en'] (P=0.00, R=0.00, F=1.00, Rel=0.35)
- **g142** `wrong_source_top` gold=`njt-decree-35-2024` retrieved=['nje-academic-regulations-en', 'oif-higher-education-institutions', 'oif-student-residence-en'] (P=0.00, R=0.00, F=1.00, Rel=0.35)
- **g133** `wrong_source_top` gold=`oif-student-residence-hu` retrieved=['sh-faq', 'sh-regulations-index', 'sh-faq'] (P=0.00, R=0.00, F=1.00, Rel=0.36)
- **g70** `wrong_source_top` gold=`sh-faq` retrieved=['sh-regulations-index', 'elte-international-guide-2026-27', 'elte-international-guide-2026-27'] (P=0.00, R=0.00, F=1.00, Rel=0.37)
- **g51** `wrong_source_top` gold=`njt-decree-35-2024` retrieved=['oif-higher-education-institutions', 'oif-student-residence-en', 'oif-student-residence-en'] (P=0.00, R=0.00, F=1.00, Rel=0.37)
- **g109** `wrong_source_top` gold=`oif-study-permit-extension` retrieved=['oif-enter-hungary-student-manual', 'elte-international-guide-2026-27', 'sh-regulations-index'] (P=0.00, R=0.00, F=0.00, Rel=0.41)
- **g35** `wrong_source_top` gold=`njt-act-xc-2023` retrieved=['oif-study-permit-extension', 'oif-student-residence-en', 'oif-enter-hungary-student-manual'] (P=0.00, R=0.00, F=1.00, Rel=0.43)
- **g150** `wrong_source_top` gold=`njt-act-xc-2023` retrieved=['elte-international-guide-2026-27', 'oif-higher-education-institutions', 'eurlex-directive-2016-801'] (P=0.00, R=0.00, F=0.97, Rel=0.46)
