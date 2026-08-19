Measured results — `bert_base_reimpl`, fairseq reimplementation of BERT-base (static masking,
segment-pair+NSP, batch 256, 1M steps, BookCorpus+Wikipedia 16GB), development-set numbers.

## vs. BERT-base published (medians over 5 seeds where noted)
| model | SQuAD 1.1/2.0 F1 | MNLI-m acc | SST-2 acc | RACE acc |
|---|---|---|---|---|
| BERT-base (published) | 88.5 / 76.3 | 84.3 | 92.8 | 64.3 |
| our reimplementation (segment-pair+NSP) | 90.4 / 78.7 | 84.0 | 92.9 | 64.2 |

## Cross-check (static-masking configuration, SQuAD2.0/MNLI-m/SST-2 only)
| | SQuAD 2.0 F1 | MNLI-m acc | SST-2 acc |
|---|---|---|---|
| reference (published) | 76.3 | 84.3 | 92.8 |
| our reimplementation (static masking) | 78.3 | 84.3 | 92.5 |

Notes: SQuAD F1 (both v1.1 and v2.0) comes in above the published reference; MNLI-m and SST-2 are
within 0.3 points of the reference in both directions; RACE matches to within 0.1 points. No task
shows a large or systematic shortfall.
