Measured results — `dynamic_masking`, BERT-base architecture, segment-pair+NSP input format, batch
256, 1M steps, BookCorpus+Wikipedia 16GB. Development-set numbers, medians over 5 seeds.

## Static vs. dynamic masking (our reimplementation)
| masking | SQuAD 2.0 F1 | MNLI-m acc | SST-2 acc |
|---|---|---|---|
| reference (published) | 76.3 | 84.3 | 92.8 |
| static (rung 1) | 78.3 | 84.3 | 92.5 |
| dynamic (rung 2) | 78.7 | 84.0 | 92.9 |

Notes: dynamic masking's SQuAD 2.0 F1 is 0.4 points above static; SST-2 is 0.4 points above static;
MNLI-m is 0.3 points below static. No tenfold data duplication is needed under dynamic masking.
