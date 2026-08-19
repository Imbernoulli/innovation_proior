Measured results — `pretrain_500k` (combined recipe, 160GB data, extended to 500,000 steps),
BERT-large architecture, 8K batch. Development-set numbers, SQuAD reported as v1.1/v2.0 F1.

| model | data | bsz | steps | SQuAD 1.1/2.0 F1 | MNLI-m acc | SST-2 acc |
|---|---|---|---|---|---|---|
| combined recipe + additional data, 100K | 160GB | 8K | 100K | 94.0 / 87.7 | 89.3 | 95.6 |
| combined recipe + additional data, 300K | 160GB | 8K | 300K | 94.4 / 88.7 | 90.0 | 96.1 |
| combined recipe + additional data, 500K (this rung) | 160GB | 8K | 500K | 94.6 / 89.4 | 90.2 | 96.4 |

Notes: 100K -> 300K step gains were +0.4/+1.0 SQuAD F1, +0.7 MNLI-m, +0.5 SST-2. 300K -> 500K step
gains are +0.2/+0.7 SQuAD F1, +0.2 MNLI-m, +0.3 SST-2 — smaller than the 100K-to-300K gains on every
column but still positive on every column; no metric stalls or reverses. This 500K-step configuration
posts the best number in every column measured across all eight rungs of this study.
