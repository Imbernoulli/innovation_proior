Measured results — `more_data` (combined recipe + additional pretraining data), BERT-large
architecture, 8K batch, 100,000 steps. Development-set numbers, SQuAD reported as v1.1/v2.0 F1.

| model | data | bsz | steps | SQuAD 1.1/2.0 F1 | MNLI-m acc | SST-2 acc |
|---|---|---|---|---|---|---|
| combined recipe, 16GB (rung 5) | 16GB | 8K | 100K | 93.6 / 87.3 | 89.0 | 95.3 |
| combined recipe + additional data (this rung) | 160GB | 8K | 100K | 94.0 / 87.7 | 89.3 | 95.6 |

Notes: holding steps fixed at 100K, adding the additional pretraining data (16GB -> 160GB) improves
every column relative to rung 5: +0.4/+0.4 SQuAD F1, +0.3 MNLI-m, +0.3 SST-2. Improvements are in the
few-tenths range on every metric, not a large jump.
