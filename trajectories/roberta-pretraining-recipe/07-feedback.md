Measured results — `pretrain_300k` (combined recipe, 160GB data, extended to 300,000 steps),
BERT-large architecture, 8K batch. Development-set numbers, SQuAD reported as v1.1/v2.0 F1.

| model | data | bsz | steps | SQuAD 1.1/2.0 F1 | MNLI-m acc | SST-2 acc |
|---|---|---|---|---|---|---|
| combined recipe + additional data, 100K (rung 6) | 160GB | 8K | 100K | 94.0 / 87.7 | 89.3 | 95.6 |
| combined recipe + additional data, 300K (this rung) | 160GB | 8K | 300K | 94.4 / 88.7 | 90.0 | 96.1 |

Notes: holding data fixed at 160GB, extending 100K -> 300K steps improves every column relative to
rung 6: +0.4/+1.0 SQuAD F1, +0.7 MNLI-m, +0.5 SST-2. The SQuAD 2.0 and MNLI-m gains here are larger
than the gains produced by the data-only increase in rung 6 (+0.4 SQuAD2.0, +0.3 MNLI-m).
