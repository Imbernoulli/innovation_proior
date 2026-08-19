Measured results — `combined_recipe_bert_large` (dynamic masking + full-sentences/no-NSP + 8K batch +
byte-level BPE), BERT-large architecture, BookCorpus+Wikipedia 16GB, 100,000 steps. Development-set
numbers, SQuAD reported as v1.1/v2.0 F1.

| model | data | bsz | steps | SQuAD 1.1/2.0 F1 | MNLI-m acc | SST-2 acc |
|---|---|---|---|---|---|---|
| combined recipe (this rung) | 16GB | 8K | 100K | 93.6 / 87.3 | 89.0 | 95.3 |
| BERT-large (published, original recipe) | 13GB | 256 | 1M | 90.9 / 81.8 | 86.6 | 93.7 |

Notes: the combined recipe at 100K steps (a tenth of the original's step count) and no additional data
exceeds BERT-large's published numbers on every column: +2.7/+5.5 SQuAD F1, +2.4 MNLI-m, +1.6 SST-2.
