Measured results — `large_batches`, BERT-base architecture, dynamic masking, full-sentences (no NSP),
BookCorpus+Wikipedia 16GB, compute-matched batch-size sweep, learning rate tuned per setting.

| bsz | steps | lr | held-out ppl | MNLI-m acc | SST-2 acc |
|---|---|---|---|---|---|
| 256 | 1M | 1e-4 | 3.99 | 84.7 | 92.7 |
| 2K | 125K | 7e-4 | 3.68 | 85.2 | 92.9 |
| 8K | 31K | 1e-3 | 3.77 | 84.6 | 92.8 |

Notes: 2K batch posts the best number on all three columns (lowest ppl, highest MNLI-m, highest
SST-2). 8K is worse than 2K on all three columns but still better than 256 on ppl and MNLI-m, and ties
256 within 0.1 on SST-2. All three settings observe the same total number of sequences.
