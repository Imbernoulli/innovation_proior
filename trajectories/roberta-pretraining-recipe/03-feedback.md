Measured results — `full_sentences_no_nsp` (4-way input-format comparison), BERT-base architecture,
dynamic masking, ~1M-step-equivalent budget, BookCorpus+Wikipedia. Development-set numbers, medians
over 5 seeds. SQuAD reported as v1.1/v2.0 F1.

| format | SQuAD 1.1/2.0 F1 | MNLI-m acc | SST-2 acc | RACE acc |
|---|---|---|---|---|
| segment-pair+NSP (our reimpl.) | 90.4 / 78.7 | 84.0 | 92.9 | 64.2 |
| sentence-pair+NSP (our reimpl.) | 88.7 / 76.2 | 82.9 | 92.1 | 63.0 |
| full-sentences, no NSP (our reimpl.) | 90.4 / 79.1 | 84.7 | 92.5 | 64.8 |
| doc-sentences, no NSP (our reimpl.) | 90.6 / 79.7 | 84.7 | 92.7 | 65.6 |
| BERT-base (published reference) | 88.5 / 76.3 | 84.3 | 92.8 | 64.3 |
| external base-scale model, K=7 (published reference) | -- / 81.3 | 85.8 | 92.7 | 66.1 |

Notes: sentence-pair+NSP is the weakest of the four on every task despite retaining the NSP loss.
Both NSP-free formats (full-sentences, doc-sentences) meet or exceed segment-pair+NSP on every column
except SST-2 (full-sentences 92.5 vs 92.9; doc-sentences 92.7 vs 92.9). doc-sentences scores marginally
above full-sentences on SQuAD 2.0 (79.7 vs 79.1) and RACE (65.6 vs 64.8), and ties it on SQuAD 1.1 and
MNLI-m.
