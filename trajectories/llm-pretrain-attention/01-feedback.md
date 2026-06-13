Measured results — `baseline:qk_norm` (`is_final,true`), seed 42 (single-seed task; mean = the seed).

## Language modeling (gpt-345m) — lower is better
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.2885 | 43.65 | 69.99 |
| **mean** | **2.2885** | **43.65** | **69.99** |

## Downstream (lm-eval-345m) — higher is better
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 55.64 | 33.41 | 63.17 | 51.30 |
| **mean** | **55.64** | **33.41** | **63.17** | **51.30** |
