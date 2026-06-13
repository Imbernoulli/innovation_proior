Measured results — `baseline:rope_qk_norm` (`is_final,true`), seed 42 (single-seed task; mean = the seed).

## Language modeling (gpt-345m) — lower is better
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.2589 | 43.44 | 67.20 |
| **mean** | **2.2589** | **43.44** | **67.20** |

## Downstream (lm-eval-345m) — higher is better
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 57.83 | 34.24 | 64.74 | 50.67 |
| **mean** | **57.83** | **34.24** | **64.74** | **50.67** |
