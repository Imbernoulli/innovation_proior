Measured results — `baseline:rope` (`is_final,true`), seed 42 (single-seed task; mean = the seed).

## Language modeling (gpt-345m) — lower is better
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.2570 | 43.17 | 65.81 |
| **mean** | **2.2570** | **43.17** | **65.81** |

## Downstream (lm-eval-345m) — higher is better
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 57.32 | 34.48 | 64.42 | 51.70 |
| **mean** | **57.32** | **34.48** | **64.42** | **51.70** |
