Measured results — `baseline:rmsnorm_parallel` (`is_final,true`), seed 42 (the only seed; per-seed = mean).

## Primary / perplexity (lower is better)
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.3112 | 45.98 | 70.96 |
| **mean** | **2.3112** | **45.98** | **70.96** |

## Downstream accuracy (higher is better)
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 54.76 | 32.93 | 64.42 | 50.2 |
| **mean** | **54.76** | **32.93** | **64.42** | **50.2** |

## Wall-clock (seconds)
| seed | elapsed (gpt-345m) | elapsed (lm-eval-345m) |
|---|---|---|
| 42 | 19747 | 406 |
| **mean** | **19747** | **406** |
