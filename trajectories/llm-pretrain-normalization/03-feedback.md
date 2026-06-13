Measured results — `baseline:rmsnorm` (`is_final,true`), seed 42 (the only seed; per-seed = mean).

## Primary / perplexity (lower is better)
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.295 | 44.75 | 68.29 |
| **mean** | **2.295** | **44.75** | **68.29** |

## Downstream accuracy (higher is better)
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 54.97 | 33.25 | 64.36 | 51.22 |
| **mean** | **54.97** | **33.25** | **64.36** | **51.22** |

## Wall-clock (seconds)
| seed | elapsed (gpt-345m) | elapsed (lm-eval-345m) |
|---|---|---|
| 42 | 20389 | 458 |
| **mean** | **20389** | **458** |
