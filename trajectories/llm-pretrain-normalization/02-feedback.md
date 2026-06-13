Measured results — `baseline:rmsnorm_post` (`is_final,true`), seed 42 (the only seed; per-seed = mean).

## Primary / perplexity (lower is better)
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.3104 | 46.8 | 72.08 |
| **mean** | **2.3104** | **46.8** | **72.08** |

## Downstream accuracy (higher is better)
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 54.76 | 33.03 | 62.46 | 50.28 |
| **mean** | **54.76** | **33.03** | **62.46** | **50.28** |

## Wall-clock (seconds)
| seed | elapsed (gpt-345m) | elapsed (lm-eval-345m) |
|---|---|---|
| 42 | 21661 | 586 |
| **mean** | **21661** | **586** |
