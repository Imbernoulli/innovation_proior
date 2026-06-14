Measured results — `baseline:deltanet` (`is_final,true`), seed 42.

## Language-model quality (lower is better)
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.3481 | 49.88 | 70.48 |
| **mean** | **2.3481** | **49.88** | **70.48** |

## Downstream zero-shot accuracy (higher is better)
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 53.58 | 32.77 | 62.95 | 49.17 |
| **mean** | **53.58** | **32.77** | **62.95** | **49.17** |

Wall-clock: `elapsed_gpt-345m` = 28265 s, `elapsed_lm-eval-345m` = 10 s.
