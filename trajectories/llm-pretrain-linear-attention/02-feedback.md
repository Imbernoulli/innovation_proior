Measured results — `baseline:gla` (`is_final,true`), seed 42.

## Language-model quality (lower is better)
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.4482 | 64.32 | 84.73 |
| **mean** | **2.4482** | **64.32** | **84.73** |

## Downstream zero-shot accuracy (higher is better)
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 53.11 | 31.10 | 62.40 | 49.88 |
| **mean** | **53.11** | **31.10** | **62.40** | **49.88** |

Wall-clock: `elapsed_gpt-345m` = 27219 s, `elapsed_lm-eval-345m` = 1242 s.
