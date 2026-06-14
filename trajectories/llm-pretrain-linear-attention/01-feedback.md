Measured results — `baseline:retnet` (`is_final,true`), seed 42.

## Language-model quality (lower is better)
| seed | val_loss | wikitext2_ppl | lambada_ppl |
|---|---|---|---|
| 42 | 2.4795 | 66.67 | 82.36 |
| **mean** | **2.4795** | **66.67** | **82.36** |

## Downstream zero-shot accuracy (higher is better)
| seed | arc_easy | hellaswag | piqa | winogrande |
|---|---|---|---|---|
| 42 | 51.47 | 31.12 | 62.40 | 52.01 |
| **mean** | **51.47** | **31.12** | **62.40** | **52.01** |

Wall-clock: `elapsed_gpt-345m` = 27106 s, `elapsed_lm-eval-345m` = 7 s.
