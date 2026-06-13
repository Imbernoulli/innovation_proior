Measured results — `baseline:mqa` (`is_final,true`), seed 42 (the task's only seed).

## Primary + efficiency (gpt-345m)
| seed | val_loss | kv_bytes_per_token | head_sharing_ratio | latent_rank_ratio |
|---|---|---|---|---|
| 42 | 2.337850 | 256.0 | 16.0 | 1.0 |
| **mean** | **2.337850** | **256.0** | **16.0** | **1.0** |

## Held-out cross-entropy (gpt-345m)
| seed | heldout_loss | wikitext2 | wikitext103 | lambada |
|---|---|---|---|---|
| 42 | 3.999302 | 3.862528 | 3.853836 | 4.281542 |
| **mean** | **3.999302** | **3.862528** | **3.853836** | **4.281542** |

## Downstream 0-shot (lm-eval-345m)
| seed | arc_easy | hellaswag |
|---|---|---|
| 42 | 53.49 | 32.46 |
| **mean** | **53.49** | **32.46** |
