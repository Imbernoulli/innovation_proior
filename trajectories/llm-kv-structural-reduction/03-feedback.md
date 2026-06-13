Measured results — `baseline:gqa` (`is_final,true`), seed 42 (the task's only seed).

## Primary + efficiency (gpt-345m)
| seed | val_loss | kv_bytes_per_token | head_sharing_ratio | latent_rank_ratio |
|---|---|---|---|---|
| 42 | 2.312553 | 1024.0 | 4.0 | 1.0 |
| **mean** | **2.312553** | **1024.0** | **4.0** | **1.0** |

## Held-out cross-entropy (gpt-345m)
| seed | heldout_loss | wikitext2 | wikitext103 | lambada |
|---|---|---|---|---|
| 42 | 3.969124 | 3.767792 | 3.845813 | 4.293768 |
| **mean** | **3.969124** | **3.767792** | **3.845813** | **4.293768** |

## Downstream 0-shot (lm-eval-345m)
| seed | arc_easy | hellaswag |
|---|---|---|
| 42 | 55.01 | 33.12 |
| **mean** | **55.01** | **33.12** |
