Measured results — `baseline:mha` (`is_final,true`), seed 42 (the task's only seed).

## Primary + efficiency (gpt-345m)
| seed | val_loss | kv_bytes_per_token | head_sharing_ratio | latent_rank_ratio |
|---|---|---|---|---|
| 42 | 2.275425 | 4096.0 | 1.0 | 1.0 |
| **mean** | **2.275425** | **4096.0** | **1.0** | **1.0** |

## Held-out cross-entropy (gpt-345m)
| seed | heldout_loss | wikitext2 | wikitext103 | lambada |
|---|---|---|---|---|
| 42 | 3.967285 | 3.873442 | 3.798263 | 4.230150 |
| **mean** | **3.967285** | **3.873442** | **3.798263** | **4.230150** |

## Downstream 0-shot (lm-eval-345m)
| seed | arc_easy | hellaswag |
|---|---|---|
| 42 | 54.88 | 33.42 |
| **mean** | **54.88** | **33.42** |
