Measured results — `baseline:mla` (`is_final,true`), seed 42 (the task's only seed).

## Primary + efficiency (gpt-345m)
| seed | val_loss | kv_bytes_per_token | head_sharing_ratio | latent_rank_ratio |
|---|---|---|---|---|
| 42 | 2.306939 | 192.0 | 16.0 | 0.25 |
| **mean** | **2.306939** | **192.0** | **16.0** | **0.25** |

## Held-out cross-entropy (gpt-345m)
| seed | heldout_loss | wikitext2 | wikitext103 | lambada |
|---|---|---|---|---|
| 42 | 3.988359 | 3.871585 | 3.861501 | 4.231989 |
| **mean** | **3.988359** | **3.871585** | **3.861501** | **4.231989** |

## Downstream 0-shot (lm-eval-345m)
| seed | arc_easy | hellaswag |
|---|---|---|
| 42 | 54.76 | 33.21 |
| **mean** | **54.76** | **33.21** |
