Measured results — `baseline:ssm` (`is_final,true`), seed {42} (the task's single configured seed; mean equals the seed-42 row).

## dense (`FFL(p_i=0.5, T=512)`)
| seed | read_error_rate | seq_error_rate | score |
|---|---|---|---|
| 42 | 0.0 | 0.0 | 1.0 |
| **mean** | **0.0** | **0.0** | **1.0** |

## sparse (`FFL(p_i=0.98, T=512)`)
| seed | read_error_rate | seq_error_rate | score |
|---|---|---|---|
| 42 | 0.06571314 | 0.277 | 0.723 |
| **mean** | **0.06571314** | **0.277** | **0.723** |

## long_ctx (`FFL(p_i=0.8, T=1024)`)
| seed | read_error_rate | seq_error_rate | score |
|---|---|---|---|
| 42 | 0.0 | 0.0 | 1.0 |
| **mean** | **0.0** | **0.0** | **1.0** |
