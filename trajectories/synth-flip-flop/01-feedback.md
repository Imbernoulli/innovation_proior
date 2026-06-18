Measured results — `baseline:transformer` (`is_final,true`), seed {42} (the task's single configured seed; mean equals the seed-42 row).

## dense (`FFL(p_i=0.5, T=512)`)
| seed | read_error_rate | seq_error_rate | score |
|---|---|---|---|
| 42 | 0.01554676 | 0.7685 | 0.2315 |
| **mean** | **0.01554676** | **0.7685** | **0.2315** |

## sparse (`FFL(p_i=0.98, T=512)`)
| seed | read_error_rate | seq_error_rate | score |
|---|---|---|---|
| 42 | 0.06631326 | 0.199 | 0.801 |
| **mean** | **0.06631326** | **0.199** | **0.801** |

## long_ctx (`FFL(p_i=0.8, T=1024)`)
| seed | read_error_rate | seq_error_rate | score |
|---|---|---|---|
| 42 | 0.09908875 | 0.998 | 0.002 |
| **mean** | **0.09908875** | **0.998** | **0.002** |
