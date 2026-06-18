Measured results — `baseline:transformer` (`is_final,true`), seed 42.

## exact (score = OOD accuracy on `T in [128, 256]`)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | score |
|---|---|---|---|---|
| 42 | 0.936523 | 0.0 | 0.936523 | 0.0 |

## abc (score = in-distribution membership accuracy)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | score |
|---|---|---|---|---|
| 42 | 0.753906 | 0.524414 | 0.229492 | 0.753906 |

## length-ood (score = retention `1 - max(0, in_dist - ood)`)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | retention | score |
|---|---|---|---|---|---|
| 42 | 0.753906 | 0.524414 | 0.229492 | 0.770508 | 0.770508 |
