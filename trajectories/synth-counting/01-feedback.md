Measured results — `baseline:lstm` (`is_final,true`), seed 42.

## exact (score = OOD accuracy on `T in [128, 256]`)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | score |
|---|---|---|---|---|
| 42 | 0.998047 | 0.0 | 0.998047 | 0.0 |

## abc (score = in-distribution membership accuracy)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | score |
|---|---|---|---|---|
| 42 | 0.994141 | 0.524414 | 0.469727 | 0.994141 |

## length-ood (score = retention `1 - max(0, in_dist - ood)`)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | retention | score |
|---|---|---|---|---|---|
| 42 | 0.994141 | 0.524414 | 0.469727 | 0.530273 | 0.530273 |
