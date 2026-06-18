Measured results — `baseline:abacus` (`is_final,true`), seed 42.

## exact (score = OOD accuracy on `T in [128, 256]`)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | score |
|---|---|---|---|---|
| 42 | 0.402344 | 0.0 | 0.402344 | 0.0 |

## abc (score = in-distribution membership accuracy)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | score |
|---|---|---|---|---|
| 42 | 0.744141 | 0.668945 | 0.075195 | 0.744141 |

## length-ood (score = retention `1 - max(0, in_dist - ood)`)
| seed | in_dist_accuracy | ood_accuracy | extrapolation_gap | retention | score |
|---|---|---|---|---|---|
| 42 | 0.744141 | 0.668945 | 0.075195 | 0.924805 | 0.924805 |
