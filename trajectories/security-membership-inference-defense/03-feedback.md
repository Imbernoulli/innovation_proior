Measured results — `baseline:confidence_penalty` (`is_final,true`), seed 42. Ordering relative to erm
and label_smoothing follows published consensus, not the leaderboard mean: the vgg16bn-cifar100 run is
a degenerate training collapse (test_acc 0.01 ≈ chance), which makes a numeric privacy_score tie-break
across the three simpler rungs unreliable; see reasoning.

## resnet20-cifar10
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.7974 | 0.7130 | 0.1012 | 0.5844 |

## vgg16bn-cifar100
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.0100 | 0.0026 | 0.0000 | 0.0100 |

## mobilenetv2-fmnist
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.9273 | 0.5602 | 0.0258 | 0.8671 |
