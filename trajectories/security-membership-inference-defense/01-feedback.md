Measured results — `baseline:erm` (`is_final,true`), seed 42. No leaderboard ordering dispute at this
rung: ERM is the no-defense floor by construction. (The three sub-leaderboard-strongest rungs below —
erm, label_smoothing, confidence_penalty — land within ~0.04 privacy_score of one another and the
CIFAR-100 numbers are confounded by a training collapse, so ordering among the simpler regularizers
follows published consensus; see each reasoning. RelaxLoss is unambiguously strongest by measurement.)

## resnet20-cifar10
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.7972 | 0.7275 | 0.0924 | 0.5697 |

## vgg16bn-cifar100
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.5045 | 0.8677 | 0.1479 | 0.1368 |

## mobilenetv2-fmnist
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.9260 | 0.5575 | 0.0211 | 0.8685 |
