Measured results — `baseline:label_smoothing` (`is_final,true`), seed 42. Ordering of label_smoothing
relative to erm and confidence_penalty follows published consensus, not the leaderboard: the three
land within ~0.04 mean privacy_score of one another and the CIFAR-100 column is confounded by a
training collapse in confidence_penalty, so a numeric tie-break would be unreliable; see reasoning.

## resnet20-cifar10
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.7878 | 0.7678 | 0.1377 | 0.5200 |

## vgg16bn-cifar100
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.5258 | 0.8686 | 0.2866 | 0.1572 |

## mobilenetv2-fmnist
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.9270 | 0.5810 | 0.0216 | 0.8460 |
