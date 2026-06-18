Measured results — `baseline:relaxloss` (`is_final,true`), seed 42. RelaxLoss is the strongest baseline
by measurement: highest mean privacy_score (0.6832) and the only method to push the attack AUC toward
0.5 on all three benchmarks.

## resnet20-cifar10
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.7949 | 0.5092 | 0.0075 | 0.7857 |

## vgg16bn-cifar100
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.6226 | 0.6490 | 0.1660 | 0.4736 |

## mobilenetv2-fmnist
| seed | test_acc | mia_auc | privacy_gap | privacy_score |
|---|---|---|---|---|
| 42 | 0.7904 | 0.4987 | -0.0012 | 0.7904 |
