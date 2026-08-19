Measured results — two-token architecture, hard-label objective, fused readout, RegNetY-16GF teacher,
rung-1 recipe: all fixed from rung 3. Only the schedule length changed. ImageNet-1k, top-1 accuracy.

## 224² training, 1000 epochs (fused readout) vs. rung-3 300-epoch reference

| model | 300 epochs (rung 3) | 1000 epochs | delta |
|---|---|---|---|
| Tiny | 74.5 | 76.6 | +2.1 |
| Small | 81.2 | 82.6 | +1.4 |
| Base | 83.4 | 84.2 | +0.8 |

## Base size, fine-tuned at 384², 1000-epoch pretrained checkpoint vs. rung-3 300-epoch reference

| schedule | top1_imagenet |
|---|---|
| 300 epochs (rung 3) | 84.5 |
| 1000 epochs | 85.2 |

Extending the schedule keeps buying accuracy at every size and both resolutions; the gain is largest for
the smallest model (Tiny, +2.1) and smallest for the largest (Base, +0.8 at 224², +0.7 at 384²), consistent
with the smaller models having had more headroom left in a fixed 300-epoch budget. No size or resolution
shows a plateau or a regression from the longer schedule.

## Qualitative record note (epoch-scaling comparison, no-distillation model)

The record separately reports that the rung-1 (no-distillation, single class-token) model's accuracy
saturates by roughly epoch 400 and gains little from training further, while the two-token distilled
model keeps improving over the same extended range — the pattern in the two tables above is consistent
with that account, though the no-distillation model's own 1000-epoch number was not separately measured
here.

## Endpoint

The Base-size, two-token architecture, hard-label distillation from RegNetY-16GF, fused readout, fine-tuned at
384² after a 1000-epoch pretraining schedule: **85.2% top-1 on ImageNet-1k, no external training data**.
This exceeds a ViT-B pretrained on ~300M private images and fine-tuned at the same 384² resolution
(84.15% top-1), using only ImageNet-1k throughout. A separate, disjoint regime — extra training data,
ViT-H at 600M parameters and 512² resolution — reaches 88.55% and is not a comparable reference point for
this no-external-data exploration.
