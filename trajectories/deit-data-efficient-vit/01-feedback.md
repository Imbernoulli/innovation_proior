Measured results — data-efficient training recipe (AdamW, wd 0.05, truncated-normal init, RandAugment +
Mixup + CutMix + Random Erasing + Stochastic Depth + Repeated Augmentation + Label Smoothing, no
dropout, 300 epochs), unmodified ViT architecture, class-token readout only, no distillation.
ImageNet-1k, top-1 accuracy.

## Main result (224² training)

| model | top1_imagenet |
|---|---|
| Tiny (5M) | 72.2 |
| Small (22M) | 79.8 |
| Base (86M) | 81.8 |

## Base size, fine-tuned at 384² (bicubic-interpolated positional embeddings)

| resolution | top1_imagenet |
|---|---|
| 160² | 79.9 |
| 224² | 81.8 |
| 320² | 82.7 |
| 384² | 83.1 |

Resolution and accuracy move together monotonically over this range at fixed recipe and fixed
architecture; the 384² fine-tune is the headline Base-size number for this rung (83.1).

## Supplementary: optimizer sensitivity, same recipe otherwise unchanged

| optimizer (pretrain / fine-tune) | 224² | 384² |
|---|---|---|
| SGD / AdamW | 74.5 | 77.3 |
| AdamW / SGD | 81.8 | 83.1 |
| AdamW / AdamW (main result) | 81.8 | 83.1 |

AdamW at pretraining time is responsible for most of the gap between the two rows above; once pretrained
with AdamW, the fine-tuning-stage optimizer (AdamW vs. SGD) makes no measurable difference.

The Base size (86M) at 81.8% (224²) already exceeds both the as-published ViT-B-on-ImageNet-1k-only number
(77.91%) and the timm-tuned starting bar (79.35%) by a wide margin, using the same architecture and the
same dataset. No distillation signal was used to produce any number above.
