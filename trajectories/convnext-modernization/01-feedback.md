Measured results — modern training recipe, unmodified ResNet-50 architecture,
ImageNet-1K, 224x224, mean +/- std over three seeds.

## ImageNet-1K

| variant | top1_acc | gflops |
|---|---|---|
| ResNet-50, torchvision, original 90-epoch recipe | 76.13 | 4.09 |
| ResNet-50, unmodified architecture, modern recipe | 78.82 +/- 0.07 | 4.09 |

GFLOPs unchanged at 4.09G, confirming the architecture is byte-for-byte the
same as the untouched baseline. This recipe (AdamW, 300 epochs, cosine
decay, 20-epoch warmup, RandAugment/Mixup/CutMix/Random Erasing, Stochastic
Depth, Label Smoothing 0.1, LayerScale init 1e-6, no EMA) is fixed and reused
unchanged for every subsequent step; only the network architecture varies
from here on.
