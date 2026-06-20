Measured result — FCN head on the dilated ImageNet-pretrained backbone (R-50-D8), Cityscapes val.
Metric: **Cityscapes val mIoU**, **higher is better**, fixed 512×1024 crop, R-50-D8, 40k schedule.

| configuration | backbone | crop | mIoU |
|---|---|---|---|
| U-Net + FCN (step 1) | UNet-S5-D16 | 512×1024 | 69.10 |
| FCN | R-50-D8 | 512×1024 | **72.25** |

Mounting the fully-convolutional refine-and-classify head on a dilated, ImageNet-pretrained ResNet-50
reaches **72.25 mIoU** (73.36 with multi-scale + flip), a +3.15 mIoU gain over the from-scratch
encoder–decoder. The pretrained semantics at output stride 8 win on the class-averaged metric. This is
the number the next rung must beat, with the backbone and crop now held fixed.
