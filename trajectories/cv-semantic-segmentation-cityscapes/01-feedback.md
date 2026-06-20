Measured result — U-Net (encoder–decoder with per-level skip connections), the FCN-UNet-S5-D16
configuration on Cityscapes val. Metric: **Cityscapes val mIoU**, **higher is better**, fixed 512×1024
crop.

| configuration | backbone | crop | mIoU |
|---|---|---|---|
| U-Net + FCN | UNet-S5-D16 | 512×1024 | **69.10** |

The symmetric encoder–decoder with skip connections reaches **69.10 mIoU** on Cityscapes val
(71.05 with multi-scale + flip test-time augmentation). It establishes that fusing matching-resolution
detail back into an upsampling decoder produces a usable dense prediction; this is the number the next
rung has to beat.
