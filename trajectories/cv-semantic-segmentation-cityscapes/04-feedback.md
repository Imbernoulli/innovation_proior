Measured result — DeepLabV3+ (separable ASPP + OS-4 decoder) on R-50-D8, Cityscapes val. Metric:
**Cityscapes val mIoU**, **higher is better**, fixed 512×1024 crop, R-50-D8, 40k schedule.

| configuration | backbone | crop | mIoU |
|---|---|---|---|
| PSPNet (step 3) | R-50-D8 | 512×1024 | 77.85 |
| DeepLabV3+ | R-50-D8 | 512×1024 | **79.61** |

Gathering multi-scale context densely with parallel atrous convs (no pooling-away of resolution) plus a
single low-level OS-4 decoder skip for the boundaries reaches **79.61 mIoU** (81.01 with multi-scale +
flip), +1.76 over PSPNet at the same backbone and crop. The headroom is smaller than the context jump
and comes mostly from boundary/thin classes, as expected. This is the strongest convolutional rung and
the bar the finale must beat.
