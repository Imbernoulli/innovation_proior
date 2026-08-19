Measured results — large kernel exploration, ImageNet-1K, 224x224, mean
+/- std over three seeds. Frozen recipe throughout.

## ImageNet-1K

| variant | top1_acc | gflops |
|---|---|---|
| rung 4 baseline (inverted bottleneck, kernel 3x3, depthwise mid-block) | 80.64 +/- 0.03 | 4.64 |
| depthwise conv moved above the 1x1 expansion (kernel still 3x3) | 79.92 +/- 0.08 | 4.07 |
| kernel 3x3 (post-reorder) | 79.92 +/- 0.08 | 4.07 |
| kernel 5x5 | 80.35 +/- 0.08 | 4.10 |
| kernel 7x7 | 80.57 +/- 0.14 | 4.15 |
| kernel 9x9 | 80.57 +/- 0.06 | 4.21 |
| kernel 11x11 | 80.47 +/- 0.11 | 4.29 |

The reorder alone costs 0.72 points (80.64 -> 79.92) and cuts FLOPs from
4.64G to 4.07G, matching the prediction of a temporary regression bought at
lower cost. From that reordered starting point, accuracy climbs with kernel
size from 79.92 (k=3) through 80.57 (k=7), then is flat from k=7 to k=9
(both 80.57) and turns down at k=11 (80.47), while FLOPs rise only slightly
across the whole sweep (4.07G to 4.29G). Gains saturate at 7x7.
