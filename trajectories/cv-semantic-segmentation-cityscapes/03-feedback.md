Measured result — PSPNet (pyramid pooling head) on R-50-D8, Cityscapes val. Metric: **Cityscapes val
mIoU**, **higher is better**, fixed 512×1024 crop, R-50-D8, 40k schedule.

| configuration | backbone | crop | mIoU |
|---|---|---|---|
| FCN (step 2) | R-50-D8 | 512×1024 | 72.25 |
| PSPNet | R-50-D8 | 512×1024 | **77.85** |

Piping multi-scale global context into every pixel's decision lifts mIoU to **77.85** (79.18 with
multi-scale + flip), a +5.60 gain over the local FCN head at the same backbone and crop — the largest
single jump on the ladder. The context-starved errors (road/sidewalk, floating mislabeled regions,
ambiguous verticals) are exactly what the pyramid clears. This is the bar the next rung must beat.
