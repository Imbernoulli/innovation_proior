Measured results — ResNeXt-ify (depthwise conv + width), ImageNet-1K,
224x224, mean +/- std over three seeds. Frozen recipe throughout.

## ImageNet-1K

| variant | top1_acc | gflops |
|---|---|---|
| rung 2 baseline (macro design) | 79.51 +/- 0.18 | 4.42 |
| depthwise 3x3 conv, width still 64 | 78.28 +/- 0.08 | 2.35 |
| + width 64 -> 96 | 80.50 +/- 0.02 | 5.27 |

Depthwise conv alone drops accuracy by 1.23 points (79.51 -> 78.28) while
cutting FLOPs nearly in half (4.42G -> 2.35G). Widening to 96 channels more
than recovers the loss (78.28 -> 80.50) and pushes FLOPs to 5.27G, above the
4.5G Swin-T reference.
