Measured results — inverted bottleneck, ImageNet-1K, 224x224, mean +/- std
over three seeds. Frozen recipe throughout.

## ImageNet-1K

| variant | top1_acc | gflops |
|---|---|---|
| rung 3 baseline (depthwise conv + width 96) | 80.50 +/- 0.02 | 5.27 |
| inverted bottleneck (narrow -> wide 4x -> narrow) | 80.64 +/- 0.03 | 4.64 |

FLOPs drop from 5.27G to 4.64G (the narrowed downsampling shortcuts more
than offset the depthwise conv now running at 4x width) while accuracy rises
slightly, 80.50 -> 80.64.
