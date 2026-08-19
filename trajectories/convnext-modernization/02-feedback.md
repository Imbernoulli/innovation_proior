Measured results — macro design, ImageNet-1K, 224x224, mean +/- std over
three seeds. Frozen recipe from rung 1 throughout.

## ImageNet-1K

| variant | top1_acc | gflops |
|---|---|---|
| rung 1 baseline (unmodified ResNet-50, modern recipe) | 78.82 +/- 0.07 | 4.09 |
| stage ratio (3,4,6,3) -> (3,3,9,3), stem unchanged | 79.36 +/- 0.07 | 4.53 |
| + patchify stem (4x4 s4 conv) | 79.51 +/- 0.18 | 4.42 |

Both sub-steps land within roughly half a point of each other; the combined
macro-design change moves the network from 78.82% to 79.51% (+0.69) while
GFLOPs rise from 4.09G to 4.42G, closer to Swin-T's 4.50G budget.
