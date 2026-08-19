Measured results — micro design (five sequential, cumulative sub-steps),
ImageNet-1K, 224x224, mean +/- std over three seeds. Frozen recipe
throughout.

## ImageNet-1K

| variant | top1_acc | gflops |
|---|---|---|
| rung 5 baseline (kernel 7x7, depthwise reordered) | 80.57 +/- 0.14 | 4.15 |
| ReLU -> GELU | 80.62 +/- 0.14 | 4.15 |
| + single activation per block | 81.27 +/- 0.06 | 4.15 |
| + single normalization per block | 81.41 +/- 0.09 | 4.15 |
| + BatchNorm -> LayerNorm | 81.47 +/- 0.09 | 4.46 |
| + separate downsampling layers, bracketing LayerNorm | 81.97 +/- 0.06 | 4.49 |

## Reference point (not a rung; fixed comparison target throughout this exploration)

| model | top1_acc | gflops |
|---|---|---|
| Swin-T | 81.30 | 4.50 |

The GELU swap moves accuracy by 0.05 points, within noise of the std
across seeds, at unchanged FLOPs — flat, as a same-position same-count
function swap. Cutting to one activation per block gains 0.65 points
(80.62 -> 81.27) at unchanged FLOPs, the single largest gain in this rung.
Cutting to one normalization per block gains a further 0.14 points
(81.27 -> 81.41), also at unchanged FLOPs. The BatchNorm -> LayerNorm
substitution gains 0.06 points (81.41 -> 81.47) at a FLOPs increase from
4.15G to 4.46G, and trains without instability. Separate downsampling with
the bracketing LayerNorm scheme (before each downsampling conv, after the
stem, after the final pooling) gains 0.50 points (81.47 -> 81.97) at 4.49G,
training stably across all three seeds; this configuration now exceeds the
81.30 reference point at matched compute (4.49G vs 4.50G).
