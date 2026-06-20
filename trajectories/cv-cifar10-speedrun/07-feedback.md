Measured result — residual-scaling + Cutout architecture; the airbench96 training (arXiv:2404.00498 §4;
`legacy/airbench96.py` header). This rung targets the harder **96% bar**, so its metric is A100-seconds to
reach 96% mean accuracy (lower is better), not directly comparable to the 94% rungs above.

| script | mean accuracy | A100-seconds | PFLOPs |
|---|---|---|---|
| airbench94.py (94% target) | 94.01% | 3.83 | 0.36 |
| airbench95.py (95% target) | 95.01% | 10.4 | 1.4 |
| **airbench96.py (96% target)** | **96.03%** | **34.7** | 4.9 |

The paper reports airbench96 at 96.05% in 46.3 A100-seconds (7.2×10¹⁵ FLOPs); the repo's preserved legacy
script measures 96.03% mean accuracy over n=400 runs at 34.7 A100-seconds. Changes vs. airbench95 (legacy
header): increased width and reduced lr & weight decay; reduced warmup with LR decaying fully to zero; a
third conv per ConvGroup (10 conv layers total); residual connections over the last two convs of each block;
12-pixel Cutout and translation strength raised 2→4 pixels; 37 training epochs. The three airbench points
follow a linear log-log FLOPs↔error relationship (Fig. 3).
