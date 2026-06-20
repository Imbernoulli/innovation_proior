Measured result — data filtering (proxy-mask hard-example selection); the airbench96_faster training
(`airbench96_faster.py` header; README method table). This rung targets the **96% bar**: metric is
A100-seconds to reach 96% mean accuracy (lower is better).

| script | mean accuracy | A100-seconds | PFLOPs |
|---|---|---|---|
| airbench96.py (legacy, step 7) | 96.03% | 34.7 | 4.9 |
| **airbench96_faster.py (data filtering)** | **96.00%** | **27.3** | 3.1 |

Measured on a 400W NVIDIA A100 with `torch==2.4.1` (README) / `torch==2.4.0+cu121` (script header), 96.00%
mean accuracy over n=200 runs at 27.3 A100-seconds, 3.1 PFLOPs. This is a post-paper record: it uses "a form
of data filtering" (README) — a small proxy network collects per-batch difficulty masks (the top-loss
`batch_size_masked=512` of each `batch_size=1024` batch), which the full-size model reuses to train only on
the hardest examples — lowering the 96% record from 34.7 to 27.3 A100-seconds (≈1.3× faster, ~37% fewer
PFLOPs) at the same accuracy.
