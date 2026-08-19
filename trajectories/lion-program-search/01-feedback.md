Measured results — PowerSign (α=e, default) and AddSign (α=1, default), tuned lr/λ, ImageNet
top-1 / ReaL top-1 / V2 top-1, three-run average. AdamW row repeated for reference.

## ViT-S/16
| optimizer | ImageNet | ReaL | V2 |
|---|---|---|---|
| AdamW | 78.89 | 84.61 | 66.73 |
| PowerSign | 77.36 | 83.39 | 65.17 |
| AddSign | 77.37 | 83.36 | 64.52 |

## ViT-B/16
| optimizer | ImageNet | ReaL | V2 |
|---|---|---|---|
| AdamW | 80.12 | 85.46 | 68.14 |
| PowerSign | 78.95 | 84.76 | 67.46 |
| AddSign | 78.50 | 84.49 | 65.95 |

Both PowerSign and AddSign fall below AdamW on all three columns at both scales. Neither clears
the bar.
