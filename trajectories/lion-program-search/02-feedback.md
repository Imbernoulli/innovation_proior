Measured results — Ablation_0.9 and Ablation_0.99 (`m = interp(g, m, β); update = sign(m)`),
tuned lr/λ, ImageNet top-1 / ReaL top-1 / V2 top-1, three-run average. AdamW and rung-1 rows
repeated for reference.

## ViT-S/16
| optimizer | ImageNet | ReaL | V2 |
|---|---|---|---|
| AdamW | 78.89 | 84.61 | 66.73 |
| PowerSign | 77.36 | 83.39 | 65.17 |
| AddSign | 77.37 | 83.36 | 64.52 |
| Ablation_0.9 | 78.23 | 84.28 | 66.13 |
| Ablation_0.99 | 78.19 | 84.17 | 65.96 |

## ViT-B/16
| optimizer | ImageNet | ReaL | V2 |
|---|---|---|---|
| AdamW | 80.12 | 85.46 | 68.14 |
| PowerSign | 78.95 | 84.76 | 67.46 |
| AddSign | 78.50 | 84.49 | 65.95 |
| Ablation_0.9 | 79.54 | 85.10 | 68.07 |
| Ablation_0.99 | 79.90 | 85.36 | 68.20 |

Both Ablation_0.9 and Ablation_0.99 beat both PowerSign and AddSign on every column at both
scales. Neither clears AdamW's ImageNet number at either scale (78.23/78.19 vs 78.89 on
ViT-S/16; 79.54/79.90 vs 80.12 on ViT-B/16); on ReaL both also stay below AdamW. On V2 at
ViT-B/16, Ablation_0.99 (68.20) edges AdamW (68.14); every other column stays below AdamW.
