Measured results — Lion (decoupled β₁=0.9 / β₂=0.99 sign-momentum), tuned lr/λ, ImageNet top-1 /
ReaL top-1 / V2 top-1, three-run average. All prior rows repeated for reference.

## ViT-S/16
| optimizer | ImageNet | ReaL | V2 |
|---|---|---|---|
| AdamW | 78.89 | 84.61 | 66.73 |
| PowerSign | 77.36 | 83.39 | 65.17 |
| AddSign | 77.37 | 83.36 | 64.52 |
| Ablation_0.9 | 78.23 | 84.28 | 66.13 |
| Ablation_0.99 | 78.19 | 84.17 | 65.96 |
| Lion | 79.46 | 85.25 | 67.68 |

## ViT-B/16
| optimizer | ImageNet | ReaL | V2 |
|---|---|---|---|
| AdamW | 80.12 | 85.46 | 68.14 |
| PowerSign | 78.95 | 84.76 | 67.46 |
| AddSign | 78.50 | 84.49 | 65.95 |
| Ablation_0.9 | 79.54 | 85.10 | 68.07 |
| Ablation_0.99 | 79.90 | 85.36 | 68.20 |
| Lion | 80.77 | 86.15 | 69.19 |

Lion beats every prior entrant — AdamW, PowerSign, AddSign, Ablation_0.9, Ablation_0.99 — on
every column at both scales. Margin over the better of the two ablations: +1.23 ImageNet /
+0.97 ReaL / +1.55 V2 at ViT-S/16; +0.87 ImageNet / +0.79 ReaL / +0.99 V2 at ViT-B/16. Margin over
AdamW: +0.57 / +0.64 / +0.95 (ViT-S/16); +0.65 / +0.69 / +1.05 (ViT-B/16).
