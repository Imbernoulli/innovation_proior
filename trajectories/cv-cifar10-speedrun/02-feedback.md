Measured result — identity (Dirac) initialization of all convolutions after the first
(arXiv:2404.00498 §3.3). Metric: A100-seconds to reach 94% mean accuracy, **lower is better**.

| configuration | epochs to 94% | A100-seconds |
|---|---|---|
| + frozen patch-whitening (§3.2) | 21 | 8.0 |
| + identity (Dirac) init (§3.3) | **18** | **6.8** |

"With this feature added, training attains 94% accuracy in 18 epochs taking 6.8 A100-seconds." Per the
paper's feature-interaction study (Fig. 4 / §5.1), removing Dirac from the final airbench94 *increases*
epochs-to-94% from 9.9 to 12.8, and adding it to the whitened baseline reduces epochs-to-94% from 21 to 18
— a consistent ≈3-epoch effect in both directions, indicating the speedups accumulate additively.
