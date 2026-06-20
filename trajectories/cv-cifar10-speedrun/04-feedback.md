Measured result — Lookahead optimization (EMA "slow weights" with periodic synchronization),
arXiv:2404.00498 §3.4. Metric: A100-seconds to reach 94% mean accuracy, **lower is better**.

| configuration | epochs to 94% | A100-seconds |
|---|---|---|
| + scalebias 64× (§3.4) | 13.5 | 5.1 |
| + lookahead (§3.4) | **12.0** | **4.6** |

"With this feature added, training reaches 94% in 12.0 epochs taking 4.6 A100-seconds." Per the
feature-interaction study (Fig. 4), lookahead's effect is among the smaller ones (≈0.8 epochs-to-94% in
both the add-to-baseline and remove-from-airbench94 directions), consistent with it harvesting residual SGD
noise after the larger inefficiencies are already gone.
