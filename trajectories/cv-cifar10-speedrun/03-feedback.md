Measured result — scalebias (64× learning-rate boost on the BatchNorm biases), arXiv:2404.00498 §3.4.
Metric: A100-seconds to reach 94% mean accuracy, **lower is better**.

| configuration | epochs to 94% | A100-seconds |
|---|---|---|
| + identity (Dirac) init (§3.3) | 18 | 6.8 |
| + scalebias 64× (§3.4) | **13.5** | **5.1** |

"With this feature added, training reaches 94% in 13.5 epochs taking 5.1 A100-seconds." It is the
largest-effect of the optimization tricks in the feature-interaction study (Fig. 4), with an effect of
roughly 4–4.5 epochs-to-94% whether removed from airbench94 or added to the whitened baseline.
