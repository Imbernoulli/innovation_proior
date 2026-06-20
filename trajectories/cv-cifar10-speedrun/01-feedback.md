Measured result — frozen patch-whitening initialization added to the unwhitened baseline
(arXiv:2404.00498 §3.2). Metric: A100-seconds to reach 94% mean accuracy, **lower is better**.

| configuration | epochs to 94% | A100-seconds |
|---|---|---|
| unwhitened baseline (§3.1) | 45 | 18.3 |
| + frozen patch-whitening init (§3.2) | **21** | **8.0** |

Patch-whitening initialization is, in the author's words, "the single most impactful feature." Adding it
to the baseline "more than doubles training speed so that we reach 94% in 21 epochs taking 8.0
A100-seconds" — a 2.3× wall-clock speedup, with accuracy held at the 94% bar.
