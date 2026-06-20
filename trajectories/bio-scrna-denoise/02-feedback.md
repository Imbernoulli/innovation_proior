Measured result — `denoise:knn` (kNN-smoothing, `k = 10`, neighbors on noisy sqrt-normalized
profiles, uniform hard average). Synthetic data: 900 cells × 1000 genes, dropout ~0.59, MCV split
seed 42. Tune = sim seed 0; held-out = sim seeds 1–3.

| Dataset | norm_MSE | norm_Poisson | combined | MSE | Poisson |
|---|---|---|---|---|---|
| tune (seed 0) | 0.7228 | 0.8822 | **0.8025** | 1.2098 | −0.5978 |
| held seed 1 | 0.7351 | 0.8611 | 0.7981 | 1.0707 | −0.7779 |
| held seed 2 | 0.7234 | 0.8697 | 0.7965 | 1.1282 | −0.7034 |
| held seed 3 | 0.7258 | 0.8713 | 0.7986 | 1.1232 | −0.7370 |
| **held-out mean** | **0.7281** | **0.8674** | **0.7977** | — | — |

Reference points: raw 0.0000; MAGIC (next rung) target ~0.64 on the real benchmark; TTT-Discover
0.71/0.73 on real PBMC/Tabula.

Notes: a large jump off the floor — neighbor averaging closes ~80% of the gap to the rate on this
synthetic data (well above the real-benchmark numbers because the synthetic manifold is cleaner).
The Poisson term (0.867) is recovered better than the MSE term (0.728): hard uniform averaging fixes
the gross under-dispersion of the raw counts but its single global `k` and noisy neighbor selection
leave the log-normalized shape rougher. Stable across held-out seeds (0.7965–0.7986), no over-fit to
the tune set. The MSE/Poisson asymmetry is the signature of the two named weaknesses — noisy neighbor
choice and a non-adaptive hard pool — which the affinity-graph diffusion rung addresses.
