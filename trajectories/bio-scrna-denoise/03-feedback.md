Measured result — `denoise:magic` (MAGIC, `t = 2`, `knn = 10`, `n_pca = 50`, alpha-decay `α = 2`,
diffusion `Pᵗ X` in sqrt-normalized space). Synthetic data: 900 cells × 1000 genes, dropout ~0.59,
MCV split seed 42. Tune = sim seed 0; held-out = sim seeds 1–3.

| Dataset | norm_MSE | norm_Poisson | combined | MSE | Poisson |
|---|---|---|---|---|---|
| tune (seed 0) | 0.8279 | 0.8866 | **0.8572** | 1.1525 | −0.6107 |
| held seed 1 | 0.8247 | 0.8713 | 0.8480 | 1.0274 | −0.8051 |
| held seed 2 | 0.8248 | 0.8759 | 0.8504 | 1.0771 | −0.7208 |
| held seed 3 | 0.8210 | 0.8753 | 0.8482 | 1.0747 | −0.7483 |
| **held-out mean** | **0.8235** | **0.8742** | **0.8488** | — | — |

Reference points: raw 0.0000; kNN-smoothing 0.7977 (held-out mean); real-benchmark MAGIC ~0.64;
TTT-Discover 0.71/0.73 on real PBMC/Tabula.

Notes: MAGIC clears kNN by +0.051 on the held-out mean (0.8488 vs 0.7977), and the gain is
concentrated where predicted — the MSE term rises from 0.728 to 0.824 (the adaptive-bandwidth
weighting and PCA-denoised embedding recover the log-normalized *shape* that hard uniform pooling
smeared), while the Poisson term improves more modestly (0.874 vs 0.867). The small `t = 2` confirms
the affinity weighting does most of the work in one step; larger `t` (tried `t = 3, 4, 6`)
over-diffused toward the global mean and lost score monotonically. Stable across held-out seeds. On
this synthetic data MAGIC scores far above its real-benchmark ~0.64 (the synthetic manifold is
cleaner). Remaining gaps: a single sqrt transform for all genes regardless of dropout, and no
refinement that targets the log-normalized scoring space or the global low-rank structure — the
endpoint rung's targets.
