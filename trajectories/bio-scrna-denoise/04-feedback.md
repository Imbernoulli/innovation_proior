Measured result — `denoise:ttt` (TTT-Discover endpoint: 3-VST gene-wise ensemble + zero-imputation
+ gene-wise multi-scale diffusion + adaptive raw/diffused blend + light truncated-SVD low-rank
refinement + two log-normalized-space diffusion passes). Synthetic data: 900 cells × 1000 genes,
dropout ~0.59, MCV split seed 42. Tune = sim seed 0; held-out = sim seeds 1–3.

| Dataset | norm_MSE | norm_Poisson | combined | MSE | Poisson |
|---|---|---|---|---|---|
| tune (seed 0) | 0.8360 | 0.9335 | **0.8847** | 1.1481 | −0.7506 |
| held seed 1 | 0.8195 | 0.9255 | 0.8725 | 1.0299 | −0.9503 |
| held seed 2 | 0.8287 | 0.9279 | 0.8783 | 1.0751 | −0.8664 |
| held seed 3 | 0.8291 | 0.9286 | 0.8788 | 1.0706 | −0.8984 |
| **held-out mean** | **0.8258** | **0.9273** | **0.8765** | — | — |

Full ladder (held-out mean, sim seeds 1–3):

| Rung | method | norm_MSE | norm_Poisson | **combined** |
|---|---|---|---|---|
| 1 | raw (no denoising) | 0.0000 | 0.0000 | **0.0000** |
| 2 | kNN-smoothing | 0.7281 | 0.8674 | **0.7977** |
| 3 | MAGIC | 0.8235 | 0.8742 | **0.8488** |
| 4 | **TTT-Discover endpoint** | 0.8258 | 0.9273 | **0.8765** |

Reference points (real OpenProblems benchmark): MAGIC ~0.64; TTT-Discover 0.71 (PBMC) / 0.73
(Tabula). The endpoint adapts test-time-training/discover (`results/denoising/denoise_ttt.py`,
arXiv:2601.16175).

Notes: the endpoint is the strongest rung, clearing MAGIC by +0.028 on the held-out mean (0.8765 vs
0.8488). The gain is concentrated on the **Poisson** term (0.927 vs 0.874, +0.053) — the
gene-adaptive multi-VST ensemble, zero-imputation, and light low-rank refinement tighten the
absolute per-gene rate — while the MSE term ties MAGIC (0.826 vs 0.824), held there by the
log-normalized-space diffusion passes against the cost of the low-rank step. Component check on the
tune set: `lowrank_weight=0` scored marginally higher on MSE but is kept at `0.10` to retain the
genuine TTT-Discover SVD-refinement component (it helps Poisson and is a defining feature); the two
log-space diffusion passes are the largest single contributor to the MSE term. Stable across
held-out seeds (0.8725–0.8788), no over-fit. On this synthetic data the endpoint scores ~0.88 versus
its real-benchmark 0.71/0.73 — the synthetic manifold is cleaner, so every rung runs high, but the
*ordering* (raw < kNN < MAGIC < TTT) and the endpoint's decisive lead reproduce the real result. No
denoiser can exceed the true-rate ceiling: the multiplicative biological over-dispersion baked into
`Λ` is irreducible noise, which is why the scores plateau below 1 rather than running away. Endpoint
of the ladder.
