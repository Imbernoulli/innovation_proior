Measured result — `denoise:raw` (identity, `X̂ = X_train`). Synthetic data: 900 cells × 1000
genes, dropout ~0.59, MCV split seed 42. Tune = sim seed 0; held-out = sim seeds 1–3.

| Dataset | norm_MSE | norm_Poisson | combined | raw MSE | raw Poisson |
|---|---|---|---|---|---|
| tune (seed 0) | 0.0000 | 0.0000 | **0.0000** | 1.6037 | 2.0329 |
| held seed 1 | 0.0000 | 0.0000 | 0.0000 | 1.4263 | 1.5303 |
| held seed 2 | 0.0000 | 0.0000 | 0.0000 | 1.4927 | 1.7328 |
| held seed 3 | 0.0000 | 0.0000 | 0.0000 | 1.4929 | 1.7196 |
| **held-out mean** | **0.0000** | **0.0000** | **0.0000** | — | — |

Reference points (real OpenProblems benchmark): MAGIC ~0.64, TTT-Discover 0.71 (PBMC) / 0.73
(Tabula).

Notes: both normalized terms are exactly `0.0000` on every dataset, as predicted — the method matrix
*is* the raw anchor, so the numerator `raw − method` is identically zero. The raw-anchor MSE (~1.5)
and Poisson NLL (~1.7) are the noise floor of the binomial split; the perfect-rate anchor sits well
below them (tune perf_MSE 1.0587, perf_Poisson −0.949), confirming the two anchors bracket the
scale correctly and the metric has no sign error. The evaluator is calibrated. The identity pools no
information across cells; the entire 0→1 distance is open.
