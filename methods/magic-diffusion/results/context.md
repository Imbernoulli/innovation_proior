# Context: single-cell RNA-seq denoising — MAGIC graph diffusion

## Research question

A single-cell RNA-seq UMI count matrix `X` (cells × genes) is undersampled, with most expressed
genes reading as zero (dropout). A denoiser maps `X` to a smoother non-negative `X̂` approximating
the expression rate, by pooling information across biologically similar cells to beat down Poisson
noise. Naive kNN-smoothing pools, but it picks neighbors on noisy profiles and uses a hard uniform
average with a single global `k`. The question here: can a geometry-aware, adaptive-bandwidth,
transitively-pooled diffusion on a cell-cell affinity graph — MAGIC (van Dijk et al., Cell 2018) —
cure those two flaws and recover the local manifold structure they smear?

## Evaluation

Molecular cross-validation (Batson 2019): thin the counts into two independent Poisson halves,
denoise `X_train`, score against `X_test`. Two metrics — log-normalized MSE and Poisson NLL — each
normalized `(raw − method)/(raw − perfect)` against raw-counts (0) and true-rate (1) anchors;
combined score is their mean, higher better. Synthetic harness: true rate `Λ` (low-rank trajectories
+ branches + over-dispersion), `X ~ Poisson(Λ)`, 900 cells × 1000 genes, dropout ~0.59, fixed split
seed; tune on one dataset, report on held-out ones.

## Method interface

Edit `denoise(X) → X̂`, same shape, non-negative. MAGIC: square-root-transform, library-normalize,
PCA-embed (so neighbor distances are measured on denoised structure), build an adaptive-bandwidth
alpha-decay affinity kernel (per-cell bandwidth = distance to the `k`-th neighbor), symmetrize and
row-normalize into a Markov transition matrix `P`, and impute by diffusion `X̂ = Pᵗ X` — powered
diffusion lets cells borrow transitively from neighbors' neighbors along the manifold. The step count
`t` is the bias-variance lever. Remaining gaps — a single transform for all genes, and no refinement
in the log-normalized scoring space or the global low-rank structure — motivate the TTT-Discover
endpoint. Baselines: raw 0; kNN-smoothing; real-benchmark MAGIC ~0.64; TTT-Discover 0.71/0.73.
