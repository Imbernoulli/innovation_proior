# Context: single-cell RNA-seq denoising — kNN-smoothing

## Research question

A single-cell RNA-seq UMI count matrix `X` (cells × genes) is heavily undersampled: most expressed
genes read as zero (dropout). A denoiser maps `X` to a smoother non-negative `X̂` approximating the
underlying expression rate. Cells in the same biological state are independent noisy measurements of
one rate, so averaging similar cells reduces Poisson sampling noise while preserving shared signal.
The question: how can neighbor-averaging be applied to recover gene expression rates from the noisy
observed profiles?

## Evaluation

Molecular cross-validation (Batson 2019): binomially thin the counts into two independent Poisson
halves `X_train`, `X_test`; denoise `X_train`, score against `X_test`. Two metrics — log-normalized
MSE and Poisson NLL — each normalized `(raw − method)/(raw − perfect)` against a raw-counts anchor
(0) and a true-rate anchor (1); combined score is their mean, higher better. Synthetic harness: true
rate `Λ` (low-rank trajectories + branches + over-dispersion), `X ~ Poisson(Λ)`, 900 cells × 1000
genes, dropout ~0.59, fixed split seed; tune on one dataset, report on held-out ones.

## Method interface

Edit `denoise(X) → X̂`, same shape, non-negative. The approach: square-root-transform (Poisson
variance stabilization), library-size-normalize, find each cell's `k` nearest neighbors on the
normalized profiles, and replace the cell with the uniform average over itself and those neighbors,
then invert the transform and restore library size. Baselines: raw counts (score 0); MAGIC ~0.64 on
the real OpenProblems benchmark; TTT-Discover 0.71/0.73.
