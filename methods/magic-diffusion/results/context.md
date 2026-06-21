# Context: single-cell RNA-seq denoising — MAGIC graph diffusion

## Research question

A single-cell RNA-seq UMI count matrix `X` (cells × genes) is undersampled, with most expressed
genes reading as zero (dropout). A denoiser maps `X` to a smoother non-negative `X̂` approximating
the expression rate, by pooling information across biologically similar cells to beat down Poisson
noise. The question: how to best pool information across cells so that the recovered expression
profiles reflect the underlying biological manifold structure?

## Evaluation

Molecular cross-validation (Batson 2019): thin the counts into two independent Poisson halves,
denoise `X_train`, score against `X_test`. Two metrics — log-normalized MSE and Poisson NLL — each
normalized `(raw − method)/(raw − perfect)` against raw-counts (0) and true-rate (1) anchors;
combined score is their mean, higher better. Synthetic harness: true rate `Λ` (low-rank trajectories
+ branches + over-dispersion), `X ~ Poisson(Λ)`, 900 cells × 1000 genes, dropout ~0.59, fixed split
seed; tune on one dataset, report on held-out ones.

## Method interface

Edit `denoise(X) → X̂`, same shape, non-negative. The existing baseline is kNN-smoothing:
library-normalize the counts, find the `k` nearest neighbors for each cell in gene space, and replace
each cell's profile with the uniform average of those neighbors. The step count `k` is the
bias-variance lever. Baselines: raw 0; kNN-smoothing.
