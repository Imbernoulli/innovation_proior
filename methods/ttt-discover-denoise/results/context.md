# Context: single-cell RNA-seq denoising — the TTT-Discover endpoint

## Research question

A single-cell RNA-seq UMI count matrix `X` (cells × genes) is undersampled, with most expressed
genes reading as zero (dropout). A denoiser maps `X` to a smoother non-negative `X̂` approximating
the expression rate. MAGIC builds an adaptive-bandwidth diffusion graph on a PCA embedding of the
Anscombe-transformed, library-size-normalized counts and applies t steps of diffusion to smooth
each gene. The question here: how can the denoising pipeline be improved beyond MAGIC's graph
diffusion to achieve higher scores on the molecular cross-validation benchmark?

## Evaluation

Molecular cross-validation (Batson 2019): thin the counts into two independent Poisson halves,
denoise `X_train`, score against `X_test`. Two metrics — log-normalized MSE and Poisson NLL — each
normalized `(raw − method)/(raw − perfect)` against raw-counts (0) and true-rate (1) anchors;
combined score is their mean, higher better. Synthetic harness: true rate `Λ` (low-rank trajectories
+ branches + multiplicative biological over-dispersion, so `Λ` is *not* exactly low-rank), `X ~
Poisson(Λ)`, 900 cells × 1000 genes, dropout ~0.59, fixed split seed; tune on one dataset, report on
held-out ones.

## Method interface

Edit `denoise(X) → X̂`, same shape, non-negative. Baselines: raw 0; kNN-smoothing; MAGIC.
Implementation uses plain numpy/scipy/sklearn (no graphtools or scprep dependency).
