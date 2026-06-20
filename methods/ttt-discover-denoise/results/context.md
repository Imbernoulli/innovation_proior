# Context: single-cell RNA-seq denoising — the TTT-Discover endpoint

## Research question

A single-cell RNA-seq UMI count matrix `X` (cells × genes) is undersampled, with most expressed
genes reading as zero (dropout). A denoiser maps `X` to a smoother non-negative `X̂` approximating
the expression rate. MAGIC's graph diffusion recovers local manifold geometry but forces every gene
through one square-root transform, never smooths in the log-normalized space the MSE is scored in,
and ignores the global low-rank structure beneath the local manifold. The question here: can an
ensemble of gene-adaptive variance-stabilizing transforms, a low-rank refinement, and a final
log-space diffusion — the denoiser TTT-Discover evolved to the top of the OpenProblems leaderboard —
close all three gaps?

## Evaluation

Molecular cross-validation (Batson 2019): thin the counts into two independent Poisson halves,
denoise `X_train`, score against `X_test`. Two metrics — log-normalized MSE and Poisson NLL — each
normalized `(raw − method)/(raw − perfect)` against raw-counts (0) and true-rate (1) anchors;
combined score is their mean, higher better. Synthetic harness: true rate `Λ` (low-rank trajectories
+ branches + multiplicative biological over-dispersion, so `Λ` is *not* exactly low-rank), `X ~
Poisson(Λ)`, 900 cells × 1000 genes, dropout ~0.59, fixed split seed; tune on one dataset, report on
held-out ones.

## Method interface

Edit `denoise(X) → X̂`, same shape, non-negative. The endpoint keeps MAGIC's adaptive-bandwidth
diffusion graph and adds: a gene-wise ensemble over three transforms (Anscombe / Freeman-Tukey /
square root, weighted by per-gene dropout); zero-imputation; gene-wise multi-scale diffusion;
adaptive raw/diffused blending; a light truncated-SVD low-rank refinement; and two diffusion passes
in the log-normalized space the MSE is computed in. It adapts the genuine TTT-Discover denoiser
(github.com/test-time-training/discover, `results/denoising/denoise_ttt.py`; arXiv:2601.16175), which
reaches 0.71 (PBMC) / 0.73 (Tabula Muris) on the real benchmark versus MAGIC's ~0.64 — rebuilt in
plain numpy/scipy/sklearn (the original uses graphtools+scprep). Baselines: raw 0; kNN-smoothing;
MAGIC; this endpoint strongest.
