# Context: single-cell RNA-seq denoising — the raw (no-denoising) baseline

## Research question

A single-cell RNA-seq experiment returns a UMI count matrix `X` (cells × genes): `X[c,g]` is the
number of unique mRNA molecules from gene `g` captured in cell `c`. Droplet capture is undersampled,
so most expressed genes read as zero (dropout). A denoiser maps `X` to a smoother non-negative `X̂`
approximating the underlying expression rate. The question is: what score does a denoiser achieve
when given access to the raw training counts?

## Evaluation

Molecular cross-validation (Batson, Royer & Webber 2019): binomially thin the observed counts,
`X_train ~ Binomial(X, 0.5)`, `X_test = X − X_train`. Thinning a Poisson splits it into two
independent Poissons, so the two halves are independent noisy views of one rate; denoise `X_train`,
score against `X_test`. Two metrics: (1) MSE in log-normalized space (library-size normalize, log1p,
MSE); (2) Poisson NLL of the held-out counts under the rescaled denoised rate. Each is normalized
`(raw − method) / (raw − perfect)` against the raw-counts anchor (0) and the true-rate anchor (1,
available on synthetic data); combined score is their mean, higher better. The harness simulates a
true rate `Λ` (low-rank trajectories + branches + multiplicative over-dispersion + lognormal library
sizes), draws `X ~ Poisson(Λ)`, and scores on a synthetic dataset (900 cells × 1000 genes, dropout
~0.59), with the MCV split seed fixed.

## Method interface

Edit one function `denoise(X) → X̂`, same shape, non-negative.
