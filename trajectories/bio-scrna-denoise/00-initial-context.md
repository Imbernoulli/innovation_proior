## Research question

A single-cell RNA-seq experiment returns a UMI count matrix `X` of shape `cells × genes`, where `X[c, g]` is the number of unique mRNA molecules from gene `g` captured in cell `c`. Because droplet chemistry captures only a small fraction of the transcripts present, most expressed genes read as hard zeros (*dropouts*). The goal is a denoiser `denoise(X) → X̂` that maps the noisy integer matrix to a non-negative matrix `X̂` approximating the expression rate each cell would show under deeper sequencing. The denoiser is scored directly on held-out signal recovery.

No clean ground truth exists on real data; evaluation uses molecular cross-validation to turn the Poisson nature of the counts into a self-supervised target.

## Prior art / Background / Baselines

- **kNN-smoothing (Wagner, Yan & Yanai, 2018).** Average each cell's profile with its `k` nearest neighbors by observed expression distance to pool molecules and reduce Poisson noise.
- **MAGIC (van Dijk et al., Cell 2018).** Build a cell-cell affinity graph with an adaptive-bandwidth kernel, normalize it to a Markov matrix `P`, and impute by `X̂ = P^t X`, diffusing expression transitively along the data manifold.
- **ALRA (Linderman et al., 2022).** Take a low-rank SVD of the normalized matrix, reconstruct from the top components, and threshold per gene to preserve biological zeros.

## Fixed substrate / Code framework

The harness is a deterministic evaluator. It simulates a true rate matrix `Λ` with continuous-trajectory and discrete-branch low-rank structure, multiplicative biological over-dispersion (so `Λ` is not exactly low-rank), and lognormal per-cell library-size variation; draws `X ~ Poisson(Λ)`; performs a binomial molecular-cross-validation split with a fixed seed; calls the denoiser on `X_train`; and computes two normalized metrics against raw-count and perfect-rate anchors. The simulator, split, metrics, and anchors are frozen. Methods are tuned on sim seed 0 and reported on sim seeds 1–3.

## Editable interface

Exactly one function is editable: `denoise(X) → X̂`, taking the integer count matrix `X_train` and returning a non-negative float matrix of the same shape.

```python
import numpy as np

def simulate(n_cells=900, n_genes=1000, seed=0):
    """True rate Λ (low-rank trajectories + branches + over-dispersion), X ~ Poisson(Λ)."""
    # ... fixed generator; returns (X, Lam) ...

def mcv_split(X, p=0.5, seed=42):
    """Binomial molecular cross-validation (Batson 2019): thin a Poisson into two."""
    rng = np.random.default_rng(seed)
    X_train = rng.binomial(X.astype(np.int64), p).astype(np.float64)
    X_test = X - X_train
    return X_train, X_test

def _lognorm(M, target=1e4, eps=1e-12):           # library-size normalize + log1p
    s = np.maximum(M.sum(axis=1, keepdims=True), eps)
    return np.log1p(M * (target / s))

def mse_lognorm(test, denoised):
    return float(np.mean((_lognorm(test) - _lognorm(np.maximum(denoised, 0))) ** 2))

def poisson_nll(train, test, denoised, eps=1e-8):
    lam = np.maximum(denoised, 0) * (test.sum() / max(train.sum(), 1.0)) + eps
    return float(np.mean(lam - test * np.log(lam)))

# ---- EDITABLE: the denoiser. Default = raw (no denoising). ----
def denoise(X):
    return X.astype(np.float64).copy()
```

Outputs must match `X` in shape, be finite, and be non-negative.

## Evaluation settings

Each dataset is `900 cells × 1000 genes`, median library size ~900 molecules after the MCV split, dropout ~0.59. Sim seed 0 is the tune set; seeds 1–3 are held-out report sets. The MCV split seed is fixed at 42, so every method sees the same train/test partition.

The two metrics are MSE in log-normalized space and Poisson negative log-likelihood on the held-out half. Each is normalized against the raw-count anchor (`X̂ = X_train`, score 0) and the perfect-rate anchor (true `Λ` scaled to each cell's observed depth, score 1). The headline score is the mean of the two normalized metrics, averaged over the three held-out datasets. The tune-set score is reported alongside to expose over-fitting.

On the real OpenProblems PBMC/Tabula benchmark, raw counts score ~0.0 and MAGIC scores ~0.64; the synthetic datasets here are cleaner, so absolute numbers run higher, but the relative ordering of baselines remains the object of study.
