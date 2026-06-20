## Research question

A single-cell RNA-seq experiment hands back a **UMI count matrix** `X` of shape `cells × genes`:
entry `X[c, g]` is the number of unique mRNA molecules from gene `g` captured in cell `c`. The
matrix is brutally undersampled. A real cell holds hundreds of thousands of transcripts, but
droplet chemistry captures only a few thousand, so most of what was expressed is simply never
seen — a gene that is genuinely on in a cell reads as a hard zero (a *dropout*) far more often than
not. The thing being designed is a **denoiser**: a function `denoise(X) → X̂` that maps the noisy
count matrix to a smoother non-negative matrix `X̂` meant to approximate the underlying expression
*rate* each cell would have shown under infinite sequencing depth. Nothing downstream is learned;
the denoiser is scored directly on how well `X̂` recovers held-out signal.

The difficulty is that there is no clean ground truth on a real dataset — we only ever observe one
noisy `X`. So the evaluation borrows a trick from the imputation literature that turns the Poisson
nature of the counts into a self-supervised target.

## How the score is defined

The evaluation is **molecular cross-validation** (Batson, Royer & Webber, 2019). Take the observed
integer matrix `X` and split each entry's molecules by binomial thinning: draw `X_train[c,g] ~
Binomial(X[c,g], 0.5)` and set `X_test = X − X_train`. The justification is exact: thinning a
Poisson count splits it into two *independent* Poisson counts, each with half the rate. So
`X_train` and `X_test` are two independent noisy views of the same latent rate — denoise `X_train`,
and a good denoiser's output should predict `X_test`. Two metrics read off that prediction.

- **MSE in log-normalized space.** Library-size-normalize each row of both `X_test` and the
  denoised output (rescale each cell to a fixed total count), apply `log1p`, and take the mean
  squared error between the two matrices. This is the metric the OpenProblems "Denoising" task
  reports first, and it rewards getting the *shape* of each cell's normalized expression right.
- **Poisson negative log-likelihood.** Treat the denoised output, rescaled to the test molecule
  budget, as a rate `λ`, and score `mean(λ − X_test · log(λ + ε))` — the (constant-dropped) Poisson
  NLL of the held-out counts under that rate. This rewards getting the *absolute* per-gene rate
  right, and it penalizes a denoiser that smooths the shape correctly but mis-scales the counts.

Each metric is normalized against two fixed anchors. The **no-denoising** anchor is the raw
`X_train` itself (`X̂ = X_train`), the honest floor — predicting the noisy training half as-is. The
**perfect-denoising** anchor is the true rate `Λ` (available here because the data is synthetic),
scaled to each cell's observed depth — the ceiling a denoiser could reach if it knew the answer.
For each metric, `normalized = (raw − method) / (raw − perfect)`, so `0` means "no better than the
raw counts" and `1` means "as good as knowing the true rate." The reported score is the mean of the
two normalized metrics, higher better. A score above `1` is possible in principle (a method can
beat the rate-only anchor on a particular noisy split) but signals over-fitting the split rather
than recovering biology; the honest target is to climb toward `1` from below.

The reference points worth keeping in view come from the real OpenProblems benchmark and the
LLM-discovery system that currently tops it:

| Reference point | combined score (real PBMC / Tabula) |
|---|---|
| No denoising (raw counts) | ~0.0 |
| **MAGIC** (van Dijk 2018) — best classical method on the OpenProblems leaderboard | **~0.64** |
| **TTT-Discover** endpoint (test-time-training/discover, arXiv:2601.16175) | **0.71 / 0.73** |

The ladder below climbs from the trivial raw baseline through the classical neighbor-averaging and
graph-diffusion methods to an adaptation of the TTT-Discover denoiser, the genuine current best.
Because the experiment here runs on a *synthetic* count matrix rather than real PBMC/Tabula tissue,
the absolute numbers will not match the leaderboard — synthetic data is cleaner, so every rung
scores higher — but the *ordering* of the rungs and the size of the gaps between them is the honest
object of study.

## Prior art before the first rung

- **kNN-smoothing (Wagner, Yan & Yanai, 2018).** Find each cell's `k` nearest neighbors by
  observed expression distance and replace its profile with the (aggregated) neighbor average,
  pooling molecules to beat down Poisson noise. *Gap:* neighbors are chosen on *noisy* profiles, so
  in the sparse high-dropout regime the wrong cells get pooled; and a single hard `k` over-smooths
  cells in dense regions while under-smoothing isolated ones.
- **MAGIC (van Dijk et al., Cell 2018).** Build a cell-cell affinity graph with an adaptive-
  bandwidth (alpha-decay) kernel, row-normalize it into a Markov transition matrix `P`, and impute
  by `X̂ = Pᵗ X` — diffusing each cell's expression across the data manifold for `t` steps. The
  powered transition matrix propagates information *transitively* along the manifold, not just to
  immediate neighbors. *Gap:* `t` and the kernel are global; too much diffusion erases real
  cell-to-cell variation, and a single variance-stabilizing transform (square root) is not optimal
  across genes with wildly different dropout rates.
- **ALRA (Linderman et al., 2022).** Adaptively-thresholded low-rank approximation: SVD the
  normalized matrix, keep the top components, and zero out reconstruction values below a
  per-gene noise threshold so true biological zeros are preserved. *Gap:* purely low-rank, so it
  recovers the dominant structure but not fine local geometry, and it does not target the
  log-normalized or Poisson metrics directly.
- **Molecular cross-validation (Batson et al., 2019).** Not a denoiser but the *evaluation* itself —
  binomial thinning gives the self-supervised train/test split this whole task is scored on, and it
  is what lets the methods above be compared without a ground-truth rate.

## The fixed substrate

The harness is a thin, deterministic evaluator. It simulates a true rate matrix `Λ` with
continuous-trajectory plus discrete-branch low-rank structure, multiplicative biological
over-dispersion (so `Λ` is *not* exactly low-rank — no diffusion or SVD can recover it perfectly),
and lognormal per-cell library-size variation; draws `X ~ Poisson(Λ)`; performs the binomial MCV
split with a fixed seed; calls the denoiser on `X_train`; and computes the two normalized metrics
against the raw and perfect anchors. The simulator, the split, the metric, and the anchors are
frozen. Methods are *tuned* on one synthetic dataset (sim seed 0) and *reported* on three held-out
datasets (sim seeds 1–3), mimicking the PBMC→Tabula generalization the real benchmark measures.

## The editable interface

Exactly one function is editable: `denoise(X) → X̂`, taking the integer count matrix `X_train`
(shape `cells × genes`) and returning a non-negative float matrix of the same shape. Every rung on
the ladder is a different body for it. The simulator, MCV split, and scorer are fixed and shown for
reference.

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

Every valid output must be the same shape as `X`, finite, and non-negative. There are no other
constraints — the denoiser is free to smooth, diffuse, factorize, or transform however it likes.

## Evaluation settings

Four synthetic datasets: sim seed 0 is the **tune** set; sim seeds 1, 2, 3 are **held-out** report
sets. Each is `900 cells × 1000 genes`, median library size ~900 molecules after the MCV split,
dropout ~0.59 — a deliberately sparse, high-dropout regime where denoising matters. The MCV split
seed is fixed at 42 so every method is scored on exactly the same train/test partition. The
headline number for each rung is the combined score (mean of normalized MSE and normalized Poisson
NLL), reported as the mean over the three held-out datasets; the tune-set score is shown alongside
to expose any over-fitting. The three reference points — raw ~0, MAGIC ~0.64, TTT-Discover
0.71/0.73 on the *real* benchmark — are the yardsticks the ladder is read against, with the standing
caveat that the synthetic numbers run higher than the real ones.
