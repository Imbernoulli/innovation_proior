**Problem.** Denoise a UMI count matrix `X_train` (cells × genes) so its output predicts the
held-out half `X_test` of a binomial molecular-cross-validation split, scored by log-normalized MSE
and Poisson NLL, each normalized so raw counts → 0 and the true rate → 1. The first rung is the raw
baseline: return the input unchanged, as a calibration of the evaluator's zero point.

**Key idea.** Do no denoising at all — `X̂ = X_train`. Because both metrics' "no-denoising" anchor
*is* the raw training matrix, this method must score exactly `0` on each normalized term on every
dataset. It pools no information across cells, so it carries the full Poisson sampling noise into
its prediction of the test half, which is precisely the honest floor every later rung is measured
above.

**Why these choices.** There is nothing to choose. The value of the rung is diagnostic: confirming
the normalization brackets correctly (raw at 0, perfect rate above all methods), that the metric has
no sign error, and that the raw and perfect anchors really do bound the methods. A clean `0.0000`
on both terms is the signal that the scale is trustworthy before any real method is read against it.

**Hyperparameters / contract.** None. Input `X_train` (cells × genes, non-negative); output the same
matrix as float64. Deterministic. The whole gap from `0` to `1` is left for the methods that
actually borrow strength across cells.

```python
import numpy as np

def denoise(X):
    """Raw baseline: no denoising. The evaluator's zero anchor."""
    return X.astype(np.float64).copy()
```
