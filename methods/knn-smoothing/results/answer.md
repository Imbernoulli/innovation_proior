**Problem.** Lift the denoiser off the raw zero by pooling information across cells. Cells in the
same biological state are independent noisy draws of one rate, so averaging similar cells beats down
Poisson noise while preserving shared signal. The crudest honest version is kNN-smoothing.

**Key idea.** Square-root-transform the counts (variance stabilization for Poisson data) and
library-size-normalize each cell to a common total, so distances reflect biology not depth. For each
cell, find its `k` nearest neighbors *on these noisy observed profiles* and replace the cell with the
uniform average over itself and its `k` neighbors. Undo the transform — square, then restore each
cell's original library size — so the output is on the count scale and keeps per-cell depth.

**Why these choices.** Square root makes count variance roughly mean-independent so loud genes do
not hijack the distance; library-size normalization removes the depth confound from neighbor
selection. Neighbors are chosen on the *raw* normalized profiles (no denoised embedding) — the
classical Wagner-style baseline — which is deliberately crude: in high dropout the noisy distances
mis-select neighbors, and a single global `k` with a hard uniform pool over-smooths dense regions
while under-smoothing isolated cells. These are the two flaws the graph-diffusion rung exists to
cure. `k ≈ 10` is tuned on the tune set as a bias-variance compromise.

**Hyperparameters / contract.** `knn = 10` (neighbors averaged, plus the cell itself). Input/output
shape preserved, output non-negative. Deterministic given the input. Neighbors via exact Euclidean
nearest-neighbor search in sqrt-normalized space.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def denoise(X, knn=10):
    """Wagner-style kNN-smoothing: neighbours on NOISY profiles, hard uniform average."""
    Xs = np.sqrt(X.astype(np.float64))                 # variance-stabilising transform
    lib = Xs.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    Xn = Xs / lib                                       # library-size normalise
    n = X.shape[0]
    k = min(knn + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(Xn)        # neighbours in noisy normalised space
    _, idx = nn.kneighbors(Xn)
    Y = Xn[idx].mean(axis=1)                            # uniform hard average over cell + k neighbours
    Y = Y * lib                                         # restore each cell's library size
    return np.square(Y)                                 # undo sqrt → count scale
```
