**Problem.** Cure the two weaknesses of kNN-smoothing: neighbors chosen on noisy profiles, and a
hard uniform pool with a single global `k`. Replace both with a geometry-aware, adaptive-bandwidth,
transitively-pooled diffusion on a cell-cell affinity graph — MAGIC.

**Key idea.** Square-root-transform and library-normalize the counts, then PCA-embed to a few dozen
components so neighbor distances are measured on *denoised* structure, not raw Poisson jitter. Build
an alpha-decay affinity kernel `exp(−(d/σ_i)^α)` whose per-cell bandwidth `σ_i` is the distance to
the `k`-th neighbor — narrow in dense regions, wide in sparse ones. Symmetrize and row-normalize
into a Markov transition matrix `P`, then impute by diffusion `X̂ = Pᵗ X`: one step is a soft
affinity-weighted neighbor average, and powers `Pᵗ` let cells borrow *transitively* from neighbors'
neighbors along the manifold. Invert (square, restore library size) back to the count scale.

**Why these choices.** PCA fixes the noisy-neighbor flaw — the leading components capture the
trajectories and branches, discarding per-gene noise. The adaptive-bandwidth kernel fixes the
single-`k` flaw — bandwidth follows local density automatically. Powered diffusion replaces the hard
cliff of kNN with smooth transitive pooling, which is exactly right on a continuum where cells a few
hops apart should share information. The step count `t` is the bias-variance lever: too small
under-smooths, too large washes every cell toward the global mean; `t = 2` is tuned small because
the affinity weighting already does most of the work in one step. `α = 2` (Gaussian-like decay) and
`k = 10` are standard.

**Hyperparameters / contract.** `knn = 10`, `t = 2`, `n_pca = 50`, `decay (α) = 2.0`. Input/output
shape preserved, output non-negative. Deterministic given a fixed PCA random state.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix

def denoise(X, knn=10, t=2, n_pca=50, decay=2.0, seed=0):
    Xs = np.sqrt(X.astype(np.float64))                      # sqrt VST
    lib = Xs.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    Xn = Xs / lib                                           # library-size normalise
    k_pca = min(n_pca, min(Xn.shape) - 1)
    emb = PCA(n_components=k_pca, random_state=seed).fit_transform(Xn)   # denoised embedding

    n = emb.shape[0]
    k = min(knn * 3 + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(emb)
    dist, idx = nn.kneighbors(emb)
    sigma = np.maximum(dist[:, knn], 1e-12)                 # adaptive bandwidth = knn-th distance
    rows = np.repeat(np.arange(n), k); cols = idx.ravel()
    aff = np.exp(-((dist.ravel() / sigma[rows]) ** decay))  # alpha-decay kernel
    A = csr_matrix((aff, (rows, cols)), shape=(n, n)); A = A + A.T   # symmetrise
    rs = np.asarray(A.sum(axis=1)).ravel(); rs[rs == 0] = 1.0
    P = A.multiply(1.0 / rs[:, None]).tocsr()               # row-stochastic transition matrix

    Y = Xn.copy()
    for _ in range(t):                                      # diffuse: X_hat = P^t X
        Y = P @ Y
    return np.square(Y * lib)                               # restore depth, undo sqrt
```
