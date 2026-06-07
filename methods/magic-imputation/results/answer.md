# MAGIC — Markov Affinity-based Graph Imputation of Cells

**Problem.** Single-cell RNA-seq captures only a small random fraction (~5-15%) of each cell's transcriptome, so the cells x genes count matrix is sparse and noisy: expressed genes are frequently recorded as zero ("dropout"). This attenuates gene-gene correlations and fractures continuous biological trajectories.

**Key idea.** Cells lie on a low-dimensional manifold, so a cell's true expression is shared with its near neighbors, and dropout is per-cell and roughly independent — what one cell lost, its neighbors kept. Build a cell-cell affinity graph, turn it into a Markov random-walk operator, and **diffuse** the expression values over the graph: replace each cell by the t-step random-walk-weighted average of its neighborhood. Spectrally this is a low-pass filter that keeps slow manifold gradients (signal) and removes fast graph modes (noise).

## Algorithm

Preprocess: filter empty cells / unexpressed genes; library-size normalize each cell (divide by its total count, x median total); square-root transform; PCA to retain most variance (default n_pca = 100; ~70% variance, 20-100 PCs).

1. **Distances.** Euclidean cell-cell distances d(i,j) in PCA space.
2. **Adaptive affinity kernel.** Per-cell bandwidth sigma_i = distance from cell i to its knn-th nearest neighbor (the "ka"-th neighbor). Affinity
   A(i,j) = exp( -( d(i,j) / sigma_i )^alpha ),
   the Gaussian kernel being alpha = 2; MAGIC uses a decay exponent alpha (default decay = 1) for softer, more robustly-connecting tails. Keep at most knn_max = 3·knn nonzero neighbors per cell.
3. **Symmetrize.** A <- (A + A^T) / 2 (additive symmetrization; pulls in outliers).
4. **Markov normalization.** Row-normalize to a transition matrix: M(i,j) = A(i,j) / sum_k A(i,k). M is row-stochastic; M(i,j) is the one-step random-walk probability from cell i to cell j.
5. **Data diffusion (imputation).** X_imputed = M^t · X. Each gene in each cell becomes the t-step transition-weighted average over its neighborhood. In M's eigenbasis (eigenvalues in [0,1], top eigenvalue 1 = stationary mode), M^t scales each mode by lambda^t, shrinking small-lambda (high-frequency noise) modes far faster than large-lambda (smooth signal) modes — a graph low-pass filter.
6. **Choosing t.** Diffusion has a fast noise-removal regime then a slow over-smoothing regime (large t collapses every cell toward the global mean). Select t at the elbow: monitor the change between successive iterates and stop once it plateaus — via R^2(X_t, X_{t-1}) = 1 − SSE/SST dropping below ~5% change (take the second t past threshold), or equivalently the Procrustes disparity between X_t and X_{t-1} falling under a small threshold. Default fixed t = 3.
7. **Rescale (optional).** Averaging compresses per-gene range; restore it by scaling each imputed gene so its max equals the 99th percentile of the original gene.

Defaults: knn = 5, knn_max = 3·knn = 15, decay (alpha) = 1, t = 3 (or 'auto'), n_pca = 100.

## Code

```python
import numpy as np
from scipy import spatial
from sklearn.preprocessing import normalize   # l1 row-normalize -> stochastic
import graphtools                              # affinity kernel + diffusion operator


def preprocess(counts):
    """Library-size normalize + sqrt; filter empties first."""
    counts = counts[counts.sum(1) > 0]
    counts = counts[:, np.asarray(counts.sum(0)).ravel() > 0]
    libsize = counts.sum(1, keepdims=True)
    counts = counts / libsize * np.median(libsize)
    return np.sqrt(counts)


class MAGIC:
    """Markov Affinity-based Graph Imputation of Cells.

    knn      : neighbor index whose distance sets the per-cell bandwidth (ka)
    knn_max  : cap on nonzero neighbors (default 3 * knn ~ 3 sigma of the kernel)
    decay    : kernel exponent alpha in exp(-(d/sigma)^alpha); Gaussian is alpha=2
    t        : diffusion steps (power of M); 'auto' finds the elbow
    n_pca    : PCA dims for the neighbor geometry
    """

    def __init__(self, knn=5, knn_max=None, decay=1, t=3, n_pca=100):
        self.knn, self.decay, self.t, self.n_pca = knn, decay, t, n_pca
        self.knn_max = knn_max if knn_max is not None else 3 * knn

    def fit(self, X):
        # PCA -> kNN distances -> adaptive affinity A(i,j)=exp(-(d/sigma_i)^decay)
        # -> additive symmetrization -> M = D^{-1} A (row-stochastic Markov op).
        self.graph = graphtools.Graph(
            X, n_pca=self.n_pca, knn=self.knn, knn_max=self.knn_max,
            decay=self.decay, thresh=1e-4)
        self.diff_op = self.graph.diff_op          # the Markov transition matrix M
        return self

    def _select_t(self, X, t_max=20, thresh=1e-3):
        # diffuse one step at a time; stop when successive iterates stop moving
        Xt = np.asarray(X, dtype=float)
        prev = Xt.copy()
        for _ in range(t_max):
            Xt = self.diff_op.dot(Xt)
            _, _, disparity = spatial.procrustes(prev, Xt)
            prev = Xt.copy()
            if disparity < thresh:
                break
        return Xt

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.t == "auto":
            return self._select_t(X)
        Mt = np.linalg.matrix_power(self.diff_op.toarray(), self.t)
        return Mt.dot(X)                            # X_imputed = M^t X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def rescale(X_imputed, X_original, pct=99):
    """Optional: restore per-gene magnitude lost to averaging."""
    p = np.percentile(X_original, pct, axis=0)
    m = X_imputed.max(axis=0)
    scale = np.divide(p, m, out=np.zeros_like(p), where=m > 0)
    return X_imputed * scale


# usage
# X = preprocess(raw_counts)
# X_magic = MAGIC(knn=5, decay=1, t='auto').fit_transform(X)
```

The affinity-to-Markov construction (steps 2-4) and the row-stochastic diffusion operator are handled by `graphtools.Graph`: it sets each cell's bandwidth to its knn-th neighbor distance, forms `A(i,j) = exp(-(d/sigma_i)^decay)`, symmetrizes additively, and exposes `diff_op = normalize(A, 'l1', axis=1) = D^{-1}A`. The diffusion `M^t X` and the t-selection are the imputation itself.
