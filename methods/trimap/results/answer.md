# TriMap, distilled

TriMap is a nonlinear dimensionality-reduction method for visualization that lays out points
by satisfying weighted **triplet** constraints `(i, j, k)` ("`i` is closer to `j` than to
`k`"). Its central design split: the **global** structure comes from a **PCA initialization**,
and the **triplets** then sharpen the **local** structure on top of that frame, using a
*saturating* per-triplet loss that stops applying force once a triplet is satisfied. It runs
in time linear in the number of points (dominated by the approximate nearest-neighbor search)
and scales to millions of points.

## Problem it solves

Embed high-dimensional data into 2D/3D preserving both local neighborhoods and global structure
(overall shape, relative placement of clusters, outliers). Pairwise neighbor-embedding methods
(t-SNE, LargeVis, UMAP) preserve local structure but leave the relative arrangement of clusters
underdetermined, so they distort the global layout — and the standard (local) scores cannot see
the distortion.

## Key ideas

1. **Triplets, not pairs.** A triplet only constrains a relative order (`i`–`j` closer than
   `i`–`k`), which is scale-free and is a higher-order summary, unlike absolute pairwise
   targets that are only meaningful for neighbors.
2. **Saturating per-triplet loss.** With Student-t similarity `s(y_a, y_b) = (1 + ||y_a −
   y_b||^2)^{-1}`,
   `l_ijk = w_ijk · s(y_i, y_k) / (s(y_i, y_j) + s(y_i, y_k)) = w_ijk / (1 + d_ik/d_ij)`,
   where `d_ij = 1 + ||y_i − y_j||^2`. It → 0 (no force) once `j` is near and `k` is far —
   unlike maximizing `Σ log p` (STE/t-STE), which keeps collapsing already-satisfied triplets.
3. **Importance weights from the high-D margin.** `w̃_ijk = δ_ik^2 − δ_ij^2 ≥ 0`, with
   locally-scaled squared distances `δ_ij^2 = ||x_i − x_j||^2 / (σ_i σ_j)`, `σ_i =` mean
   distance from `x_i` to its 4th–6th nearest neighbors (self-tuning density normalization).
   Then shift by the minimum and apply a tempered logarithm:
   `w_ijk = log_t(1 + w̃_ijk − w_min)`, `log_t(u) = (u^{1−t} − 1)/(1 − t)`, default `t = 0.5`
   (gentle compression so a few huge-margin triplets don't dominate).
4. **`O(n)` triplet sampling.** Per point: `m` nearest neighbors as `j`, `m'` random outliers
   `k` per neighbor (`m·m'` near-neighbor triplets), plus `r` random triplets (oriented by
   high-D distance, downweighted ×0.1). The default counts are `m = 12`, `m' = 4`,
   `r = 3`, so each point contributes 48 near-neighbor triplets and 3 random triplets.
   Approximate NN via random-projection trees.
5. **PCA initialization carries the global structure.** Initialize `Y` to the (down-scaled,
   ×0.01) PCA embedding. The near-neighbor triplets — over 90% of the set — can drive the loss to
   ~0 while destroying global layout, so they cannot supply global structure; PCA does. Because
   PCA already separates far points, the triplet forces only sharpen neighborhoods without
   tearing the global frame, and convergence from a structured start is fast.
6. **Full-batch optimization.** Triplets sampled once; minimize `Σ l_ijk` by full-batch
   gradient descent with momentum (0.5 for the first 250 iters, 0.8 after; gradient evaluated
   at the look-ahead `Y + γ·vel`) and per-coordinate delta-bar-delta adaptive gains, ~400 iters.

## Per-triplet gradient

With `L = w · d_ij/(d_ij + d_ik)`, `y_ij = y_i − y_j`, `y_ik = y_i − y_k`, and prefactor
`w' = w/(d_ij + d_ik)^2` (the global factor 2 absorbed into the learning rate):

```
gs = w' · d_ik · y_ij        # pull i toward j
go = w' · d_ij · y_ik        # push i away from k
grad_i += gs - go ;  grad_j -= gs ;  grad_k += go
```

`w'` shrinks as a triplet becomes well satisfied (`d_ik` large, `d_ij` small), which is the
loss saturation showing up in the forces. Each triplet touches three rows, so a full-batch
gradient over `O(n)` triplets costs `O(n)` per iteration.

## Global score (companion measure)

A quantitative measure of global accuracy: `err(Y|X) = min_A ||X − A Y||_F^2` (best linear
reconstruction, `A* = X Y^T (Y Y^T)^{-1}`, which absorbs rotation/scaling), and
`GS(Y|X) = exp(−(err(Y|X) − err_PCA)/err_PCA) ∈ [0, 1]`, equal to 1 for PCA (the
linear-reconstruction-optimal embedding). Higher GS = better global structure.

## Working code

Compact implementation of the algorithm; `fit_transform` returns the `(n, n_components)`
embedding. The neighbor search is written with scikit-learn here, in the same slot occupied by
an approximate nearest-neighbor index at large scale.

```python
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

INIT_PCA_SCALE = 0.01
RAND_WEIGHT_SCALE = 0.1


def tempered_log(u, t):
    """log_t(u) = (u^{1-t} - 1)/(1 - t);  -> log(u) as t -> 1."""
    if abs(t - 1.0) < 1e-5:
        return np.log(u)
    return (np.power(u, 1.0 - t) - 1.0) / (1.0 - t)


def preprocess(X, pca_dim=100):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] > pca_dim:
        X = X - X.mean(axis=0)
        X = TruncatedSVD(n_components=pca_dim, random_state=0).fit_transform(X)
        return X, True
    X = X - np.min(X)
    X = X / max(np.max(X), 1e-12)
    X = X - X.mean(axis=0)
    return X, False


def generate_triplets(X, n_inliers=12, n_outliers=4, n_random=3,
                      n_extra=50, weight_temp=0.5, rng=None):
    """Sample O(n) weighted triplets: near-neighbor triplets + a few random ones."""
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]
    k = min(n_inliers + n_extra, n)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    knn_dist, nbrs = nn.kneighbors(X)                       # self at column 0

    # self-tuning local scale: mean distance to the 4th-6th neighbors
    sig = np.maximum(knn_dist[:, 4:7].mean(axis=1), 1e-10)
    P = -knn_dist ** 2 / (sig[:, None] * sig[nbrs])         # = -delta_ij^2

    triplets, weights = [], []
    for i in range(n):                                       # near-neighbor triplets
        order = np.argsort(-P[i])
        for a in range(n_inliers):
            j = nbrs[i, order[a + 1]]                        # skip self
            p_sim = P[i, order[a + 1]]                       # = -delta_ij^2
            rejects = set(nbrs[i, order[:a + 2]])
            for _ in range(n_outliers):
                k_ = rng.integers(n)
                while k_ in rejects:
                    k_ = rng.integers(n)
                d_ik2 = np.sum((X[i] - X[k_]) ** 2) / (sig[i] * sig[k_])
                triplets.append((i, j, k_))
                weights.append(p_sim + d_ik2)                # delta_ik^2 - delta_ij^2 >= 0

    for i in range(n):                                       # random triplets (faint global)
        for _ in range(n_random):
            j = rng.integers(n)
            while j == i:
                j = rng.integers(n)
            k_ = rng.integers(n)
            while k_ == i or k_ == j:
                k_ = rng.integers(n)
            d_ij2 = np.sum((X[i] - X[j]) ** 2) / (sig[i] * sig[j])
            d_ik2 = np.sum((X[i] - X[k_]) ** 2) / (sig[i] * sig[k_])
            if d_ij2 > d_ik2:                                # orient: j is the closer one
                j, k_, d_ij2, d_ik2 = k_, j, d_ik2, d_ij2
            triplets.append((i, j, k_))
            weights.append(RAND_WEIGHT_SCALE * (d_ik2 - d_ij2))

    triplets = np.asarray(triplets, dtype=np.int32)
    weights = np.nan_to_num(np.asarray(weights, dtype=np.float64))
    weights -= weights.min()                                 # shift smallest margin to 0
    weights = tempered_log(1.0 + weights, weight_temp)       # gentle compression, t=0.5
    return triplets, weights


def trimap_grad(Y, triplets, weights):
    """Sum l_ijk and its gradient: three local updates per triplet."""
    n, dim = Y.shape
    grad = np.zeros((n, dim))
    loss = 0.0
    for t in range(triplets.shape[0]):
        i, j, k = triplets[t]
        y_ij = Y[i] - Y[j]
        y_ik = Y[i] - Y[k]
        d_ij = 1.0 + y_ij @ y_ij                             # 1 + ||y_i - y_j||^2 (floors /0)
        d_ik = 1.0 + y_ik @ y_ik
        loss += weights[t] / (1.0 + d_ik / d_ij)             # saturates to 0 once satisfied
        w = weights[t] / (d_ij + d_ik) ** 2                  # prefactor w'
        gs = y_ij * d_ik * w
        go = y_ik * d_ij * w
        grad[i] += gs - go
        grad[j] -= gs
        grad[k] += go
    return grad, loss


class CustomDimReduction:
    """TriMap: triplet-based dimensionality reduction. PCA init for global structure;
    weighted, saturating triplet loss for local structure."""

    def __init__(self, n_components=2, random_state=None,
                 n_inliers=12, n_outliers=4, n_random=3, n_iters=400, lr=0.1):
        self.n_components = n_components
        self.random_state = random_state
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.n_iters = n_iters
        self.lr = lr

    def fit_transform(self, X):
        X, pca_solution = preprocess(X)
        rng = np.random.default_rng(self.random_state)

        triplets, weights = generate_triplets(
            X, self.n_inliers, self.n_outliers, self.n_random, rng=rng)

        if pca_solution:
            Y = INIT_PCA_SCALE * X[:, :self.n_components]
        else:
            Y = INIT_PCA_SCALE * PCA(
                n_components=self.n_components,
                random_state=self.random_state).fit_transform(X)
        Y = Y.astype(np.float64)

        vel = np.zeros_like(Y)
        gain = np.ones_like(Y)
        for it in range(self.n_iters):
            gamma = 0.8 if it > 250 else 0.5                 # momentum: calm, then accelerate
            grad, _ = trimap_grad(Y + gamma * vel, triplets, weights)   # look-ahead gradient
            flip = np.sign(vel) != np.sign(grad)
            gain = np.where(flip, gain + 0.2, np.maximum(gain * 0.8, 0.01))  # delta-bar-delta
            vel = gamma * vel - self.lr * gain * grad
            Y += vel
        return Y


def global_loss(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
    return np.mean((X.T - A @ Y.T) ** 2)


def global_score(X, Y):
    Y_pca = PCA(n_components=Y.shape[1]).fit_transform(X)
    err_pca = global_loss(X, Y_pca)
    err = global_loss(X, Y)
    return np.exp(-(err - err_pca) / err_pca)
```

## Relation to prior methods

- **t-SNE / UMAP / LargeVis**: pairwise neighbor-embedding; preserve local structure, leave
  the global cluster arrangement underdetermined. TriMap reuses t-SNE's Student-t kernel for
  the low-D similarity but constrains *relative* triplet orders and anchors the global layout
  with PCA instead of a small random start.
- **STE / t-STE** (van der Maaten & Weinberger, 2012): triplet embedding by maximizing
  `Σ log p_ijℓ` over *given* triplets. TriMap (a) samples informative triplets from the
  high-D features automatically, (b) weights them by their high-D margin, (c) uses a
  *saturating* ratio loss that stops pulling satisfied triplets (vs. the heavy-tailed
  log-prob that keeps collapsing them), and (d) initializes from PCA.
- **Semi-supervised metric learning from relative comparisons** (Amid, Gionis & Ukkonen,
  2016): the conceptual origin — refine an initial representation using relative comparisons.
  TriMap refines the PCA layout with triplets sampled from the data itself, with no human
  labels and at visualization scale.
- **Self-tuning local scaling** (Zelnik-Manor & Perona, 2005) supplies the per-point density
  normalization `σ_i σ_j`; the **tempered logarithm** (Naudts, 2002) supplies the weight
  compression.
