# UMAP, distilled

UMAP (Uniform Manifold Approximation and Projection) is a nonlinear dimension-reduction method
that represents data as a weighted neighbor graph — a *fuzzy simplicial set* — and finds a
low-dimensional layout of that graph by minimizing the **cross-entropy** between the
high-dimensional and low-dimensional fuzzy graphs via stochastic gradient descent with edge and
negative sampling. It scales roughly linearly in the number of graph edges after neighbor search,
preserves local structure while giving the optimizer a spectral global scaffold, and places no
restriction on the embedding dimension.

## Problem it solves

Embed `X = {x_1, ..., x_N} ⊂ R^n` (assumed to lie near a low-dimensional manifold `M`) into
`R^d` so that local neighborhoods remain recognizable and coarse graph structure has a useful
starting arrangement, at a scale of hundreds of thousands to millions of points, with design
choices grounded in theory rather than tuned to a benchmark.

## Key idea

Two phases over the same fuzzy graph.

**1. Build a fuzzy neighbor graph.** Justification: a `k`-neighbor graph approximates the
topology of `M` (open cover → Vietoris–Rips complex → Nerve theorem), but the clean theory needs
uniformly sampled data. Real data is not uniform, so *assume* uniformity and infer a per-point
Riemannian metric — which collapses to normalizing each point's distances by the scale of its
own `k`-neighborhood. Concretely, for each point `i` with sorted neighbor distances:

- `rho_i` = distance to the nearest neighbor (local-connectivity offset);
- `sigma_i` chosen by binary search so that `sum_{j=1}^k exp(-max(0, d(x_i,x_j) - rho_i)/sigma_i) = log2(k) * bandwidth`
  (`bandwidth = 1` by default);
- directed membership `v_{j|i} = exp(-max(0, d(x_i,x_j) - rho_i)/sigma_i)`, with the nearest
  neighbor getting weight 1 (forces local connectivity; reading the *excess* over `rho_i`
  defeats high-dimensional distance concentration).

Reconcile the two directed views of each edge by the probabilistic t-conorm (fuzzy union,
"probability at least one directed edge exists"):

`v_ij = v_{j|i} + v_{i|j} - v_{j|i} v_{i|j}`,  i.e.  `B = A + A^T - A∘A^T`.

**2. Lay out the graph by cross-entropy minimization.** Both graphs share the same reference set
of possible edges; each weight is a Bernoulli edge-existence probability (`v` high-dim, `w`
low-dim), so the matching divergence is cross-entropy, not KL-between-distributions:

`C = sum_e [ v log(v/w) + (1 - v) log((1 - v)/(1 - w)) ]`.

The `v log(v/w)` term attracts (large `v`, small `w` → pull together); the `(1 - v) log((1 - v)/(1 - w))`
term repels (small `v`, large `w` → push apart) — present for *every* pair, which is the
symmetric global-structure penalty KL lacks. Dropping the `v`-only constant terms leaves a
normalization-free objective `-sum_e [ v log w + (1 - v) log(1 - w) ]`, optimizable by SGD with
edge sampling (attraction) and negative sampling (repulsion), with no partition function and no
required LargeVis-style `gamma` term in the derived objective.

The low-dimensional membership is a smooth, differentiable kernel
`w = Phi(y_i, y_j) = (1 + a ||y_i - y_j||^{2b})^{-1}`, with `a, b` fit by nonlinear least squares
to the offset-exponential target `Psi(d) = 1` for `d ≤ min_dist`, else `exp(-(d - min_dist)/spread)`.
`a = b = 1` recovers t-SNE's Student-t, so this generalizes the heavy-tailed embedding kernel;
`min_dist = 0.1` with `spread = 1` gives `a ≈ 1.929`, `b ≈ 0.7915`.

Initialize with the spectral embedding (bottom eigenvectors of `G`'s symmetric normalized
Laplacian `L = I - D^{-1/2} A D^{-1/2}`), since the graph Laplacian approximates the
Laplace–Beltrami operator of `M` under the usual sampling assumptions and gives SGD a coherent
global starting layout.

## Gradients (the forces)

With `D = ||y_i - y_j||^2`, `w = (1 + a D^b)^{-1}`, and `dD/dy_i = 2(y_i - y_j)`:

- attractive (from `v log w`):
  `grad_{y_i} = -[2 a b D^{b-1}/(1 + a D^b)] (y_i - y_j)`, scaled by `v`;
- repulsive (from `(1 - v) log(1 - w)`):
  `grad_{y_i} = [2 b / ((eps + D)(1 + a D^b))] (y_i - y_j)`, scaled by `(1 - v)`,
  with `eps = 0.001` flooring `D → 0`.

Gradients are clipped to `[-4, 4]` per coordinate for SGD stability; the learning rate decays
linearly `alpha = 1 - epoch/n_epochs` (annealing); `n_neg_samples = 5`; `n_epochs = 500` for
`N ≤ 10000`, else `200`. Total cost `O(k N)`.

## Defaults and why

- `n_neighbors = k = 15` — the local scale at which `M` is approximated as flat; small `k`
  captures fine detail, large `k` favors global structure.
- `min_dist = 0.1` — minimum spacing of points in the layout (the embedding-side analogue of
  `rho`); small packs tightly and faithfully, large spreads for legibility.
- calibration target `log2(k) * bandwidth` — fixes the (smoothed) fuzzy-neighborhood
  cardinality; with default `bandwidth = 1`, the empirical target is `log2(k)`.
- t-conorm union — principled merge of incompatible local metrics under edge-existence
  semantics.
- spectral init + linear LR decay — coherent graph-based start, then anneal the stochastic
  optimizer on a non-convex objective.

## Algorithm

```
UMAP(X, k, d, min_dist, n_epochs):
    for each x in X:                                   # construct the fuzzy graph
        knn, dists = ApproxNearestNeighbors(X, x, k)
        rho   = dists[1]                               # nearest-neighbor distance
        sigma = SmoothKNNDist(dists, k, rho)           # default target: sum exp(...) = log2(k)
        for y in knn:
            v[x,y] = exp(-max(0, dist(x,y) - rho)/sigma)
    G = fuzzy_union over x of v                        # v_ij = v_j|i + v_i|j - v_j|i v_i|j
    a, b = fit (1 + a r^{2b})^{-1} to offset-exp(min_dist, spread)
    Y = SpectralEmbedding(G, d)                        # normalized-Laplacian eigenvectors
    for epoch in 1..n_epochs:                          # SGD: cross-entropy layout
        for edge ([i,j], v) in G sampled with prob ∝ v:
            Y_i += alpha * grad(log Phi)(Y_i, Y_j)     # attraction
            for s in 1..n_neg_samples:
                c = random vertex
                Y_i += alpha * grad(log(1 - Phi))(Y_i, Y_c)   # repulsion
        alpha = 1 - epoch/n_epochs
    return Y
```

## Relation to prior methods

- **Laplacian Eigenmaps** — supplies the spectral initialization (and the manifold-Laplacian
  justification); UMAP refines it with force-directed cross-entropy instead of stopping at the
  eigensolve, and removes the rigid uniform-sampling assumption via the inferred per-point
  metric.
- **t-SNE** — same family of input affinities, but UMAP uses cross-entropy of edge existence
  instead of KL of normalized distributions: no all-pairs partition function (hence `O(kN)` vs
  `O(N^2)` and Barnes–Hut), and a symmetric repulsion term that better preserves global
  structure. Setting `a = b = 1` makes the embedding kernel exactly t-SNE's Student-t.
- **LargeVis** — both drop embedding-side normalization and use SGD with edge + negative
  sampling; UMAP's objective is the *full* cross-entropy, so its baseline repulsion weight
  `(1 - v)` is derived rather than introduced as a separate likelihood-balance term.

## Working code

Plain-Python rendering of the canonical implementation kernels. The production package uses
NN-Descent and numba; the graph construction, `find_ab_params`, gradient coefficients,
`eps = 0.001`, `[-4,4]` clip, edge/negative sampling schedule, and defaults below match the
canonical formulas and schedules.

```python
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Per-point rho (local-connectivity offset) and sigma (bandwidth) such that
    sum_j exp(-max(0, d - rho)/sigma) = log2(k) * bandwidth."""
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])
    mean_distances = np.mean(distances)
    for i in range(distances.shape[0]):
        lo, hi, mid = 0.0, np.inf, 1.0
        ith = distances[i]
        nonzero = ith[ith > 0.0]
        if nonzero.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interp = local_connectivity - index
            if index > 0:
                rho[i] = nonzero[index - 1]
                if interp > SMOOTH_K_TOLERANCE:
                    rho[i] += interp * (nonzero[index] - nonzero[index - 1])
            else:
                rho[i] = interp * nonzero[0]
        elif nonzero.shape[0] > 0:
            rho[i] = np.max(nonzero)
        for _ in range(n_iter):                              # binary search for sigma
            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                psum += np.exp(-(d / mid)) if d > 0 else 1.0
            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid; mid = (lo + hi) / 2.0
            else:
                lo = mid; mid = mid * 2 if hi == np.inf else (lo + hi) / 2.0
        result[i] = mid
        if rho[i] > 0.0:
            result[i] = max(result[i], MIN_K_DIST_SCALE * np.mean(ith))
        else:
            result[i] = max(result[i], MIN_K_DIST_SCALE * mean_distances)
    return result, rho


def fuzzy_simplicial_set(knn_indices, knn_dists, N, k):
    """High-dim fuzzy graph with t-conorm symmetrization A + A^T - A∘A^T."""
    sigmas, rhos = smooth_knn_dist(knn_dists, k)
    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float64)
    n_nb = knn_indices.shape[1]
    for i in range(N):
        for j in range(n_nb):
            nb = knn_indices[i, j]
            if nb == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0                                    # nearest neighbor -> membership 1
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / sigmas[i]))
            rows[i * n_nb + j] = i
            cols[i * n_nb + j] = nb
            vals[i * n_nb + j] = val
    A = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(N, N))
    A.eliminate_zeros()
    AT = A.transpose()
    G = A + AT - A.multiply(AT)                              # probabilistic fuzzy union
    G.eliminate_zeros()
    return G.tocoo()


def find_ab_params(spread, min_dist):
    """Fit a, b so (1 + a r^{2b})^{-1} matches the offset-exponential target."""
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    xv = np.linspace(0, spread * 3, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    (a, b), _ = curve_fit(curve, xv, yv)
    return a, b


def spectral_layout(G, dim, random_state):
    """Bottom eigenvectors of the symmetric normalized Laplacian L = I - D^-1/2 A D^-1/2."""
    N = G.shape[0]
    deg = np.asarray(G.sum(axis=1)).flatten()
    sqrt_deg = np.sqrt(np.maximum(deg, 1e-12))
    D_inv = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, N, N)
    L = scipy.sparse.identity(N) - D_inv * G * D_inv
    k = dim + 1
    vals, vecs = scipy.sparse.linalg.eigsh(L.tocsc(), k, which="SM",
                                           v0=np.ones(N), maxiter=N * 5)
    order = np.argsort(vals)
    return vecs[:, order[1:k]]                               # drop the trivial first eigenvector


def noisy_scale_coords(coords, random_state, max_coord=10.0, noise=0.0001):
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float64)
    return coords + random_state.normal(scale=noise, size=coords.shape)


def clip(v):
    return np.clip(v, -4.0, 4.0)


def make_epochs_per_sample(weights, n_epochs):
    """Canonical edge schedule: sample an edge in proportion to its graph weight."""
    result = np.full(weights.shape[0], -1.0, dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    positive = n_samples > 0
    result[positive] = float(n_epochs) / n_samples[positive]
    return result


def optimize_layout(head, tail, weights, Y, n_epochs, a, b,
                    n_neg_samples=5, gamma=1.0, random_state=None):
    """Minimize -sum_e [v log w + (1-v) log(1-w)] by SGD with edge + negative sampling."""
    N, dim = Y.shape
    rng = np.random.default_rng(random_state)
    epochs_per_sample = make_epochs_per_sample(weights, n_epochs)
    epochs_per_neg = epochs_per_sample / n_neg_samples
    next_sample = epochs_per_sample.copy()
    next_neg = epochs_per_neg.copy()
    alpha = 1.0
    for epoch in range(n_epochs):
        for e in range(head.shape[0]):
            if next_sample[e] > epoch:
                continue
            j, k_ = head[e], tail[e]
            cur = Y[j]; oth = Y[k_]
            d2 = np.sum((cur - oth) ** 2)
            if d2 > 0.0:                                     # attraction
                coeff = -2.0 * a * b * d2 ** (b - 1.0)
                coeff /= a * d2 ** b + 1.0
            else:
                coeff = 0.0
            grad = clip(coeff * (cur - oth))
            Y[j] = cur + grad * alpha
            Y[k_] = oth - grad * alpha
            next_sample[e] += epochs_per_sample[e]

            n_neg = int((epoch - next_neg[e]) / epochs_per_neg[e])
            for _ in range(n_neg):                           # repulsion (negative samples)
                c = rng.integers(N)
                oth = Y[c]
                d2 = np.sum((Y[j] - oth) ** 2)
                if d2 > 0.0:
                    coeff = 2.0 * gamma * b
                    coeff /= (0.001 + d2) * (a * d2 ** b + 1.0)
                    grad = clip(coeff * (Y[j] - oth))
                elif j == c:
                    continue
                else:
                    grad = np.zeros(dim)
                Y[j] = Y[j] + grad * alpha
            next_neg[e] += n_neg * epochs_per_neg[e]
        alpha = 1.0 - epoch / float(n_epochs)
    return Y


def umap_fit_transform(X, n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0,
                       n_epochs=None, random_state=None):
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    knn_dists, knn_indices = nn.kneighbors(X)

    G = fuzzy_simplicial_set(knn_indices, knn_dists, N, float(n_neighbors))

    if n_epochs is None:
        n_epochs = 500 if N <= 10000 else 200
    G.data[G.data < G.data.max() / float(n_epochs)] = 0.0    # prune un-samplable edges
    G.eliminate_zeros()

    a, b = find_ab_params(spread, min_dist)

    Y = spectral_layout(G, n_components, random_state).astype(np.float64)
    Y = noisy_scale_coords(Y, rng)
    Y = 10.0 * (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))

    Y = optimize_layout(G.row, G.col, G.data, Y, n_epochs, a, b,
                        random_state=random_state)
    return Y
```
