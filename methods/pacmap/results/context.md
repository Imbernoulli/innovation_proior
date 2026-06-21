# Context: dimension reduction for visualization (circa 2018-2020)

## Research question

Take a dataset `X` of `N` points in `R^P` (`P` in the hundreds, e.g. raw image
pixels) and place each point at a position `y_i` in `R^2` so that a human looking at
the scatter can read off the structure of the data. "Structure" has two parts. *Local
structure* is each point's neighborhood — which points are near which, and the rank
order of nearby distances; preserving it makes clusters compact and keeps same-class
points together. *Global structure* is the arrangement of the neighborhoods relative
to each other — if cluster A sits between clusters B and C in high dimensions, that
betweenness should survive; preserving it makes the overall layout (a curve, a
hierarchy, a continuum) legible. A low-dimensional embedding has limited capacity: it
provably cannot preserve every pairwise relationship, so any method must *choose* what
to keep. The broad question is how to design an embedding algorithm — its graph
construction, loss, and optimization — that uses only the data (no class labels at fit
time) and scales to large `N` (working from a sparse, chosen subset of point
relationships rather than all `N^2` pairs).

## Background

Dimension-reduction methods for visualization split into two camps. **Global methods**
— PCA (Pearson, 1901) and classical MDS (Torgerson, 1952) — find a linear projection
that best preserves variance or pairwise distances; they capture the coarse layout but,
being linear, cannot unfold a curved manifold. **Local methods** grew out of manifold
learning — Isomap (Tenenbaum et al., 2000), LLE (Roweis & Saul, 2000), Laplacian
Eigenmaps (Belkin & Niyogi, 2001) — which try to preserve local Euclidean distances.
A diagnostic fact shaped the field: in high dimensions pairwise distances
**concentrate**, becoming nearly identical, so the field moved from preserving distances
to preserving **graph structure** — the neighbor relation itself.

Stochastic Neighbor Embedding (Hinton & Roweis, 2003) converts distances into
neighbor probabilities and matches the high- and low-dimensional probability
distributions by KL divergence, optimized by gradient descent (readable as a
spring/force system). SNE suffered the **crowding problem**: in low dimensions there
is not enough room, so moderately distant points pile on top of each other. Symmetric
SNE (Cook et al., 2007) symmetrized the loss and added a repulsion term.

A graph-based DR method has two separable parts:
- a **graph construction** that picks, from the high-dimensional data, a set of *graph
  components* (edges, or triplets of points) to care about, possibly with weights;
- a **loss** over those components that, under gradient descent, induces **attractive**
  forces (pull chosen pairs together) and **repulsive** forces (push others apart) in
  the low-dimensional space.

## Baselines

**PCA (Pearson, 1901; Hotelling, 1933).** Project onto the top-2 principal components
of the centered data. Fast, deterministic, preserves coarse global layout.

**t-SNE (van der Maaten & Hinton, JMLR 2008).** For each point define Gaussian
neighbor probabilities `p_{j|i} ∝ exp(-||x_i - x_j||^2 / 2σ_i^2)`, symmetrize to
`p_{ij}`, and in the low-dimensional space use a heavy-tailed Student-t kernel
`q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Σ_{k≠l}(1 + ||y_k - y_l||^2)^{-1}`. Minimize
`Σ_{ij} p_{ij} log(p_{ij}/q_{ij})` by gradient descent; the gradient is
`∂L/∂y_i = 4 Σ_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}`. The Student-t
tail defeats the crowding problem by giving distant points more room. t-SNE is
excellent at local structure and clustering. The gradient decomposes into an attractive
term `4 p_{ij} d_{ij}(1+d_{ij}^2)^{-1} e_{ij}` and a repulsive term
`4 q_{ij} d_{ij}(1+d_{ij}^2)^{-1} e_{ij}`. Modern implementations zero the attractive
term beyond `3·Perplexity` neighbors. The method uses all `N^2` pairs, making it slow
for large `N`.

**LargeVis (Tang et al., 2016).** A t-SNE-style neighbor-embedding that builds an
approximate k-NN graph and optimizes a likelihood with negative sampling, scaling to
large `N`. It applies attraction on neighbors and repulsion on sampled non-neighbors.

**UMAP (McInnes, Healy & Melville, 2018).** Build a fuzzy k-NN graph: per-point
`ρ_i` (distance to nearest neighbor) and `σ_i` (a normalizer) give membership weights
`w(x_i,x_j) = exp(-max(0, Distance_{ij} - ρ_i)/σ_i)`, symmetrized to `w̄_{ij}`. In low
dimensions use a smooth kernel `(1 + a||y_i - y_j||^{2b})^{-1}` (with `a,b` fit from a
`min_dist` parameter, default `n_neighbors=15`, `min_dist=0.1`), and minimize the fuzzy
cross-entropy between the high- and low-dimensional graphs; in practice the gradient is
approximated, with attractive steps on graph edges and repulsive steps on
negatively-sampled non-edges. Initialized by spectral embedding. Faster than t-SNE,
strong local structure.

**TriMap (Amid & Warmuth, 2019).** Move from edges to **triplets**: sample triplets
`(i,j,k)` with `j` a near point and `k` a far point, weight each triplet `ω_{i,j,k}`
(a tempered-log function of the high-dimensional distances), and minimize
`Σ ω_{i,j,k} · s(y_i,y_k)/(s(y_i,y_j)+s(y_i,y_k))` with `s(y_a,y_b) = (1+||y_a-y_b||^2)^{-1}`.
Roughly 50 of each point's 55 triplets contain one of its 10 nearest neighbors; the
other 5 are random. Initialized by PCA, optimized with delta-bar-delta momentum.

## Evaluation settings

The yardsticks already in use, all label-aware only at evaluation time (labels removed
before fitting):
- **Datasets.** Image collections with class labels — MNIST and Fashion-MNIST
  (784-dim grayscale, 10 classes), COIL-20 (object images); text reduced to a modest
  dimension (e.g. 20 Newsgroups via TF-IDF + truncated SVD to ~50 dims); and synthetic
  manifolds with known geometry where global structure is the whole point — an S-curve
  (with a hole), a Mammoth point cloud, a folded 2D curve.
- **Local-structure metrics.** k-NN classification accuracy in the 2D embedding (a
  classifier trained on the embedded coordinates; high accuracy means same-class points
  stayed together); **trustworthiness** (fraction of embedding-space neighbors that
  were also neighbors in the original space) and **continuity** (the reverse), with a
  fixed small `k` (e.g. 7).
- **Global-structure metrics.** Random-triplet accuracy (fraction of random triplets
  whose distance ordering is preserved) and centroid-triplet accuracy (the same on
  class centroids); and qualitative inspection of whether a known manifold's shape is
  recovered.
- **Protocol.** Reduce to 2 dimensions; compare methods under their default
  hyperparameters and across initializations (random vs. structured) to probe
  robustness; on synthetic manifolds, judge against the ground-truth shape.

## Code framework

The substrate is a generic graph-based DR harness that already exists: choose a small
set of point relationships from the high-dimensional data, define a differentiable loss
over the embedded positions whose gradient is a set of attractive/repulsive forces,
initialize the 2D points, and run a gradient optimizer. The pieces that are settled
beforehand — exact or approximate k-nearest-neighbor search, PCA, and a first-order
adaptive optimizer (Adam: per-parameter EMAs of the gradient and squared gradient, bias
corrected) — are library primitives. What is **not** settled is *which* relationships to
select, *what* loss/force to put on each, and *how* to drive the optimization. The
scaffold keeps those choices behind neutral function hooks.

```python
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def select_components(X, rng):
    """From high-dimensional X, choose the point relationships the loss will sum over.
    Which relationships to pick -- and whether one binary near/far split is enough --
    is part of what we design."""
    # nearest-neighbor search and PCA are available primitives:
    #   nbrs = NearestNeighbors(...).fit(X).kneighbors(X)
    # TODO: the set(s) of graph components we will define here.
    pass


def loss_and_forces(Y, components, weights):
    """Given embedded positions Y and the chosen components, return the gradient
    (attractive + repulsive forces) of the loss. The functional form of the loss --
    and how it balances attraction against repulsion -- is what we design."""
    grad = np.zeros_like(Y)
    # TODO: the loss term(s) and their gradients we will define here.
    return grad


def weight_schedule(itr, num_iters):
    """The coefficient(s) on each kind of component at iteration itr.
    Whether/how these change over the optimization is part of what we design."""
    # TODO: the schedule we will define here.
    pass


def fit_transform(X, n_components=2, *, num_iters, lr, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    # optional dimensionality preprocessing / normalization
    if X.shape[1] > 100:
        X = X - X.mean(axis=0)
        X = TruncatedSVD(n_components=100, random_state=random_state).fit_transform(X)
        pca_init = None
    else:
        X = X - X.min()
        X = X / max(X.max(), 1e-12)
        X = X - X.mean(axis=0)
        pca_init = PCA(n_components=n_components, random_state=random_state).fit(X)

    components = select_components(X, rng)

    # initialize the embedding (PCA / random are known options)
    Y = X[:, :n_components] if pca_init is None else pca_init.transform(X)

    # first-order adaptive optimizer
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    beta1, beta2, eps = 0.9, 0.999, 1e-7
    for itr in range(num_iters):
        weights = weight_schedule(itr, num_iters)
        g = loss_and_forces(Y, components, weights)
        lr_t = lr * np.sqrt(1 - beta2 ** (itr + 1)) / (1 - beta1 ** (itr + 1))
        m += (1 - beta1) * (g - m)
        v += (1 - beta2) * (g ** 2 - v)
        Y -= lr_t * m / (np.sqrt(v) + eps)
    return Y
```
