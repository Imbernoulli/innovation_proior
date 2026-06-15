# Context: dimension reduction for visualization (circa 2018-2020)

## Research question

Take a dataset `X` of `N` points in `R^P` (`P` in the hundreds, e.g. raw image
pixels) and place each point at a position `y_i` in `R^2` so that a human looking at
the scatter can read off the structure of the data. "Structure" has two parts, and
the central difficulty is that a 2D canvas cannot hold both at once. *Local structure*
is each point's neighborhood — which points are near which, and the rank order of
nearby distances; preserving it makes clusters compact and keeps same-class points
together. *Global structure* is the arrangement of the neighborhoods relative to each
other — if cluster A sits between clusters B and C in high dimensions, that betweenness
should survive; preserving it makes the overall layout (a curve, a hierarchy, a
continuum) legible. A low-dimensional embedding has limited capacity: it provably
cannot preserve every pairwise relationship, so any method must *choose* what to keep
and what to sacrifice.

The precise goal is an embedding algorithm that preserves **both** local and global
structure, and does so **robustly** — in particular not depending on a lucky
initialization or fragile hyperparameter tuning — using only the data (no class labels
at fit time). It must scale to large `N` (tens of thousands), since computing or
storing all `N^2` pairwise relationships is infeasible, so it has to work from a
sparse, chosen subset of point relationships. The methods of the day each preserve one
kind of structure well and the other poorly, or get the global layout right only by
accident of their initialization. Closing that gap — and doing it from an explicit
account of *why* a loss and a graph work, rather than another pile of choices that
happen to work — is the problem.

## Background

Dimension-reduction methods for visualization split into two camps. **Global methods**
— PCA (Pearson, 1901) and classical MDS (Torgerson, 1952) — find a linear projection
that best preserves variance or pairwise distances; they capture the coarse layout but,
being linear, cannot unfold a curved manifold. **Local methods** grew out of manifold
learning — Isomap (Tenenbaum et al., 2000), LLE (Roweis & Saul, 2000), Laplacian
Eigenmaps (Belkin & Niyogi, 2001) — which try to preserve local Euclidean distances.
A diagnostic fact undercut the distance-preserving approach: in high dimensions
pairwise distances **concentrate**, becoming nearly identical, so preserving raw
distances no longer preserves *which points are neighbors*. The field therefore moved
from preserving distances to preserving **graph structure** — the neighbor relation
itself.

Stochastic Neighbor Embedding (Hinton & Roweis, 2003) converts distances into
neighbor probabilities and matches the high- and low-dimensional probability
distributions by KL divergence, optimized by gradient descent (readable as a
spring/force system). SNE suffered the **crowding problem**: in low dimensions there
is not enough room, so moderately distant points pile on top of each other. Symmetric
SNE (Cook et al., 2007) symmetrized the loss and added a repulsion term.

A graph-based DR method has two separable parts, and the prevailing wisdom does not
say much about either in principle:
- a **graph construction** that picks, from the high-dimensional data, a set of *graph
  components* (edges, or triplets of points) to care about, possibly with weights;
- a **loss** over those components that, under gradient descent, induces **attractive**
  forces (pull chosen pairs together) and **repulsive** forces (push others apart) in
  the low-dimensional space.

Two empirical observations about *existing* methods set up the problem and are
knowable before any new method exists. First, the attractive and repulsive forces of
the leading methods **decay rapidly with low-dimensional distance**: once two points
are more than a few units apart in the embedding, essentially no force acts between
them, attractive or repulsive — the forces have a narrow "working zone." This is
visible in the distribution of embedded distances (most pairs sit well outside the
working zone) and follows from the functional form of the losses. Second, the
**initialization** of the low-dimensional points has an outsized effect: with random
starts, the global layout produced by some methods is consistently poor even after
convergence, while a structured start (PCA, spectral) produces a good global layout —
which means the global structure those methods display is, at least in part, inherited
from the initializer rather than created by the optimization.

A third recurring fact: DR algorithms are **not scale-invariant**. Scaling the
initial embedding by a constant changes the result, because whether forces fall inside
their working zone depends on the absolute distances; start everything too far apart
and all forces vanish (the optimizer freezes at iteration one), start too close and
strong repulsion dominates.

## Baselines

A new method would be measured against, and reacts to, these.

**PCA (Pearson, 1901; Hotelling, 1933).** Project onto the top-2 principal components
of the centered data. Fast, deterministic, preserves coarse global layout. **Gap:** a
linear map cannot represent a nonlinear manifold, so local neighborhood structure on a
curved manifold is distorted or lost.

**t-SNE (van der Maaten & Hinton, JMLR 2008).** For each point define Gaussian
neighbor probabilities `p_{j|i} ∝ exp(-||x_i - x_j||^2 / 2σ_i^2)`, symmetrize to
`p_{ij}`, and in the low-dimensional space use a heavy-tailed Student-t kernel
`q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Σ_{k≠l}(1 + ||y_k - y_l||^2)^{-1}`. Minimize
`Σ_{ij} p_{ij} log(p_{ij}/q_{ij})` by gradient descent; the gradient is
`∂L/∂y_i = 4 Σ_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}`. The Student-t
tail is what defeats the crowding problem (it gives distant points room). t-SNE is
excellent at local structure and clustering. **Gap:** decompose the gradient into an
attractive term `4 p_{ij} d_{ij}(1+d_{ij}^2)^{-1} e_{ij}` and a repulsive term
`4 q_{ij} d_{ij}(1+d_{ij}^2)^{-1} e_{ij}`. The attractive term depends on `p_{ij}`,
which is tiny for points that are not neighbors in the original space (modern
implementations zero it beyond `3·Perplexity` neighbors). The repulsive term, writing
    `a_{ij} = 1/(1+d_{ij}^2)` and `B_{ij} = Σ_{kl≠ij}(1+d_{kl}^2)^{-1}`, equals
    `4 a_{ij} d_{ij} a_{ij} /(B_{ij}+a_{ij})`, or
    `4d_{ij}/((1+d_{ij}^2)(B_{ij}(1+d_{ij}^2)+1))`. It is bounded above by a
    `Θ(1/d_{ij})` envelope and, with `B_{ij}` fixed, has leading decay
    `Θ(1/d_{ij}^3)`; its derivative also vanishes for large `d_{ij}`. So both forces are
near zero for points that end up far apart, and far points are treated almost
identically regardless of how far they actually are in the original space. The method
cannot tell a moderately distant point from a very distant one; the global arrangement
of clusters is not constrained. It is also slow, since it uses all `N^2` pairs.

**LargeVis (Tang et al., 2016).** A t-SNE-style neighbor-embedding that builds an
approximate k-NN graph and optimizes a likelihood with negative sampling, scaling to
large `N`. **Gap:** inherits t-SNE's local-only character — attraction on neighbors,
repulsion on sampled non-neighbors — so the same blindness to relative far-distances
remains.

**UMAP (McInnes, Healy & Melville, 2018).** Build a fuzzy k-NN graph: per-point
`ρ_i` (distance to nearest neighbor) and `σ_i` (a normalizer) give membership weights
`w(x_i,x_j) = exp(-max(0, Distance_{ij} - ρ_i)/σ_i)`, symmetrized to `w̄_{ij}`. In low
dimensions use a smooth kernel `(1 + a||y_i - y_j||^{2b})^{-1}` (with `a,b` fit from a
`min_dist` parameter, default `n_neighbors=15`, `min_dist=0.1`), and minimize the fuzzy
cross-entropy between the high- and low-dimensional graphs; in practice the gradient is
approximated, with attractive steps on graph edges and repulsive steps on
negatively-sampled non-edges. Initialized by spectral embedding. Faster than t-SNE,
strong local structure. **Gap:** its attractive and repulsive forces also decay
rapidly with low-dimensional distance — beyond the narrow working zone a far point
feels essentially no force regardless of its true distance — so, like t-SNE, it does
not distinguish among further points and need not preserve global structure; the good
global layouts it shows lean on the spectral initialization.

**TriMap (Amid & Warmuth, 2019).** Move from edges to **triplets**: sample triplets
`(i,j,k)` with `j` a near point and `k` a far point, weight each triplet `ω_{i,j,k}`
(a tempered-log function of the high-dimensional distances), and minimize
`Σ ω_{i,j,k} · s(y_i,y_k)/(s(y_i,y_j)+s(y_i,y_k))` with `s(y_a,y_b) = (1+||y_a-y_b||^2)^{-1}`.
Roughly 50 of each point's 55 triplets contain one of its 10 nearest neighbors; the
other 5 are random. Initialized by PCA, optimized with delta-bar-delta momentum.
Better global structure than t-SNE/UMAP. **Gaps:** (a) its global-structure advantage
turns out to track the **PCA initialization**, not the triplets — removing the random
triplets barely changes the output, while removing PCA init ruins the global layout;
(b) the triplet weights, after their tempered-log transform, come out almost all equal,
so the elaborate weighting does little; (c) along the axis where two far points are too
close, the triplet loss exerts little force to separate them — it gets away with this
only because the PCA start rarely places far points close together.

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
