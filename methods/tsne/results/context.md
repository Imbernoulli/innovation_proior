# Context: nonlinear embedding for high-dimensional data visualization (circa 2002-2008)

## Research question

We have a data set of high-dimensional vectors `X = {x_1, ..., x_n}` — pixel-intensity vectors
of handwritten digits (784 dims), word-count vectors of documents (thousands of dims), images
of objects seen from many viewpoints — and we want to *look* at it: assign each `x_i` a
location `y_i` in a two-dimensional map that we can plot as a scatterplot and have it reveal
the structure that is really there. The structure is of two kinds: **local** structure (points
that are similar in the high-dimensional space should land close together, so neighborhoods are
preserved and a cluster of one digit class shows up as a coherent blob), and **global**
structure (the map should show how clusters sit relative to one another — that there are roughly
ten digit clusters, that several appearance-manifolds of one object form a loop, that classes
separate at several scales). The task is to capture both in a *single* two-dimensional map,
without using class labels (labels, when they exist, are only allowed to color the plot
afterwards, never to place the points).

## Background

Dimensionality reduction converts `X` into a low-dimensional `Y = {y_1, ..., y_n}` that can be
displayed. The methods on the table differ in what kind of structure they try to preserve.

**Distance-preserving / spectral methods.** The oldest and most robust line keeps pairwise
distances or their spectral surrogates. Principal Components Analysis (Hotelling, 1933) and
classical multidimensional scaling (Torgerson, 1952) are linear; they find a projection that
minimizes squared error between high- and low-dimensional pairwise distances. Graph-spectral
methods (Locally Linear Embedding, Roweis & Saul 2000; Laplacian Eigenmaps, Belkin & Niyogi
2002) build a `k`-nearest-neighbor graph and embed via eigenvectors; Isomap (Tenenbaum et al.
2000) replaces Euclidean by geodesic graph distances and then applies classical scaling.

**The probabilistic-neighbor frame.** A different frame turns the geometry into probability.
Instead of asking "what is the distance from `x_i` to `x_j`", ask "if `x_i` were to pick one
neighbor at random, with probability falling off as a Gaussian in distance, how likely is it to
pick `x_j`?" This converts each point's local geometry into a probability distribution over the
other points, with a built-in way to cope with varying data density across the space: the width
`σ_i` of the Gaussian centered on `x_i` can be set *per point*. The natural way to set it is
to fix the entropy of each point's neighbor distribution, captured by the **perplexity**,
`Perp(P_i) = 2^{H(P_i)}` with `H(P_i) = -Σ_j p_{j|i} log_2 p_{j|i}` measured in bits.
Perplexity rises monotonically with `σ_i`, so a binary search on `σ_i` hits a chosen
perplexity, which reads as a smooth "effective number of neighbors" — denser regions get
smaller `σ_i`. Typical perplexities are between 5 and 50, and performance is fairly robust to
the exact value.

**The crowding phenomenon.** It is well documented that local methods which try to honor
moderate distances in a two-dimensional map encounter crowding: because the area available at
moderate radius in 2-D is far smaller than what is available at moderate radius on a
higher-dimensional manifold, the many moderately-distant points compete for limited map area.
This is not specific to any one method; it appears in multidimensional-scaling methods such as
Sammon mapping and in the probabilistic-neighbor methods alike.

A useful piece of theory in this frame: matching one probability distribution to another has a
natural asymmetric cost, the **Kullback-Leibler divergence** `KL(P||Q) = Σ p log(p/q)`, which
is large when a small `q` models a large `p` (placing far-apart map points for nearby data
points) but only small when a large `q` models a small `p`. That asymmetry concentrates the
cost on getting the *local* structure right.

## Baselines

These are the prior methods a new visualization technique would be measured against and would
react to.

**Classical scaling / PCA (Torgerson, 1952; Hotelling, 1933).** Find a linear map minimizing
`Σ_{ij} (||x_i - x_j|| - ||y_i - y_j||)^2` (or its inner-product form).

**Sammon mapping (Sammon, 1969).** Repair classical scaling's bias toward large distances by
dividing each squared error by the original distance, so small distances count more:

```
C = (1 / Σ_{ij} ||x_i - x_j||) · Σ_{i≠j} (||x_i - x_j|| - ||y_i - y_j||)^2 / ||x_i - x_j||.
```

**Isomap (Tenenbaum et al., 2000).** Classical scaling on geodesic distances along a `k`-NN
graph, so curvature of the manifold is followed.

**LLE (Roweis & Saul, 2000) and Laplacian Eigenmaps (Belkin & Niyogi, 2002).** Embed by
reconstructing each point from its neighbors / by the smallest eigenvectors of the graph
Laplacian.

**Stochastic Neighbor Embedding (Hinton & Roweis, 2002) — the direct predecessor.** Work in
the probabilistic-neighbor frame. Convert high-dimensional distances to per-point conditional
probabilities,

```
p_{j|i} = exp(-||x_i - x_j||^2 / 2σ_i^2) / Σ_{k≠i} exp(-||x_i - x_k||^2 / 2σ_i^2),
```

with `σ_i` set by the perplexity binary search, and define analogous conditionals in the map
using a Gaussian of fixed width (absorbing the variance into the exponent),

```
q_{j|i} = exp(-||y_i - y_j||^2) / Σ_{k≠i} exp(-||y_i - y_k||^2).
```

If the map is faithful then `q_{j|i} ≈ p_{j|i}`, so it minimizes the summed Kullback-Leibler
divergence of the conditional distributions by gradient descent,

```
C = Σ_i KL(P_i || Q_i) = Σ_i Σ_j p_{j|i} log(p_{j|i} / q_{j|i}),
```

whose gradient with respect to a map point is

```
δC/δy_i = 2 Σ_j (p_{j|i} - q_{j|i} + p_{i|j} - q_{i|j}) (y_i - y_j).
```

The optimization uses simulated annealing — Gaussian noise added to the map points each
iteration with a slowly decaying variance, plus momentum and step-size schedules.

**UNI-SNE (Cook et al., 2007).** An attempt to address crowding by adding a slight uniform
repulsion: mix a small uniform background of weight `ρ` into the low-dimensional model, so that
`q_ij` can never fall below `2ρ / (n(n-1))` over the `n(n-1)/2` pairs, giving a mild repulsion
between far-apart points. It is applied by first optimizing standard SNE with annealing, then
raising `ρ`.

## Evaluation settings

The natural yardsticks for a visualization method, all pre-existing:

- **Data sets.** Handwritten digits (MNIST; 60,000 grayscale images, 28×28 = 784 pixels, of
  which a few thousand are typically subsampled), face images of many individuals under
  viewpoint/expression variation (Olivetti), object images under 72 rotations (COIL-20), and
  high-dimensional word/document vectors. Each comes with class labels used *only* to color the
  scatterplot, never to place the points.
- **Preprocessing protocol.** Reduce each data set to about 30 dimensions with PCA first, to
  speed up the pairwise-distance computation and suppress some noise without badly distorting
  the interpoint distances; then embed the 30-dimensional representation into two dimensions and
  show the result as a scatterplot.
- **What is judged.** Visually, whether the natural classes come apart and whether local
  structure within a class (orientation of the digits, the loop of viewpoints) is visible.
  Quantitatively, the generalization error of a nearest-neighbor classifier trained on the
  two-dimensional map (e.g. a 1-NN error measured by 10-fold cross-validation), which scores
  whether class structure survived the embedding. Standard neighborhood-preservation scores
  such as trustworthiness and continuity also ask whether neighbors in one space remain
  neighbors in the other.
- **Comparison set.** Sammon mapping, Isomap, LLE, curvilinear components analysis, SNE,
  Maximum Variance Unfolding, and Laplacian Eigenmaps, run with their own neighborhood/cost
  parameters (e.g. `k = 12` for Isomap and LLE).

## Code framework

A generic embedding harness can already compute pairwise distances, turn the high-dimensional
distances into per-point Gaussian neighbor distributions whose widths are set by perplexity,
initialize a low-dimensional map, and run a gradient-descent-with-momentum loop that repeatedly
evaluates a cost-and-gradient and updates the map points. The unsettled parts are how map points
should induce their own similarities, and what objective and gradient should drive the updates.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_high_dim_affinities(X, perplexity):
    """Turn high-dimensional distances into per-point Gaussian neighbor distributions,
    with each point's bandwidth set by a binary search on sigma so that the distribution's
    perplexity (effective number of neighbors) matches the target."""
    D = squareform(pdist(X, "sqeuclidean"))
    P = np.zeros_like(D)
    target = np.log(perplexity)
    for i in range(X.shape[0]):
        # binary search sigma_i so that H(P_i) matches log(perplexity)
        beta_lo, beta_hi, beta = -np.inf, np.inf, 1.0    # beta = 1 / (2 sigma_i^2)
        Di = np.delete(D[i], i)
        for _ in range(50):
            Pi = np.exp(-Di * beta)
            sumPi = max(Pi.sum(), 1e-12)
            H = np.log(sumPi) + beta * np.dot(Di, Pi) / sumPi   # entropy in nats
            if H > target:
                beta_lo = beta
                beta = beta * 2 if beta_hi == np.inf else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = beta / 2 if beta_lo == -np.inf else (beta + beta_lo) / 2
        Pi = np.exp(-Di * beta)
        P[i, np.arange(X.shape[0]) != i] = Pi / max(Pi.sum(), 1e-12)
    return P                                              # row-normalized conditionals p_{j|i}


def low_dim_affinities(Y):
    """The map-space similarity model -- the object we will define here."""
    # TODO: how a low-dimensional map induces pairwise similarities between map points.
    pass


def cost_and_gradient(P, Y):
    """The objective comparing high-dim affinities P to the map's affinities, and its
    gradient with respect to the map points."""
    # TODO: the cost we will minimize and its gradient w.r.t. Y.
    pass


def embed(X, perplexity, n_iter, eta, n_components=2, random_state=None):
    """Generic gradient-descent-with-momentum embedding harness."""
    rng = np.random.RandomState(random_state)
    P = compute_high_dim_affinities(X, perplexity)
    Y = 1e-4 * rng.standard_normal((X.shape[0], n_components))     # tiny initial map
    Y_prev = Y.copy()
    for t in range(n_iter):
        C, grad = cost_and_gradient(P, Y)
        momentum = 0.5 if t < 250 else 0.8
        Y_new = Y - eta * grad + momentum * (Y - Y_prev)
        Y_prev, Y = Y, Y_new
    return Y
```

The high-dimensional affinity step is fixed by the probabilistic-neighbor frame; the map-space
similarity model, the cost, and the gradient are the empty slots the method will fill.
