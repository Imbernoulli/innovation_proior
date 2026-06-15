# Context: nonlinear embedding for high-dimensional data visualization (circa 2002-2008)

## Research question

We have a data set of high-dimensional vectors `X = {x_1, ..., x_n}` — pixel-intensity vectors
of handwritten digits (784 dims), word-count vectors of documents (thousands of dims), images
of objects seen from many viewpoints — and we want to *look* at it: assign each `x_i` a
location `y_i` in a two-dimensional map that we can plot as a scatterplot and have it reveal
the structure that is really there. The structure is of two kinds at once, and that is the
crux. There is **local** structure: points that are similar in the high-dimensional space
(two slightly different 7s, two near-duplicate documents) should land close together, so that
neighborhoods are preserved and a cluster of one digit class shows up as a coherent blob.
And there is **global** structure: the map should show how the clusters sit relative to one
another — that there are roughly ten digit clusters, that several appearance-manifolds of one
object form a loop, that classes separate at several scales. A good method must capture both
of these in a *single* two-dimensional map, without using class labels (the labels, when they
exist, are only allowed to color the plot afterwards, never to place the points).

This is hard because the data lies on or near a curved, nonlinear manifold of intrinsic
dimension well above two, and we are squashing it into two dimensions. A linear projection
cannot follow a curved manifold. And there is a deeper geometric obstruction specific to
squashing many dimensions into two: in an `m`-dimensional space, the volume of a ball around a
point grows like `r^m`, so the number of points sitting at a *moderate* distance from a given
point can be very large; a two-dimensional map simply does not have enough area at moderate
radius to place all of them at the right distance. Any method that insists on honoring those
moderate distances will be forced to pile the moderately-distant points up, and the many small
attractions they exert will crush everything toward the center of the map, erasing the gaps
between the natural clusters. A method that solves the visualization problem has to keep
similar points together, keep dissimilar points apart, *and* somehow make room so that the
clusters do not collapse into one ball.

## Background

Dimensionality reduction converts `X` into a low-dimensional `Y = {y_1, ..., y_n}` that can be
displayed. The methods on the table differ in what kind of structure they try to preserve, and
the field's accumulated experience is that most of them preserve one kind at the cost of the
other.

**Distance-preserving / spectral methods.** The oldest and most robust line keeps pairwise
distances or their spectral surrogates. Principal Components Analysis (Hotelling, 1933) and
classical multidimensional scaling (Torgerson, 1952) are linear; they find a projection that
minimizes squared error between high- and low-dimensional pairwise distances. Because squared
error is dominated by the *large* distances, these methods are good at keeping dissimilar
points apart and poor at preserving the small distances that carry the local manifold
structure, and being linear they cannot follow a curved manifold at all. Graph-spectral
methods (Locally Linear Embedding, Roweis & Saul 2000; Laplacian Eigenmaps, Belkin & Niyogi
2002) build a `k`-nearest-neighbor graph and embed via eigenvectors; Isomap (Tenenbaum et al.
2000) replaces Euclidean by geodesic graph distances and then applies classical scaling.

**The probabilistic-neighbor frame.** A different and, for this problem, more promising frame
turns the geometry into probability. Instead of asking "what is the distance from `x_i` to
`x_j`", ask "if `x_i` were to pick one neighbor at random, with probability falling off as a
Gaussian in distance, how likely is it to pick `x_j`?" This converts each point's local
geometry into a probability distribution over the other points, and it has a built-in way to
cope with the fact that data density varies across the space: the width `σ_i` of the Gaussian
centered on `x_i` can be set *per point*. The natural way to set it is to fix the entropy of
each point's neighbor distribution, which is captured by the **perplexity**,
`Perp(P_i) = 2^{H(P_i)}` with `H(P_i) = -Σ_j p_{j|i} log_2 p_{j|i}` measured in bits.
Perplexity rises monotonically with `σ_i`, so a binary search on `σ_i` hits a chosen
perplexity, which reads as a smooth "effective number of neighbors" — denser regions get
smaller `σ_i`. Typical perplexities are between 5 and 50, and performance is fairly robust to
the exact value.

**The crowding phenomenon.** It is well documented that local methods which try to honor
moderate distances in a two-dimensional map suffer crowding: because the area available at
moderate radius in 2-D is far too small for the number of moderately-distant points on a
higher-dimensional manifold, those points get pushed too far out, and the aggregate of their
many weak attractions pulls the clusters together and prevents gaps from forming between the
natural classes. This is not specific to any one method; it appears in multidimensional-scaling
methods such as Sammon mapping and in the probabilistic-neighbor methods alike. A separate
empirical observation from the field is that even a semi-supervised variant of one strong
method (Maximum Variance Unfolding) cannot cleanly separate handwritten digits into their
classes — a blunt reminder that most existing techniques, however principled, do not actually
produce visualizations in which the clusters come apart.

A useful piece of theory in this frame: matching one probability distribution to another has a
natural asymmetric cost, the **Kullback-Leibler divergence** `KL(P||Q) = Σ p log(p/q)`, which
is large when a small `q` models a large `p` (placing far-apart map points for nearby data
points) but only small when a large `q` models a small `p`. That asymmetry is a feature here:
it makes the cost concentrate on getting the *local* structure right.

## Baselines

These are the prior methods a new visualization technique would be measured against and would
react to.

**Classical scaling / PCA (Torgerson, 1952; Hotelling, 1933).** Find a linear map minimizing
`Σ_{ij} (||x_i - x_j|| - ||y_i - y_j||)^2` (or its inner-product form). **Gap:** dominated by
large distances, so it keeps dissimilar points apart but discards the small distances that
encode local structure, and being linear it cannot model a curved manifold.

**Sammon mapping (Sammon, 1969).** Repair classical scaling's bias toward large distances by
dividing each squared error by the original distance, so small distances count more:

```
C = (1 / Σ_{ij} ||x_i - x_j||) · Σ_{i≠j} (||x_i - x_j|| - ||y_i - y_j||)^2 / ||x_i - x_j||.
```

**Gap:** the weight `1/||x_i - x_j||` makes the cost of a small pairwise distance depend on
small *differences* in that distance — a tiny modeling error on two points that are extremely
close together produces an enormous contribution, so the method spends its effort on the most
fragile distances rather than assigning roughly equal importance across the local structure;
it is still a distance-matching method and still crowds.

**Isomap (Tenenbaum et al., 2000).** Classical scaling on geodesic distances along a `k`-NN
graph, so curvature of the manifold is followed. **Gap:** a single noisy edge "short-circuits"
the graph and corrupts many geodesics; it focuses on large geodesic distances rather than
local ones; and it needs a connected neighborhood graph, so widely separated submanifolds
cannot be embedded together.

**LLE (Roweis & Saul, 2000) and Laplacian Eigenmaps (Belkin & Niyogi, 2002).** Embed by
reconstructing each point from its neighbors / by the smallest eigenvectors of the graph
Laplacian. **Gap:** the only thing preventing all map points from collapsing to a single point
is a constraint on the covariance of the embedding, and that constraint is cheaply satisfied by
a "curdled" map — most points piled near the center, a few placed far out to supply the
variance — which carries little real structure; such methods also cannot display two or more
widely separated submanifolds, since the data does not give a connected graph.

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

**Gaps:** (1) That cost function is hard to optimize. The conditional normalizations make the
gradient complicated, and in practice the optimization needs simulated annealing — Gaussian
noise added to the map points each iteration with a slowly decaying variance, plus carefully
chosen momentum and step-size schedules — and even then it is common to rerun the whole
optimization several times to find good parameters. (2) It crowds. Using a Gaussian to convert
map-space distances into probabilities, exactly as in the high-dimensional space, leaves no
way to make room for the many moderately-distant points that a higher-dimensional manifold
produces, so the clusters are pulled together and gaps do not form.

**UNI-SNE (Cook et al., 2007).** An attempt to fight crowding by adding a slight uniform
repulsion: mix a small uniform background of weight `ρ` into the low-dimensional model, so that
`q_ij` can never fall below `2ρ / (n(n-1))` over the `n(n-1)/2` pairs, giving a mild repulsion
between far-apart points. **Gap:** optimizing this directly fails, because two far-apart map
points get almost all of their `q_ij` from the uniform floor, so a small change in their
separation has a vanishingly small *proportional* effect on `q_ij` and hence exerts no
restoring force — once two parts of a cluster separate early in the optimization, nothing pulls
them back. It only works at all by first optimizing standard SNE with annealing, then raising
`ρ` — so it inherits SNE's tedious optimization on top.

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
