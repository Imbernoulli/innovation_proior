## Research question

Given `n` points in high dimension, `{x_i in R^D}`, produce a low-dimensional embedding
`{y_i in R^d}` with `d = 2` or `3` that a human can look at and trust. "Trust" has two
faces that pull against each other. The first is **local structure**: points that are
neighbors in `R^D` should stay neighbors in the map, so the fine-grained cluster membership
of each point is faithful. The second is **global structure**: the overall shape of the data,
the relative placement of the clusters with respect to one another, the macroscopic geometry
of a curved manifold, and the existence of far-flung outliers. A genuinely useful map has to
get both right at once: it must show which points share a fine-grained neighborhood and how
larger groups, manifolds, and outliers sit with respect to one another.

The pain point is that the methods in wide use, and the metrics used to judge them, have
quietly optimized only the first face. They preserve neighborhoods beautifully and scramble
the global layout, and because the standard scores are themselves local, the scrambling goes
unmeasured. So the problem is twofold: build an embedding method that preserves global
structure as well as a linear projection does while keeping competitive local fidelity and
scaling to millions of points; and build a quantitative score that actually *measures* global
accuracy, since none of the established scores can. It also has to run on commodity hardware
in minutes, without forming any `n x n` matrix.

## Background

The earliest dimensionality-reduction tool is **PCA** (Pearson, 1901): project onto the top-`d`
orthogonal directions of highest variance. PCA only ever uses the aggregate second-order
statistics of the data, never the local neighborhood of any individual point. That is exactly
why it is excellent at the global picture — the overall shape, the placement of clusters, the
outliers — and exactly why it is poor at local structure: a flat linear projection cannot
unfold a curved manifold. Silva & Tenenbaum (2003) drew this global-versus-local distinction
explicitly. PCA carries one more property that turns out to be load-bearing: among all
embeddings it admits the most accurate *linear inverse map* back to high dimension. Given any
low-D embedding `Y`, the best linear reconstruction of `X` is a ridgeless least-squares fit,
`min_A ||X - A Y||_F^2`, solved in closed form by `A* = X Y^T (Y Y^T)^{-1}` (which absorbs any
rotation/scaling of `Y`); PCA achieves the smallest such reconstruction error of any DR method.

The dominant *nonlinear* methods of the time all share a common shape: build a `k`-nearest-
neighbor graph in high dimension, turn edges into target (dis)similarities, and lay the points
out in 2D to match.

- **t-SNE** (van der Maaten & Hinton, 2008) converts high-D distances into a probability
  distribution `P` with a per-point Gaussian (bandwidth set by a perplexity target) and
  matches it to a low-D distribution `Q` by minimizing `KL(P||Q)`. Its decisive design choice
  is the low-D kernel: a Student-t distribution with one degree of freedom,
  `q_ij ∝ (1 + ||y_i - y_j||^2)^{-1}`. The heavy tail solves the *crowding problem* — in a
  Gaussian-vs-Gaussian match, the sheer number of moderately-distant pairs exerts a small
  attraction that crushes everything to the center; the heavy tail lets a moderate high-D
  distance map faithfully to a much larger low-D distance, and it creates long-range forces
  that can pull separated clusters back together.

- **LargeVis** (Tang et al., 2016) keeps the kNN-graph idea but replaces the `O(n^2)` pairwise
  objective with a sampled, negative-sampling objective on graph edges, scaling to large `n`.

- **UMAP** (McInnes et al., 2018) builds a fuzzy simplicial set from the kNN graph (a
  locally-adaptive fuzzy membership per edge) and minimizes a cross-entropy between the high-
  and low-D fuzzy graphs, with attractive forces on edges and repulsive forces from negative
  sampling. Defaults `n_neighbors = 15`, `min_dist = 0.1`.

A separate line learns embeddings from **ordinal / triplet** information of the form
`(i, j, k)`: "point `i` is closer to `j` than to `k`."

- **Stochastic Triplet Embedding and its heavy-tailed variant** (van der Maaten &
  Weinberger, 2012) define, for each triplet, a probability that it is satisfied,
  `p_ijℓ = exp(-||x_i - x_j||^2) / (exp(-||x_i - x_j||^2) + exp(-||x_i - x_ℓ||^2))` (STE), or
  the same ratio with a Student-t kernel of `α` degrees of freedom (t-STE), and maximize the
  sum of log-satisfaction-probabilities `max Σ log p_ijℓ` over the given triplets. The
  heavy-tailed kernel deliberately keeps *collapsing* similar points and *repelling*
  dissimilar points even after a triplet is already satisfied, because its tails are not flat.

- **Semi-supervised kernel metric learning from relative comparisons** (Amid, Gionis &
  Ukkonen, 2016) starts from a feature-derived kernel `K_0` and learns a kernel that stays
  close to `K_0` (by a log-det Bregman divergence) while satisfying a set of relative-distance
  constraints "of `{i, j, k}`, which is the outlier." The governing idea: an *initial*
  representation is *refined* by relative comparisons rather than built from them alone.

Two further pieces of standard machinery are on the shelf. **Self-tuning local scaling**
(Zelnik-Manor & Perona, 2005) replaces a single global bandwidth by a per-point one,
`d^2(s_i, s_j) / (σ_i σ_j)` with `σ_i` the distance from `s_i` to its `K`-th neighbor, so that
affinities are normalized to local density and a tight cluster sitting inside a sparse one is
still resolved. The **tempered (deformed) logarithm** (Naudts, 2002),
`log_t(u) = (u^{1-t} - 1)/(1 - t)` for `t ≠ 1`, with `log_t → log` as `t → 1`, is a
concave-saturating transform that compresses large values relative to small ones. And the
**delta-bar-delta** rule (a per-coordinate adaptive learning-rate gain that grows while the
gradient sign opposes the current velocity direction and shrinks when they have the same
sign), combined with a momentum schedule, is the optimizer already used to train t-SNE.

A motivating diagnostic ties these together. On the classic 3D **S-curve** manifold, and on
the activations of a hidden layer of an image classifier, the local-focused methods recover
the neighborhoods but visibly *distort the global layout*: t-SNE and UMAP get high values of a
local score yet fail to unveil the curved shape of the S-curve or the macroscopic hierarchy of
super-clusters, whereas the linear PCA projection — poor locally — recovers the global shape.
This phenomenon is the heart of the matter: local accuracy and global accuracy are genuinely
different, and the standard methods buy the first by sacrificing the second.

## Baselines

These are the prior methods a new visualization method is measured against and reacts to.

**PCA** (Pearson, 1901). Project onto the top-`d` eigendirections of the data covariance; the
optimal linear DR by variance preserved and by linear reconstruction error. *Limitation:* a
single linear projection cannot unfold nonlinear manifold structure, so neighborhoods that are
close along a curved sheet but far in the ambient chord distance are torn apart; local fidelity
is weak.

**t-SNE** (van der Maaten & Hinton, 2008). Gaussian `P` (perplexity-tuned) vs. Student-t `Q`,
minimize `KL(P||Q)` by gradient descent with momentum and per-coordinate adaptive gains; the
heavy-tailed low-D kernel fixes crowding. *Limitation:* the objective is built from *pairwise*
neighbor probabilities and is dominated by the local neighborhood, so the relative placement
of separate clusters is essentially unconstrained — different runs and different perplexities
produce wildly different global arrangements of the same clusters; the naive cost is `O(n^2)`
(Barnes-Hut brings it to `O(n log n)`); and it is highly sensitive to initialization,
converging well only from a small random start near the origin.

**LargeVis** (Tang et al., 2016) and **UMAP** (McInnes et al., 2018). kNN-graph neighbor
embeddings with sampled attractive/repulsive objectives that scale to large `n` and run fast.
*Limitation:* the forces still act essentially between neighbors (attraction on graph edges)
and randomly-sampled negatives (repulsion); the objective can be driven near zero — all
neighbors close, all sampled non-neighbors apart — while the macroscopic arrangement of the
clusters remains arbitrary. The global layout is not what these losses constrain.

**STE / t-STE** (van der Maaten & Weinberger, 2012). Triplet embedding by maximizing
`Σ log p_ijℓ` over a *given* set of triplets, with a heavy-tailed kernel in `p`.
*Limitation:* (a) the triplets are assumed supplied (human similarity judgements); there is no
mechanism to *sample* informative triplets from a feature representation, nor any notion that
some triplets carry more evidence than others — every triplet counts equally. (b) Maximizing
log-satisfaction-probability keeps applying force to triplets that are *already satisfied* (the
heavy tail never lets go), pulling satisfied near-pairs ever tighter and pushing satisfied
far-pairs ever farther — useful for denoising human labels, but it over-compresses when the
triplets come from real feature distances. (c) It is initialized from a small random
configuration, with no anchor to the data's global geometry.

**Semi-supervised metric learning from relative comparisons** (Amid, Gionis & Ukkonen, 2016).
Refine an initial feature-kernel `K_0` to satisfy relative-distance constraints while staying
close to `K_0`. *Limitation:* it targets a *metric/kernel* for downstream clustering, learned
from a modest set of human-provided comparisons in a kernel-learning (PSD-constrained)
formulation; it is not a large-scale 2D-visualization method and does not address sampling
constraints automatically from high-dimensional features or scaling to millions of points.

The common thread across the nonlinear baselines: their objective, when minimized, pins down
the *local* neighborhoods but leaves the *relative arrangement of clusters* — the thing PCA
gets for free — underdetermined.

## Evaluation settings

The natural pre-existing yardsticks. Datasets used for DR visualization: **MNIST** (70k,
784-dim handwritten digit images), **Fashion-MNIST** (70k, 784-dim clothing images), **USPS**
(16x16 digits), **COIL-20** (object images at rotations), **20 Newsgroups** (text, TF-IDF then
reduced), **Epileptic Seizure** (EEG), single-cell **Tabula Muris**, and progressively larger
sets up to millions of points (Covertype, RCV1, Character Font Images, KDDCup99, HIGGS) to
stress scaling. A standard synthetic probe is the 3D **S-curve** manifold (5000 points
uniformly sampled from an S-shaped sheet), whose true global shape is known. Hidden-layer
activations of a small CNN on CIFAR-10 are used to inspect class separation.

Local-accuracy metrics already in use: **nearest-neighbor (kNN) accuracy** of a classifier
in the embedding; **trustworthiness and continuity** (Venna & Kaski, 2005), the fraction of
embedding neighbors that are also original neighbors and vice versa; and the AUC of a
precision-recall view of neighborhood retrieval (Venna et al., 2010). All of these score the
*local* neighborhood; there is no established score for *global* accuracy. As an external
sanity check on any proposed global score, one can `k`-means-cluster (e.g. `k = 100`) in high
and low dimension and compare the cluster-center distance matrices via the Mantel test
(yielding a Pearson correlation coefficient). Standard protocol: identical default parameters
across methods, reduce to ~100 dimensions with PCA before the neighbor search when the ambient
dimension is large, fixed iteration budget, single commodity machine with a wall-clock limit.

## Code framework

A generic neighbor-embedding harness already has a high-dimensional matrix `X`, optional
preprocessing before neighbor search, an approximate `k`-nearest-neighbor index
(random-projection trees / ANNOY) to get the graph cheaply, a place to turn the graph into
layout constraints, a starting low-D configuration, a global-accuracy scoring hook, and a
full-batch gradient-descent loop with momentum and per-coordinate adaptive gains.

```python
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors   # stand-in for an approx-NN index


def preprocess_for_neighbors(X, pca_dim=100):
    """Prepare the data used by the neighbor search."""
    if X.shape[1] > pca_dim:
        X = TruncatedSVD(n_components=pca_dim, random_state=0).fit_transform(
            X - X.mean(axis=0))
    else:
        X = X - X.min()
        X = X / max(X.max(), 1e-12)
        X = X - X.mean(axis=0)
    return X


def neighbor_graph(X, n_neighbors):
    """Build the kNN graph the layout will be derived from."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    knn_dist, knn_idx = nn.kneighbors(X)
    return knn_idx, knn_dist


def build_relations(X, knn_idx, knn_dist):
    """Turn the neighbor graph into the constraints the layout optimizes."""
    # TODO: the constraints and their weights we will design.
    pass


def init_embedding(X, n_components):
    """Produce the starting low-D configuration the optimizer refines."""
    # TODO: the initial configuration we will choose.
    pass


def embedding_loss_and_grad(Y, relations):
    """Objective on the embedding and its gradient wrt Y."""
    # TODO: the objective on Y and d(loss)/dY.
    pass


def global_accuracy_score(X, Y):
    """Score global structure using a linear reconstruction baseline."""
    # TODO: the score we will define.
    pass


def optimize(Y, relations, n_iters=400, lr=0.1):
    """Full-batch gradient descent with momentum + per-coordinate adaptive gain
    (delta-bar-delta), the same optimizer the neighbor-embedding methods use."""
    vel = np.zeros_like(Y)
    gain = np.ones_like(Y)
    for it in range(n_iters):
        gamma = 0.8 if it > 250 else 0.5                 # momentum schedule
        grad = embedding_loss_and_grad(Y + gamma * vel, relations)   # look-ahead grad
        flip = np.sign(vel) != np.sign(grad)
        gain = np.where(flip, gain + 0.2, np.maximum(gain * 0.8, 0.01))
        vel = gamma * vel - lr * gain * grad
        Y += vel
    return Y


def fit_transform(X, n_components=2, n_neighbors=10):
    X = preprocess_for_neighbors(X)
    knn_idx, knn_dist = neighbor_graph(X, n_neighbors)
    relations = build_relations(X, knn_idx, knn_dist)
    Y = init_embedding(X, n_components)
    return optimize(Y, relations)
```
