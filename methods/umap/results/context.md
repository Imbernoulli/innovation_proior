## Research question

We have a dataset `X = {x_1, ..., x_N}` of `N` points living in a high-dimensional ambient
space `R^n` (pixels of images, gene-expression vectors, word embeddings), and we want a map
into a low-dimensional space `R^d` — usually `d = 2` for a picture, but sometimes a handful of
dimensions as features for downstream learning — that keeps the structure of the data
recognizable. The working hypothesis behind nearly all of this is the *manifold hypothesis*:
the data does not fill `R^n` but lies on (or near) some unknown low-dimensional manifold `M`
embedded in it, and "structure" means the shape of `M` — which points are neighbors, how the
neighborhoods knit together into a global form, where the clusters and the holes are.

A linear projection (PCA) is fast and global but can only ever see linear subspaces; it
flattens a curved manifold and throws away exactly the nonlinear structure we care about. So
the goal is a *nonlinear* embedding. The precise demands are: (1) preserve **local**
neighborhood structure — points close on `M` should stay close in `R^d`; (2) preserve enough
**global** structure that the relative arrangement of clusters is meaningful, not just blobs in
random positions; (3) **scale** to large `N` (hundreds of thousands to millions of points) and
to high ambient `n`, in time and memory that a workstation can handle; (4) impose **no
restriction on the embedding dimension** `d`, so the same method works for visualization and
for feature extraction; and (5) ideally rest each design choice on a **principled
justification** rather than on having been tuned to make one benchmark look good — unsupervised
problems have no held-out accuracy to appeal to, so the only durable arguments are theoretical
ones. Every existing method meets some of these and fails others; closing the gap is the
problem.

## Background

The dominant lens is the **neighbor graph**. Almost every nonlinear method first turns the
point cloud into a weighted graph whose nodes are the data points and whose edges connect
nearby points with weights that decay with distance, and then lays that graph out in `R^d`.
The differences between methods are differences in (a) how the graph is built and weighted and
(b) what objective the layout optimizes.

A few load-bearing ideas underpin this.

**Distances become similarities through a kernel.** A raw distance `d(x_i, x_j)` is turned
into an affinity by an exponential kernel `exp(-d^2 / 2 sigma^2)` or similar, with a bandwidth
`sigma` that sets the scale of "near". Choosing `sigma` is delicate: one global bandwidth is
wrong because density varies across the data — a radius that connects a dense region sensibly
leaves a sparse region disconnected, and vice versa.

**The curse of dimensionality distorts neighborhoods.** In high `n`, pairwise distances
concentrate: the distance to the nearest neighbor and to the tenth nearest neighbor become
nearly equal, and absolute distances carry little information. Methods that threshold on an
absolute radius therefore behave badly in high dimensions; what remains informative is the
*ordering* and the *relative* spacing of a point's nearest neighbors, not the absolute
distances.

**The graph Laplacian approximates a differential operator.** For a weighted graph with
adjacency `W` and diagonal degree `D`, the (unnormalized) Laplacian is `L = D - W` and the
symmetric normalized Laplacian is `L_sym = D^{-1/2}(D - W) D^{-1/2}`. A classical result is
that, for a graph built from points sampled on a manifold, `L` converges (in the limit of
infinite, **uniformly sampled** data) to the Laplace–Beltrami operator of `M`. Its low-lying
eigenvectors are the smoothest functions on the manifold and give a natural set of global
coordinates. This is the basis of spectral embedding.

**Uniform sampling is the assumption everything leans on.** The clean theory — both the
spectral-convergence result and the guarantee that a cover of balls actually captures the
topology — holds when the data is uniformly distributed on `M`. Real data is not: it clumps and
thins, so a fixed-radius or fixed-bandwidth construction over-covers the dense regions and
under-covers the sparse ones. The diagnostic failure is concrete and reproducible: build balls
of a single radius around sampled points of a manifold, and a small radius shatters the graph
into many disconnected components while a large radius fuses everything into one blob — there is
no single radius that works across a non-uniform sample.

**Topological data analysis** offers a language for "structure" beyond pairwise distance. Given
an open cover of a space (e.g. a ball around each point), the Čech complex (a simplex per
collection of sets with common intersection) — or its cheaper cousin the Vietoris–Rips complex,
determined by its points and edges alone — produces a combinatorial object, and the **Nerve
theorem** guarantees it is homotopy-equivalent to the union of the cover. Restricting attention
to 0- and 1-simplices makes the object a graph again, which is why neighbor-graph layout can be
read as approximate topology. **Fuzzy sets** (Zadeh 1965), where membership is a value in
`[0,1]` rather than yes/no, and their category-theoretic treatment as sheaves on the unit
interval (Barr 1986), let edge weights be read as membership strengths; Spivak's *Metric
realization of fuzzy simplicial sets* (2012) constructs adjoint functors translating between
finite metric spaces and fuzzy simplicial sets, so that "how strongly is `x_j` in the
neighborhood of `x_i`" has a precise meaning and metric spaces can be converted into fuzzy
combinatorial objects (and back) in a canonical way.

**Scalability tools exist.** Exact `k`-nearest-neighbor search is `O(N^2)`, but approximate
methods make it cheap: Nearest-Neighbor-Descent (Dong, Moses & Li 2011) builds an approximate
kNN graph for *any* dissimilarity measure with empirical complexity around `O(N^1.14)`. On the
optimization side, the **negative-sampling** trick from word2vec (Mikolov et al. 2013) replaces
an intractable sum over all non-pairs with a handful of randomly sampled "negative" pairs per
update, turning an `O(N^2)` objective into something an SGD step can touch in `O(1)`.

## Baselines

**Principal Component Analysis (Hotelling 1933).** Project onto the top-`d` eigenvectors of
the data covariance — the directions of greatest variance. Linear, fast, deterministic, and
its axes are interpretable. *Limitation:* being linear, it cannot represent a curved manifold;
points far apart along a fold of `M` are placed close in the projection, so nonlinear
neighborhood structure is lost.

**Multidimensional scaling / Sammon mapping / Isomap (Kruskal 1964; Sammon 1969; Tenenbaum
et al. 2000).** Preserve the full matrix of pairwise distances (Isomap uses *geodesic*
distances along a neighbor graph). Strong at global metric fidelity. *Limitation:* they spend
their representational budget trying to honor large distances, which crowds out local detail;
and computing/optimizing over the full distance matrix is expensive, scaling poorly past tens
of thousands of points (Isomap failed to complete on datasets larger than a few thousand
images in practice).

**Laplacian Eigenmaps (Belkin & Niyogi 2003).** Build a weighted neighbor graph with affinity
`w_ij` and find the embedding minimizing `sum_{ij} w_ij ||y_i - y_j||^2` subject to a scale
constraint `Y^T D Y = I`. The solution is the bottom nontrivial generalized eigenvectors of
`L y = lambda D y`. Beautifully principled — the embedding coordinates are the smooth functions
on the approximated manifold — and a single eigendecomposition, no iterative tuning.
*Limitations:* the correctness argument requires the data to be uniformly sampled on `M` (and
even then holds only in the infinite-data limit), which real data violates; a single global
quadratic objective tends to compress the embedding and is rigid; and the eigensolve is a fixed
output with no easy control over how tightly local clusters are packed for viewing.

**t-SNE (van der Maaten & Hinton 2008; Barnes–Hut acceleration van der Maaten 2014).** The
state of the art for visualization. In the input space it builds per-point Gaussian affinities
`v_{j|i} = exp(-||x_i - x_j||^2 / 2 sigma_i^2)`, with each `sigma_i` calibrated so the
distribution `p_{.|i}` has a user-set *perplexity*; these are row-normalized to conditionals
`p_{j|i} = v_{j|i} / sum_{k != i} v_{k|i}`, then symmetrized and normalized over the whole
matrix to a joint distribution `p_ij = (p_{j|i} + p_{i|j}) / (2N)`. In the embedding it uses a
heavy-tailed Student-t affinity `w_ij = (1 + ||y_i - y_j||^2)^{-1}`, again normalized over all
pairs to `q_ij = w_ij / sum_{k != l} w_kl`, and minimizes the Kullback–Leibler divergence
`C = sum_{i != j} p_ij log(p_ij / q_ij)`. The heavy tail deliberately fixes the *crowding
problem* — there is not enough room in low `d` to honor all distances, so the t-distribution
lets moderate input distances map to large output distances. Local structure comes out
excellent. *Limitations:* both `p_ij` and especially `q_ij` are normalized over **all pairs**,
so `q_ij` carries a partition function `sum_{k != l} w_kl` that couples every pair of points;
the gradient depends on this global sum, which is `O(N^2)` per step and only made tractable by
Barnes–Hut / tree approximations, capping practical `N`. Empirically t-SNE preserves *local*
neighborhoods far better than the *global* arrangement of clusters — the relative positions of
well-separated clusters are largely arbitrary. Its common accelerated implementations rely on
space trees whose cost becomes unattractive as embedding dimension grows, so the method is
practically tied to two- or three-dimensional visualization rather than arbitrary-dimensional
feature extraction. The KL cost is also asymmetric in a telling way: it penalizes placing
originally-near points far apart (a large `p_ij` with small `q_ij`) heavily, but barely penalizes
placing originally-far points near each other.

**LargeVis (Tang, Liu, Zhang & Mei 2016).** Keeps t-SNE's input affinities (using *approximate*
kNN for scalability) but makes a crucial change in the embedding: it **abandons the matrix-wide
normalization** of the output weights. Instead of a normalized `q_ij` it works with the
unnormalized `w_ij = (1 + ||y_i - y_j||^2)^{-1}` directly and maximizes a likelihood
`C_LV = sum_{i != j} p_ij log w_ij + gamma sum_{i != j} log(1 - w_ij)`, where the first term
pulls together pairs with large input affinity and the second pushes apart all pairs, weighted
by a positive constant `gamma`. Because there is no partition function, this is optimizable by
plain SGD with edge sampling (sample an edge with probability proportional to its weight) and
negative sampling (approximate the second, all-pairs sum by a few random pairs) — yielding
roughly linear-time training that scales to millions of points. *Limitations:* the objective is
assembled somewhat heuristically — the first term uses the raw `w_ij` rather than a quantity
matched to the input affinities, and the relative strength of attraction and repulsion is left
to a free hyperparameter `gamma` with no principled value; the input affinities still inherit
t-SNE's perplexity calibration and Gaussian/uniform-sampling assumptions.

## Evaluation settings

The natural yardsticks already in use for dimension reduction:

- **Datasets.** PenDigits (1797 8×8 digit images, treated as 64-dim vectors); COIL-20 / COIL-100
  (objects under rotation, very high ambient dimension); MNIST (70000 28×28 = 784-dim
  handwritten digits) and Fashion-MNIST (70000 784-dim clothing images); single-cell RNA-seq
  and flow-cytometry datasets (tens of thousands to a million points); the GoogleNews word2vec
  vectors (3 million 300-dim word embeddings, compared under cosine distance). A spread from
  small/clustered to massive/continuous, so that both local fidelity and scaling are exercised.
- **Distance.** Euclidean distance on the raw vectors for image and tabular data; cosine /
  angular distance for word vectors.
- **Quality metrics.** *Trustworthiness* — the fraction of a point's neighbors in the embedding
  that were also neighbors in the input (penalizes false neighbors introduced by the
  embedding). *Continuity* — the fraction of a point's input neighbors that remain neighbors in
  the embedding (penalizes true neighbors torn apart). A `k`-NN classifier accuracy computed in
  the embedding, as a proxy for how well class/cluster structure survives. These are computed
  at a fixed small neighbor count `k` (e.g. 7). For global structure one can also compare the
  embedding to a reference (e.g. PCA) up to rotation/scale by Procrustes alignment.
- **Performance.** Wall-clock time and memory as functions of `N`, of ambient dimension `n`,
  and of embedding dimension `d`; ability to complete at all on the largest datasets.
- **Protocol.** Reproducibility under a fixed random seed (the methods are stochastic — both
  the approximate-kNN step and the SGD); stability of the embedding under subsampling of the
  data.

## Code framework

The substrate is the generic two-phase neighbor-graph pipeline that every method in this class
shares: build a weighted neighbor graph, then optimize a low-dimensional layout of it. The data
loader, the (approximate) nearest-neighbor routine, the sparse-graph container, the spectral
solver used for initialization, and the stochastic-gradient outer loop already exist. What is
not settled is what information the graph should carry and what update rule should improve the
layout. Those are left as neutral empty slots below.

```python
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(X, k):
    """Existing (approximate) kNN: returns indices and distances of the k nearest
    neighbors of every point. Treated as a black box."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    knn_dists, knn_indices = nn.kneighbors(X)
    return knn_indices, knn_dists


def build_graph(knn_indices, knn_dists, N):
    """Turn the kNN distances into a weighted graph."""
    rows, cols, vals = [], [], []
    for i in range(N):
        for j_idx in range(knn_indices.shape[1]):
            j = knn_indices[i, j_idx]
            d = knn_dists[i, j_idx]
            # TODO: the graph relation we will design
            w = 0.0  # placeholder
            rows.append(i); cols.append(j); vals.append(w)
    raw_graph = coo_matrix((vals, (rows, cols)), shape=(N, N))
    # TODO: turn the relation records into the graph the layout will use
    graph = raw_graph  # placeholder
    return graph


def spectral_init(graph, d):
    """Existing spectral embedding: bottom eigenvectors of the graph's normalized
    Laplacian, used as a starting layout."""
    ...


def optimize_layout(graph, Y, n_epochs, n_neg_samples):
    """Stochastic-gradient layout. The harness that samples edges and applies
    per-coordinate updates already exists; the objective being descended does not."""
    alpha = 1.0
    for epoch in range(n_epochs):
        for (i, j, w) in sample_edges(graph):
            # TODO: the update rule we will design
            pass
        alpha = 1.0 - epoch / n_epochs       # learning-rate decay
    return Y


def fit_transform(X, k, d, n_epochs, n_neg_samples):
    N = X.shape[0]
    knn_indices, knn_dists = nearest_neighbors(X, k)
    graph = build_graph(knn_indices, knn_dists, N)
    Y = spectral_init(graph, d)
    Y = optimize_layout(graph, Y, n_epochs, n_neg_samples)
    return Y
```

The pipeline supplies neighbors, a sparse graph, a spectral starting point, and an SGD loop; the
graph relation and the layout update are the slots to fill.
