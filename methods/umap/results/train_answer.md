We have $N$ points $X = \{x_1, \dots, x_N\}$ in a high-dimensional ambient space $\mathbb{R}^n$ — image pixels, gene-expression vectors, word embeddings — and we believe they do not fill that space but lie on or near some unknown curved low-dimensional manifold $M$. We want a map into $\mathbb{R}^d$ (usually $d = 2$ for a picture, sometimes a handful of dimensions as features) that keeps the shape of $M$ recognizable: who is near whom locally, and roughly how the clusters are arranged globally. The demands are concrete — preserve local neighborhoods, preserve enough global structure that cluster arrangement is meaningful, scale to hundreds of thousands or millions of points, impose no restriction on the embedding dimension, and rest each design choice on a principled argument rather than on tuning, because the problem is unsupervised and there is no held-out accuracy to anchor on. PCA is the cheap first reach and the wrong tool by construction: being linear, it can only find linear subspaces, so it flattens any fold of $M$ and places points on opposite sides of a curl right next to each other — it throws away exactly the nonlinearity that is the whole point. MDS, Sammon mapping, and Isomap honor the full matrix of pairwise distances and so spend their representational budget on large distances, crowding out local detail, and the full-matrix optimization scales poorly past tens of thousands of points. Laplacian Eigenmaps is beautifully principled — its embedding coordinates are the smooth functions on the manifold's approximated Laplacian — but its convergence to the Laplace–Beltrami operator requires uniformly sampled data, which real data violates, and its single eigensolve is rigid. t-SNE has superb local structure, but it normalizes both the input and output affinities over all pairs into probability distributions and minimizes $\mathrm{KL}(P\,\|\,Q)$; the output partition function $\sum_{k \ne l} w_{kl}$ couples every pair, making the gradient $O(N^2)$ (tractable only with Barnes–Hut trees, and tied to two or three dimensions), and the KL is lopsided — it punishes putting originally-near points far apart but barely penalizes putting originally-far points near each other, so its clusters are locally crisp yet float arbitrarily at the global scale. LargeVis fixes the scaling by dropping the output-side normalization and optimizing $\sum p_{ij} \log w_{ij} + \gamma \sum \log(1 - w_{ij})$ by edge plus negative sampling, but the objective is assembled by hand: it pairs an input affinity with a raw output affinity, and the attraction/repulsion balance is a free knob $\gamma$ with no principled value. We want the scaling of LargeVis and a cost whose attraction/repulsion balance is fixed by the comparison itself.

I propose UMAP — Uniform Manifold Approximation and Projection — which represents the data as a weighted fuzzy neighbor graph and finds a low-dimensional layout by minimizing the cross-entropy between the high- and low-dimensional fuzzy graphs with stochastic gradient descent over edge and negative samples. The justification starts in topology: covering the data with a ball around each point and forming the Čech complex (or its cheaper Vietoris–Rips truncation to points and edges, i.e. a graph) yields, by the Nerve theorem, an object homotopy-equivalent to the union of the cover — so "build a neighbor graph and lay it out" is secretly "approximate the topology of $M$ by a cover and embed the resulting complex." But the moment we try to pick the ball radius we hit a wall: too small and the cover shatters into disconnected components, too large and every ball overlaps and the topology is buried. There is a sweet spot only if the data is uniformly sampled — exactly the assumption Laplacian Eigenmaps also needs, and exactly what real, clumping-and-thinning data violates. The escape is to turn the problem on its head: we cannot make the data uniform, so we *assume* it is uniform and ask what that forces the geometry to be. If points look denser here and sparser there yet are uniform, then distance itself must vary across $M$ — a Riemannian metric not inherited from the ambient space. Making this precise hands us the edge weights. Under a locally constant diagonal metric $g$, the volume of an ambient-radius-$r$ ball is the Euclidean volume scaled by $\sqrt{\det g}$; demanding every such ball enclose the same fixed number of points and fixing that volume to the unit ball's gives $\sqrt{\det g}\, r^n = 1$, so $\det g = 1/r^{2n}$. With only a scalar neighbor distance available, the isotropic choice $g_{ij} = (1/r^2)\,\delta_{ij}$ has exactly that determinant, and under it the geodesic distance from $p$ to a nearby $q$ is $\tfrac{1}{r}\, d_{\mathbb{R}^n}(p, q)$. So the whole construction collapses to normalizing each point's distances by the scale of its own $k$-neighborhood, with $r$ the distance to the $k$-th neighbor — and the standard $k$-NN graph hack falls out as choosing unit balls in the per-point metric. Now $k$ is the interpretable resolution at which $M$ is approximated as flat, a far more natural knob than an absolute radius.

To keep the metric information we worked for, membership is fuzzy rather than binary, decaying exponentially in the local-metric distance with a per-point normalizer $\sigma_i$ playing the role of $r$, set so the total fuzzy membership out of $i$ hits a fixed target — a one-dimensional monotone equation solved by binary search. The plain exponential still has a high-dimensional failure: distances concentrate, so $\exp(-d_{ij}/\sigma_i)$ can be uniformly tiny and many points become effectively isolated, which should not happen if $M$ is locally connected. The cure is to measure the kernel from each point's nearest-neighbor distance $\rho_i$ outward. With $\rho_i$ the distance to $i$'s nearest nonzero-distance neighbor, the directed membership is
$$v_{j\mid i} = \exp\!\Big(\!-\frac{\max(0,\, d(x_i, x_j) - \rho_i)}{\sigma_i}\Big),$$
which makes the nearest neighbor sit at $d = \rho_i$ with weight exactly $1$ (guaranteed connection) and reads the *excess* over $\rho_i$ — precisely the relative spacing that survives in high dimensions, so the same offset cures both isolation and concentration. The calibration target with the offset is $\sum_{j=1}^k \exp(-\max(0, d_{ij} - \rho_i)/\sigma_i) = \log_2(k)\cdot\text{bandwidth}$; $\log_2(k)$ is the empirically chosen fuzzy cardinality, small enough that the tail does not dominate and growing slowly with $k$. These weights are directed, and the two local opinions $v_{j\mid i}$ and $v_{i\mid j}$ disagree because $i$ and $j$ have different $\rho$ and $\sigma$. Reading each weight as the probability that an endpoint believes the edge exists, the principled reconciliation is the probabilistic union — the edge exists if at least one endpoint vouches for it:
$$v_{ij} = v_{j\mid i} + v_{i\mid j} - v_{j\mid i}\, v_{i\mid j}, \qquad B = A + A^\top - A \circ A^\top,$$
which is the fuzzy-set t-conorm, not a heuristic mean, and which softly pulls reverse-neighbors into the graph.

For the layout we build the same kind of fuzzy structure over the embedded points $Y = \{y_i\} \subset \mathbb{R}^d$ and make it match. Two simplifications come for free: the target manifold is plain $\mathbb{R}^d$ with one global metric, so there is no per-point metric to estimate, and the connectivity offset $\rho$ — computed from data we no longer have — is promoted to a hyperparameter $\text{min\_dist}$, the distance below which points may be fully together. The match is measured by reading both graphs as vectors of Bernoulli edge-existence probabilities over the same reference edge set: $v(e)$ high-dimensional, $w(e)$ low-dimensional. The right divergence is then not KL-between-distributions (the weights do not sum to one) but the edgewise cross-entropy
$$C = \sum_{e} \Big[\, v(e)\,\log\frac{v(e)}{w(e)} + (1 - v(e))\,\log\frac{1 - v(e)}{1 - w(e)} \,\Big].$$
The asymmetry that crippled t-SNE is gone. The first term is large when $v$ is high but $w$ is low — an edge that should exist but does not in the layout — so it *attracts*, pulling points together; the second is large when $v$ is low but $w$ is high — an edge that should not exist but does — so it *repels*, and it is present and weighted by $(1 - v)$ for *every* pair, including the originally-far ones, which is exactly the global-structure pressure t-SNE lacked and the term LargeVis's $\gamma$ stands in for. Expanding, the $v$-only bracket $\sum_e [v \log v + (1-v)\log(1-v)]$ depends only on the fixed input graph and drops as a constant, leaving the normalization-free objective $-\sum_e [\, v \log w + (1-v)\log(1-w)\,]$ — $w$ enters only through $\log w$ and $\log(1-w)$, never a partition function — so the gradient decomposes edge by edge and plain SGD works: edge sampling for the attractive sum, negative sampling for the all-pairs repulsive sum.

The low-dimensional membership must be smooth and differentiable. We take
$$\Phi(y_i, y_j) = \big(1 + a\,\|y_i - y_j\|^{2b}\big)^{-1},$$
where $a = b = 1$ recovers exactly t-SNE's Student-t $(1 + d^2)^{-1}$, so the family *generalizes* the heavy-tailed kernel (and its crowding-problem fix) while gaining freedom to match $\text{min\_dist}$. We fit $a, b$ by nonlinear least squares to the offset-exponential target $\Psi(d) = 1$ for $d \le \text{min\_dist}$ and $\exp(-(d - \text{min\_dist})/\text{spread})$ beyond — the low-dimensional analogue of the input kernel, with $\text{min\_dist}$ in the role of $\rho$; defaults $\text{min\_dist} = 0.1$, $\text{spread} = 1$ give $a \approx 1.929$, $b \approx 0.7915$. The forces are the algorithm, so I derive them exactly. Writing $D = \|y_i - y_j\|^2$, $w = (1 + a D^b)^{-1}$, and $dD/dy_i = 2(y_i - y_j)$, the attractive gradient of $v \log w$ is
$$\nabla_{y_i}(v \log w) = v \cdot \frac{-2 a b\, D^{b-1}}{1 + a D^b}\,(y_i - y_j),$$
a pull stronger for heavier edges, and the repulsive gradient of $(1 - v)\log(1 - w)$, using $\tfrac{d}{dD}\log(1-w) = b/(D(1 + a D^b))$, is
$$\nabla_{y_i}\big((1-v)\log(1-w)\big) = (1 - v)\cdot \frac{2 b}{D\,(1 + a D^b)}\,(y_i - y_j),$$
a push. As $D \to 0$ the repulsive $1/D$ blows up, so we floor the denominator with $\varepsilon = 0.001$ to give $2b/((\varepsilon + D)(1 + a D^b))$, and we clip each gradient coordinate to $[-4, 4]$ against the occasional large step. The optimizer samples edge $e$ with frequency proportional to its weight $v(e)$ (an edge is due every $\text{max\_v}/v(e)$ epochs), applies the attractive update to both endpoints, then for the pulled vertex draws $n_{\text{neg}} = 5$ random vertices as non-edges and applies the repulsive update, paced by a parallel counter so repulsion stays proportional to attraction. The learning rate anneals linearly $\alpha = 1 - \text{epoch}/n_{\text{epochs}}$ so the non-convex layout cools into a local minimum; $n_{\text{epochs}}$ is $500$ for $N \le 10000$ else $200$, and total work is $O(kN)$. Finally, the same fuzzy graph $G$ serves twice: because its symmetric normalized Laplacian $L = I - D^{-1/2} A D^{-1/2}$ approximates the Laplace–Beltrami operator of $M$ under the usual sampling assumptions, its bottom nontrivial eigenvectors give a globally coherent set of manifold coordinates that initialize the layout, so the local cross-entropy forces refine a coherent global scaffold rather than inventing one from random coordinates.

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
