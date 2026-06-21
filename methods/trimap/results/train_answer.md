I want a 2D or 3D picture of high-dimensional data that a human can actually trust, and "trust" has two faces that pull against each other. One is local structure: points that are neighbors in $\mathbb{R}^D$ should stay neighbors in the map. The other is global structure: the overall shape of a curved manifold, the relative placement of clusters with respect to one another, the existence of far-flung outliers. A genuinely useful map gets both right at once. The trouble is that the methods in wide use, and the scores used to judge them, have quietly optimized only the first face. On the S-curve — a few thousand points on a smooth curled sheet in 3D whose true shape I know — t-SNE and UMAP keep the local neighborhoods but unroll the curl into something that no longer tells me which end of the S was near which; PCA, by contrast, gives an ugly flat shadow that is locally weak but globally honest. And the nearest-neighbor accuracy and the trustworthiness scores can both be high for those distorted maps, so by every local number I have the embedding looks "good." The pairwise neighbor-embedding methods all work the same way — build a kNN graph, turn each edge into a target distance or similarity, and lay the points out to match — and a pairwise target is *absolute*: it commits to a specific distance, and only neighbors get a meaningful target because the Gaussian similarity of two far-apart points is numerically zero, indistinguishable from any other far pair. So the objective says nothing about how far apart two clusters should sit relative to two others; the global arrangement is left dangling and the optimizer fills it in arbitrarily. The triplet-embedding methods (STE / t-STE) are relative rather than absolute, but they assume the triplets are supplied by humans, weight every triplet equally, and — because they maximize a heavy-tailed log-satisfaction-probability — keep applying force to triplets that are already satisfied, over-compressing real feature data; and they start from a small random configuration with no anchor to the data's geometry. So I need two things tangled together: a method that keeps the global layout the way PCA does while staying competitive locally and scaling to millions of points, and a number that actually measures global accuracy.

Take the measurement first, because I cannot claim global accuracy if I cannot measure it. PCA is the natural gold standard: among all $d$-dimensional embeddings it admits the most accurate linear inverse map back to high dimension. So I define global accuracy as closeness to what PCA preserves, operationalized as linear reconstruction. Given a centered embedding $Y$ and centered data $X$, the best linear reconstruction is the least-squares fit $\mathrm{err}(Y\mid X) = \min_A \|X - AY\|_F^2$; differentiating $\|X-AY\|_F^2$ in $A$ and setting it to zero gives $-2(X-AY)Y^\top = 0$, so $A^\star = X Y^\top (Y Y^\top)^{-1}$, which automatically absorbs any rotation, reflection, or scaling of $Y$ — exactly right, since a visualization is only defined up to those. By the variational characterization of PCA, $\mathrm{err}_{\mathrm{PCA}}$ is the smallest such error achievable by any $d$-dimensional embedding, a floor. I normalize against it:
$$\mathrm{GS}(Y\mid X) = \exp\!\Big(-\tfrac{\mathrm{err}(Y\mid X) - \mathrm{err}_{\mathrm{PCA}}}{\mathrm{err}_{\mathrm{PCA}}}\Big) \in (0,1],$$
which is $1$ when $Y$ is as globally faithful as PCA and decays as the layout degrades.

For the method itself — I propose TriMap. Its central design split is that the global structure comes from a PCA initialization and the triplets only sharpen the local structure on top of that frame, using a saturating per-triplet loss that stops applying force once a triplet is satisfied. The constraint unit is a triplet $(i,j,k)$ meaning "$i$ is closer to $j$ than to $k$." A triplet demands only an ordering, not a distance, which is scale-free and robust, and it is a higher-order summary: by letting $k$ range over far points, a triplet can speak about the placement of $i$ relative to distant structure. For the loss I want force only while the order is wrong or marginal, and zero once it is right. Using a Student-t low-D similarity $s(y_a,y_b) = (1 + \|y_a - y_b\|^2)^{-1}$ — the same heavy-tailed kernel that beats crowding in t-SNE and supplies long-range forces — the natural quantity to minimize is the share of similarity mass sitting on the wrong side,
$$l_{ijk} = w_{ijk}\,\frac{s(y_i,y_k)}{s(y_i,y_j) + s(y_i,y_k)} = \frac{w_{ijk}}{1 + d_{ik}/d_{ij}},\qquad d_{ij} := 1 + \|y_i - y_j\|^2.$$
The $+1$ does double duty: it is the Student-t shape and it floors the denominator so coinciding embedding points never divide by zero. When $j$ is near $i$ and $k$ is far, $d_{ik}\gg d_{ij}$ and the loss vanishes — it saturates to zero rather than continuing to reward an already-satisfied triplet, which is precisely where maximizing $\sum\log p$ goes wrong. This saturation is what makes refining-from-PCA safe later: most neighbor triplets start nearly satisfied, so their forces are gentle and will not chew up a good global frame.

The weight $w_{ijk}$ encodes that not all triplets are equally informative: a triplet whose $k$ is enormously farther than $j$ in high dimension is strong evidence, while a near-tie is noise. So the weight grows with the high-D margin $\tilde w_{ijk} = \delta_{ik}^2 - \delta_{ij}^2 \ge 0$. Raw squared distances are unfair across density regions, so I use self-tuning local scaling, $\delta_{ij}^2 = \|x_i - x_j\|^2 / (\sigma_i \sigma_j)$, with $\sigma_i$ the mean distance from $x_i$ to its 4th–6th nearest neighbors — far enough out to be stable, close enough to stay local, and not the jittery 1st neighbor. The margins are heavy-tailed, so a few cross-cluster giants would dominate the gradient; I keep the ranking but compress the dynamic range with a tempered logarithm $\log_t(u) = (u^{1-t}-1)/(1-t)$, which becomes ordinary $\log$ as $t\to1$ and saturates more gently for $t<1$. With $t=0.5$ (square-root-flavored compression) and a shift so the smallest margin is zero, $w_{ijk} = \log_t(1 + \tilde w_{ijk} - w_{\min})$, the $1+$ keeping the argument $\ge 1$ so weights stay nonnegative.

Which triplets? Not all $O(n^3)$. For each $i$ I take its $m=12$ nearest neighbors as candidate $j$'s and, per neighbor, sample $m'=4$ random outliers $k$ from outside the closer neighbors — $48$ near-neighbor triplets per point, linear in $n$, so the only real cost is the approximate nearest-neighbor search shared with every method in this family. But I have to check whether these neighbor triplets buy the global structure I am chasing, and they do not: every one has $j$ as a neighbor of $i$, so the optimizer can drive the loss to near zero while folding the global layout into almost any shape — relax the curled S-sheet flat and locally every neighbor is still nearest. The loss can be zero amid a complete sacrifice of global structure. Adding a few random far-far triplets ($r=3$ per point, oriented by high-D distance and downweighted by $0.1$) gives only a faint, noisy long-range signal that the 48 neighbor triplets would overwhelm. The real resolution is not to discover the global layout from forces at all but to start from one that is already globally correct and let the triplets only refine it: initialize $Y$ to the PCA embedding, scaled down by $0.01$ so the Student-t kernel operates in its sensitive range. PCA has already placed the clusters faithfully and keeps far points far apart, so from that start there are essentially no triplets demanding that far points be pushed apart; the only work left is to tighten true neighbors and unfold the local manifold without any large force that would tear the global frame. The neighbor triplets, a liability for global structure from a random start, become exactly right when the frame is already correct — and a structured start converges far faster than the tiny random seed the pairwise methods require.

For the gradient, take one triplet's contribution $L = w\,d_{ij}/(d_{ij}+d_{ik})$ with $y_{ij}:=y_i-y_j$, $y_{ik}:=y_i-y_k$. Since $\partial d_{ij}/\partial y_i = 2y_{ij}$, $\partial d_{ij}/\partial y_j = -2y_{ij}$, $\partial d_{ik}/\partial y_i = 2y_{ik}$, $\partial d_{ik}/\partial y_k = -2y_{ik}$, and the quotient gives $\partial L/\partial d_{ij} = w\,d_{ik}/(d_{ij}+d_{ik})^2$, $\partial L/\partial d_{ik} = -w\,d_{ij}/(d_{ij}+d_{ik})^2$, I define the common prefactor $w' := w/(d_{ij}+d_{ik})^2$ and fold the global factor $2$ into the learning rate. With $gs := w'\,d_{ik}\,y_{ij}$ (a pull along $i$–$j$) and $go := w'\,d_{ij}\,y_{ik}$ (a push along $i$–$k$), the accumulation is local: $\mathrm{grad}_i \mathrel{+}= gs - go$, $\mathrm{grad}_j \mathrel{-}= gs$, $\mathrm{grad}_k \mathrel{+}= go$. Because the shared denominator is squared, an already-satisfied triplet ($d_{ik}$ large, $d_{ij}$ small) exerts a small total force — the saturation made visible in the forces. Each triplet touches three rows, so a full-batch gradient over $O(n)$ triplets costs $O(n)$ per iteration. The triplet set is sampled once and never resampled, so the objective is smooth, deterministic, and full-batch; I minimize it with the optimizer the neighbor-embedding methods already use — gradient descent with momentum (a calmer $\gamma = 0.5$ for the first 250 iterations, then $0.8$ to accelerate, gradient evaluated at the look-ahead point $Y + \gamma\,\mathrm{vel}$) and per-coordinate delta-bar-delta adaptive gains that grow by $0.2$ when velocity and gradient disagree and damp to $\max(0.8\,\mathrm{gain}, 0.01)$ when they agree — for 400 iterations from the PCA start. Global structure from PCA, local fidelity from saturating weighted triplets, linear time because the cost is just the neighbor search.

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
