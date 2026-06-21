I want to *look* at high-dimensional data: a few thousand handwritten digits as 784-pixel vectors, or documents as word-count vectors with thousands of entries, laid out in a single two-dimensional scatterplot that tells me the truth about them — that the 7s cluster together and the 1s cluster together, that within the 7s the crossbarred ones drift to one side, that there are roughly ten blobs sitting visibly apart from each other. The map has to do two jobs at once, and they pull against each other. Locally, points close in 784 dimensions must land close in the plane, so neighborhoods are preserved; globally, the blobs must sit sensibly relative to one another with *gaps* between them so I can read them as separate clusters. And I am not allowed to use the class labels to place anything — labels only color the dots afterward — so this is purely unsupervised: out of nothing but the pairwise geometry, manufacture a faithful planar picture.

The existing tools each break in a way that becomes my design spec. The distance-matching family — PCA, classical scaling — minimizes squared error between high- and low-dimensional pairwise distances; but a sum of *squared* errors is dominated by the big distances, so these methods keep faraway things faraway and barely care about the small distances that carry the local manifold structure, and being linear they cannot follow a curved manifold at all. Sammon mapping patches the bias by dividing each squared error by the original distance, $C = (1/\sum_{ij}\|x_i-x_j\|)\sum_{i\neq j}(\|x_i-x_j\| - \|y_i-y_j\|)^2/\|x_i-x_j\|$, up-weighting small distances; but the weight $1/\|x_i-x_j\|$ blows up exactly where the distance is tiny, so the cost becomes obsessed with small *differences* between already-tiny distances, hostage to the most fragile near-coincident pairs rather than spreading its attention evenly across the local structure. Isomap, LLE, and Laplacian Eigenmaps go spectral, but Isomap short-circuits when one noisy edge bridges the graph, the spectral methods avoid collapsing the entire map to a point only via a covariance constraint that is cheaply gamed by a "curdled" map (everything piled at the center with a couple of outliers placed far out to supply the variance), and none of them can display two genuinely separated submanifolds because the data does not give a connected graph. Underlying all of these, for the local methods, is a deeper geometric obstruction: on a manifold of intrinsic dimension well above two, the volume of a ball grows like $r^m$, so there are vastly more *moderately* distant neighbors than near ones, and a two-dimensional map has nowhere near enough area at moderate radius to seat them all — a method that insists on honoring those moderate distances is forced to crush everything inward, erasing the gaps.

The frame that feels right is the probabilistic one: stop talking about distances and talk about neighbors-as-probabilities. Center a Gaussian on $x_i$ and ask how likely $x_i$ is to pick $x_j$ if it drew one neighbor with probability falling off as that Gaussian, giving the conditional $p_{j|i} = \exp(-\|x_i-x_j\|^2/2\sigma_i^2)/\sum_{k\neq i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)$ with $p_{i|i}=0$. The width $\sigma_i$ is *per point*, set principled-ly by fixing the entropy of each neighbor distribution: the perplexity $\mathrm{Perp}(P_i)=2^{H(P_i)}$ with $H(P_i)=-\sum_j p_{j|i}\log_2 p_{j|i}$ reads as a smooth "effective number of neighbors," and because it rises monotonically with $\sigma_i$ I binary-search $\sigma_i$ per point to hit a target perplexity (typically 5–50, robust to the exact value), so denser regions automatically get smaller $\sigma_i$. The predecessor, SNE, mirrors this on the map side with fixed-width Gaussians $q_{j|i}=\exp(-\|y_i-y_j\|^2)/\sum_{k\neq i}\exp(-\|y_i-y_k\|^2)$ and minimizes the summed conditional Kullback-Leibler divergence $C=\sum_i\mathrm{KL}(P_i\|Q_i)$. I like the *asymmetry* of KL here: it punishes a small $q$ modeling a large $p$ — placing two map points far apart when their data points are close — far more than the reverse, which tilts the whole objective toward getting the local structure right. But SNE has two flaws that are precisely what I must fix. Its conditional normalizations — one denominator per point in both $p$ and $q$ — make the gradient awkward and force simulated annealing (Gaussian noise injected each iteration with a slowly decaying variance, plus hand-tuned momentum and step schedules, often rerun several times) to optimize at all. And it crowds: a Gaussian map kernel decays so fast that "moderately dissimilar" can only be represented by a modest map distance, and there is no room for modest distances times that many moderately-distant points, so their summed weak attractions crush the map inward and the clusters never separate.

I propose t-SNE, t-Distributed Stochastic Neighbor Embedding. Two moves fix the two flaws, and they compose. The first move attacks the optimization: match a *single joint* distribution over all pairs instead of one conditional distribution per point, minimizing one global KL, $C=\mathrm{KL}(P\|Q)=\sum_i\sum_j p_{ij}\log(p_{ij}/q_{ij})$ with $p_{ii}=q_{ii}=0$. A single global normalization makes $p_{ij}=p_{ji}$ and $q_{ij}=q_{ji}$ by construction, and with only one denominator to differentiate the gradient collapses to something clean. The obvious joint — one global Gaussian $p_{ij}\propto\exp(-\|x_i-x_j\|^2/2\sigma^2)$ — fails on outliers: a point far from everything has every $p_{ij}$ minuscule, so $y_i$ barely enters the cost and wanders off undetermined, exactly the robustness the per-point conditionals had given me. The fix is to keep the conditionals I trust and *build* the joint from them by symmetrizing,
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}.$$
Now $\sum_j p_{ij} = \frac{1}{2n}\sum_j(p_{j|i}+p_{i|j}) = \frac{1}{2n}(1 + \sum_j p_{i|j}) \geq \frac{1}{2n}$, because each row $\sum_j p_{j|i}=1$ regardless of how outlying $i$ is — so every point contributes at least $1/(2n)$ of the mass and is forced to matter. With a Gaussian joint $q_{ij}\propto\exp(-\|y_i-y_j\|^2)$ the gradient becomes $\delta C/\delta y_i = 4\sum_j(p_{ij}-q_{ij})(y_i-y_j)$ — a single $(p_{ij}-q_{ij})$ per pair instead of SNE's four-term expression. But symmetrization bought only simplicity; with a Gaussian still on the map, the volume argument still crushes the map.

The second move attacks crowding, and the leverage is that I am now matching joint *probabilities*, not distances, so the kernel that turns a map distance into a similarity need not be the same Gaussian I use in the high-dimensional space. Fixing crowding requires that a *moderate* $p_{ij}$ be representable by a *larger* map distance than a Gaussian allows, so the moderately-dissimilar points can spread far out without paying a large attractive penalty and the gaps can open. That is precisely what a heavy-tailed kernel does. I take a Student-t with one degree of freedom, the Cauchy, defining the map joint
$$q_{ij} = \frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\neq l}(1+\|y_k-y_l\|^2)^{-1}}.$$
Why exactly one degree of freedom? Its far field is inverse-square: for large $d$, $(1+d^2)^{-1}\to d^{-2}$. Rescaling the whole map by a constant $c$ multiplies the unnormalized similarity of every far pair by the same $c^{-2}$, which factors out of the normalization, so $q_{ij}$ is approximately invariant to the map's scale for far-apart pairs — I need not get the global scale right, and a distant cluster interacts with the rest of the map almost exactly as a single point would, because its internal scale becomes irrelevant to its long-range pull. The optimization then behaves the same at all scales except the very finest, which is what lets it organize globally without the annealing crutch. Two more reasons: a Student-t is an infinite mixture of Gaussians, a close cousin of the high-dimensional kernel rather than something exotic; and $(1+d^2)^{-1}$ has no exponential, making it far cheaper across the $O(n^2)$ pairs evaluated every iteration. Critically, the heavy tail does *not* reintroduce the disease of the earlier uniform-background fix (UNI-SNE), where two far-apart points drew essentially all their $q$ from a constant floor so that moving them changed $q$ only by a vanishing *proportional* amount and exerted no restoring force. The Student-t's tail is a *function of the distance*, not a constant: $(1+d^2)^{-1}$ keeps depending on $d$ no matter how large, so there is always a real proportional change and always a restoring force, and a cluster split early in the optimization can still be pulled back.

Deriving the gradient of this Student-t KL confirms the heavy tail does not wreck the clean form. Write $d_{ij}=\|y_i-y_j\|$ and $Z=\sum_{k\neq l}(1+d_{kl}^2)^{-1}$, so $q_{ij}=(1+d_{ij}^2)^{-1}/Z$. Only $-\sum p_{ij}\log q_{ij}$ depends on the map (the $\sum p_{ij}\log p_{ij}$ term is constant), and moving $y_i$ changes only the ordered distances $d_{ij},d_{ji}$ for all $j$, with $\partial d_{ij}/\partial y_i=(y_i-y_j)/d_{ij}$ (the zero-distance case resolved by continuity after the factors cancel), giving $\delta C/\delta y_i=\sum_j[\delta C/\delta d_{ij}+\delta C/\delta d_{ji}](y_i-y_j)/d_{ij}$, the two scalar terms equal by symmetry. Since $\log q_{kl}=-\log(1+d_{kl}^2)-\log Z$ and $\sum_{k\neq l}p_{kl}=1$,
$$C = \text{const} + \sum_{k\neq l}p_{kl}\log(1+d_{kl}^2) + \log Z,$$
so differentiating, with $\delta Z/\delta d_{ij}=-2d_{ij}(1+d_{ij}^2)^{-2}$ and the second term equal to $-2q_{ij}d_{ij}/(1+d_{ij}^2)$,
$$\frac{\delta C}{\delta d_{ij}} = 2\,p_{ij}\,\frac{d_{ij}}{1+d_{ij}^2} + \frac{1}{Z}\frac{\delta Z}{\delta d_{ij}} = 2(p_{ij}-q_{ij})\frac{d_{ij}}{1+d_{ij}^2}.$$
Substituting both ordered-pair contributions yields the working gradient,
$$\frac{\delta C}{\delta y_i} = 4\sum_j (p_{ij}-q_{ij})(y_i-y_j)(1+\|y_i-y_j\|^2)^{-1},$$
barely more than the symmetric-SNE gradient — just the extra $(1+\|y_i-y_j\|^2)^{-1}$ riding along. Read as forces, this is a sum of pairwise pushes along $y_i-y_j$: when $p_{ij}>q_{ij}$ the descent direction is $-(y_i-y_j)$ and pulls the pair together; when $p_{ij}<q_{ij}$ it pushes them apart; the extra factor softens the force for far pairs. The strong, finite repulsion of dissimilar-but-too-close points carves the gaps, while the surviving long-range attractions reunite split clusters — which is exactly why simulated annealing can be dropped: the long-range forces themselves do the global-organization job annealing was hacking around.

The optimization is plain momentum gradient descent, $Y^{(t)}=Y^{(t-1)}-\eta\,\nabla_Y C+\alpha(t)(Y^{(t-1)}-Y^{(t-2)})$, with $\alpha=0.5$ while the map is a disorganized blob and $\alpha=0.8$ once clusters form and I want to coast, from a tiny start — either $N(0,10^{-4}I)$ or a PCA projection rescaled so the first component has standard deviation $10^{-4}$. A reasonable learning rate is on the order of 100–200, with an automatic default $\eta=\max(n/\text{early\_exaggeration}/4,\,50)$. Per-coordinate adaptive gains (the Jacobs scheme — grow the gain where the descent direction is stable, shrink it where it flips, clip at a small floor) replace SNE's hand-tuned schedules. Two further tricks earn their keep. *Early compression*: an optional small L2 penalty $\propto\sum_i\|y_i\|^2$ early keeps the points near the origin so clusters can pass through one another while organizing. *Early exaggeration*, the more important: for the first several dozen iterations multiply all $p_{ij}$ by a constant (4 in the basic algorithm, 12 in the implementation default) before computing the cost; the $q_{ij}$ still sum to one and cannot match the inflated $p_{ij}$, so the only way to reduce the KL is to make the $q_{ij}$ of genuinely-similar pairs as large as possible, condensing each true cluster into a tight, widely separated knot whose surrounding empty space is exactly what lets the clusters rearrange into a good global layout; the $p_{ij}$ are divided back down afterward. The cost and memory are $O(n^2)$ because of the all-pairs $Z$ and gradient, capping the plain method near ten thousand points; for larger $n$ the same map objective runs on high-dimensional affinities computed from random walks on a $k$-NN graph, where the walk steps from $x_i$ to $x_j$ with probability $\propto\exp(-\|x_i-x_j\|^2)$ and $p_{j|i}$ is the fraction of walks from landmark $i$ that first reach landmark $j$, integrating over all paths and so resisting the short-circuits that wreck geodesic methods.

```python
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(X, perplexity):
    """High-dimensional affinities: per-point Gaussian conditionals with sigma_i set by a
    binary search on perplexity, symmetrized into a normalized joint p_ij = (p_{j|i}+p_{i|j})/(2n)."""
    D = squareform(pdist(X, "sqeuclidean"))
    n = X.shape[0]
    P = np.zeros((n, n))
    target = np.log(perplexity)                          # entropy target in nats
    for i in range(n):
        beta_lo, beta_hi, beta = -np.inf, np.inf, 1.0    # beta = 1 / (2 sigma_i^2)
        idx = np.arange(n) != i
        Di = D[i, idx]
        for _ in range(50):                              # binary search for sigma_i
            Pi = np.exp(-Di * beta)
            sumPi = max(Pi.sum(), MACHINE_EPSILON)
            H = np.log(sumPi) + beta * np.dot(Di, Pi) / sumPi
            if H > target:                               # entropy too high -> raise beta (shrink sigma)
                beta_lo = beta
                beta = beta * 2 if beta_hi == np.inf else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = beta / 2 if beta_lo == -np.inf else (beta + beta_lo) / 2
        Pi = np.exp(-Di * beta)
        P[i, idx] = Pi / max(Pi.sum(), MACHINE_EPSILON)  # conditional p_{j|i}
    P = P + P.T                                          # symmetrize
    P = np.maximum(P / max(P.sum(), MACHINE_EPSILON), MACHINE_EPSILON)  # joint, normalized
    np.fill_diagonal(P, 0.0)
    return P


def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    """KL(P||Q) with Student-t Q, and its gradient w.r.t. the embedding."""
    Y = params.reshape(n_samples, n_components)
    # Q: Student-t, normalized over all pairs; 2-D uses degrees_of_freedom = 1.
    dist = pdist(Y, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    Pf = squareform(P)                                  # condensed P
    kl = 2.0 * np.dot(Pf, np.log(np.maximum(Pf, MACHINE_EPSILON) / Q))

    # For 2-D: grad_i = 4 sum_j (p_ij - q_ij)(1 + ||y_i-y_j||^2)^-1 (y_i-y_j)
    grad = np.empty((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((Pf - Q) * dist)                   # sklearn's (P - Q) * dist trick
    for i in range(n_samples):
        grad[i] = np.dot(PQd[i], Y[i] - Y)
    grad = grad.ravel() * (2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom)
    return kl, grad


def _gradient_descent(p0, args, n_iter, eta, momentum, min_gain=0.01):
    """Batch gradient descent with momentum and per-coordinate adaptive gains (Jacobs 1988)."""
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    for _ in range(n_iter):
        _, grad = _kl_divergence(p, *args)
        inc = (update * grad) < 0.0
        gains[inc] += 0.2
        gains[~inc] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - eta * grad
        p += update
    return p


def tsne(X, n_components=2, perplexity=30.0, early_exaggeration=12.0,
         learning_rate="auto", n_iter=1000, init="pca", random_state=None):
    """t-SNE embedding of X into n_components dimensions."""
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    if learning_rate == "auto":
        learning_rate = max(n / early_exaggeration / 4.0, 50.0)

    P = _joint_probabilities(X, perplexity)
    if init == "pca":
        Y = PCA(n_components=n_components, random_state=rng).fit_transform(X).astype(np.float32)
        Y = Y / np.std(Y[:, 0]) * 1e-4
    else:
        Y = 1e-4 * rng.standard_normal((n, n_components)).astype(np.float32)
    Y = Y.ravel()
    degrees_of_freedom = max(n_components - 1, 1)        # sklearn's rule; equals 1 for 2-D
    args = (P, degrees_of_freedom, n, n_components)

    # Stage 1: early exaggeration, low momentum (exploration)
    explore = min(250, n_iter)
    P *= early_exaggeration
    Y = _gradient_descent(Y, args, explore, learning_rate, momentum=0.5)
    P /= early_exaggeration

    # Stage 2: real objective, higher momentum
    remaining = n_iter - explore
    if remaining > 0:
        Y = _gradient_descent(Y, (P, degrees_of_freedom, n, n_components),
                              remaining, learning_rate, momentum=0.8)
    return Y.reshape(n, n_components)
```
