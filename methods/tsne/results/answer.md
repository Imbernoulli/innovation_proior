# t-SNE, distilled

t-SNE (t-Distributed Stochastic Neighbor Embedding) embeds high-dimensional data into two (or
three) dimensions for visualization. It converts high-dimensional distances into a joint
probability distribution `P` over pairs of points using per-point Gaussian neighborhoods, and
defines a second joint distribution `Q` over the map points using a heavy-tailed Student-t (one
degree of freedom) kernel, then moves the map points to minimize the Kullback-Leibler
divergence `KL(P||Q)` by gradient descent. The two design choices that distinguish it are the
single symmetrized joint objective (instead of a sum of per-point conditional KLs) and the
heavy-tailed map kernel, which together fix the "crowding problem" and make the optimization
work without simulated annealing.

## Problem it solves

Given high-dimensional vectors `X = {x_1, ..., x_n}`, place each at a location `y_i` in a 2-D
map (without using labels) so that the scatterplot preserves local neighborhoods *and* shows
global cluster structure with visible gaps between clusters. Linear and distance-matching
methods lose local structure; existing local methods crowd all points toward the center and
fail to separate clusters.

## Key idea

1. **High-dimensional affinities (per-point Gaussian, perplexity).** Conditional probability
   that `x_i` would pick `x_j` as neighbor:

   ```
   p_{j|i} = exp(-||x_i - x_j||^2 / 2 sigma_i^2) / sum_{k != i} exp(-||x_i - x_k||^2 / 2 sigma_i^2),   p_{i|i} = 0.
   ```

   Each `sigma_i` is set by a binary search so the distribution's perplexity
   `Perp(P_i) = 2^{H(P_i)}`, with `H(P_i) = -sum_j p_{j|i} log2 p_{j|i}`, matches a user target
   (a smooth effective number of neighbors; typical 5-50). Denser regions get smaller `sigma_i`.

2. **Symmetrize into a single joint.** Instead of a sum of per-point conditional KLs (the SNE
   approach), use one joint distribution and one KL. Building the joint from the conditionals,

   ```
   p_ij = (p_{j|i} + p_{i|j}) / (2n),
   ```

   guarantees `sum_j p_ij >= 1/(2n)` for every `i`, so even an outlier contributes to the cost
   (a naive single Gaussian joint would leave an outlier's position undetermined). This yields a
   much simpler gradient.

3. **Heavy-tailed map kernel (the "t").** Because the objective matches joint *probabilities*
   (not distances), the map kernel may differ from the high-dimensional one. Use a Student-t
   with one degree of freedom (Cauchy):

   ```
   q_ij = (1 + ||y_i - y_j||^2)^{-1} / sum_{k != l} (1 + ||y_k - y_l||^2)^{-1},   q_ii = 0.
   ```

   Its slow tail lets a moderate `p_ij` be matched by a *larger* map distance, removing the
   unwanted attraction from the many moderately-distant points and so opening gaps between
   clusters (this cures crowding). One degree of freedom gives an inverse-square far-field
   `(1 + d^2)^{-1} -> d^{-2}`, making the map approximately invariant to scale for far-apart
   pairs and letting distant clusters interact like single points; unlike a uniform-background
   floor, it stays distance-dependent, so real long-range restoring forces survive and split
   clusters can be pulled back together. It is also cheap (no exponential).

4. **Objective and gradient.** Minimize `C = KL(P||Q) = sum_i sum_j p_ij log(p_ij / q_ij)`.
   The derivative used by gradient descent is:

   ```
   dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}.
   ```

## Gradient derivation

Let `d_ij = ||y_i - y_j||` and `Z = sum_{k != l} (1 + d_kl^2)^{-1}`, so
`q_ij = (1 + d_ij^2)^{-1} / Z`. Only `-sum p_ij log q_ij` depends on the map (the
`sum p_ij log p_ij` term is constant). Moving `y_i` changes only the ordered distances
`d_ij`, `d_ji` for all `j`. For nonzero `d_ij`, `dd_ij/dy_i = (y_i - y_j) / d_ij`; the final
product is interpreted by continuity at zero, so

```
dC/dy_i = sum_j [dC/dd_ij * (y_i - y_j)/d_ij + dC/dd_ji * (y_i - y_j)/d_ij].
```

The two scalar derivatives are equal by symmetry. For `dC/dd_ij`, write

```
C = constant + sum_{k != l} p_kl log(1 + d_kl^2) + log Z,
```

because `sum p_kl = 1`. Hence

```
dC/dd_ij = 2 p_ij d_ij/(1 + d_ij^2) + (1/Z) dZ/dd_ij
         = 2 p_ij d_ij/(1 + d_ij^2) - 2 d_ij (1 + d_ij^2)^{-2}/Z
         = 2 (p_ij - q_ij) d_ij (1 + d_ij^2)^{-1}.
```

Substituting both ordered-pair contributions:

```
dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}.
```

## Optimization and tricks

- **Init:** the basic algorithm samples the map from `N(0, 1e-4 * I)`; sklearn uses coordinates
  with standard deviation `1e-4`, or a PCA projection rescaled so the first component has std
  `1e-4`.
- **Update:** momentum gradient descent subtracts the returned derivative,
  `Y^{(t)} = Y^{(t-1)} - eta * grad + alpha(t)(Y^{(t-1)} - Y^{(t-2)})`, with `alpha = 0.5`
  early and `0.8` later. In sklearn, `learning_rate="auto"` is
  `max(n_samples / early_exaggeration / 4, 50)`. A
  per-coordinate adaptive learning rate (Jacobs gains: grow where the descent direction is stable,
  shrink where it flips, clip at a floor) speeds convergence.
- **Early exaggeration:** multiply all `p_ij` by a constant (4 in the basic algorithm; 12 in
  sklearn) for the first ~50-250 iterations. The `q_ij` cannot match the inflated
  `p_ij`, so the optimizer condenses true clusters into tight, widely separated knots,
  creating empty space in which the clusters can rearrange to a good global layout. Remove it
  afterward.
- **Early compression:** an optional small `L2` penalty `propto sum_i ||y_i||^2` early keeps
  points near the origin so clusters can pass through one another while organizing.
- No simulated annealing is needed: the heavy-tailed kernel's long-range forces do the
  global-organization job directly.

The cost and memory are `O(n^2)`. For large `n`, keep the same map objective but compute the
high-dimensional affinities from random walks on a `k`-NN graph (the walk's edge probability is
`propto exp(-||x_i - x_j||^2)`, and `p_{j|i}` is the fraction of walks from landmark `i` that
first reach landmark `j`), which integrates over all paths and resists short-circuits.

## Defaults and why

- **Student-t, 1 dof:** inverse-square far-field gives scale-invariance for far pairs and cheap
  evaluation; for embeddings into more than 3 dimensions, more degrees of freedom are
  appropriate.
- **Per-point `sigma_i` via perplexity (not one global `sigma`):** data density varies; equal
  perplexity sizes each neighborhood to its local density.
- **Symmetrized joint `(p_{j|i} + p_{i|j})/(2n)`:** anchors outliers and simplifies the
  gradient.
- **Momentum 0.5 -> 0.8, exaggeration factor 4 in the basic algorithm or 12 in sklearn:** low momentum and inflated `p_ij`
  while the map is disorganized; coast once clusters form.

## Working code

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

This mirrors sklearn's implementation structure: `_joint_probabilities`
(perplexity binary search + symmetrize + normalize), `_kl_divergence` (Student-t `Q` and the
`(P - Q) * dist` gradient trick), and the two-stage `_gradient_descent` with momentum and
adaptive gains.
