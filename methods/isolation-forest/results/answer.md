# Isolation Forest (iForest), distilled

Isolation Forest detects anomalies by *isolating* points instead of profiling normal ones. It
builds an ensemble of random *isolation trees*, each grown on a small sub-sample by recursively
cutting on a random attribute at a random split value until points are separated. Because
anomalies are "few and different," they are fenced off in fewer cuts, so they sit at shorter
root-to-leaf path lengths. The anomaly score is the average path length over the ensemble,
normalized by the average path length of a random tree and mapped monotonically into (0, 1].
No distance or density is ever computed; cost and memory are set by the sub-sample size, not by
the data set size.

## Problem it solves

Unsupervised anomaly detection / ranking on unlabelled tabular data that must scale to large n,
high dimension, and many irrelevant attributes — where distance/density methods are too costly
(near-/super-linear with heavy constants) and profiling methods aim at describing the normal
bulk rather than separating the rare anomalies.

## Key idea

Isolate, don't profile. Recursive random partitioning isolates anomalies early (short path)
and normal points late (long path); average path length over a forest of random trees is a
principled anomaly signal. Two structural choices make it work and scale:

- **Sub-sampling (small psi, without replacement, fresh per tree)** relieves *swamping* (normal
  points crowding an anomaly inflate its path length) and *masking* (a large dense anomaly
  cluster shields its own members) — both are too-much-data pathologies — and caps cost at psi.
- **Normalization by c(psi)** makes path lengths comparable across tree sizes and bounds the
  score.

## Final definitions

Path length `h(x)`: number of edges from the root to the external node where x terminates.

Catalan tree-shape check: for `|X|` sorted univariate points, let `j = |X| - 1` and let `C_j`
be the Catalan number counting binary-tree shapes. If `t_{lmj}` counts shapes in which the
m-th point has depth l, then `P(h(x_m)=l) = t_{lmj}/C_j` and
`E(h(x_m)) = sum_l P(h(x_m)=l) l`. Summing the depth contribution over all tree shapes gives

```
h_{mj} =
  2 * binom(2m, m) * binom(2j - 2m, j - m)
    * (2m + 1) * (2j - 2m + 1) / ((j + 1)(j + 2))
  - C_j,

E(h(x_m)) = h_{mj} / C_j,      m = 0, ..., j.
```

This profile is symmetric and dome-shaped: fringe points have much lower expected path length
than core points, and the peak height is about `4 sqrt(j/pi)`.

Average path length of a random isolation tree on n points (= average unsuccessful-search depth
of a random BST; derived from internal path length `E[I_{n-1}] = 2 n H_{n-1} - 4(n-1)` and the
external-internal identity `E_m = I_m + 2m`):

```
c(n) = 2 H(n-1) - 2(n-1)/n     (n > 2),     c(2) = 1,     c(n) = 0  (n <= 1),
H(i) = sum_{k=1}^i 1/k ~= ln(i) + gamma,     gamma = 0.5772156649.
```

Anomaly score, with `E(h(x))` the mean path length over the t trees and psi the sub-sample size:

```
s(x, psi) = 2^( - E(h(x)) / c(psi) ) ,      s in (0, 1] .
```

`E(h) -> 0  => s -> 1` (anomaly); `E(h) -> psi-1` pushes s toward 0 (strictly positive for
finite psi, vanishing in the large-psi limit); `E(h) = c(psi) => s = 0.5` (exactly average).
Higher s = more anomalous. When a path stops early at a node holding
`Size > 1` unisolated points (height limit reached), add `c(Size)` to its edge count as the
estimate of the unbuilt remainder.

## Algorithms

```
iForest(X, t, psi):                                # t trees, sub-sample size psi
    l = ceil(log2(psi))                            # height limit ~ average tree height
    Forest = {}
    for i in 1..t:
        X' = sample(X, psi)  without replacement
        Forest += iTree(X', 0, l)
    return Forest

iTree(X, e, l):                                    # e = current depth, l = height limit
    if e >= l or |X| <= 1 or all rows of X equal:
        return exNode{ Size = |X| }
    q = random attribute
    p = random split in ( min_q(X), max_q(X) )
    return inNode{ Left  = iTree(X[q <  p], e+1, l),
                   Right = iTree(X[q >= p], e+1, l),
                   SplitAtt = q, SplitValue = p }

PathLength(x, T, e):
    if T is external: return e + c(T.Size)         # c(.) above; c(1)=0
    if x[T.SplitAtt] < T.SplitValue: return PathLength(x, T.Left,  e+1)
    else:                            return PathLength(x, T.Right, e+1)
```

Defaults: `t = 100`, `psi = 256` (`max_samples="auto"` means `min(256, n)` in the common
library API), height limit `ceil(log2 psi)`. With height-limited trees, training is
`O(t psi log psi)`, space is `O(t psi)`, and scoring is `O(t n_test log psi)`; a fully grown
variant has worst-case `O(t psi^2)` construction and `O(t n_test psi)` traversal.

## Working code (self-contained, mirrors the algorithms above)

```python
import numpy as np


def _average_path_length(n):
    # average path length to an external node of a random isolation tree on n points
    n = np.asarray(n, dtype=float)
    shape = n.shape
    n = n.reshape(-1)
    out = np.zeros_like(n, dtype=float)
    mask_1 = n <= 1.0
    mask_2 = n == 2.0
    not_mask = ~(mask_1 | mask_2)
    out[mask_2] = 1.0
    out[not_mask] = (
        2.0 * (np.log(n[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n[not_mask] - 1.0) / n[not_mask]
    )
    out = out.reshape(shape)
    return out.item() if out.shape == () else out


class _Node:
    __slots__ = ("left", "right", "split_att", "split_val", "size", "is_leaf")


def _grow(X, e, hlim, rng):
    nd = _Node()
    nd.left = nd.right = None
    n = X.shape[0]
    mins, maxs = (X.min(0), X.max(0)) if n else (None, None)
    splittable = np.flatnonzero(maxs > mins) if n else np.array([], int)
    if e >= hlim or n <= 1 or splittable.size == 0:
        nd.is_leaf, nd.size = True, n
        return nd
    q = splittable[rng.integers(splittable.size)]          # random attribute
    p = rng.uniform(mins[q], maxs[q])                       # random split value
    m = X[:, q] < p
    nd.is_leaf, nd.split_att, nd.split_val = False, int(q), float(p)
    nd.left = _grow(X[m], e + 1, hlim, rng)                 # q < p
    nd.right = _grow(X[~m], e + 1, hlim, rng)               # q >= p
    return nd


def _path(x, nd, e):
    if nd.is_leaf:
        return e + float(_average_path_length(nd.size))     # + c(Size) for unbuilt remainder
    if x[nd.split_att] < nd.split_val:
        return _path(x, nd.left, e + 1)
    return _path(x, nd.right, e + 1)


class CustomAnomalyDetector:
    def __init__(self, n_estimators=100, max_samples=256, random_state=0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        psi = min(self.max_samples, n)                      # "auto" = min(256, n)
        self.c_psi_ = float(_average_path_length(psi))
        hlim = int(np.ceil(np.log2(max(psi, 2))))           # l = ceil(log2(psi))
        self.trees_ = [
            _grow(X[rng.choice(n, psi, replace=False)], 0, hlim, rng)  # sub-sample w/o replacement
            for _ in range(self.n_estimators)
        ]
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        depths = np.zeros(X.shape[0])
        for tree in self.trees_:
            for i in range(X.shape[0]):
                depths[i] += _path(X[i], tree, 0)
        mean_depth = depths / len(self.trees_)              # E(h(x))
        if self.c_psi_ == 0.0:
            return np.ones(X.shape[0])                      # sklearn's single-sample guard
        return 2.0 ** (-mean_depth / self.c_psi_)           # s(x, psi); higher = more anomalous
```

## Library wrapper

The widely used implementation is `sklearn.ensemble.IsolationForest`, which PyOD wraps as
`pyod.models.iforest.IForest`. scikit-learn computes per-leaf `_average_path_length`
(`2*(ln(n-1)+np.euler_gamma) - 2*(n-1)/n`, with 0 and 1 for n<=1 and n=2), sums node depths plus
the leaf adjustment over the forest, and its internal `_compute_score_samples` forms
`2**(-depth/denominator)`. The public `score_samples` returns the negative of that raw score,
`decision_function` subtracts the offset used for thresholding, and PyOD inverts the public
decision function so that higher values mean more anomalous.

```python
class CustomAnomalyDetector:
    def __init__(self):
        from pyod.models.iforest import IForest
        self.model = IForest(
            n_estimators=100,        # t
            max_samples="auto",      # psi = min(256, n)
            contamination=0.1,       # only sets the predict threshold, not the ranking score
            random_state=0,
        )

    def fit(self, X):
        self.model.fit(X)            # X standardized, no labels
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)   # higher = more anomalous
```
