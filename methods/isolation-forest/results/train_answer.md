I have a pile of unlabelled tabular vectors and I need to surface the rare, unusual ones — the anomalies — and rank every point by how anomalous it is. The two facts I am handed about anomalies are almost too simple to be useful: they are *few* (a small minority) and they are *different* (their attribute values stand apart from the bulk). The detector has to do two things well at once: produce a faithful ranking so the truly unusual points float to the top, and do so cheaply enough to scale — to very many instances, high dimensionality, and many attributes that are simply irrelevant to whether a point is anomalous. The trouble is that every method I reach for buys ranking quality at a cost that does not scale, or scales at a cost in ranking quality, and they all aim their effort somewhere other than the rare minority I actually care about.

The whole model-based tradition does the same thing: build a profile of what "normal" looks like — fit a distribution, draw a one-class boundary, cluster the bulk — and flag whatever falls outside. The objective is aimed at the wrong target. I am optimizing a model to *describe the normal majority* when the thing I care about is the rare minority at the edges, so a model can fit the bulk beautifully and still be badly calibrated for the handful of points I want, yielding a flood of false alarms or missed anomalies. And the methods that rank best make me pay for it. Distance-based detection — score a point by its distance to its $k$-th nearest neighbour — is conceptually clean, but the naive version is $O(n^2)$, and even the clever pruning of ORCA, which processes points in random order and keeps a running cutoff, is only near-linear *in practice* and stays quadratic in the worst case; the cost *is* the distance computation, and it climbs with dimensionality. Density-based LOF compares a point's local reachability density to its neighbours', $\mathrm{LOF}_k(A) = \frac{1}{|N_k(A)|}\sum_{B\in N_k(A)} \mathrm{lrd}_k(B)/\mathrm{lrd}_k(A)$, so it copes with clusters of different density, but it too computes a $k$-NN neighbourhood for every point — the same distance bill — and being *local* it can be fooled: a point in a tight little local group has $\mathrm{LOF}\approx 1$ even when it is a glaring *global* outlier. So I keep hitting the same wall: good rankings cost distance computations that do not scale, and cheap profiles aim at the wrong objective.

So I throw out the assumption that I must model normality at all, and exploit "few and different" directly. I propose Isolation Forest. Suppose I chop the feature space up at random — pick an attribute, pick a split value somewhere between its smallest and largest value in the current group, send each point left or right, and recurse on each piece. A typical normal point is buried in a dense region with many neighbours just like it, so a random cut almost never lands right next to it; it survives cut after cut and is isolated only after many cuts. An anomaly is *different* — its values sit out in a sparse region — so a random cut on the right attribute, anywhere in the wide empty gap, peels it off immediately, and it is *few*, so there are not many other points to shield it. Anomalies are *easier to isolate*. If I keep recursively cutting until every point is alone, the process is a binary tree: each internal node is one random cut, every point ends at its own leaf, and the number of edges from the root to $x$'s leaf is the path length $h(x)$. Short path means anomalous. No distance, no density, no profile of normality — just how deep a random partitioning has to go to fence each point off, which is wonderfully cheap: a cut is a comparison and scoring a point is one traversal.

One such tree is a single noisy random draw, so the cure is the standard one for high-variance random trees: build an ensemble, each from its own independent randomness, and *average* the path length of $x$ across the trees, turning the noisy per-tree $h(x)$ into a stable estimate of $E(h(x))$ while unlucky cuts wash out and irrelevant attributes are diluted. I want to *believe*, not just narrate, that extreme points have shorter expected path length. Counting the binary-tree shapes the random cuts can make on $|X|$ sorted univariate points, with $j = |X|-1$ and $C_j$ the Catalan number, the expected depth of the $m$-th point is $E(h(x_m)) = \sum_l P(h(x_m)=l)\,l$ with $P(h(x_m)=l) = t_{lmj}/C_j$ where $t_{lmj}$ counts shapes giving $x_m$ depth $l$. Summing the depth contribution over all shapes has a closed form,
$$h_{mj} = \frac{2\binom{2m}{m}\binom{2j-2m}{j-m}(2m+1)(2j-2m+1)}{(j+1)(j+2)} - C_j,\qquad E(h(x_m)) = \frac{h_{mj}}{C_j},\quad m=0,\dots,j.$$
This is symmetric under $m \mapsto j-m$, smallest at the two ends and largest near the middle, a dome with peak height about $4\sqrt{j/\pi}$. Fringe (extreme) points genuinely sit in shallower leaves — exactly the geometric fact the method rests on.

Now an average path length is not yet a usable score: a depth of $8$ means different things for a tree over a thousand points versus sixteen, and raw $E(h)$ is neither comparable across tree sizes nor bounded. I divide by the *typical* path length of such a tree. Here is the key observation: a fully-grown isolation tree has the same structure as a *binary search tree* built by inserting the points — each split is a comparison, and isolating a point is precisely an *unsuccessful search* falling off at an external node. So the average path length to an external node equals the average unsuccessful-search cost of a random BST, a classical exact result. For a random BST on $m$ keys the expected internal path length is $E[I_m] = 2(m+1)H_m - 4m$ with $H_m$ the $m$-th harmonic number, and the external–internal identity $E_m = I_m + 2m$ holds for any binary tree, so the average external depth is $E_m/(m+1)$. A tree built from $n$ points has $n$ external nodes, i.e. $m = n-1$ keys, and substituting gives
$$c(n) = \frac{E_{n-1}}{n} = \frac{I_{n-1} + 2(n-1)}{n} = 2H_{n-1} - \frac{2(n-1)}{n}.$$
This checks against direct counts: $c(2)=1$, $c(3)=2H_2-\tfrac{4}{3}=\tfrac{5}{3}$, and $c(n)=0$ for $n\le 1$. I approximate $H(i)\approx \ln(i) + \gamma$ with $\gamma = 0.5772156649$, Euler's constant. The score then maps short $E(h)$ to anomalous in a bounded, monotone way:
$$s(x, \psi) = 2^{-E(h(x))/c(\psi)},\qquad s \in (0,1].$$
As $E(h)\to 0$ the exponent vanishes and $s\to 1$ (anomaly); as $E(h)$ approaches the deepest value $s$ moves toward $0$; and at the calibration point $E(h)=c(\psi)$ the exponent is $-1$ and $s = 0.5$ exactly, so $0.5$ is the average-depth anchor — well above it is anomalous, well below is normal, and an all-$0.5$ data set has no distinct anomaly.

Two structural choices make it work and scale, and the first is the one I would never have guessed. The obvious move is to build each tree from the *entire* data set, but the known failure modes forbid it. *Swamping*: when many normal points crowd near an anomaly, it takes more cuts to fence the anomaly off, so its path length creeps up toward a normal point's. *Masking*: when anomalies form a *large, dense* cluster, its members shield each other and all look normal. Both are driven by *too much data* in the tree, so I turn it around and build each tree from a small random *sub-sample drawn without replacement*, fresh per tree: a small sub-sample thins the crowd of interfering normals and thins a dense anomaly clump, restoring the empty space that lets anomalies isolate in a cut or two, while the ensemble's many sub-samples cover the data. This also caps cost — building each tree is governed by the sub-sample size $\psi$, not by $n$ — and a small power-of-two default $\psi = 256$ (or all rows, if fewer) is the practical choice. The second choice is a height limit. Growing every tree to the bottom wastes effort resolving the deep, already-normal part; since the average depth scale grows like $\log\psi$, I cap the height at $l = \lceil \log_2 \psi \rceil$. When a traversal stops early at a node still holding $\mathrm{Size} > 1$ unisolated points, the path would have continued on average by $c(\mathrm{Size})$ more edges, so I return the edges traversed *plus* $c(\mathrm{Size})$ — preserving the average-depth accounting while skipping the wasted work ($c(1)=0$, so a singleton leaf adds nothing).

Putting it together: pick the number of trees $t$ (default $100$, controlling the variance of the averaged depth) and sub-sample size $\psi$, set $l = \lceil \log_2 \psi \rceil$, and for each tree draw $\psi$ points without replacement and grow an isolation tree. Growing on a group $X$ at depth $e$: if $e \ge l$, or $|X| \le 1$, or all rows are identical, make an external node recording $\mathrm{Size} = |X|$; otherwise pick a random splittable attribute $q$, a split $p$ uniform in $(\min_q X, \max_q X)$, send $X[q<p]$ left and $X[q\ge p]$ right, and recurse. To score $x$, run it down each tree following the stored split rules, count edges, add $c(\mathrm{Size})$ at the terminating node, average over the $t$ trees to get $E(h(x))$, and return $s(x,\psi) = 2^{-E(h(x))/c(\psi)}$, normalizing by $c(\psi)$ because every tree was grown from $\psi$ points. The result is a ranking in $(0,1]$, higher meaning more anomalous, at a cost set by $\psi$ and $t$ rather than $n$.

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
