Let me start from what actually hurts. I have a pile of unlabelled tabular vectors and I need to surface the rare, unusual ones — the anomalies — and rank everything by how anomalous it is. The two facts I'm handed about anomalies are almost too simple to be useful: they are *few* (a small minority) and they are *different* (their attribute values stand apart from the bulk). And yet every method I reach for spends its effort somewhere else. The whole model-based tradition does the same thing: build a profile of what "normal" looks like — fit a distribution, draw a one-class boundary, cluster the bulk — and then flag whatever falls outside. The trouble is the objective is aimed at the wrong target. I'm optimizing a model to *describe the normal majority*, when the thing I care about is the rare minority at the edges. A model can fit the bulk beautifully and still be badly calibrated for the handful of points I actually want, and then I get a flood of false alarms or I miss the real anomalies — because all the modelling capacity went into the part of the data that isn't the question.

And the methods that rank best make me pay for it. Distance-based detection — score a point by its distance to its k-th nearest neighbour, far-from-everything is anomalous — is conceptually clean, but the naive version is O(n^2), and even the clever pruning of something like ORCA, which processes points in random order and keeps a running cutoff so it can prune candidates whose partial score already fell below the n-th best, is only near-linear *in practice* and stays quadratic in the worst case; the cost *is* the distance computation, and it climbs with dimensionality. Density-based LOF is subtler — it compares a point's local reachability density to its neighbours', `LOF_k(A) = (1/|N_k(A)|) sum_{B in N_k(A)} lrd_k(B)/lrd_k(A)`, with `lrd` the inverse average reachability distance — so it can cope with clusters of different density, which plain distance methods can't. But it too computes a k-NN neighbourhood for every point, the same distance bill, and being *local* it can be fooled: a point in a tight little local group has `LOF ~ 1` even when it's a glaring *global* outlier. So I keep hitting the same wall: the good rankings cost distance computations that don't scale, and the cheap profiles aim at the wrong objective.

So let me throw out the assumption that I have to model normality at all and stare at the two given facts directly. Anomalies are few and different. What does "few and different" *do* for me, mechanically, if I stop trying to describe the normal cloud? Suppose I take the data and start chopping the feature space up at random — pick an attribute, pick a split value somewhere between its smallest and largest value in the current group, and that cut sends each point left or right. Repeat on each piece. Where does a typical normal point end up? It's buried in a dense region with lots of neighbours just like it, so a random cut almost never lands right next to it; it survives cut after cut, sitting in a shrinking-but-still-populated box, and it takes many cuts before it's finally alone. Now where does an anomaly end up? It's *different* — its values sit out in a sparse region — so a random cut on the right attribute, anywhere in the wide empty gap between it and the bulk, peels it off immediately. And it's *few*, so there aren't many other points to shield it. The anomaly gets separated early; the normal point gets separated late. So "few and different" isn't only a description of anomalies — it seems to mean anomalies are *easier to isolate*. If that holds up, it's something I can exploit without ever computing a distance or a density.

Let me make "separated early vs. late" precise, because it has to become a number. If I keep recursively cutting until every point is alone, the whole process is a binary tree: each internal node is one random cut, the two children are the two sides of the cut, and every point ends up at its own leaf. "How many cuts did it take to isolate point x" is exactly the number of edges from the root down to x's leaf — the *path length* h(x). Anomalies, isolated early, would sit near the top: short path length. Normal points, isolated late, deep: long path length. If that's right, the anomaly score is just: short path = anomalous. No distance, no density, no profile of normality — just how deep a random partitioning has to go to fence each point off by itself. And it would be cheap: a cut is a comparison, building the tree is linear-ish, and scoring a point is one traversal.

But one such tree is a single random draw — one particular sequence of random cuts — and it'll be noisy. A normal point might get unlucky and be isolated early by a freak cut; an anomaly might survive a few cuts by chance. The cure is the standard one for high-variance random trees: build a whole ensemble of them, each from its own independent randomness, and *average* the path length of x across the trees. The law of large numbers turns the noisy per-tree h(x) into a stable estimate of the expected path length E(h(x)), and the random cuts that happened to be unlucky in one tree wash out across many. As a bonus, the random splits make the whole thing trivially parallel and robust to irrelevant attributes — an irrelevant attribute is just one of the many a tree might cut on, and averaging over many random trees dilutes its effect. So the object is: an ensemble of random partitioning trees, score by average path length.

That last clause is exactly the kind of thing I'd be tempted to just assert, so before I hang the whole method on it I want to *believe* "anomalies have shorter expected path length," not just narrate it. Let me actually count. Forget the data for a second and think about a one-dimensional sorted set of points and the random partitioning over it. Take the simplest non-trivial case, three points `x_0 < x_1 < x_2`. There are only two distinct tree shapes the random cuts can make. If the very first cut falls in the gap between `x_0` and `x_1`, it peels off `x_0` at depth 1 and then `x_1, x_2` need one more cut — call that tree A. If the first cut falls between `x_1` and `x_2`, it peels off `x_2` at depth 1 — tree B. Under a uniform random split, the probability the first cut lands in a given gap is just that gap's width over the total span. So `P(tree A) = D_{0,1}/D_{0,2}` and `P(tree B) = D_{1,2}/D_{0,2}`, writing `D_{i,j}` for the distance from `x_i` to `x_j`. Now look at the middle point `x_1`: it's isolated at depth 2 in *both* trees, so `h(x_1) = 2` always. The extreme points `x_0` and `x_2` each get isolated at depth 1 in one tree and depth 2 in the other, so their expected depth is a mix of 1 and 2.

Let me put numbers on it so I'm not waving my hands. Set `x = (0, 9, 10)`, so `x_0` sits far out: `D_{0,1} = 9`, `D_{1,2} = 1`, `D_{0,2} = 10`. Then `x_0` is isolated at depth 1 with probability `9/10`, giving `E(h(x_0)) = 0.9*1 + 0.1*2 = 1.1`; `x_1` is always at depth 2; and `x_2` is isolated at depth 1 with probability `1/10`, giving `E(h(x_2)) = 0.1*1 + 0.9*2 = 1.9`. To be sure I haven't fooled myself with the gap argument, I simulate it directly — 200,000 trees, drawing a uniform first cut in `(0, 10)` and recursing — and read back the mean depths: `x_0 -> 1.099`, `x_1 -> 2.000`, `x_2 -> 1.901`. That matches the hand computation, and it's the shape I wanted: the far-out fringe point `x_0` has the *shortest* expected path length, the central point `x_1` the longest, and pushing `x_0` further out (a wider surrounding gap) only pulls its expected depth closer to 1. So in the smallest case the "easier to isolate" claim is not a story — it's a computed fact.

Does that survive scaling up past three points? The hand-count gets hopeless fast — at `|X|` points there are `C_j` possible binary-tree shapes, the Catalan number with `j = |X| - 1` (already 42 shapes at six points), so I can't enumerate by eye. But there's a closed form I can lean on. Under the uniform-over-tree-shapes idealization the expected path length of the m-th sorted point is a weighted sum `E(h(x_m)) = sum_l P(h(x_m) = l) * l` over the possible depths l, with `P(h(x_m) = l) = t_{lmj}/C_j` and `t_{lmj}` the number of shapes in which `x_m` has depth l. The sum of the depth contributions over all l collapses; if `h_{mj}` is the total path-length contribution of `x_m` over all `C_j` shapes, Knuth gives

```
h_{mj} =
  2 * binom(2m, m) * binom(2j - 2m, j - m)
    * (2m + 1) * (2j - 2m + 1) / ((j + 1)(j + 2))
  - C_j,

E(h(x_m)) = h_{mj} / C_j,      m = 0, ..., j.
```

I don't want to take that on faith for a formula this fiddly, so let me check it against a brute-force enumeration of every tree shape. I recurse over contiguous runs of points: for a run, each gap can be the root cut, the two sides recurse, every resulting shape is weighted equally, and I sum the depth of each point index across all shapes. For `j = 4` (five points, 14 shapes) the enumeration gives `E(h) = (2.000, 3.000, 3.286, 3.000, 2.000)` reading m from 0 to 4; the closed form gives `(2.000, 3.000, 3.286, 3.000, 2.000)`. They agree to machine precision, and the same holds for `j = 2, 3, 5, 6` — every index, zero difference. So the formula is right, and now I can read its shape with confidence: it's symmetric (swapping m for `j - m` leaves it unchanged), smallest at the two ends, largest near the middle — a dome. The fringe points sit in markedly shallower leaves than the core points, by construction of the random-tree shapes themselves and not by any appeal to a picture. (A standard asymptotic estimate puts the peak height at about `4 sqrt(j/pi)`; my enumerated peaks sit well below that at these tiny j — 3.29 vs 4.51 at `j = 4` — which just tells me that estimate is a large-j approximation, not something to read at small samples. What I actually need here, the dome ordering, is what the exact check confirms.)

Now I have to turn an average path length into a usable anomaly score, and there's a normalization problem I have to face squarely. A path length of, say, 8 means completely different things depending on how many points the tree was built from: in a tree over a thousand points, depth 8 is shallow; in a tree over sixteen points, depth 8 is about as deep as it gets. The maximum height of one of these trees grows roughly linearly in n, the average height only like log n, so a raw E(h) isn't comparable across trees of different sizes and isn't even bounded in a fixed range. I need to divide E(h) by the *typical* path length for a tree of that size, so the score becomes "how short is this point's path relative to a normal point's path." So what *is* the typical / average path length of one of these fully-grown random partitioning trees?

Let me look hard at the structure of a fully-grown isolation tree — recursively split until every point is alone — and see if it matches anything I already know the cost of. Each split is a comparison `q < p`; each point lands at its own external node; isolating a point is exactly the process of an item falling off the bottom of a *binary search tree* at an external node — an *unsuccessful search*. So the average path length to an external node of my random tree should be the same quantity as the average cost of an unsuccessful search in a random BST, which is a classical, exact result. Let me re-derive that result so I actually trust the constant rather than just borrowing it. For a random BST built from m keys, the expected *internal* path length (sum of depths of the internal nodes) is `E[I_m] = 2(m+1)H_m - 4m`, with `H_m` the m-th harmonic number — this comes from the recurrence where a uniformly random key is the root and splits the rest into two random sub-BSTs, telescoping into the harmonic sum. The *external* path length (sum of depths of the m+1 external nodes, where an unsuccessful search terminates) satisfies the identity `E_m = I_m + 2m` for any binary tree with m internal nodes. So the average depth of an external node, i.e. the average unsuccessful-search path length, is `E_m/(m+1)`. For my tree built from n points, all isolated, there are n external nodes, which corresponds to a BST on `m = n - 1` keys; substituting and simplifying,

```
c(n) = E_{n-1}/n = (I_{n-1} + 2(n-1))/n
     = (2 n H_{n-1} - 4(n-1) + 2(n-1))/n
     = 2 H_{n-1} - 2(n-1)/n .
```

Let me sanity-check the small cases against the direct count. `n = 2`: `c(2) = 2 H_1 - 2(1)/2 = 2 - 1 = 1`, so two points are split apart by one cut and every point has depth 1 — right. `n = 3`: `c(3) = 2 H_2 - 2(2)/3 = 2(1.5) - 4/3 = 3 - 1.333... = 1.667`, i.e. `5/3`; and that should equal the average over the three points of the dome I already computed at `j = 2`, namely `(1.5 + 2 + 1.5)/3 = 5/3` — it does, so the BST cost and the tree-shape count are telling the same story from two directions. And `c(n) = 0` for `n <= 1` since a lone point needs no cut. For the implementation I'll approximate the harmonic number by `H(i) = ln(i) + gamma` with `gamma = 0.5772156649`, the cheap form used in practice, giving `c(n) = 2(ln(n-1) + gamma) - 2(n-1)/n` for `n > 2`. I should know how lossy that approximation is, so I compare it to the exact harmonic sum: at `n = 256` it gives `10.24477` against the exact `10.24869` — four digits — but at `n = 3` it gives `1.2074` against the exact `1.6667`, badly off. So the `ln + gamma` form is only safe at the scale where I'll actually use the normalization (sub-samples of a couple hundred), and I keep the exact small-n branches `c(2) = 1`, `c(n<=1) = 0` rather than letting the approximation handle them.

Now the score itself. I want short E(h) to map to "anomalous" and long E(h) to "normal," in a bounded, monotone way, with `c(n)` as the natural reference point. The clean choice is

```
s(x, n) = 2^( - E(h(x)) / c(n) ) .
```

Read off its behaviour and it's exactly what I want. When `E(h(x)) -> 0`, a point isolated almost immediately, the exponent goes to 0 and `s -> 1`. When `E(h(x))` moves toward the deepest possible value, around `n - 1`, the exponent becomes very negative and the score moves toward 0; in the finite tree it is still positive, and in the large-n limit it vanishes. And the calibration point is exact: when `E(h(x)) = c(n)`, the exponent is -1 and `s = 2^{-1} = 0.5`. So 0.5 is the natural average-depth anchor, points well above 0.5 are anomalous, points well below 0.5 are safely normal, and if everything scores around 0.5 the data has no distinct anomaly at all. The `2^{-(.)}` is monotone decreasing in E(h), bounded in (0, 1], and the `/c(n)` makes it comparable across tree sizes. That's the scoring rule.

Now the part that I'd never have guessed and that the swamping/masking pathologies force on me. The obvious thing is to build each tree from the *entire* data set — more data, better estimate, surely. Let me check that intuition against the two known failure modes. *Swamping*: when lots of normal points crowd in near an anomaly, it takes more and more cuts to fence the anomaly off from its near-normal neighbours, so the anomaly's path length creeps up toward a normal point's — it gets *swamped*, its short-path signal blurred. The more normal points participate, the worse. *Masking*: when the anomalies themselves form a *large, dense* cluster, that cluster behaves like a legitimate dense region — its members shield each other, each one now has plenty of nearby points, so it takes many cuts to isolate any of them and they all look normal. Both pathologies are driven by *too much data* sitting in the tree: too many interfering normals (swamping), or too big and dense an anomaly clump (masking). So building each tree from the full data set is feeding exactly the thing that breaks me — the obvious "more data" move makes both failure modes worse, not better.

Turn it around: build each tree from a small random *sub-sample* of the data, drawn without replacement, a fresh one per tree. A small sub-sample thins out the crowd of normals interfering near an anomaly, so the anomaly is once again surrounded by empty space and isolates in a cut or two. And a small sub-sample thins a dense anomaly cluster too, so its members no longer shield each other as strongly. Each tree's sub-sample contains a different handful of anomalies, or none, and averaging over the ensemble covers the data through many such views. This is the opposite of the "more data is better" instinct, and it falls straight out of swamping and masking once I see them as crowding problems. It also caps the cost: if every tree is built from a fixed sub-sample of size psi regardless of how big the data set is, then building each tree is governed by psi, not by n, and the whole forest scales linearly in the number of rows scored or sampled. How big should psi be? Big enough that a typical sub-sample still contains useful structure, small enough to reduce crowding and stay cheap. A small power-of-two default, `psi = 256`, is a practical fixed choice, and if the data set has fewer rows than that I just use all rows.

One more efficiency point, and it's not just efficiency. If I grow every tree all the way down until every sub-sampled point is alone, I'm spending most of my effort resolving the *deep* part of the tree. But the short paths are the informative ones; once a point has already travelled to ordinary depth, extra cuts mostly refine the normal bulk. The average depth scale grows like log of the sub-sample size, so I set a height limit `l = ceil(log2(psi))`. When a traversal of a test point runs into a node that I stopped growing, that node still holds `Size > 1` unisolated points; the path would have continued a bit further had I grown it, and I know exactly how much further on average — it's `c(Size)`, the same average-unsuccessful-search formula applied to the remaining sub-group. So when I terminate a path early at such a node, I return the actual edges traversed *plus* `c(Size)` as the estimate of the unbuilt remainder. The height limit skips the wasted work of fully resolving the bottom of every tree while preserving the average-depth accounting.

So the whole training stage is: pick the number of trees t and the sub-sample size psi, set the height limit `l = ceil(log2(psi))`, and for each of the t trees draw a sub-sample `X'` of psi points without replacement and grow an isolation tree on it down to height l. Growing one tree, on a group `X` at current depth e: if `e >= l` or the group has one or zero points or all its points are identical, make it an external node recording its `Size = |X|`; otherwise pick a random attribute q, pick a random split p uniformly between the min and max of q over `X`, send the points with `q < p` left and `q >= p` right, and recurse on each side at depth `e + 1`, storing q and p at the internal node. The number of trees t controls the variance of the averaged path length; a fixed default such as `t = 100` keeps that variance low while leaving the cost bounded.

The evaluation stage: for a test point x, run it down each of the t trees by following the stored split rules — at each internal node compare `x[q]` to the node's split value and go left if smaller, else right — counting edges, and when it hits an external node return `edges + c(node.Size)` (the adjustment is 0 when the node holds a single isolated point, since `c(1) = 0`). Average that over the t trees to get `E(h(x))`, then `s(x, psi) = 2^{-E(h(x))/c(psi)}`, where I normalize by `c(psi)` because every tree was grown from psi points. The score is in (0, 1], higher means more anomalous, exactly the ranking I needed and at a cost set by psi and t, not by n.

Let me write it the way it actually runs. I'll keep the height-limit recursion and the average-path-length adjustment, and at score time I'll work in the natural "average depth over trees" and exponentiate at the end. A library implementation can precompute each leaf's depth and its `_average_path_length(Size)` adjustment, but the accounting is the same.

```python
import numpy as np


def _average_path_length(n):
    # Average path length to an external node of a random binary tree on n points,
    # = average unsuccessful-search depth of a random BST = 2 H(n-1) - 2(n-1)/n.
    # scikit-learn uses H(i) ~= ln(i) + np.euler_gamma, with exact small cases.
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

    def __init__(self):
        self.left = self.right = None
        self.split_att = -1
        self.split_val = 0.0
        self.size = 0
        self.is_leaf = False


def _grow(X, e, height_limit, rng):
    # iTree(X, e, l): isolate points by random cuts until the height limit, |X|<=1,
    # or all rows identical -- short paths fall out for the few-and-different points.
    node = _Node()
    n = X.shape[0]
    if e >= height_limit or n <= 1:
        node.is_leaf = True
        node.size = n
        return node
    # stop if every row is identical (no attribute can split them)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    splittable = np.flatnonzero(maxs > mins)
    if splittable.size == 0:
        node.is_leaf = True
        node.size = n
        return node
    q = splittable[rng.integers(splittable.size)]   # random attribute
    p = rng.uniform(mins[q], maxs[q])               # random split in (min, max)
    left_mask = X[:, q] < p
    node.split_att, node.split_val = int(q), float(p)
    node.left = _grow(X[left_mask], e + 1, height_limit, rng)     # q < p
    node.right = _grow(X[~left_mask], e + 1, height_limit, rng)   # q >= p
    return node


def _path_length(x, node, e):
    # PathLength(x, T, e): edges to the terminating node, plus c(Size) for the
    # unbuilt remainder when a path stops early at a node holding Size > 1 points.
    if node.is_leaf:
        return e + _average_path_length(node.size)
    if x[node.split_att] < node.split_val:
        return _path_length(x, node.left, e + 1)
    return _path_length(x, node.right, e + 1)


class CustomAnomalyDetector:
    """Isolation-based anomaly scorer: an ensemble of random isolation trees,
    each grown on a small sub-sample; score = 2^(-mean path length / c(psi))."""

    def __init__(self, n_estimators=100, max_samples=256, random_state=0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples       # sub-sample size psi (small on purpose)
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        psi = min(self.max_samples, n)                  # "auto" = min(256, n)
        self.psi_ = psi
        self.c_psi_ = float(_average_path_length(psi))  # normalization constant
        height_limit = int(np.ceil(np.log2(max(psi, 2))))   # l = ceil(log2(psi)) ~ avg height
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=psi, replace=False)     # sub-sample WITHOUT replacement
            self.trees_.append(_grow(X[idx], 0, height_limit, rng))
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        m = X.shape[0]
        depths = np.zeros(m)
        for tree in self.trees_:
            for i in range(m):
                depths[i] += _path_length(X[i], tree, 0)
        mean_depth = depths / len(self.trees_)           # E(h(x)) over the ensemble
        if self.c_psi_ == 0.0:
            return np.ones(m)                            # sklearn's single-sample guard
        return 2.0 ** (-mean_depth / self.c_psi_)         # s(x, psi); higher = more anomalous
```

Every individual piece checked out, but the pieces interact, so before I trust the whole pipeline I want to run it on data where I know the answer. I draw 480 normal points from a 2D unit Gaussian and plant four obvious anomalies far out — `(8,8), (-7,6), (7,-7), (0,9)` — fit the detector with `t = 100, psi = 256`, and score all 484 points. `c(256)` comes out `10.245`, matching the constant I derived. The four planted anomalies score `0.832, 0.795, 0.808, 0.756`; the 480 normals average `0.424` with a maximum of `0.657`. So the anomalies all land above the `0.5` anchor and the normals all sit below it, exactly the partition the calibration predicted; the four planted points come out as the top four scored points, and the ranking AUC against the held-out labels is `1.0`. That's not a proof the method is good in general, but it confirms the code computes what the derivation says it should — short averaged path lengths for the far-out points, mapping through `2^{-E(h)/c(psi)}` to high scores — and that the sub-sampling and height limit don't break the ranking.

The causal chain, start to finish: profiling normality aims at the wrong target and the good rankings cost distances that don't scale, so I drop normality-modelling and exploit "few and different" directly — anomalies are *easier to isolate* under random partitioning, which makes their root-to-leaf path length short (checked: a far-out point's expected depth is computed and simulated to be shorter than a central point's); one random tree is noisy so I average path length over an ensemble; the dome-shape count over all random trees, verified against brute-force enumeration, shows fringe (extreme) points genuinely have shorter expected path length; an isolation tree is a random BST so its average path length is the classical unsuccessful-search cost `c(n) = 2H(n-1) - 2(n-1)/n`, which normalizes the raw depth into the bounded score `s = 2^{-E(h)/c(psi)}` anchored at 0.5; building each tree from a *small sub-sample without replacement* relieves swamping and masking (both too-much-data pathologies) and pins the cost to psi rather than n; and a height limit at the average height `ceil(log2 psi)` with a `c(Size)` adjustment skips the wasted work of fully resolving the deep, already-normal points.
