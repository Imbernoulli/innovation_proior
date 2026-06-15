# PaCMAP, distilled

PaCMAP (Pairwise Controlled Manifold Approximation Projection) is a nonlinear
dimension-reduction method for visualization that preserves **both** local and global
structure. It optimizes the low-dimensional embedding with three kinds of point pairs —
**neighbor**, **mid-near**, and **further** — under a simple separable pairwise loss,
using a three-phase weight schedule that builds global structure first and refines local
structure last. It is derived from explicit *principles of a good loss function* and a
proof that triplet losses are unnecessary: a separable edge loss with the right force
shapes obeys all the principles, and mid-near pairs supply the global structure that
neighbor/further pairs alone cannot.

## Problem it solves

Embed `N` points from `R^P` into `R^2` so a human can read both the local neighborhoods
(which points are near which) and the global arrangement (how neighborhoods sit relative
to each other), robustly to initialization, using no labels at fit time, and scalably
(working from a chosen sparse set of pairs, never all `N^2`).

## Key ideas

1. **Compare methods by their forces, not their losses.** For a triplet `(i,j,k)` where
   `i` attracts `j` and repulses `k`, plot the loss gradient over `(d_ij, d_ik)`. Good
   methods (t-SNE, UMAP, TriMap) share a force pattern captured by six **principles**:
   monotone (attract near, repulse far); once a far point is far enough, stop pushing it
   and focus on neighbors; push hard a far point that is too close; small force on an
   already-close neighbor; large force on a too-close far point; and the attractive
   force is **unimodal in `d_ij`** — strongest at moderate distance, vanishing for
   neighbors that are hopelessly far (limited capacity: give up on what you cannot
   preserve).

2. **Separable losses suffice — triplets are unnecessary (Proposition).** If
   `Loss = Σ Loss_attr(d_ij) + Σ Loss_rep(d_ik)` with `f = ∂Loss_attr/∂d_ij` and
   `g = -∂Loss_rep/∂d_ik` both nonnegative, unimodal, and vanishing at `0` and `∞`, then
   all six principles hold (each follows from the limits + unimodality; the per-triplet
   gradient components decouple as `f(d_ij)` and `-g(d_ik)`). So an edge loss does
   everything a triplet loss does for local structure.

3. **Principles only buy local structure; global structure needs forces on
   non-neighbors.** A 0-1 loss confined to near/too-close pairs can reach zero loss while
   destroying the global layout, because the relative distances among *moderately* far
   points never enter the objective. Concretely, t-SNE's force decomposes into
   `F_attract = 4 p_ij d_ij(1+d_ij^2)^{-1}` (tiny for non-neighbors, since `p_ij` is tiny)
   and `F_repulse = 4 q_ij d_ij(1+d_ij^2)^{-1}`. With
   `q_ij = a_ij/(B_ij+a_ij)` and `a_ij=(1+d_ij^2)^{-1}`, this is bounded by a `1/d_ij`
   envelope and, for fixed `B_ij`, decays as `Θ(1/d_ij^3)` with derivative
   `Θ(1/d_ij^4)` — so far points are indistinguishable. UMAP is the same; TriMap's
   global edge comes from PCA init, not its triplets (removing random triplets barely
   changes it; removing PCA init ruins it).

4. **Mid-near pairs.** Add a third pair type at *moderate* distance, attracted, so the
   objective itself carries global structure. Each is the **second-closest of 6 uniformly
   sampled points** (a cheap moderate-distance draw, no full ranking). Neighbors are the
   `n_NB` nearest by a **scaled distance** `||x_i - x_j||^2/(σ_i σ_j)`, `σ_i` = average
   distance to the 4th-6th Euclidean neighbors (used only for selection); further pairs
   are random non-neighbors.

5. **Three-phase schedule (coarse-to-fine).** Because the forces have narrow working
   zones, global structure placed early stays put while local structure is refined. So:
   phase 1 build global structure (mid-near weight high, decaying); phase 2 stabilize;
   phase 3 refine local (mid-near off). This also avoids flinging neighbors so far early
   that their attraction saturates (false clusters).

## Final algorithm

Inputs: `n_NB` (default 10), `MN_ratio` (0.5 → `n_MN = round(MN_ratio·n_NB) = 5`), `FP_ratio`
(2.0 → `n_FP = 20`), `n_iters = (100, 100, 250)`, PCA init.

Pairs (per point `i`):
- **Neighbors:** over-fetch `min(n_NB+50, N-1)` Euclidean neighbors, re-rank by
  `d²_select = ||x_i - x_j||^2/(σ_i σ_j)`, keep top `n_NB`.
- **Mid-near:** `n_MN` pairs, each the 2nd-closest of 6 random samples.
- **Further:** `n_FP` random non-neighbors.

Loss, with `d̃_ab = ||y_a - y_b||^2 + 1`:
```
Loss = w_NB · Σ_neighbors  d̃_ij/(10 + d̃_ij)
     + w_MN · Σ_mid-near    d̃_ik/(10000 + d̃_ik)
     + w_FP · Σ_further     1/(1 + d̃_il)
```
Forces (gradient on `y_i`, attractive terms point descent toward `j`, repulsive away):
```
neighbor   :  + w_NB · 20/(10 + d̃)^2 · (y_i - y_j)
mid-near   :  + w_MN · 20000/(10000 + d̃)^2 · (y_i - y_j)
further    :  - w_FP · 2/(1 + d̃)^2 · (y_i - y_j)
```
(derived: `d/dd̃ [ d̃/(c+d̃) ] = c/(c+d̃)^2`, chain rule `∂d̃/∂y_i = 2(y_i-y_j)`;
`d/dd̃ [ 1/(1+d̃) ] = -1/(1+d̃)^2`.)

Weight schedule (`itr` zero-indexed, as in the implementation; phase lengths 100/100/250):
```
0   <= itr < 100:  w_NB = 2,  w_MN = (1 - itr/100)·1000 + (itr/100)·3,  w_FP = 1
100 <= itr < 200:  w_NB = 3,  w_MN = 3,                                  w_FP = 1
200 <= itr < 450:  w_NB = 1,  w_MN = 0,                                  w_FP = 1
```
The first phase starts at `w_MN = 1000`, anneals it toward `3`, and the second phase
sets it exactly to `3`.

Init: PCA (scaled by 0.01 to stay inside the force working zones). Random initialization
uses `1e-4 * N(0, I)`; the mid-near phase is the mechanism intended to create global
structure rather than inherit it. Optimizer: Adam, `lr = 1.0`, `betas = (0.9,
0.999)`, `eps = 1e-7`, 450 iterations. If `P > 100`, PCA-preprocess to 100 dims first.

## Why the design choices

- **`d̃ = d^2 + 1` numerator** (from t-SNE's Student-t): slow growth near 0 → small force
  on already-close neighbors; the rational saturates → give-up tail. Raw `d^2` would not
  saturate (violates the unimodality principle).
- **Constants 10 / 10000 / 1**: working-zone tuners. Neighbor `10` → narrow local zone;
  mid-near `10000` → a *much wider* zone so mid-near attraction persists across moderate
  distances and organizes global structure; further `1` → repulsion only when too close.
  Other functions with the same characteristics would also work; what is load-bearing is
  separability, the principle-obeying shapes, and a *separate* mid-near attraction.
- **Mid-near = 2nd of 6**: a cheap order statistic landing in the lower-middle of the
  distance distribution (moderate, not nearest, not random-far) without ranking.
- **Scaled distance for neighbor selection only**: neighborhoods differ in scale across
  the manifold; `σ_i σ_j` normalizes them. Not used in the loss, which acts on embedded
  distances.
- **Three phases**: simulated-annealing/early-exaggeration spirit, but exaggerating
  *mid-near* (global) pull, not neighbor pull.
- **Adam, `lr = 1`**: the three force terms span very different magnitudes (`w_MN` starts
  at 1000) and change across phases; per-coordinate adaptive steps keep the effective step
  sane. Forces are bounded rationals, so a large base rate is safe.

## Working code

Compact numpy/scipy/sklearn implementation of the same core algorithm; exact sklearn
kNN is fine for `N <= 5000`:

```python
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def _draw_excluding(n, size, rng, banned):
    banned = set(int(x) for x in banned)
    out = []
    while len(out) < size:
        k = int(rng.integers(n))
        if k not in banned:
            out.append(k)
            banned.add(k)
    return np.asarray(out, dtype=np.int64)


def _select_components(X, n_neighbors, n_MN, n_FP, rng):
    n = X.shape[0]
    # neighbor pairs: nearest by scaled distance ||x_i-x_j||^2 / (sig_i sig_j)
    n_extra = min(n_neighbors + 50, n - 1)
    dist, nbrs = NearestNeighbors(n_neighbors=n_extra + 1).fit(X).kneighbors(X)
    dist, nbrs = dist[:, 1:], nbrs[:, 1:]                       # drop self
    sig = np.maximum(dist[:, 3:6].mean(axis=1), 1e-10)          # avg dist to 4th-6th NN
    scaled = (dist ** 2) / (sig[:, None] * sig[nbrs])
    order = np.argsort(scaled, axis=1)[:, :n_neighbors]
    rows = np.repeat(np.arange(n), n_neighbors)
    nb = np.stack([rows, np.take_along_axis(nbrs, order, axis=1).ravel()], 1)

    # mid-near pairs: 2nd-closest of 6 uniform samples
    mn = np.empty((n * n_MN, 2), dtype=np.int64); t = 0
    for i in range(n):
        picked_for_i = []
        for _ in range(n_MN):
            s = _draw_excluding(n, 6, rng, [i] + picked_for_i)
            d = ((X[s] - X[i]) ** 2).sum(1)
            picked = int(s[np.argsort(d)[1]])
            picked_for_i.append(picked)
            mn[t] = (i, picked); t += 1
    mn = mn[:t]

    # further pairs: random non-neighbors
    nbr_of = nb[:, 1].reshape(n, n_neighbors)
    fp = np.empty((n * n_FP, 2), dtype=np.int64); t = 0
    for i in range(n):
        banned = set(nbr_of[i].tolist()) | {i}
        for _ in range(n_FP):
            k = int(_draw_excluding(n, 1, rng, banned)[0])
            banned.add(k)
            fp[t] = (i, k); t += 1
    return nb, mn, fp[:t]


def _grad(Y, nb, mn, fp, w_nb, w_mn, w_fp):
    grad = np.zeros_like(Y)
    for pairs, c, w in ((nb, 10.0, w_nb), (mn, 10000.0, w_mn)):   # attraction
        i, j = pairs[:, 0], pairs[:, 1]
        diff = Y[i] - Y[j]
        dt = (diff ** 2).sum(1) + 1.0                            # d~ = ||y_i-y_j||^2 + 1
        upd = (w * (2.0 * c) / (c + dt) ** 2)[:, None] * diff
        np.add.at(grad, i, upd); np.add.at(grad, j, -upd)
    i, j = fp[:, 0], fp[:, 1]                                    # repulsion
    diff = Y[i] - Y[j]
    dt = (diff ** 2).sum(1) + 1.0
    upd = (w_fp * 2.0 / (1.0 + dt) ** 2)[:, None] * diff
    np.add.at(grad, i, -upd); np.add.at(grad, j, upd)
    return grad


def _weights(itr, p1, p2):
    if itr < p1:                                                # phase 1: build global
        return 2.0, (1 - itr / p1) * 1000.0 + (itr / p1) * 3.0, 1.0
    if itr < p1 + p2:                                           # phase 2: stabilize
        return 3.0, 3.0, 1.0
    return 1.0, 0.0, 1.0                                        # phase 3: refine local


def pacmap_fit_transform(X, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
                         num_iters=(100, 100, 250), lr=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] > 100:                                        # denoise / speed kNN
        X = X - X.mean(0)
        X = TruncatedSVD(n_components=100, random_state=random_state).fit_transform(X)
        pca_solution = True
        pca_init = None
    else:
        X = X - X.min()
        X = X / max(X.max(), 1e-12)
        X = X - X.mean(0)
        pca_solution = False
        pca_init = PCA(n_components=n_components, random_state=random_state).fit(X)
    n_MN = int(round(n_neighbors * MN_ratio))
    n_FP = int(round(n_neighbors * FP_ratio))
    nb, mn, fp = _select_components(X.astype(np.float32), n_neighbors, n_MN, n_FP, rng)

    Y = (0.01 * X[:, :n_components] if pca_solution             # PCA init, scaled small
         else 0.01 * pca_init.transform(X))
    Y = np.ascontiguousarray(Y, dtype=np.float64)

    beta1, beta2, eps = 0.9, 0.999, 1e-7                        # Adam
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    p1, p2, _ = num_iters
    for itr in range(sum(num_iters)):
        w_nb, w_mn, w_fp = _weights(itr, p1, p2)
        g = _grad(Y, nb, mn, fp, w_nb, w_mn, w_fp)
        lr_t = lr * np.sqrt(1 - beta2 ** (itr + 1)) / (1 - beta1 ** (itr + 1))
        m += (1 - beta1) * (g - m)
        v += (1 - beta2) * (g ** 2 - v)
        Y -= lr_t * m / (np.sqrt(v) + eps)
    return Y
```
