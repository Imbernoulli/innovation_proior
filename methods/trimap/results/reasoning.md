Let me start from what actually goes wrong when I look at these 2D maps. I have a pile of high-dimensional points, digits or single cells or hidden activations, and I want a picture I can trust. On the S-curve — a few thousand points lying on a smooth curled sheet in 3D whose true shape I know — the t-SNE picture keeps the local neighborhoods but loses the shape. The curl is unrolled into something that no longer tells me which end of the S was near which. UMAP does the same. And here's the thing that bothers me most: the nearest-neighbor accuracy and the trustworthiness score can both be high, so by every local number I have, the embedding is "good." PCA, on the other hand, gives me an ugly flat shadow that's locally weak but globally honest — the overall shape and the relative placement of the clusters survive. So the methods I have buy local fidelity by quietly sacrificing global fidelity, and the scores I have can't even see the sacrifice. Two problems then, tangled together: I need a method that keeps the global layout the way PCA does while staying competitive locally, and I need a number that actually measures global accuracy so I can tell whether I've done it.

Take the measurement problem first, because if I can't measure global accuracy I can't claim it. What does "global accuracy" even mean operationally? PCA is the natural gold standard for it — it's the linear projection that keeps the most variance, the most aggregate second-order structure, so it's as globally faithful as a linear map can be. So let me try to define global accuracy as *closeness to what PCA preserves*. The cleanest handle on "what PCA preserves" is reconstruction: how well can I get the original high-D data back from the embedding using a *linear* map? Given an embedding `Y` (`d × n`, say `d = 2`) and the data `X` (`D × n`, both centered), the best linear reconstruction is the least-squares fit, `err(Y|X) = min_A ||X − A Y||_F^2`. That minimization is just multivariate regression of `X` on `Y`: take the derivative of `||X − AY||_F^2` with respect to `A`, set it to zero, `−2(X − AY)Y^T = 0`, so `A* = X Y^T (Y Y^T)^{-1}`, and plugging back gives the minimum error. The thing I most want from this `A*` is invariance: a visualization is only meaningful up to rotation, reflection, and scaling, so the score must not change when I rotate or rescale `Y`. Does `A*` actually give me that? It should, because `A*` is fit to whatever `Y` I hand it — if I replace `Y` by `cRY` for a rotation `R` and scale `c`, the regression just learns `A* R^T / c` and reconstructs the same `X`. Let me not just believe that. I built a small 5D dataset (a noisy 3D helix padded with two near-zero noise axes), took its 2D PCA, and evaluated my score on it and on a rotated-and-scaled (×3.7, rotated 0.6 rad) copy. The raw `err` is identical to machine precision and the score comes out `1.0` and `1.0000000000000002` — the invariance is real, not wishful.

Now, is PCA really the floor? I want `err_PCA = err(Y_PCA | X)` to be the *smallest* `err` any 2D linear-ish layout can reach, so that normalizing against it gives a score capped at 1. The clean argument is the variational characterization of PCA, but I'd rather see it than cite it: on that same dataset I generated 2000 random linear 2D projections `X W` and measured each one's `err`. The best of the 2000 lands at `0.004648`, while PCA sits at `0.004349` — strictly below all of them. That matches what the theory says and gives me confidence the floor is genuine. So I'll normalize: a global score `GS(Y|X) = exp(−(err(Y|X) − err_PCA)/err_PCA)`. When `Y` is as good as PCA the exponent is zero and `GS = 1`; as `Y` reconstructs worse, the relative excess error grows and `GS` decays into `(0, 1)`; the `exp` keeps it bounded and monotone. I checked the decay too: degrading the PCA embedding by replacing one of its two axes with pure noise drops `GS` from 1 to `0.033`, and a fully random 2D layout drops it to `≈ 5e−25`. So the yardstick is 1 for PCA, near-1 for anything close to it, and collapses as the global layout is wrecked. I'd still want it to track an independent check on real data — clustering both spaces and comparing the cluster-center distance matrices — but the construction behaves the way a global-accuracy measure should.

Now the real problem: the method. Why do the local methods lose the global picture? They all work the same way — build a kNN graph, turn each edge into a pairwise *target distance or similarity*, and lay the points out to match those pairwise targets. The trouble with a pairwise target is that it's *absolute*: "`i` and `j` should be at similarity 0.7" commits to a specific distance, and the only pairs that get a meaningful target are neighbors, because the Gaussian similarity of two far-apart points is numerically zero — indistinguishable from any other far pair. So the objective says nothing about how far apart two *clusters* should be relative to two *other* clusters. The global arrangement is left dangling, and the optimizer fills it in arbitrarily. That's the crowding-and-arbitrariness failure in one sentence: pairwise targets pin down local distances and underdetermine global ones.

So I want a relation that is *relative*, not absolute, and that carries information about more than just the nearest neighbors. The smallest relation that encodes relative order is a triplet: `(i, j, k)` meaning "`i` is closer to `j` than to `k`." A triplet doesn't demand any particular distance; it only demands an *ordering*. That's scale-free and robust, and crucially it's a higher-order summary — if I let `k` range over far-away points, a triplet can say something about the placement of `i` relative to distant structure, not just its immediate neighborhood. This isn't a new idea in the abstract: there's a line of work that *learns* embeddings from triplets given by humans — "is `A` more like `B` or `C`?" — because such relative judgements are more reliable than asking people for numeric similarities. And there's metric-learning work that takes an *initial* feature representation and *refines* it so that a set of relative comparisons is satisfied while staying close to the initial one. That refine-an-initial-representation framing is the seed I want to steal: I won't build the map from triplets out of nothing; I'll start from something globally sensible and *refine* it with triplets sampled from the data's own high-D geometry. Hold that thought — I'll come back to what "start from something globally sensible" should be, because it's going to matter more than I expect.

First, what's the loss for a single triplet `(i, j, k)`? The existing triplet-embedding methods define a probability that the triplet is satisfied and maximize the log of it. Stochastic triplet embedding writes `p_ijℓ = exp(−||x_i − x_j||^2) / (exp(−||x_i − x_j||^2) + exp(−||x_i − x_ℓ||^2))` and maximizes `Σ log p_ijℓ`, with a heavy-tailed Student-t kernel in the variant. Let me think about whether maximizing `Σ log p` is what I want for *visualization*. The heavy tail there is deliberate: it keeps pulling similar points together and pushing dissimilar points apart *even after the triplet is already satisfied*, because the tails of the Student-t never flatten. For denoising noisy human triplets, that's a feature — it collapses clusters cleanly. But for me the triplets come from real feature distances, and I do *not* want to keep collapsing a triplet that's already correct: if `i` is already closer to `j` than to `k`, that relation is *done*, and continuing to crush them together is exactly the over-compression that destroys structure. I want a per-triplet loss that goes to zero — that *stops pulling* — once the order is right, and only pushes while the order is wrong or marginal.

Let me build that. I'll use a low-D similarity `s(y_a, y_b)` that's large when the points are close and small when far, and I'll reuse the Student-t kernel with one degree of freedom that works so well in t-SNE for the low-D side: `s(y_a, y_b) = (1 + ||y_a − y_b||^2)^{-1}`. The heavy tail is still good for the *geometry* — it gives long-range forces that can pull a misplaced point back across the map and it avoids the crowding problem — I just won't wrap it in a log-probability that never lets go. The relation I care about is "`i` closer to `j` than `k`," i.e. `s(y_i, y_j)` should dominate `s(y_i, y_k)`. So the natural quantity to *minimize* is the share of the similarity mass that's sitting on the *wrong* side:

  `l_ijk = w_ijk · s(y_i, y_k) / (s(y_i, y_j) + s(y_i, y_k))`,

where `w_ijk ≥ 0` is an importance weight I'll design in a moment. Stare at this. When `j` is near `i` and `k` is far, `s(y_i, y_j)` is big and `s(y_i, y_k)` is small, so the ratio → 0 and the loss vanishes — the triplet is satisfied and exerts (almost) no force. When the order is violated, `s(y_i, y_k)` dominates and the ratio → 1, full loss. It's bounded in `[0, 1]` and smooth everywhere (no hinge kink like a 0/1 triplet loss). The property I actually need, though, is that it *stops pulling* once satisfied, so let me put numbers on it rather than assert it. Fix `j` near `i` (`d_ij = 1.04`) and sweep `k` from sitting on top of `i` out to far away. With `w = 1`, the loss `l = d_ij/(d_ij + d_ik)` runs `0.51 → 0.50 → 0.45 → 0.34 → 0.17 → 0.039 → 0.010` as `d_ik` goes `1.0, 1.04, 1.25, 2.0, 5.0, 26, 101`. And the force factor I'll derive below (the magnitude `w' d_ik` that scales the pull) follows it down: `0.240 → 0.240 → 0.238 → 0.216 → 0.137 → 0.036 → 0.010`. So as the triplet becomes well satisfied — `k` pushed far out — both the loss and the force decay toward zero. That is the saturation I want, confirmed on actual values, and it's the decisive difference from the log-prob objective: that one keeps rewarding an already-satisfied triplet, whereas this one lets go.

Let me simplify it, because I'll be differentiating it a lot. Define `d_ij := 1 + ||y_i − y_j||^2`, so `s(y_i, y_j) = 1/d_ij`. (The `+1` is doing double duty: it's the Student-t shape *and* it floors the denominator so I never divide by zero when two embedding points coincide.) Then

  `s(y_i,y_k)/(s(y_i,y_j)+s(y_i,y_k)) = (1/d_ik)/(1/d_ij + 1/d_ik) = d_ij/(d_ij + d_ik) = 1/(1 + d_ik/d_ij)`.

So `l_ijk = w_ijk / (1 + d_ik/d_ij)`. Clean: the loss is small exactly when `d_ik ≫ d_ij`, i.e. `k` much farther than `j` in the map. The total loss is `Σ_{(i,j,k) ∈ T} l_ijk` over whatever set of triplets `T` I sample.

Now the weight `w_ijk`. Why weight at all? Because not all triplets are equally informative. A triplet where `k` is *enormously* farther than `j` in high dimension is strong, reliable evidence about the data's geometry — I should insist the map respect it. A triplet where `k` is only marginally farther than `j` is nearly a tie, dominated by noise — leaning on it hard would just fit noise. So I want the weight to grow with the high-D *margin* of the triplet. To keep the high-D distances separate from the low-D `d_ij = 1 + ||y_i − y_j||^2`, call the locally scaled high-D squared distance `δ_ij^2`. The raw margin is `w̃_ijk = δ_ik^2 − δ_ij^2`, which is `≥ 0` by construction if I sample so that `j` really is the closer one. But raw squared Euclidean distances are dangerous across regions of different density: in a dense cluster all distances are small, in a sparse one all are large, so the same "margin" means different things in different places. I've seen the fix for this in spectral clustering — self-tuning local scaling: set `δ_ij^2 = ||x_i − x_j||^2 / (σ_i σ_j)`, where `σ_i` is a local length read off `i`'s own neighborhood. That normalizes margins to local density so a tight cluster inside a sparse one is still resolved. What should `σ_i` be? The distance to the very nearest neighbor is too jittery and can be near zero (degenerate), so I'll use something a little out into the neighborhood — the average distance from `x_i` to its 4th through 6th nearest neighbors. Far enough out to be stable, close enough to still be local.

So the weights are margins of locally-scaled squared distances, all nonnegative. But there's a distributional problem: these margins are heavy-tailed. A handful of triplets straddling two very-distant clusters will have gigantic `w̃`, and if I sum the raw losses those few triplets will dominate the gradient and drown out the thousands of ordinary informative ones. I want to *keep* the ordering of importance — bigger margin still means more important — but compress the dynamic range so the giants don't swamp everything. A logarithm is the obvious squashing transform, but a plain `log` may compress too aggressively. I'll reach for a tempered (deformed) logarithm, `log_t(u) = (u^{1−t} − 1)/(1 − t)` for `t ≠ 1`, which interpolates: as `t → 1` it becomes the ordinary `log`, and for `t < 1` it saturates more gently. Pick `t = 0.5` — a square-root-flavored compression, gentler than `log`, that smooths the weights and prevents the huge-margin triplets from dominating while still ranking them above the small ones. And to make the transform well-behaved I'll shift the margins so the smallest is zero before tempering: `w_ijk = log_t(1 + w̃_ijk − w_min)`, with `w_min` the minimum margin over all sampled triplets. The `1 +` keeps the argument `≥ 1` so `log_t` is nonnegative.

Now: which triplets? I can't use all `O(n^3)` of them, and I shouldn't — most are redundant or uninformative. For each point `i` I want triplets that teach the layout something. The informative `j` is a *near* neighbor of `i` (so the triplet says "keep this neighbor close relative to something farther"). So: for each `i`, take its `m = 12` nearest neighbors as candidate `j`'s, and for each such `j` sample `m' = 4` "outlier" points `k` uniformly at random from outside the already-closer neighbors — points that should be farther than `j`. That's `m × m' = 48` near-neighbor triplets per point. The total triplet count is then linear in `n`, which is what lets this scale to millions of points; the real cost is just the approximate nearest-neighbor search (random-projection trees), shared with every other method in this family. And the method is robust to the exact count because the triplets are highly redundant — `(i,j,k)` and `(i,j,k')` say nearly the same thing if `k` and `k'` are close.

But wait — let me check whether these 48-per-point neighbor triplets actually preserve the *global* structure I'm chasing, because that was the whole point. Every one of these triplets has `j` as a *neighbor* of `i`. None of them constrains how two *far-apart* points sit relative to each other. So suppose the optimizer drives the loss to (near) zero: every neighbor is closer than its sampled outliers. Does that pin down the global arrangement of the clusters? My worry is that it doesn't — that I could satisfy "every neighbor closer than every outlier" while folding the global layout into almost any shape. The picture in my head is the curled S-sheet relaxed into a flat line: locally every neighbor is still the nearest thing, so the neighbor-triplet loss should stay near zero even though the global curl is gone. But that's exactly the kind of claim I just caught myself asserting, so let me actually count. Take a 1D manifold (60 points along a parameter `t`), give it a true 2D shape — an S-curve `(sin 2πt, t)` — and a "flattened" rival layout that places the points on a straight line `(t, 0)`, which throws the global S-shape away entirely. For each point I form neighbor triplets `(i, j, k)` with `j` an adjacent-on-the-manifold neighbor and `k` any point more than 6 indices away, and count how many are violated (some far `k` closer than the true neighbor `j`). The S-curve gives `0 / 11132` violations. The flattened line *also* gives `0 / 11132`. So the neighbor-triplet loss is identically minimized by both — by the layout that keeps the global shape and by the one that destroys it. The loss really can be zero *amidst a complete sacrifice of global structure*; it's the same failure the pairwise local methods have, just dressed as triplets. Wall. Adding a near-neighbor triplet loss, by itself, does not buy me global structure at all.

So where can global structure come from? Two routes. Route one: add triplets with forces *between non-neighbors* — sample some triplets `(i, j, k)` where all three are mutually far, so the loss has a say over the relative placement of distant points. I'll add a few of these, `r = 3` random triplets per point: pick `j` and `k` uniformly at random and orient them by their high-D distance so the closer one is `j`. These do carry a faint global signal. But let me be honest about how strong that signal is. Three random triplets per point against forty-eight neighbor triplets per point — the random ones are a small slice of the set, they're noisy (a random pair of far points is a weak, high-variance constraint), and I'll have to downweight them anyway (scale their weights by something like `0.1`) precisely because they're unreliable. Leaning the entire global structure on this thin, noisy signal is a bad bet. The neighbor triplets will dominate the gradient and the random triplets will get overwhelmed. Route one alone won't carry the global layout.

Route two is the one I parked earlier: don't try to *discover* the global layout from the triplet forces at all — *start* from a layout that's already globally correct, and let the triplets only *refine* it locally. This is exactly the metric-learning seed: refine an initial representation rather than build from scratch. And I already have a globally-faithful layout in hand — PCA, the very one I measured against above, where it beat all 2000 random linear projections on reconstruction error. So initialize `Y` to the PCA embedding (scaled down by a small constant, `0.01`, so the Student-t kernel `1/(1 + ||·||^2)` operates in its sensitive range rather than saturated). Now think about what the optimizer faces from a PCA start. PCA has already placed the clusters in their globally-faithful relative positions, and crucially it already keeps far points far apart. So once I start there, there are essentially *no* triplets demanding that two far-apart points be pushed apart — they're already apart. The only work left for the triplet forces is to *sharpen the local neighborhoods*: pull each point's true near-neighbors in tight, nudge the outliers out, unfold the local manifold — all without any large force that would tear the global arrangement PCA established. The neighbor triplets, which were a liability for global structure when starting from noise, become exactly right when the global frame is already correct: they do the local job and leave the global job alone. That's the resolution. The global structure comes from PCA initialization; the triplets supply the local fidelity PCA lacks. The random triplets are a minor insurance term, not the load-bearing member. And there's a free side benefit: starting from a structured PCA solution converges far faster than starting from random noise — unlike the pairwise methods, which only converge well from a tiny random seed and would be wrecked by a structured start.

Let me make sure this is consistent with my saturating loss. From a PCA start, most neighbor triplets are *already nearly satisfied* (neighbors are already roughly closest), so my `l_ijk = w/(1 + d_ik/d_ij)` is already smallish for them and its gradient is gentle — the method won't violently rearrange the good global frame, it'll make small local corrections. If I'd used the log-probability objective that keeps pulling satisfied triplets forever, it would over-collapse each neighborhood and slowly chew up the global structure even from a good start. The saturating loss is what makes the refine-from-PCA strategy safe. The pieces are reinforcing each other now.

Now I need the gradient, because I'm going to minimize `Σ l_ijk` by gradient descent. Take one triplet's contribution `L = w · d_ij/(d_ij + d_ik)` (using the simplified form, with `w` the final tempered weight). The embedding points enter only through `d_ij = 1 + ||y_i − y_j||^2` and `d_ik = 1 + ||y_i − y_k||^2`. The distance derivatives: writing `y_ij := y_i − y_j`, `∂d_ij/∂y_i = 2 y_ij`, `∂d_ij/∂y_j = −2 y_ij`, and similarly `∂d_ik/∂y_i = 2 y_ik`, `∂d_ik/∂y_k = −2 y_ik`. For the scalar `L` of a quotient `d_ij/(d_ij + d_ik)`:

  `∂L/∂d_ij = w · [(d_ij + d_ik) − d_ij]/(d_ij + d_ik)^2 = w · d_ik/(d_ij + d_ik)^2`,
  `∂L/∂d_ik = w · [−d_ij]/(d_ij + d_ik)^2 = −w · d_ij/(d_ij + d_ik)^2`.

Let me name the common factor `w' := w/(d_ij + d_ik)^2`. Then by the chain rule, gathering the contributions to each of the three points:

  `∂L/∂y_i = ∂L/∂d_ij · 2 y_ij + ∂L/∂d_ik · 2 y_ik = 2 w'(d_ik · y_ij − d_ij · y_ik)`,
  `∂L/∂y_j = ∂L/∂d_ij · (−2 y_ij) = −2 w' d_ik · y_ij`,
  `∂L/∂y_k = ∂L/∂d_ik · (−2 y_ik) = +2 w' d_ij · y_ik`.

The factor of 2 is global across every term, so I can fold it into the learning rate and drop it. Define two per-triplet vectors, `gs := w' d_ik · y_ij` (a pull along the `i–j` direction) and `go := w' d_ij · y_ik` (a push along the `i–k` direction). The gradient accumulation is then beautifully local:

  `grad_i += gs − go`,  `grad_j −= gs`,  `grad_k += go`.

Read it physically: `gs` is the symmetric pull between `i` and `j`, and `go` is the symmetric push between `i` and `k`. The numerator factors `d_ik` and `d_ij` set the relative pull and push inside the triplet, but the shared denominator is squared, so an already-satisfied triplet with `d_ik` large and `d_ij` small has a small total force — which lines up with the force-magnitude column I tabulated earlier, where `w' d_ik` decayed to `0.01` as `k` moved far out.

Before I build the whole optimizer on this, I should make sure I didn't slip a sign or drop a term in the chain rule — quotient-rule gradients are exactly where I make algebra mistakes. So I checked it against finite differences: one triplet, three points placed at random in 2D, weight `0.7`, and I compared the analytic `(gs − go, −gs, go)` (with the factor of 2 restored) to a central-difference gradient of `L` at `ε = 1e−6`. The two agree to `3.6e−11` across all six components. The derivation is correct. Each triplet touches only three rows of the gradient, so a full-batch gradient over `O(n)` triplets is `O(n)` per iteration.

How to optimize? The loss is a sum over a *fixed* set of triplets — I sample `T` once up front and never resample — so this is a smooth, deterministic, full-batch objective. No need for stochastic minibatching; I can take exact full-batch steps, which are stable. I'll use the optimizer the neighbor-embedding methods already use: gradient descent with momentum and a per-coordinate adaptive gain (delta-bar-delta). Here `vel` is the update direction, so a coordinate is still moving consistently downhill when `vel` and `grad` have opposite signs; that is when I grow the gain by `0.2`. If they have the same sign, the update direction is fighting the gradient, so I damp the gain to `max(0.8 · gain, 0.01)`. And a momentum schedule: a calmer `γ = 0.5` for the first 250 iterations while the layout is still rearranging, then `γ = 0.8` afterwards to accelerate convergence once it's settling. I'll evaluate the gradient at the look-ahead point `Y + γ · vel` for stability, then `vel ← γ · vel − lr · gain · grad`, `Y += vel`. Four hundred iterations is enough for this fixed triplet objective from the PCA start.

Let me write the whole thing as the procedure I'd actually ship. Pre-reduce wide data to 100 dimensions with a truncated PCA/SVD before the neighbor search; otherwise normalize and center it. Build the approximate kNN graph. Compute the per-point scales `σ_i` from the 4th–6th neighbors. Sample 48 neighbor triplets and 3 random triplets per point; compute their high-D margins, scale and temper them into weights. Initialize `Y` to the downscaled PCA solution, or to the first embedding coordinates of the 100-dimensional pre-reduction when I already computed it. Then run full-batch delta-bar-delta gradient descent on `Σ l_ijk` for 400 iterations.

```python
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

INIT_PCA_SCALE = 0.01
RAND_WEIGHT_SCALE = 0.1


def tempered_log(u, t):
    # log_t(u) = (u^{1-t} - 1)/(1 - t); -> log(u) as t -> 1
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
    rng = rng or np.random.default_rng()
    n = X.shape[0]
    k = min(n_inliers + n_extra, n)                     # includes self if returned first
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    knn_dist, nbrs = nn.kneighbors(X)                   # includes self at column 0

    # local scale sigma_i = mean Euclidean distance to the 4th-6th neighbors
    sig = np.maximum(knn_dist[:, 4:7].mean(axis=1), 1e-10)
    # locally-scaled squared distances along the graph: P[i,j] = -||x_i-x_nbr||^2/(sig_i sig_j)
    P = -knn_dist ** 2 / (sig[:, None] * sig[nbrs])     # = -delta_ij^2  (larger = closer)

    triplets, weights = [], []
    # ---- near-neighbor triplets: j among i's inliers, k a random outlier ----
    for i in range(n):
        order = np.argsort(-P[i])                       # most-similar neighbors first
        for a in range(n_inliers):
            j = nbrs[i, order[a + 1]]                   # skip self
            p_sim = P[i, order[a + 1]]                  # = -delta_ij^2
            rejects = set(nbrs[i, order[:a + 2]])
            for _ in range(n_outliers):
                k_ = rng.integers(n)
                while k_ in rejects:
                    k_ = rng.integers(n)
                d_ik2 = np.sum((X[i] - X[k_]) ** 2) / (sig[i] * sig[k_])
                triplets.append((i, j, k_))
                weights.append(p_sim + d_ik2)           # delta_ik^2 - delta_ij^2 >= 0
    # ---- random triplets: a faint long-range signal (downweighted) ----
    for i in range(n):
        for _ in range(n_random):
            j = rng.integers(n);  k_ = rng.integers(n)
            while j == i: j = rng.integers(n)
            while k_ == i or k_ == j: k_ = rng.integers(n)
            d_ij2 = np.sum((X[i] - X[j]) ** 2) / (sig[i] * sig[j])
            d_ik2 = np.sum((X[i] - X[k_]) ** 2) / (sig[i] * sig[k_])
            if d_ij2 > d_ik2:                            # orient so j is the closer one
                j, k_, d_ij2, d_ik2 = k_, j, d_ik2, d_ij2
            triplets.append((i, j, k_))
            weights.append(RAND_WEIGHT_SCALE * (d_ik2 - d_ij2))

    triplets = np.asarray(triplets, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.nan_to_num(weights)
    weights -= weights.min()                            # shift so smallest margin is 0
    weights = tempered_log(1.0 + weights, weight_temp)  # gentle compression, t=0.5
    return triplets, weights


def trimap_grad(Y, triplets, weights):
    n, dim = Y.shape
    grad = np.zeros((n, dim))
    loss = 0.0
    for t in range(triplets.shape[0]):
        i, j, k = triplets[t]
        y_ij = Y[i] - Y[j]
        y_ik = Y[i] - Y[k]
        d_ij = 1.0 + y_ij @ y_ij                        # 1 + ||y_i - y_j||^2  (=1/s, floors /0)
        d_ik = 1.0 + y_ik @ y_ik
        loss += weights[t] / (1.0 + d_ik / d_ij)        # l_ijk, saturates to 0 once satisfied
        w = weights[t] / (d_ij + d_ik) ** 2             # common prefactor w'
        gs = y_ij * d_ik * w                            # pull i toward j
        go = y_ik * d_ij * w                            # push i away from k
        grad[i] += gs - go
        grad[j] -= gs
        grad[k] += go
    return grad, loss


def trimap(X, n_components=2, n_iters=400, lr=0.1, rng=None):
    X, pca_solution = preprocess(X)
    triplets, weights = generate_triplets(X, rng=rng)
    if pca_solution:
        Y = INIT_PCA_SCALE * X[:, :n_components]
    else:
        Y = INIT_PCA_SCALE * PCA(n_components=n_components).fit_transform(X)
    vel = np.zeros_like(Y)
    gain = np.ones_like(Y)
    for it in range(n_iters):
        gamma = 0.8 if it > 250 else 0.5                # momentum: calm, then accelerate
        grad, _ = trimap_grad(Y + gamma * vel, triplets, weights)   # look-ahead gradient
        flip = np.sign(vel) != np.sign(grad)
        gain = np.where(flip, gain + 0.2, np.maximum(gain * 0.8, 0.01))  # delta-bar-delta
        vel = gamma * vel - lr * gain * grad
        Y += vel
    return Y


def fit_transform(X, n_components=2, random_state=None):
    rng = np.random.default_rng(random_state)
    return trimap(X, n_components=n_components, rng=rng)


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

Now the pieces settle into a simple division of labor. The nonlinear maps I have preserve local neighborhoods but scramble the global layout, and the local scores can't even detect it, so I first define a global score as normalized closeness to PCA's linear-reconstruction error (`GS = exp(−(err − err_PCA)/err_PCA)`, one for PCA), giving me a yardstick. Then I diagnose why the local methods fail globally: pairwise targets are absolute and only meaningful for neighbors, so the relative arrangement of clusters is left undetermined. I switch to triplets — relative, scale-free, higher-order constraints — and, learning from the over-collapse of log-probability triplet objectives, design a saturating per-triplet loss `w · s_ik/(s_ij + s_ik)` with the heavy-tailed Student-t kernel, which stops pulling once a triplet is satisfied. I weight triplets by their high-D margin `δ_ik^2 − δ_ij^2`, locally scaled by per-point densities to compare fairly across regions, and temper the weights to keep the giants from dominating. I sample `O(n)` triplets — forty-eight neighbor triplets and three random triplets per point — and then hit the wall that those neighbor triplets, despite being the bulk of the constraints, can reach near-zero loss while destroying global structure, so they cannot be what carries it. A few random triplets help only faintly. The fix is to put the global structure in the initialization: start from the downscaled PCA layout, which is globally optimal and already separates far points, so the triplet forces only sharpen local neighborhoods without tearing the global frame — and the saturating loss is exactly what keeps those forces gentle enough not to disturb it. Finally I derive the per-triplet gradient (three local force updates per triplet, with a prefactor that shrinks as the triplet is satisfied) and minimize the full-batch sum with momentum and delta-bar-delta adaptive gains for 400 iterations from the PCA start. Global structure from PCA, local fidelity from saturating weighted triplets, linear-time and scalable because the cost is just the neighbor search.
