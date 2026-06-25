Let me start from the thing that actually breaks. I have a dataset, no labels, a distance function, and I want to hand each point a number that says how exceptional it is. The intuitive target is Hawkins's: a point that looks like it came from a different mechanism than the rest. The cleanest operational version I know is the distance-based outlier definition — call an object `O` a `DB(p, D)`-outlier if at least a fraction `p` of all objects lie farther than `D` from it. I like this a lot. It needs only a metric, so it isn't trapped in one dimension or one assumed distribution the way the old discordancy tests are; it doesn't build a convex hull, so it doesn't die at `k ≥ 3` the way depth-based methods do; and it even swallows the classical tests as special cases — a `3σ` Gaussian outlier is exactly a `DB(0.9988, 0.13σ)`-outlier, a Poisson(3) tail point a `DB(0.9892, 1)`-outlier — so I'm not throwing away the statistics, I'm generalizing it. Ramaswamy, Rastogi and Shim then sand off a rough edge: instead of asking me to name a distance `D`, rank every point by `Dᵏ(p)`, the distance to its `k`-th nearest neighbor, and take the top `n`. That hands me two things `DB(p, D)` lacked — a ranking, a notion of *strength* — and a more intuitive knob, "how many outliers do I want," instead of a raw distance.

So I have a good, scalable, distribution-free, rankable distance-based detector. Why am I not done? Let me draw the case that worries me. Picture a loose, low-density cluster `C₁` and, off to the side, a tight, high-density cluster `C₂`. Now drop in a stray `o₂`, sitting just off the edge of the dense `C₂`. It is an outlier to my eye, because relative to how tightly `C₂` packs, it is conspicuously detached — yet its absolute distance to `C₂` is small, perhaps smaller than the ordinary spacing *inside* the loose `C₁`. That coincidence is exactly where I worry `Dᵏ` breaks, so let me stop hand-waving and compute it on a case small enough to check by hand. Put `C₂` on the line at `0,1,2,3,4` (spacing 1), `C₁` at `20,25,30,35,40` (spacing 5), and the stray at `x = 8` — detached from `C₂` (nearest `C₂` point is at `4`, distance `4`) but with that gap of `4` *smaller* than `C₁`'s internal spacing of `5`. Take `MinPts = 2`. Then `Dᵏ` is the distance to the 2nd nearest neighbor: for `o₂` at `8` the two nearest are `4` and... the next is `3` at distance `5`, so `Dᵏ(o₂) = 5`; for the `C₁` interior point at `30` the 2nd neighbor is at distance `5` likewise, and the `C₁` endpoints at `20,40` reach `10`. Ranking everything by `Dᵏ` descending puts `C₁_20` and `C₁_40` at `10` on top, then `C₁_25, C₁_30, C₁_35` and `o₂` all tied at `5`. So the genuine anomaly `o₂` is buried — two ordinary `C₁` members rank strictly above it and three more tie with it. The global ranking has failed exactly as feared. And `DB(p, D)` is no better: any single distance `D` is one number for the whole dataset — small enough to catch `o₂`'s gap of `4` and it condemns most of `C₁`, whose internal spacing is `5`; large enough to spare `C₁` and `o₂`'s gap slips under it. Both methods compare every point to a single global scale, and a single global scale cannot express "anomalous relative to *this* neighborhood." `o₂` is not far in absolute terms; it is far *for the company it keeps*.

So the property I'm missing has a name now that I look at it squarely: I need the verdict to be *local*. Not "is `O` far from the bulk of the data," but "is `O`'s surroundings much emptier than the surroundings of the points around it." Two more demands fall out of the same picture. First, it has to be a *degree*, a real number, not a yes/no — because the whole problem was that `o₂` and a loose-cluster point can be indistinguishable on a binary global test, and the only way to separate them is to measure *how much* each deviates from its own context, and rank. `DB(p, D)` is binary, depth is binary, DBSCAN's noise/not-noise is binary; all of them throw away exactly the gradation I need. Second, whatever number I assign, it must mean *the same thing everywhere* — a point deep inside the loose `C₁` and a point deep inside the dense `C₂` should score identically "ordinary," even though their absolute densities differ by an order of magnitude. If the score didn't have that property, I couldn't compare across regions, which was the entire point.

How do I measure "surroundings"? I don't want to assume a distribution (that's the statistical-test trap) and I don't want a global volume threshold (that's DBSCAN). The cheapest distribution-free density proxy I have is already in my hands from the distance-based methods: the distances to the nearest neighbors. If a point's `k` nearest neighbors are all close, it sits in a dense region; if they're far, a sparse one. I'll reuse the one good idea from density-based clustering — define a neighborhood by a *count* of objects, `MinPts`, rather than by a fixed radius, so I never have to commit to a volume in advance. So fix a count, call it `MinPts` (it's the `k` from `Dᵏ`, the `n_neighbors` I'll set in code). For a point `p`, its `k-distance` is the distance to its `MinPts`-th nearest neighbor, and its `k-distance neighborhood` `N(p)` is every object within that distance — I should be careful that under ties this set can hold *more* than `MinPts` points, so I'll always write `|N(p)|` and never assume it equals `MinPts`.

First instinct for "local density of `p`": just average the distances from `p` to its `MinPts` neighbors and invert. Small average distance, high density. Let me stress-test that before I commit. Take `p` very near a single neighbor `o` that lives in the dense cluster. The raw distance `d(p, o)` could be tiny — smaller than the typical spacing *inside* the dense cluster — and a couple of such small distances would make `p` look absurdly dense, denser than `o`'s own cluster, just because of where `p` happened to fall. That's noise in the estimate: the raw pairwise distances of nearby points fluctuate, and I'm about to build a ratio out of these densities, so jitter in the denominator will swing every score. I want to damp it.

The instability comes from letting `d(p, o)` get arbitrarily small for `p` close to `o`. So floor it: when computing how far `p` is "reachable" from `o`, never let the distance drop below `o`'s own `k-distance` — the radius of `o`'s neighborhood. Define

    reach-dist(p, o) = max{ k-distance(o), d(p, o) }.

Read it: if `p` is comfortably inside `o`'s neighborhood (`d(p,o) ≤ k-distance(o)`), I pretend `p` is exactly at `o`'s neighborhood radius, `k-distance(o)`; if `p` is outside, I use the true distance. The effect is that *all* the points sitting inside `o`'s neighborhood get the same reach-distance to `o`, namely `k-distance(o)` — the local fluctuations among nearby points collapse to a single stable number, and `k-distance(o)` itself is a smoothed, density-of-`o` quantity. Crank `MinPts` up and the smoothing strengthens, because `k-distance(o)` then averages over a wider neighborhood. Note the asymmetry, and notice it's the *right* asymmetry: `reach-dist(p, o)` is floored by `o`'s `k-distance`, not `p`'s, because what I'm trying to estimate stably is the density *around `o`*, the point whose neighborhood I'm probing. `reach-dist(p, o) ≠ reach-dist(o, p)` in general, and that's fine — they're estimating different local densities.

Now the local density of `p`. I want "one over a typical reach-distance from `p` to its neighbors," so I average the reach-distances over `N(p)` and invert. Call it the local reachability density:

    lrd(p) = 1 / ( ( Σ_{o ∈ N(p)} reach-dist(p, o) ) / |N(p)| ).

Why the *average* and not the sum? Because under ties `|N(p)|` varies from point to point, and if I used the bare sum a point that happened to have more neighbors would look artificially less dense; dividing by `|N(p)|` cancels that, so `lrd` is genuinely "inverse of a typical distance" and is comparable from point to point. Inverse so that larger means denser, which is what "density" should mean. Good — `lrd(p)` is now a stable, count-normalized, distribution-free estimate of how packed `p`'s immediate surroundings are.

But `lrd` is still an *absolute* density — large in `C₂`, small in `C₁` — so on its own it's just `Dᵏ` in disguise and inherits the same global-scale disease. The locality has to come from a *comparison*. What I actually claimed I wanted was: is `p`'s surroundings much emptier than the surroundings of the points around it? That's a ratio. Compare `p`'s density to the densities of `p`'s own neighbors, and average:

    LOF(p) = ( Σ_{o ∈ N(p)} lrd(o) / lrd(p) ) / |N(p)|
           = ( average lrd over p's neighbors ) / ( lrd(p) ).

Stare at this and predict what it should do. The absolute scale of density should cancel in the ratio — if I'm deep in the dense `C₂`, both `lrd(p)` and the `lrd(o)` of my neighbors are large and roughly equal, so the ratio ought to be ≈ 1; deep in the loose `C₁`, both are small and roughly equal, ratio again ≈ 1. So a point well inside *any* cluster, regardless of that cluster's absolute density, should score ≈ 1 — the "means the same thing everywhere" property I insisted on. And `o₂` should break out: it sits in a sparse pocket (its `lrd` is small) but its neighbors are the dense `C₂` points (their `lrd` is large), so the ratio `lrd(o)/lrd(p)` should run well above 1.

Rather than trust the prediction, I run this very quantity on the same 1-D toy that defeated `Dᵏ` above (`MinPts = 2`). Tracing the reachability/density/ratio arithmetic through gives:

    point     lrd      LOF
    C₂ (0..4) ~0.67    1.25  (one interior C₂ point dips to 0.67/LOF 0.67)
    C₁(20..40)~0.13    1.25  (one interior C₁ point dips to 0.20/LOF 0.67)
    o₂ (=8)    0.22    3.00

The `lrd`s differ by a factor of five between `C₂` (~0.67) and `C₁` (~0.13) — the absolute density really is an order of magnitude apart — yet *both* clusters' members land at `LOF ≈ 1.25` (or 0.67), the same "ordinary" band, while `o₂` stands alone at `3.00`. That is the prediction confirmed where `Dᵏ` failed: on this exact dataset `Dᵏ` ranked two `C₁` points strictly above `o₂` and tied three more with it, but `LOF` puts `o₂` cleanly at the top with no inlier near it. The ratio has untangled the two cases the global scale confused. (The inliers sit at `1.25` rather than a clean `1.0` because `MinPts = 2` is tiny and noisy — consistent with the deep-point lemma giving exactly `1` only as `ε → 0`, which I'll return to when I pick `MinPts`.) And it's a real number, so I can rank by it. `LOF ≈ 1`: as dense as your neighbors, an inlier. `LOF > 1`: emptier surroundings than your neighbors, an outlier; the larger, the stronger. (`LOF < 1` would mean denser than your neighbors, which I'll treat as solidly inlier.)

The cancellation makes interior-≈-1 plausible, but I don't trust a hand-wave here — the whole credibility of "≈ 1 means ordinary" rests on it, so I want to bound it. Take a cluster `C`. Let `reach-dist-min` and `reach-dist-max` be the minimum and maximum reach-distance among pairs within `C`, and set `ε = reach-dist-max/reach-dist-min − 1`, a measure of how non-uniform `C`'s internal density is. Consider a point `p` buried deep enough in `C` that (i) all of `p`'s neighbors lie in `C`, and (ii) all of *their* neighbors lie in `C` too. Then every reach-distance entering `lrd(p)` is between `reach-dist-min` and `reach-dist-max`, so the average is too, and inverting,

    1/reach-dist-max ≤ lrd(p) ≤ 1/reach-dist-min.

By condition (ii) the identical bound holds for each neighbor `o` of `p`. Therefore each ratio `lrd(o)/lrd(p)` is squeezed between `(1/reach-dist-max)/(1/reach-dist-min) = reach-dist-min/reach-dist-max` and its reciprocal `reach-dist-max/reach-dist-min`. Averaging ratios that all lie in `[reach-dist-min/reach-dist-max, reach-dist-max/reach-dist-min]` keeps `LOF(p)` in the same interval, i.e.

    1/(1+ε) ≤ LOF(p) ≤ (1+ε).

So a deep point's `LOF` should be pinned within a factor `(1+ε)` of 1, and the tighter the cluster (the smaller `ε`), the closer to exactly 1. Let me actually drive `ε` to its extreme and see whether the bound bites. The cleanest near-uniform cluster I can build is a regular grid, where reach-distances are essentially constant. On a 15×15 unit grid with `MinPts = 8`, I restrict to the 49 interior points that sit ≥ 4 cells from every edge (so their neighbors and neighbors-of-neighbors all stay inside, satisfying (i)–(ii)), and I compute `ε` directly from the within-interior reach-distances: it comes out `0.0000`. The predicted interval `[1/(1+ε), 1+ε]` is therefore `[1.0000, 1.0000]`, and the actual interior LOFs are min = max = mean = `1.0000`. The bound doesn't just hold loosely — at `ε = 0` it collapses to the single point `1`, and every interior LOF lands exactly there. That is the guarantee in its sharpest form: in a perfectly uniform cluster, however dense or sparse in absolute units, an interior point scores precisely `1`. A real cluster has `ε > 0` and the interior smears a little around 1, but it is genuinely pinned, not merely "tends to be."

Now the points I actually care about are *not* deep inside — `o₂` is on the fringe. Can I bound `LOF` for an arbitrary `p`, with no clustering assumption? Let me push the same averaging argument but keep separate handles on `p`'s neighborhood and its neighbors' neighborhoods. Define, over `p`'s direct neighbors,

    direct_min(p) = min_{o ∈ N(p)} reach-dist(p, o),     direct_max(p) = max_{o ∈ N(p)} reach-dist(p, o),

and, over the neighbors of `p`'s neighbors (the "indirect" neighborhood),

    indirect_min(p) = min { reach-dist(o, o') : o ∈ N(p), o' ∈ N(o) },   indirect_max similarly with max.

For the lower bound: every reach-distance in `lrd(p)`'s average is ≥ `direct_min(p)`, so the average is ≥ `direct_min(p)`, so `lrd(p) ≤ 1/direct_min(p)`. And for each neighbor `o`, every reach-distance in `lrd(o)`'s average is ≤ `indirect_max(p)`, so `lrd(o) ≥ 1/indirect_max(p)`. Substituting into the LOF average,

    LOF(p) = (1/|N(p)|) Σ_o lrd(o)/lrd(p) ≥ (1/|N(p)|) Σ_o (1/indirect_max(p)) / (1/direct_min(p)) = direct_min(p)/indirect_max(p).

The mirror computation — `lrd(p) ≥ 1/direct_max(p)`, `lrd(o) ≤ 1/indirect_min(p)` — gives the upper bound, so

    direct_min(p)/indirect_max(p) ≤ LOF(p) ≤ direct_max(p)/indirect_min(p).

This holds for *any* `p`, deep or fringe, and it says precisely the right thing: `LOF` is governed by the ratio of `p`'s own reach-distances (direct) to those of its neighbors (indirect). If `p` sits in a sparse pocket whose neighbors are dense, `direct` ≫ `indirect`, and the lower bound forces `LOF` well above 1. Concretely, if `p`'s smallest reach-distance is four times its neighbors' largest, and its largest reach-distance is six times its neighbors' smallest, then `LOF(p) ∈ [4, 6]` — `p` is between 4 and 6 times more isolated than its surroundings, a clean, interpretable statement of degree.

How tight are these bounds — is `[4, 6]` typical, or could the spread be uselessly wide? Let me see what controls `LOF_max − LOF_min`. Suppose the reach-distances inside the direct and indirect neighborhoods each fluctuate by a fraction `pct` around their means: `direct_max = direct·(1 + pct/100)`, `direct_min = direct·(1 − pct/100)`, and the same for `indirect`. Then

    (LOF_max − LOF_min) / (direct/indirect)
      = (direct_max/indirect_min − direct_min/indirect_max) / (direct/indirect)
      = (1+pct/100)/(1−pct/100) − (1−pct/100)/(1+pct/100)
      = [ (1+pct/100)² − (1−pct/100)² ] / (1 − (pct/100)²)
      = (4·pct/100) / (1 − (pct/100)²).

Good: the *relative* spread of `LOF` depends only on `pct`, the fluctuation of the reach-distances — not on their absolute magnitudes, not on `direct/indirect` itself. That is the local property made quantitative: how sharp the score is depends only on how homogeneous the neighborhood is, the same in the dense cluster and the sparse one. If the `MinPts`-neighbors of `p` all live in one well-behaved cluster, `pct` is small, the spread is small, and `LOF` is nailed down — close to 1 for an interior point, sharply above 1 for `o₂`. As `pct → 100` the denominator vanishes and the spread blows up, which warns me of the one regime where the bound goes slack.

That regime is when `p`'s neighbors come from *several* clusters of different densities at once — then the reach-distances within `N(p)` genuinely span a wide range, `pct` is large, and the single-ratio bound is loose. I can do better there by not lumping the neighbors together. Partition `N(p)` into the groups `C₁, …, C_n` it draws from, let `ξᵢ = |Cᵢ|/|N(p)|` be the fraction of `p`'s neighbors in group `i`, and define `direct^i`, `indirect^i` as the min/max reach-distances restricted to group `i`. The average reach-distance for `p` is now bounded below by `Σᵢ ξᵢ direct^i_min(p)` and above by `Σᵢ ξᵢ direct^i_max(p)`. For a neighbor in group `i`, its density is bounded below by `1/indirect^i_max(p)` and above by `1/indirect^i_min(p)`, so the average neighbor density is bounded below by `Σᵢ ξᵢ/indirect^i_max(p)` and above by `Σᵢ ξᵢ/indirect^i_min(p)`. And since `1/lrd(p)` is the average reach-distance from `p` to its direct neighbors, `LOF(p) = (1/lrd(p)) · (1/|N(p)|)Σ_o lrd(o)`. The bounds therefore become products of weighted sums,

    LOF(p) ≥ ( Σᵢ ξᵢ·direct^i_min(p) ) · ( Σᵢ ξᵢ / indirect^i_max(p) ),
    LOF(p) ≤ ( Σᵢ ξᵢ·direct^i_max(p) ) · ( Σᵢ ξᵢ / indirect^i_min(p) ).

Each group contributes in proportion to how much of `p`'s neighborhood it owns. Sanity check the degenerate case: with a single group, `n = 1`, `ξ₁ = 1`, and these collapse straight back to `direct_min/indirect_max ≤ LOF ≤ direct_max/indirect_min` — the earlier bound is exactly the one-partition special case, so the two are consistent and I haven't introduced a contradiction. When the neighbors really do span clusters of differing density, the weighted version is genuinely tighter because it never multiplies a dense group's small reach-distance against a sparse group's large one inside the same min/max.

Now the one knob, `MinPts`. I have to choose it, and I want to understand its behavior before I pick a default. (This is also where my first attempt at "interior ≈ 1" tripped: I had reached for a 200-point Gaussian blob to demonstrate it, expecting clean 1s, and didn't get them — which is what sent me to the uniform grid above for a cluster with `ε` small enough that the lemma actually bites. Let me go back to that blob now and read off what it *does* do as `MinPts` grows, since it's the right instrument for the smoothing question.) Naively I'd hope `LOF` moves monotonically as I raise `MinPts` — more neighbors, more smoothing, scores settle. At the extreme `MinPts = 2`, `lrd` is built from essentially the raw distance to the single nearest neighbor, so the density estimate has the full statistical fluctuation of one sample; that jitter should pass straight into the ratio. I compute the spread of interior LOF over the blob (no outliers planted) at `MinPts = 2, 5, 10, 20`:

    MinPts= 2:  mean=1.33  std=0.97  p95=2.55  max=11.77
    MinPts= 5:  mean=1.13  std=0.27  p95=1.53  max= 3.37
    MinPts=10:  mean=1.13  std=0.26  p95=1.53  max= 3.57
    MinPts=20:  mean=1.13  std=0.26  p95=1.62  max= 3.15

The `MinPts = 2` column is alarming on its own terms: with nothing outlying in the data, ordinary points reach LOF up to `11.8` and a std near `1` — at that neighborhood size I'd flag pure noise as anomalies. Raising to `5` collapses the std from `0.97` to `0.27` and the max from `11.8` to `3.4`; that is the averaging killing the one-sample variance, exactly as expected. But then it *stops* improving: from `5` to `20` the std sits flat at `≈0.26` and the mean refuses to fall below `≈1.13`. So two things are now established by computation rather than hope. The variance is suppressed only after `MinPts` leaves the single digits — small `MinPts` is genuinely dangerous. And the score does *not* keep settling toward 1; it plateaus, because a finite Gaussian has a real density gradient and the `MinPts`-neighbors of an edge point and a center point are never drawn from quite the same local density, so the residual `≈1.13`/`0.26` is structural, not removable by more smoothing. `LOF` is therefore not monotone-decreasing-to-1 in `MinPts`. On structured data the non-monotonicity is sharper still — with several clusters of different sizes, a point's `LOF` rises and falls as the growing neighborhood swallows first its own cluster, then a neighboring one — but even the single-blob numbers already refute the naive "just keep raising `MinPts`."

Two consequences. First, a floor on `MinPts`. While `MinPts` is small the variance of the `lrd` estimate — and hence the spread of `LOF` across identical inliers — stays large, so I'd flag perfectly ordinary points; the averaging only quiets the fluctuations once `MinPts` reaches roughly the low tens, so I want `MinPts` at least ~10 to be safe. There's a second, structural reading of the lower bound too: `MinPts` is effectively the minimum number of points a group must contain before a nearby point can register as a local outlier *to* that group — if a "cluster" has fewer than `MinPts` members, a point's neighborhood reaches right through it and the cluster can't anchor a local-density comparison. So the lower bound is also "the smallest cluster I want to be able to be an outlier relative to." Second, since `LOF` is non-monotonic, no single `MinPts` is universally right. The clean answer is to pick a *range* `[MinPtsLB, MinPtsUB]`, compute `LOF` at each value in it, and score each point by the *maximum* `LOF` it attains over the range. Maximum, deliberately — taking the minimum would erase a point's outlying nature at the one `MinPts` where it looks innocent, and the mean would dilute it; the maximum reports the scale at which the point is *most* exceptional, which is the honest summary of "is this point an outlier at some sensible neighborhood size." For the lower bound, ≥ 10; the upper bound is the largest group of "close by" points I'd still want to be able to flag as a clutch of local outliers, which is domain-dependent — for most datasets something in the 10–20 region works well as a starting point.

I should also handle the degenerate case the density estimate can hit: if a point has `MinPts` or more *exact duplicates*, all its reach-distances can be zero, the average reach-distance is zero, and `lrd = 1/0` blows up. The clean theory assumed no duplicates; in code I'll just floor the denominator with a tiny `ε` so a zero average reachability gives a huge-but-finite `lrd` instead of a NaN, which is the right behavior — a point sitting on top of `MinPts` copies of itself genuinely is maximally dense.

Let me also pin the scale-independence I'll lean on in practice: `LOF` is a pure ratio of reach-distances, so multiplying every distance by a constant — changing all units by the same factor — leaves every `lrd(o)/lrd(p)` unchanged. The score is invariant to the global distance scale, exactly as a "relative to local density" quantity should be. What it is *not* invariant to is a feature-by-feature rescaling that changes the neighbor graph itself, so the metric geometry has to be chosen before the neighbor search begins; after that, the score only sees the distances it is given.

Now write it as the procedure I'd actually run. The mathematical definition keeps the tie-inclusive `N(p)`, so `|N(p)|` may exceed `MinPts`; the fixed-width nearest-neighbor table used in code resolves ties by returning `n_neighbors_` rows. That changes the table convention, not the reachability, density, or ratio algebra. For the fitted training set, each point must be scored against its nearest training neighbors without counting itself; once those neighbor rows are materialized, the `k-distance` of any neighbor `o` is just the last stored distance in `o`'s row. That makes the reachability step an elementwise maximum between the query-to-neighbor distance matrix and those gathered last-column distances. Then I average, invert, and take the neighbor-density ratio. One implementation detail is only a sign convention: store the training score as `negative_outlier_factor_ = -LOF`, so larger stored values mean more normal and inliers sit near `-1`; a wrapper that wants "higher = more anomalous" just negates that opposite-LOF score.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors


class LOF:
    """Local-density ratio with the standard opposite-LOF scoring convention."""

    def __init__(
        self,
        n_neighbors=20,
        *,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination="auto",
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors          # MinPts: neighborhood size / smoothing knob
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.n_jobs = n_jobs

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        # +1 because a training point is its own 0-distance neighbor; drop that column.
        self._nn = NearestNeighbors(
            n_neighbors=self.n_neighbors_ + 1,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        ).fit(X)
        distances, neighbors = self._nn.kneighbors(X)
        self._distances_fit_X_ = distances[:, 1:]
        neighbors = neighbors[:, 1:]

        self._lrd = self._local_reachability_density(self._distances_fit_X_, neighbors)
        lrd_ratios = self._lrd[neighbors] / self._lrd[:, None]
        self.negative_outlier_factor_ = -np.mean(lrd_ratios, axis=1)

        if self.contamination == "auto":
            self.offset_ = -1.5                 # inliers are around -1
        else:
            self.offset_ = np.percentile(
                self.negative_outlier_factor_,
                100.0 * self.contamination,
            )
        self.decision_scores_ = -self.negative_outlier_factor_  # higher = more anomalous
        return self

    def _local_reachability_density(self, distances_X, neighbors_indices):
        # reach-dist(p, o) = max{ k-distance(o), d(p, o) }  -- floor by the NEIGHBOR's k-distance
        dist_k = self._distances_fit_X_[neighbors_indices, self.n_neighbors_ - 1]
        reach = np.maximum(distances_X, dist_k)             # smooths the density estimate
        # lrd(p) = 1 / average reach-dist over p's neighbors  (average, not sum: |N| cancels)
        return 1.0 / (reach.mean(axis=1) + 1e-10)           # eps floors the duplicate case

    def score_samples(self, X):
        """Opposite LOF for new query points; larger values mean more normal."""
        X = np.asarray(X, dtype=float)
        distances_X, neighbors_X = self._nn.kneighbors(
            X,
            n_neighbors=self.n_neighbors_,
        )
        X_lrd = self._local_reachability_density(distances_X, neighbors_X)
        lrd_ratios = self._lrd[neighbors_X] / X_lrd[:, None]
        return -np.mean(lrd_ratios, axis=1)

    def decision_function(self, X):
        """Shifted opposite LOF; zero is the inlier/outlier threshold."""
        return self.score_samples(X) - self.offset_

    def anomaly_score(self, X):
        """Positive LOF-style score for APIs where higher means more anomalous."""
        return -self.score_samples(X)
```

The causal chain, start to finish: distance-based detection (`DB(p, D)`, then `Dᵏ`) made outlier-finding distribution-free, multidimensional and scalable, but it compares every point to a *single global distance scale*, so it cannot tell a genuine anomaly next to a dense cluster from an ordinary member of a sparse cluster — the absolute distances coincide. The cure is to make the verdict local and graded: estimate each point's density from its `MinPts`-nearest-neighbor distances (no distribution, no global volume), stabilize that estimate by flooring every distance to a neighbor `o` at `o`'s own `k-distance` (the reachability distance, which smooths fluctuations and strengthens with `MinPts`), turn it into a local reachability density (inverse of the average reach-distance, count-normalized so it's comparable across points), and finally take the *ratio* of a point's neighbors' density to its own — the local outlier factor. The ratio cancels absolute density, so interior points of any cluster score ≈ 1 — provably, within a factor `(1+ε)` of 1 by the deep-point lemma, and bounded by `direct/indirect` reach-distance ratios for any point, with a relative sharpness that depends only on neighborhood homogeneity `pct`, refinable by a `ξ`-weighted multi-cluster bound when neighborhoods straddle clusters — while a point in a sparse pocket surrounded by dense neighbors scores well above 1. Because `LOF` is non-monotonic in `MinPts`, the robust reading scores each point by the maximum `LOF` over a range `MinPts ≥ 10`, and the computation is a neighbor materialization followed by vectorized reachability-density ratios, distribution-free, any-metric, scalable.
