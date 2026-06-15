Let me start from what actually goes wrong when I try to flag outliers in a real tabular dataset with no labels. I have `n` points in `d` dimensions, and I want a score per point — high means "rare, anomalous". The thing that keeps biting me isn't the modeling, it's the constraints around it. I have no labels, so the moment my method has a knob — `k` for a neighbor count, a kernel bandwidth, a histogram bin width, the number of trees — I have no honest way to set it, because there is no validation signal to optimize against; tuning becomes guessing. And the datasets are getting wide: `d` comparable to or bigger than `n`. Anything that estimates a multivariate density or computes all-pairs distances falls apart there, both in speed and in accuracy, because density estimation in `d` dimensions needs the sample size to grow exponentially in `d`, and distances stop discriminating as dimension climbs. On top of that, when I do flag a point as fraud or a bad sensor reading, someone is going to ask *why*, and a single opaque number is not an answer. So the real target is narrow and demanding: a scoring rule with literally nothing to tune, that scales to large `n` and `d`, and that can say which features made a point look strange.

What do I actually mean by "outlier"? The definition that organizes everything is the rare-event one: an outlier sits in a low-probability region of the data distribution. For a unimodal distribution those low-probability regions are the tails — the far left and far right ends. That's exactly why the old heuristics are tail rules: three-sigma flags anything more than three standard deviations from the mean, and the 1.5·IQR rule flags anything outside the quartile fences. I like that they're tuning-free and they reason per feature, which is already half of what I want. But they describe each feature's distribution by two numbers — mean and standard deviation, or the quartiles — and nothing else. That implicitly assumes a symmetric, roughly Gaussian shape; the moment a marginal is skewed or heavy-tailed, "three sigmas" is the wrong yardstick for how extreme a tail value really is. I'm throwing away almost everything the data tell me about the actual shape of the tail. I want to use the whole distribution of each feature, not a two-number caricature of it, and I want to do it without assuming a parametric form.

So let me make the rare-event definition precise and see where it leads. Suppose I knew the true distribution. The cleanest measure of "how extreme is point `x` toward the small side" is the probability of drawing something at least that small: the cumulative distribution function `F(x) = P(X ≤ x)`. If `F(X_i)` is tiny, then almost nothing in the population is as small as `X_i` along the way the inequality reads — that's the definition of being in the left tail, of being rare. For the large side I want the mirror event, the probability of drawing something at least as large. In one dimension the complement `1 − F(z)` gives `P(X > z)`, which already differs from the non-strict left-tail event at ties; in several dimensions the complement of a joint left-tail event is not the same as the event that every coordinate is large. So the right tail has to be its own mirror tail probability, not just a casual `1 − F` substitution. This is exactly the rare-event intuition, but now expressed as tail probability instead of a hand-set sigma count, and it uses the entire distribution because the CDF integrates all of it.

The catch is `F` is unknown and I'm in `d` dimensions. The honest left-side object is the *joint* CDF `F(x) = P(X^{(1)} ≤ x^{(1)}, …, X^{(d)} ≤ x^{(d)})` — the probability that a fresh draw is at least as small as `x` in every coordinate simultaneously. The mirror right-side object is `P(X^{(1)} ≥ x^{(1)}, …, X^{(d)} ≥ x^{(d)})`. If either of those joint tail probabilities is minuscule, `x` is jointly extreme. Conceptually right. But how do I estimate it? The nonparametric way is the empirical CDF: replace probabilities by sample fractions. In one dimension the ECDF `F_n(z) = (1/n) Σ_i 1{X_i ≤ z}` is a gorgeous estimator — it assumes nothing about the distribution's shape and has nothing to tune, and it comes with guarantees. Glivenko–Cantelli says it converges uniformly to the truth, `sup_z |F_n(z) − F(z)| → 0` almost surely; and DKW pins the rate, `P(sup_z |F_n − F| > ε) ≤ 2 exp(−2nε²)`. Read that bound carefully: it depends only on `n` and `ε`. Not on the distribution, not on anything dimension-like. In one dimension the ECDF is a tuning-free, dimension-free estimator of tail mass. That's precisely the kind of estimator my no-labels, no-knobs constraint is begging for.

So the obvious thing is: estimate the *joint* CDF by the *joint* ECDF, count the fraction of training points dominated by `x` in all `d` coordinates at once. Let me try it and see if it survives. The joint ECDF is `(1/n) Σ_i 1{X_i ≤ x in every coordinate}`. The problem is immediate. For a point out near the joint tail, almost no training point dominates it across *all* `d` coordinates simultaneously, so the count is 0 or 1 even when each individual coordinate isn't all that extreme — the indicator is the AND of `d` conditions, and the AND collapses to nothing as `d` grows. And the convergence rate of the joint ECDF to the true joint CDF degrades with dimension; the lovely dimension-free DKW guarantee is a one-dimensional fact, it does not carry over to the `d`-dimensional indicator. So the joint ECDF inherits the very curse of dimensionality I was trying to escape. The dimension-free estimator I trust lives in one dimension; the object I want lives in `d`. That's the wall.

I need to get the joint quantity out of one-dimensional pieces. What relates a joint CDF to its marginals? If the `d` features were *independent*, the joint distribution would factor: `F(x) = Π_{j=1}^d F^{(j)}(x^{(j)})`, a product of univariate CDFs. Then the joint left-tail probability is just the product of `d` one-dimensional tail probabilities, and each of those I can estimate with the univariate ECDF — the good, dimension-free estimator. I know independence is not literally true; features in real data are correlated. But look at the trade I'm being offered: in exchange for a structural assumption on the dependence, I get to replace the cursed joint ECDF with `d` clean univariate ECDFs, each converging at the dimension-free DKW rate, computable in linear time, and trivially parallel across columns. That is exactly the kind of "wrong but useful" assumption that fast detectors live on — it costs me sensitivity to outliers that only show up in the joint correlation structure, but it buys scalability, tuning-freedom, and per-feature transparency, which are precisely my three pressures. I'll take it, eyes open. (There's a more careful framing lurking here via copulas — Sklar's theorem says any joint CDF factors as a copula applied to the marginals, with the copula carrying all the dependence — and the independence assumption is just the statement that the copula is the trivial independence copula, under which the copula machinery collapses back to the product of marginals. I don't need the copula language to proceed; the product form is the operative object. But it tells me independence is a clean point in a more general family, not an ad hoc hack.)

So the plan crystallizes: per dimension `j`, estimate the univariate left-tail ECDF
`F̂_left^{(j)}(z) = (1/n) Σ_i 1{X_i^{(j)} ≤ z}`, and the joint left-tail probability of a point `x` is `Π_j F̂_left^{(j)}(x^{(j)})`. Symmetrically I want the right tail. The naive move is `1 − F̂_left`, but let me look at that carefully because there's an asymmetry that will bite. `1 − F^{(j)}(z) = 1 − P(X^{(j)} ≤ z) = P(X^{(j)} > z)` — note the *strict* inequality, whereas `F^{(j)}` used non-strict `≤`. Worse, empirically: the largest sample value `z*` has `F̂_left(z*) = 1` exactly, so `1 − F̂_left(z*) = 0`, and I'm about to take logs — `log 0 = −∞` would detonate the score for the single most-right point, which is exactly a point I care about. The fix is to estimate the right tail *symmetrically* with its own ECDF using a non-strict inequality the other way: `F̂_right^{(j)}(z) = (1/n) Σ_i 1{X_i^{(j)} ≥ z}`. Now the right tail of the maximum value is `1/n`, not `0` — a small positive number, log-safe — and the left and right constructions are mirror images of each other, which they should be. In code this is just `F̂_right^{(j)}(z) = F̂_left of −X` evaluated at `−z`: negate the column, run the same left ECDF. One routine, two tails.

Now the product `Π_j F̂^{(j)}(x^{(j)})` has a second, purely numerical problem: it's a product of `d` numbers each in `(0, 1]`, so as `d` grows it underflows toward zero and I lose all resolution between points. The standard cure is to work in negative-log space. Take `−log` of the product:
`−log Π_j p_j = −Σ_j log p_j`. The log is monotone, so smaller tail probability maps to larger negative-log — exactly the direction I want for an outlier score (rarer ⇒ higher). The sum of `d` terms is numerically stable where the product was not, and each term `−log F̂^{(j)}(x^{(j)})` is a clean per-dimension contribution that I can read off individually. That last point is a gift I didn't go looking for: because the joint score is *additively* decomposed across dimensions, I can attribute it — `−log F̂^{(j)}(x^{(j)})` is "how much dimension `j` contributed to this point's outlyingness", directly comparable across dimensions and against a reference like `−log(0.01) ≈ 4.6`. The interpretability I needed is a free consequence of the log-product form, not a bolt-on. (And it's no accident this looks like a log-likelihood — summing per-feature negative log scores under an independence assumption is exactly what a histogram-density detector like HBOS does, `Σ_j −log hist_j(x^{(j)})`. The difference is *what* I'm summing the negative log of: HBOS uses a binned point-density estimate, which needs a bin width to be chosen and produces discretization artifacts; I'm using the ECDF tail mass, which is the integral of the density, needs nothing chosen, and — given that "outlier = tail event" — is the more directly appropriate object. So I'm taking the same additive aggregation that already works and swapping the tuned density estimator for a tuning-free tail-mass estimator.)

So far I have two candidate scores per point: a left-only score `O_left(x) = −Σ_j log F̂_left^{(j)}(x^{(j)})` and a right-only score `O_right(x) = −Σ_j log F̂_right^{(j)}(x^{(j)})`. Which do I use? Let me stress-test them. Picture a 2-D dataset where the inliers cluster in one corner — say both coordinates tend large — and a scatter of outliers lands on the *small* side of both coordinates. The outliers all sit in the left tail of each marginal. `O_left` nails them: small left-tail probability per dimension, large negative log, high score, done. `O_right` is a disaster: the outliers are small, so their *right*-tail probabilities are near 1 (almost everything is bigger than them), tiny negative log, low score — and meanwhile the genuinely large inliers get high right-tail scores and get flagged. So `O_right` doesn't just miss the outliers, it actively inverts and flags the wrong points. Flip the construction so outliers fall on the large side and the verdicts swap: `O_right` is perfect, `O_left` fails. So neither one-sided score is safe in general; each is right exactly when the outliers happen to live on its side.

The first reflex is to combine them — average the two tail probabilities per dimension and score off that. Let me check the same scenario. With outliers on the small side, the left tail is tiny and the right tail is near 1; averaging gives roughly `(small + 1)/2 ≈ 1/2`, which is not extreme — so the true outliers get a middling score, and any inlier that's extreme on *either* side also gets a middling score. Averaging is a compromise that's good at nothing: it dilutes a genuine one-sided signal with the uninformative opposite tail.

What I actually need is, per dimension, to *pick the right tail* — left for dimensions where outliers are on the small side, right where they're on the large side. The brute-force version is to try all `2^d` assignments of left/right across the `d` dimensions and take the most outlier-revealing — utterly infeasible for any real `d`. I need a cheap, per-dimension signal for which tail is the outlying one. Go back to the toy case: when the outliers populate the left tail and the inlier mass sits on the right, the marginal has a *long left tail and its mass concentrated on the right* — that is precisely a negatively skewed distribution. When outliers are on the large side, the marginal has a long right tail, positive skew. So the *sign of the per-dimension skewness* tells me which tail to trust: negative skew ⇒ the long, sparse tail is on the left ⇒ use the left tail probability; positive skew ⇒ use the right. And skewness is cheap and tuning-free — the sample skewness coefficient

`γ_j = [ (1/n) Σ_i (X_i^{(j)} − X̄^{(j)})³ ] / [ (1/(n−1)) Σ_i (X_i^{(j)} − X̄^{(j)})² ]^{3/2}`,

a third central moment over the cubed standard deviation, computable in one pass over the column. The aggregate mathematical version can be written with indicators as a third, *skewness-corrected* score that selects the tail per dimension by the sign of `γ_j`:

`O_auto(x) = −Σ_j [ 1{γ_j < 0} · log F̂_left^{(j)}(x^{(j)}) + 1{γ_j ≥ 0} · log F̂_right^{(j)}(x^{(j)}) ]`.

The `γ_j = 0` branch is a convention in this aggregate formula. For continuous nonconstant data it is a measure-zero event, but in code I still need to know exactly what the branch-free expression does at zero.

Now, do I just use `O_auto` and call it done? Let me not trust the skewness estimate too far. Skewness is a third-moment statistic; it's noisy on small samples, and a marginal can be near-symmetric (skew ≈ 0) while still having outliers on a definite side, or genuinely two-sided so that *no* single tail per dimension is right. If `O_auto` picks the wrong tail for some dimension, it can suppress a real one-sided outlier exactly the way `O_right` did above. But I still have `O_left` and `O_right` sitting right there, and each is the correct score when the outliers are uniformly on one side. So rather than bet everything on the skewness selector, take the *most extreme* verdict among the three: if a point is genuinely a left-tail outlier across the board, `O_left` will be large and will speak up even if `O_auto` mis-selected; if right-tail, `O_right`; and `O_auto` catches the mixed-skew cases the pure scores can't. The final score is

`O(x) = max{ O_left(x), O_right(x), O_auto(x) }`.

This is belt-and-suspenders, and it stays parameter-free — there's nothing to tune in a `max`. It's the safe aggregation: I never *lose* a one-sided outlier to a bad skew call, because the corresponding pure tail score is always in the running, and I gain the per-dimension tail selection on the cases where it helps.

Let me sanity-check the whole pipeline against the constraints I started with, because it's easy to build something elegant that secretly has a knob. Hyperparameters: the ECDF has none; the skewness coefficient has none; the `max` and the sum have none. Zero knobs — that pressure is fully resolved. Scalability: after the per-column ranks are available, tail scoring and aggregation are linear in the matrix size; the sort-based rank helper I am about to write pays `O(d n log n)` to materialize the ECDF values, then the skewness and per-point reductions are `O(nd)`, and every dimension is independent enough to parallelize across columns. Interpretability: the additive log-product structure hands me `−log F̂^{(j)}(x^{(j)})` per dimension as a contribution I can rank, plot, and threshold. All three pressures met by construction, no tuning anywhere.

Now let me get the ECDF semantics exactly right, because the whole score rides on them and a sloppy implementation will misrank the extremes. For a column, I want `F̂_left^{(j)}(z) = (#\{i : X_i^{(j)} ≤ z\}) / n` — the rank-based ECDF. Sort the column; the value at sorted rank `r` (1-indexed) gets probability `r/n`, and for ties every equal value takes the *largest* rank among them, so that `F̂_left(z)` is genuinely `P(X ≤ z)` (the count of values at-or-below, ties included). This gives values in `{1/n, 2/n, …, 1}` — never `0`, so `log` is always finite on the left tail. For the right tail, the same routine on the negated column: `F̂_right^{(j)}(z) = (#\{i : X_i^{(j)} ≥ z\})/n`, computed as the left ECDF of `−X` at `−z`; the maximum value then gets `1/n`, log-safe, as I wanted.

Let me also pin the bookkeeping for the fit/score split. A practical detector can keep the fitted reference matrix and rebuild rank-based tail arrays on the matrix it is currently scoring; when it scores new rows after fitting, the implementation I want to mirror pools the stored training rows with the new rows, computes the empirical tails on that pooled matrix, and then returns only the new rows' scores. Concretely I'll build the two `n × d` arrays of per-dimension negative-log tail probabilities, `U_left = −log F̂_left` and `U_right = −log F̂_right`, and the skewness sign per column, then assemble the scores. For the branch-free elementwise expression, let `s_j = sign(γ_j) ∈ {−1, 0, +1}`. Form `U_skew = U_left · (−sign(s_j − 1)) + U_right · sign(s_j + 1)`. Check it: for `s_j = −1`, `−sign(−1−1) = −sign(−2) = +1` and `sign(−1+1) = sign(0) = 0`, so `U_skew = U_left` — left, correct. For `s_j = +1`, `−sign(1−1) = −sign(0) = 0` and `sign(1+1) = +1`, so `U_skew = U_right` — right, correct. For `s_j = 0`, `−sign(0−1) = +1` and `sign(0+1) = +1`, so `U_skew = U_left + U_right`. The indicator formula above breaks the zero tie toward the right tail; this branch-free implementation sums both at exact zero. For continuous nonconstant data that distinction is rarely active, and for a constant column both tail logs are zero anyway.

I should also decide, when I take the max of the three scores, whether to take it over the three *total* sums or elementwise per dimension and then sum. The clean object I derived is the maximum of the three total scores, `O(x) = max{ O_left, O_right, O_auto }` — three sums, one max — and that's the aggregate algorithm, because each of the three is a coherent joint negative-log tail probability and "most extreme of the three" has a clear meaning. The canonical PyOD realization takes the per-dimension max of `U_left, U_right, U_skew` and then sums, which is a different aggregation: `Σ_j max{U_left^{(j)}, U_right^{(j)}, U_skew^{(j)}}`, an upper bound on the max of the three sums. I need to keep both honest: the aggregate derivation lands on max-of-three-sums, and the code path grounded in PyOD uses sum-of-per-dimension-maxes.

Let me write the algorithm out as the loop I'd actually run:

```
Input: X ∈ R^{n×d}
For each dimension j = 1..d:
    F̂_left^{(j)}(z)  = (1/n) Σ_i 1{X_i^{(j)} ≤ z}      # rank-based ECDF of column j
    F̂_right^{(j)}(z) = (1/n) Σ_i 1{X_i^{(j)} ≥ z}      # = left ECDF of −X_j at −z
    γ_j = [ (1/n) Σ_i (X_i^{(j)}−X̄^{(j)})³ ] / [ (1/(n−1)) Σ_i (X_i^{(j)}−X̄^{(j)})² ]^{3/2}
For each point X_i:
    O_left(X_i)  = −Σ_j log F̂_left^{(j)}(X_i^{(j)})
    O_right(X_i) = −Σ_j log F̂_right^{(j)}(X_i^{(j)})
    O_auto(X_i)  = −Σ_j [ 1{γ_j<0} log F̂_left^{(j)}(X_i^{(j)}) + 1{γ_j≥0} log F̂_right^{(j)}(X_i^{(j)}) ]
    O_i = max{ O_left(X_i), O_right(X_i), O_auto(X_i) }
Return O = (O_1, …, O_n)
```

Now let me turn this into vectorized array code that fills the scoring slot, grounded in how the ECDF is actually computed. The one nontrivial primitive is the rank-based column ECDF with the correct tie handling; everything else is elementwise NumPy.

```python
import numpy as np


def _column_ecdf(matrix):
    """Per-column empirical CDF: for value z in column j, return (#{X_ij <= z}) / n.
    Ties take the largest rank, so F̂_left(z) = P(X <= z) exactly. Values lie in
    {1/n, ..., 1}, never 0, so the subsequent log is finite."""
    n = matrix.shape[0]
    # probability assigned to sorted rank r (1-indexed) is r/n, per column
    probabilities = np.linspace(np.ones(matrix.shape[1]) / n, np.ones(matrix.shape[1]), n)
    sort_idx = np.argsort(matrix, axis=0)
    matrix = np.take_along_axis(matrix, sort_idx, axis=0)        # sort each column ascending
    # tie handling: equal values all take the largest rank's probability (P(X <= z))
    for cx in range(probabilities.shape[1]):
        for rx in range(probabilities.shape[0] - 2, -1, -1):
            if matrix[rx, cx] == matrix[rx + 1, cx]:
                probabilities[rx, cx] = probabilities[rx + 1, cx]
    # undo the sort so probabilities line up with the original rows
    reordered = np.ones_like(probabilities)
    np.put_along_axis(reordered, sort_idx, probabilities, axis=0)
    return reordered


def ecdf_outlier_scores_aggregate(X):
    # per-dimension negative-log tail probabilities (n x d):
    U_left = -1.0 * np.log(_column_ecdf(X))     # −log F̂_left^{(j)}(X_ij), left-tail rarity
    U_right = -1.0 * np.log(_column_ecdf(-X))   # −log F̂_right^{(j)}(X_ij) via left ECDF of −X

    # per-dimension skewness sign: −1 long left tail, +1 long right tail
    from scipy.stats import skew as _skew
    s = np.sign(np.nan_to_num(_skew(X, axis=0)))   # {−1, 0, +1} per column

    # skewness-corrected per-dimension pick (branch-free):
    #   s=−1 -> U_left ; s=+1 -> U_right ; s=0 -> U_left + U_right
    U_skew = U_left * -1.0 * np.sign(s - 1) + U_right * np.sign(s + 1)

    # three joint negative-log tail scores per point (sum over dimensions):
    O_left = U_left.sum(axis=1)
    O_right = U_right.sum(axis=1)
    O_auto = U_skew.sum(axis=1)

    # aggregate-sum specification: the most extreme of the three aggregate sums
    O = np.maximum(np.maximum(O_left, O_right), O_auto)
    return O.ravel()


def ecdf_outlier_scores_pyod(X):
    U_left = -1.0 * np.log(_column_ecdf(X))
    U_right = -1.0 * np.log(_column_ecdf(-X))
    from scipy.stats import skew as _skew
    s = np.sign(np.nan_to_num(_skew(X, axis=0)))
    U_skew = U_left * -1.0 * np.sign(s - 1) + U_right * np.sign(s + 1)
    # PyOD implementation: take the per-dimension max first, then sum over dimensions.
    O = np.maximum(np.maximum(U_left, U_right), U_skew)
    return O.sum(axis=1).ravel()
```

And dropping the library aggregation into the detector harness — `fit` computes training scores before storing the reference matrix, and later calls to `decision_function` pool the reference matrix with the new rows before returning only the new scores:

```python
import numpy as np
from scipy.stats import skew as _skew


class AnomalyDetector:
    """Per-dimension ECDF tail-probability outlier detector. No hyperparameters."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if hasattr(self, "X_train"):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        U_left = -np.log(_column_ecdf(X))           # −log F̂_left per dim (n x d)
        U_right = -np.log(_column_ecdf(-X))         # −log F̂_right per dim
        s = np.sign(np.nan_to_num(_skew(X, axis=0)))  # skewness sign per dim
        U_skew = U_left * -np.sign(s - 1) + U_right * np.sign(s + 1)  # per-dim tail pick
        O = np.maximum(np.maximum(U_left, U_right), U_skew).sum(axis=1)
        if hasattr(self, "X_train"):
            O = O[-original_size:]
        return O.ravel()
```

Let me trace the causal chain one more time to make sure each piece is forced, not chosen. I started needing an outlier score with no hyperparameters, scalable to large `n` and `d`, and interpretable — and the no-labels constraint is what makes "no hyperparameters" non-negotiable, since I can't tune what I can't validate. The rare-event definition said outliers live in tails, so the right quantities are left and mirror-right tail probabilities that use the whole distribution unlike the two-number sigma rules. Estimating them nonparametrically points straight at the ECDF, which Glivenko–Cantelli and DKW certify as consistent, tuning-free, and — crucially — *dimension-free in one dimension*. The joint ECDF wrecks that guarantee under the curse of dimensionality, so I factored the joint tail probability across dimensions under an independence assumption, trading dependence-sensitivity for `d` clean univariate ECDFs. The product of `d` tail probabilities underflows, so I moved to a negative-log sum — which also, for free, made the score additively attributable per dimension, giving interpretability. The right tail needed its own `≥`-ECDF rather than `1 − F̂` to avoid the strict/non-strict asymmetry and the `log 0` at the maximum value. One-sided scores fail on the opposite-side outliers and averaging the tails smears the populations, so I selected the outlying tail per dimension by the sign of the skewness coefficient — and, distrusting that selector on noisy or two-sided marginals, kept the left-only, right-only, and skewness-corrected views all available. The mathematical specification takes the maximum of the three aggregate sums, `O(x) = max{ O_left, O_right, O_auto }`; the PyOD-style implementation takes the maximum per dimension and then sums, a slightly stronger aggregation that still uses the same three tail views and the same parameter-free ingredients. Either way, the score is built from per-dimension `−log` tail probabilities, parallel across dimensions, and interpretable, which is exactly the conjunction I couldn't get from any single prior method.
