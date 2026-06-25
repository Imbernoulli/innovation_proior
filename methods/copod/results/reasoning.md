Let me start from what actually hurts. I have a single batch of unlabeled tabular data, `n` rows of `d` features each, and I need to hand back one number per row that says how anomalous it is, with nothing telling me which rows are normal and which are not. The two things I can lean on are weak but real: anomalies are rare, and their feature values differ noticeably from the bulk. Everything I'd reach for off the shelf trips on the same handful of rocks. The proximity methods — score a point by how far it sits from its `k` nearest neighbors, or for LOF by how much sparser its neighborhood is than its neighbors' neighborhoods — are accurate when `d` is two or three, but the neighbor search is `O(n^2)` and falls over when `n` is in the hundreds of thousands, and distances concentrate as `d` grows so the signal washes out exactly where I need it. The one-class SVM fits a kernel boundary around the bulk, but I have to pick the kernel, the bandwidth, and `nu` with no labels to pick them against. Clustering and ensemble detectors hide the same problem inside a base clusterer or base detectors plus a combination rule, each with its own knobs, and the output isn't even deterministic across runs. The density methods are the closest to what I want — a histogram per feature, or a kernel density estimate — but they still ask me to choose a bin count or a bandwidth, and those choices move the answer. So the thing I keep wanting and never getting is a *scoring rule* that is at once cheap (linear-ish in `n` and `d`), unbothered by high `d`, deterministic, free of any knob I'd have to tune blind, and ideally interpretable down to "this feature is what made the row weird." That last property nobody even tries for. I'm not sure all five can coexist — interpretability and high-`d`-robustness usually pull against speed — but I want to push on the cheapest, most knob-free family and see how far it carries.

The fast knob-free option I most want to learn from is the histogram score, HBOS. Its shape is exactly right and I should pin down *why* before I touch it. For each feature on its own it builds a univariate histogram, normalizes each so its tallest bar is height 1 — so a tall narrow feature and a flat wide feature both contribute on the same footing rather than one drowning the other — and then reads off the bar height `hist_i(p)` where the point `p` lands in feature `i`. The score is

```
HBOS(p) = sum_{i=1}^{d} log( 1 / hist_i(p) ).
```

Sit with that for a second. It's a product of inverse per-feature densities, `prod_i 1/hist_i(p)`, with a log wrapped around it. The product-of-per-feature-factors is an independence assumption: it's the score of a discrete naive-Bayes model that treats the features as independent and asks "how surprising is this point if each coordinate is judged on its own." And the log is a numerical safety move — on extremely unbalanced features the product can get astronomically large or small, and `log(a*b) = log(a) + log(b)` turns the runaway product into a tame sum without changing the ranking, since log is monotone. So the architecture HBOS hands me is "sum over features of a per-feature anomaly contribution," cheap and parameter-light and embarrassingly parallel across features. I want to keep that skeleton.

But what is the per-feature contribution actually measuring, and is it the right thing? It's `-log(density)` — the height of the histogram bar. Picture a feature whose distribution is bimodal, two humps with a valley between them. A point that lands in the valley sits in a low-density region, so HBOS scores it as anomalous. But it isn't *extreme* — it's smack in the middle of the feature's range, with plenty of mass on both sides of it. For outlier detection that's the wrong call: "rare because it's far out in a tail" and "rare because it happens to fall between two clumps" are different animals, and density conflates them. On top of that the density estimate is bumpy — slide the bin edges and `hist_i(p)` jumps — and that bin count is the very knob I swore off. So HBOS's per-feature contribution has two diseases at once: it measures density when I want *extremeness*, and it needs binning to measure even that. The fix has to replace the per-feature quantity with something monotone in "how far out toward a tail this value is," and that needs no bins. Let me find that quantity.

What does "how far out toward a tail" mean precisely for a single feature? If a feature `X` has continuous CDF `F`, then `F(x) = P(X <= x)` is literally the fraction of the distribution lying at or below `x` — `F(x)` near 0 means `x` is deep in the lower tail, near 1 means deep in the upper tail. And there's a clean fact I can exploit: the probability integral transform says that if `X` has continuous CDF `F`, then `U = F(X)` is Uniform(0,1), i.e. `P[F(X) <= u] = u`. So `F(x)` isn't just *a* measure of position, it's the *canonical* one — it maps any feature, whatever its shape, onto a common uniform yardstick where "deep in the lower tail" always means "`F(x)` close to 0." That's exactly the "extremeness" I wanted, and it's a probability, not a density — no bins, no bandwidth. The empirical version costs nothing and assumes nothing: the empirical CDF

```
F_hat(x) = (1/n) * sum_{i=1}^{n} I(X_i <= x),
```

just the fraction of observed points at or below `x`, a step function supported on `{1/n, 2/n, ..., 1}`. So per feature `j`, feed the value `x_{j,i}` into the empirical CDF `F_hat_j` and out comes `u_{j,i} = F_hat_j(x_{j,i})`, the empirical probability of seeing something as small as this along feature `j`. Small `u` means a left-tail outlier on that feature. That's my parameter-free, bin-free, monotone-in-extremeness per-feature contribution.

Now I need to combine `d` of these per-feature tail probabilities into one joint statement about the whole row. The honest joint quantity is the joint left tail, `F_X(x_i) = P(X_1 <= x_{1,i}, ..., X_d <= x_{d,i})` — the probability of a point being at least as small as `x_i` in *every* coordinate at once. And here copula theory is exactly the right machine, because it's built to separate the marginals (which I've just nailed with the empirical CDFs) from the dependence (which I'd rather not have to estimate). A `d`-copula `C: [0,1]^d -> [0,1]` is the joint CDF of a vector with Uniform(0,1) margins, `C(u) = P(U_1 <= u_1, ..., U_d <= u_d)`, and Sklar's theorem says any joint CDF factors as `F(x) = C(F_1(x_1), ..., F_d(x_d))`. So my joint left tail *is* the copula evaluated at the per-feature tail probabilities I already have. Replace `F_j` by the empirical `F_hat_j`, replace `C` by the empirical copula `C_hat(u) = (1/n) sum_i I(U_hat_{1,i} <= u_1, ..., U_hat_{d,i} <= u_d)` — which is nonparametric, knob-free, and converges to the true copula as `n` grows — and I have an estimate of the joint tail probability of every point, with nothing to tune. A point whose joint tail probability is tiny is rare in the precise sense I want, and I'll call it an outlier.

Before I trust this I want to watch `C_hat(u)` actually behave as `d` grows, because the joint indicator makes me nervous. Write out the indicator for the joint event as a product of per-coordinate indicators, since "below in every coordinate" is the AND of "below in coordinate `j`":

```
C_hat(u_1, ..., u_d) = (1/n) sum_i I(U_hat_{1,i} <= u_1) * ... * I(U_hat_{d,i} <= u_d).
```

If I let go of the cross-coordinate dependence and treat the coordinates as roughly independent — the independence copula, where the joint CDF is just the product of marginals `C(u) = u_1 * u_2 * ... * u_d` — this collapses to the product of the per-coordinate probabilities,

```
C_hat(u) ≈ P(U_1 <= u_1) * P(U_2 <= u_2) * ... * P(U_d <= u_d).
```

Now the trouble. Each factor is a probability in `[0,1]`, so each is at most 1, and multiplying many numbers below 1 drives the product toward 0 exponentially in `d`. For a point that's even mildly into the tails on many features, the product underflows to numerical zero, and then I can't tell one "zero" from another — every high-dimensional point looks infinitely rare, which is useless for ranking. This is the curse of dimensionality biting the joint-probability estimate directly. Wall.

But this is the same disease HBOS already cured, and the cure is staring at me from the structure I copied. The product is exploding (toward zero); take logs to turn it into a sum, exactly as HBOS does, and the underflow goes away while the ranking is preserved because log is monotone:

```
-log( C_hat(u) ) = -log( prod_j P(U_j <= u_j) ) = - sum_{j=1}^{d} log( P(U_j <= u_j) ).
```

And there's a small gift here. The copula's margins are Uniform(0,1) by construction, so `P(U_j <= u_j) = u_j` exactly. The negative log of the joint tail probability is just

```
- sum_{j=1}^{d} log( u_j ),
```

the sum, over features, of the negative log of each per-feature tail probability. So I'm right back at the HBOS skeleton — a sum over features of a per-feature contribution — except the contribution is now `-log(empirical tail probability)` instead of `-log(histogram density)`. Bigger means rarer, the diseases of density-versus-extremeness and binning are both gone, and the "sum of negative logs" has the same justification as a log-likelihood: it's the log of an independence-model probability. Intuitively I'm computing an anomalous p-value for the whole row, the more negative its log the more extreme.

I've only done one tail, though. `u_j = F_hat_j(x_{j,i})` is small when `x` is far into the *lower* end of feature `j`. But an outlier can be far into the *upper* end too — a value much larger than everything else — and that has a *large* left-tail probability `F_hat(x) ≈ 1`, so `-log(u_j) ≈ 0`, and my score would call a giant outlier perfectly normal. I need the right tail as well. In the continuous copula picture that is the survival event `P(X_1 > x_1, ..., X_d > x_d)`, and the survival copula links it to the marginal survival functions: `P(X_1 > x_1, ..., X_d > x_d) = C_bar(F_bar_1(x_1), ..., F_bar_d(x_d))` with `F_bar_j(x) = 1 - F_j(x)`. So by the same independence-plus-log argument the continuous right-tail contribution per feature would be `-log(F_bar_j(x_j))`, equivalently `-log(1 - u_j)`.

Let me try to compute that and see if it's as clean as the left tail. The naive route is `1 - u_j = 1 - F_hat_j(x)`, feed `1 - u_j` in. But `F_hat_j` isn't a true continuous CDF — it lives on the discrete grid `{1/n, ..., 1}`, and its largest value is exactly 1, attained by the maximum observed point in that feature. For that point `u_j = 1`, so `1 - u_j = 0`, and `-log(0)` is `+infinity`. Worse, in the joint right-tail copula, `P(U_j >= 1)` collapses the whole product to zero — no uniform exceeds 1 — so the most extreme point on a feature, which is *exactly the point I most want to flag*, produces a degenerate zero / infinite-log that I can't combine with anything downstream. The boundary of the empirical support has bitten me. Wall.

I need the right-tail empirical CDF without that boundary degeneracy. Here's the move: the right tail of `X` is the left tail of `-X`. If I negate every value in the feature, "far into the upper end of `X`" becomes "far into the lower end of `-X`," and I can reuse the *exact same* left-tail empirical-CDF machinery — an ECDF computed on `-X` — which is well-behaved at its small end. Concretely, the empirical right-tail probability I want is `F_hat_{-X,j}(-x)`: substituting `-X_i` and `-x` into the empirical-CDF count, `(1/n) sum_i I(-X_i <= -x) = (1/n) sum_i I(X_i >= x)`, which is exactly the empirical probability of being at least as large as `x`, ties included, and it never hits the broken `1 - 1 = 0` corner because it's a left-tail computation on the negated data. So I keep two empirical CDFs per feature: the left-tail one from `X`, call it `F_hat_j`, and the right-tail one from `-X`, call it `F_bar_hat_j`. Define `U_{j,i} = F_hat_j(x_{j,i})` (left-tail probability input) and `V_{j,i} = F_bar_hat_j(x_{j,i})` (right-tail probability input); in contribution space I will call `U_l = -log(U)` and `U_r = -log(V)`.

So now I have, per row, a left-tail score `p_l = -sum_j log(U_{j,i})` and a right-tail score `p_r = -sum_j log(V_{j,i})`. How do I use both? The obvious thing is to average them, treat the row as anomalous if it's extreme on either side on balance. Let me test averaging on a one-sided feature rather than reason about it in the abstract, because I suspect it dilutes. Take a strongly right-skewed column: nine clustered small values and one big outlier, `(0, 0.1, 0.2, ..., 0.8, 50)`. The right-tail contribution `U_r` here is `(0.000, 0.105, 0.223, 0.357, 0.511, 0.693, 0.916, 1.204, 1.609, 2.303)` — monotone increasing, peaking on the `50`, exactly right; its argmax is the outlier. Now the two-tail average `(U_l + U_r)/2` comes out `(1.151, 0.857, 0.714, 0.636, 0.602, 0.602, 0.636, 0.714, 0.857, 1.151)` — a U-shape, and its *argmax is the `0` row*, tied with the `50` row at `1.151`. So the blind average has been pulled toward flagging the smallest, perfectly-normal value just as hard as the true outlier, because the left tail it folds in lights up on the dense left end where there are no anomalies to find. The average doesn't merely miss; it actively splits its attention onto the wrong end. If the outliers were instead all on the upper side, the roles flip — the left tail becomes the misleading half. So a blind two-tail average is not it; I need to look the *right way* per feature.

What tells me which way to look? The asymmetry of the feature is exactly its skewness. A feature with a long left tail skews negative (the mass concentrates on the right, the thin tail stretches left), and that left tail is where its rare points are — so use the left-tail contribution. A feature with a long right tail skews positive — use the right tail. So compute, per feature, the standard skewness coefficient

```
b_d = ( (1/n) sum_i (x_i - x_bar)^3 ) / ( sqrt( (1/(n-1)) sum_i (x_i - x_bar)^2 )^3 ),
```

the third central moment over the cubed standard deviation. Its sign tells me which negative-log contribution should be trusted on that feature: negative skew means use the left-tail contribution, positive skew means use the right-tail contribution, and exact zero gives me no reason to choose a side, so the neutral contribution should carry both tails. In contribution units that means a skewness-corrected term that is `U_l` for sign `-1`, `U_r` for sign `+1`, and `U_l + U_r` for sign `0`. Now each feature is judged on its informative side when the skew has a direction, and a perfectly symmetric feature does not get arbitrarily sent to one tail.

I want to be honest that the skewness sign is itself an estimate that can be unreliable — near a symmetric feature `b_d` is close to zero and its sign is essentially a coin flip on sampling noise, and the same is true mid-fit on a small split. If I commit hard to the skew-chosen tail and the sign is wrong, I look entirely the wrong way on that feature. So I don't want to throw the two-tail average away entirely; I want it as a fallback that catches a feature when the skew-targeted tail misses it. The right way to combine "the targeted tail" with "the symmetric safety net" so that a feature is flagged if *either* says extreme is to take, per feature, the larger of the skewness-corrected contribution and the average of the two tails. Per feature `j`,

```
O_{j,i} = max( U_skew_{j,i},  (U_l_{j,i} + U_r_{j,i}) / 2 ),
```

and the row's score is the sum over features `sum_j O_{j,i}`. I want the max taken per feature *before* summing, rather than summing each variant over all features and then maxing the three totals, so that each feature can independently use whichever of "look the targeted way" or "look both ways" makes it most extreme, instead of forcing one global decision for the whole row. Let me check this on the same right-skewed column to confirm it doesn't undo the skew selection I just fought for. With positive skew the targeted tail is `U_r`, and `O = max(U_r, average)` per row comes out `(1.151, 0.857, 0.714, 0.636, 0.602, 0.693, 0.916, 1.204, 1.609, 2.303)`. The `50` outlier still tops out at `2.303`, so the safety net hasn't derailed the detection — good. But the check also shows the cost honestly: the `0` row is now lifted to `1.151` (its average term winning over its tiny `U_r`), which is the *second*-highest score. So the two-tail safety net is not free — it does re-elevate the dense left end somewhat — but it elevates it strictly below the true outlier rather than tying it, which is the price I'll pay to be protected when a skew sign comes out wrong. A feature whose skew sign is mis-estimated gets rescued by its two-tail average; a feature where the targeted tail is sharply more extreme keeps that sharper signal and the average never overtakes it. Both terms are already in the `-log(probability)` units, so the max and the sum are comparing and adding the same kind of quantity.

There's a nice consequence I should record while it's in front of me: because the row score is `sum_j O_{j,i}`, a sum of nonnegative per-feature contributions, I can read off *which features* drove a row's score by looking at the individual `O_{j,i}`. Plot them, draw a band at, say, `-log(0.01) = 4.61` or at the contamination percentile, and the features whose contribution pokes above the band are the ones that made the row an outlier. The interpretability I wanted for free falls straight out of the additive structure — it's the same property that made the sum-of-logs numerically nice.

Let me make the empirical CDF concrete, because the implementation hinges on getting `F_hat` exactly right per column. For one feature column of `n` values: sort it ascending, and assign the sorted values the probabilities `1/n, 2/n, ..., 1` so the smallest observed value gets `1/n` and the largest gets `1` — that is exactly `F_hat(x) = #{X_i <= x}/n` evaluated at each observed point. Ties need care: if several rows share a value they must all receive the *same* `F_hat`, namely the largest (rightmost) probability among them, because `F_hat` counts everything `<= x` and tied points are all `<= x` together. So after sorting I sweep from the top down and, wherever a value equals the one above it, copy down the higher probability. Then I unsort back to the original row order. Done per column, this is `O(n log n)` per feature from the sort, linear in `d` across features — cheap, deterministic, parameter-free, scales to large `n` and `d`. The right-tail CDF is the identical routine run on the negated column.

Let me run a single column by hand to be sure the tie handling, the two tails, and the negation trick actually do what I'm claiming, before I wire them together. Take one feature with five values, deliberately including a tie and an obvious upper outlier: `x = (1, 2, 2, 3, 100)`. Sorted it's the same order, with raw probabilities `1/5, 2/5, 3/5, 4/5, 5/5 = (0.2, 0.4, 0.6, 0.8, 1.0)`. The two `2`s tie, so the lower one must take the higher of their two probabilities: the sweep copies `0.6` down over the `0.4`, giving the left ECDF `u = (0.2, 0.6, 0.6, 0.8, 1.0)`. Good — both `2`s read `0.6 = #{X_i <= 2}/5 = 3/5`, the count interpretation, and `100` reads `1.0`. The left-tail contributions are `U_l = -log(u) = (1.609, 0.511, 0.511, 0.223, 0.000)`: the smallest value `1` is the most extreme on the left (`1.609`), and `100` is treated as totally unremarkable on the left (`0.000`) — which is exactly the blind spot I worried about, a giant value scoring as normal under a one-tailed left score.

Now the right tail via the negated column. ECDF of `-x` gives, back in original order, `(1.0, 0.8, 0.8, 0.4, 0.2)` — i.e. `#{X_i >= x}/5`, so `100` reads `0.2 = 1/5` (only itself is `>= 100`) and `1` reads `1.0`. Then `U_r = -log(...) = (0.000, 0.223, 0.223, 0.916, 1.609)`, and now `100` is the most extreme point (`1.609`). This is the concrete payoff of the negation move: I also have to check the boundary degeneracy it was meant to dodge actually bites. The naive right tail would be `-log(1 - u)`, and at the maximum point `u = 1.0`, so `1 - u = 0` and `-log(0) = +inf` — confirmed degenerate, and it strikes precisely at `100`, the row I most want to score. The negated-ECDF route gives `0.2` there instead of `0`, a finite, usable number. So the trick isn't cosmetic; without it the single most anomalous point in this column produces an infinity I can't sum.

Now the whole thing in vectorized form, which is how I'd actually ship it. Build a matrix of per-feature left-tail contributions `U_l = -log( ecdf(X) )` and right-tail contributions `U_r = -log( ecdf(-X) )`, both `n` by `d`. Get the per-feature skewness sign `s = sign(skew(X, axis=0))`, a length-`d` vector in `{-1, 0, +1}`. I want a skewness-corrected matrix `U_skew` that equals `U_l` on negatively-skewed features, `U_r` on positively-skewed ones. I can do that branch-free with the sign arithmetic: `U_skew = U_l * (-1) * sign(s - 1) + U_r * sign(s + 1)`. Check the three cases. If `s = -1`: `sign(-1 - 1) = sign(-2) = -1`, so the first term is `U_l * (-1) * (-1) = U_l`; `sign(-1 + 1) = sign(0) = 0`, so the second term vanishes — `U_skew = U_l`, the left tail, correct for negative skew. If `s = +1`: `sign(1 - 1) = sign(0) = 0`, first term gone; `sign(1 + 1) = +1`, second term `U_r` — `U_skew = U_r`, right tail, correct for positive skew. If `s = 0` (a feature that came out exactly symmetric): `sign(-1) = -1` gives `U_l`, `sign(+1) = +1` gives `U_r`, so `U_skew = U_l + U_r` — it just uses both, the natural neutral choice when there's no skew to break the tie. Then take the per-feature max against the two-tail average and sum across features:

```
O = maximum( U_skew, (U_l + U_r) / 2 )
score = O.sum(axis=1)
```

higher meaning more anomalous, comparative across rows. There's one bookkeeping subtlety for scoring held-out rows: the tail probabilities are only meaningful relative to a reference population, so when I score a fresh batch I form the empirical CDFs on the training rows concatenated with the new rows and then read off only the new rows' scores at the end — the new points are ranked against the established distribution rather than against themselves alone.

Notice what I never had to choose: no number of neighbors, no clusters, no kernel, no bandwidth, no bin count, no architecture, no training, no randomness. The only externally supplied number is a contamination rate, and that does *not* enter the score at all — it only sets where to draw the binary inlier/outlier threshold on the already-computed scores (flag the top fraction). The ranking itself is entirely determined by the data. So the method is genuinely parameter-free in the part that matters, deterministic, `O(n log n)` per feature and linear in `d`, and interpretable down to the per-feature contribution. The high-`d` robustness I'm claiming rests on the log-sum taming the underflow — which I checked is real on the product form — but whether the *ranking quality* actually holds up as `d` grows from a handful of columns to thousands is the one bar I haven't verified here; that needs the benchmark tables (ODDS/DAMI) and an AUROC sweep over dimensionality, and I'd want to see it before claiming the bar is cleared rather than just argued.

Let me write the scoring rule into the detector harness, filling the one empty slot:

```python
import numpy as np
from scipy.stats import skew as skew_sp


def skew(X, axis=0):
    # per-feature skewness; nan_to_num guards constant columns (0 variance -> nan)
    return np.nan_to_num(skew_sp(X, axis=axis))


def column_ecdf(matrix):
    """Per-column empirical CDF F_hat(x) = #{X_i <= x}/n, support {1/n, ..., 1}.
    Ties take the largest (rightmost) probability, matching the <= count."""
    assert matrix.ndim == 2
    n, d = matrix.shape
    # probabilities 1/n, 2/n, ..., 1 broadcast across the d columns
    probabilities = np.linspace(np.ones(d) / n, np.ones(d), n)
    sort_idx = np.argsort(matrix, axis=0)
    sorted_mat = np.take_along_axis(matrix, sort_idx, axis=0)
    # propagate the higher probability down across equal values (ties)
    for c in range(d):
        for r in range(n - 2, -1, -1):
            if sorted_mat[r, c] == sorted_mat[r + 1, c]:
                probabilities[r, c] = probabilities[r + 1, c]
    # undo the sort so each original row gets its F_hat
    reordered = np.empty_like(probabilities)
    np.put_along_axis(reordered, sort_idx, probabilities, axis=0)
    return reordered


class Detector:
    """Unsupervised tabular outlier detector. Scores each row by the negative
    log of an empirical-copula tail probability, summed over features."""

    def __init__(self, contamination=0.1):
        self.contamination = contamination  # threshold only; not in the ranking

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        # contamination sets the binary label cut on the scores (labels only)
        self.threshold_ = np.quantile(self.decision_scores_, 1 - self.contamination)
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        # rank new rows against the established training distribution
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)
        # per-feature tail contributions: left from X, right from -X (avoids the
        # F_hat(max)=1 boundary that would make -log(1-u) blow up)
        U_l = -1 * np.log(column_ecdf(X))    # -log P(X_j <= x)  (left tail)
        U_r = -1 * np.log(column_ecdf(-X))   # -log P(X_j >= x)  (right tail)
        # pick the informative tail per feature by skewness sign (branch-free):
        #   skew<0 -> U_l, skew>0 -> U_r, skew==0 -> U_l + U_r
        s = np.sign(skew(X, axis=0))
        U_skew = U_l * -1 * np.sign(s - 1) + U_r * np.sign(s + 1)
        # per feature: max(skew-targeted tail, two-tail average safety net), then sum
        O = np.maximum(U_skew, np.add(U_l, U_r) / 2)
        if hasattr(self, 'X_train'):
            scores = O.sum(axis=1)[-original_size:]
        else:
            scores = O.sum(axis=1)
        return scores.ravel()
```

The causal chain, start to finish: I needed a scoring rule that is cheap, high-dimensional-proof, deterministic, knob-free, and interpretable, and nothing on the shelf was all five at once. HBOS gave me the right skeleton — a sum over features of a per-feature anomaly contribution, with a log to keep the arithmetic stable under an independence assumption — but its contribution was histogram *density*, which confuses extreme points with low-density-valley points and still needs a bin count. Swapping density for the empirical-CDF tail probability `F_hat(x)` fixed both: it measures genuine extremeness, it's monotone, and it needs no bins, because the probability integral transform makes `F(x)` the canonical position-in-its-own-distribution. Copula theory let me assemble the per-feature tail probabilities into a joint tail probability while keeping the marginals separate from the dependence; the empirical copula made that nonparametric and knob-free. The joint product underflowed in high `d` — the curse of dimensionality — so I took logs, turning the product into the sum `-sum_j log(u_j)`, with the copula's uniform margins making each factor exactly the per-feature tail probability. One tail only flags small-side outliers, so I added the right tail via the survival copula; the empirical right tail hit a `1 - 1 = 0` boundary degeneracy, which I dodged by computing the left-tail CDF on the negated data. A blind two-tail average dilutes detection when a feature's outliers are one-sided, so I select the tail per feature by skewness sign, and guard against an unreliable sign by taking, per feature, the max of the skew-targeted contribution and the two-tail average before summing. The result is parameter-free where it counts, deterministic, linear in the features and `O(n log n)` in the rows, and interpretable down to which feature made each row an outlier.
