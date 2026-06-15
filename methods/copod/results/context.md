# Context: unsupervised outlier detection on tabular data (circa 2019-2020)

## Research question

Given a single batch of unlabeled tabular data — `n` rows, each a `d`-dimensional feature
vector, with no indication of which rows are normal and which are anomalous — assign every row
a score whose magnitude reflects how anomalous the row is, so that the rare, deviant rows float
to the top of the ranking. This is the *unsupervised* regime: there is no clean training set of
"normal only" examples and no labels at fit time; the algorithm sees only feature values and
must decide what "normal structure" looks like from the data itself, then flag deviations from
it. The two working assumptions are the standard ones for the problem — anomalies are rare, and
their feature values differ noticeably from the bulk of the data.

A method that would actually be useful in practice has to clear several bars that the prior art
clears only one or two of at a time. It must **not collapse in high dimensions**: many detectors
are accurate on two or three features and degrade sharply as `d` grows, yet real tables have
tens to thousands of columns. It must be **cheap**: pairwise-distance and kernel methods are
`O(n^2)`, infeasible when `n` reaches the hundreds of thousands or millions. It should be
**deterministic and free of fiddly hyperparameters** — the number of neighbors, the number of
clusters, a kernel bandwidth, the `nu` of a one-class boundary, a neural architecture — because
each such knob is a subjective choice that changes the output and demands tuning that, by
construction, has no labels to tune against. And ideally it should be **interpretable**: a
practitioner who is handed a high-scoring row wants to know *which features* made it one.
Closing all of these at once — high-dimensional, fast, parameter-free, interpretable, and
competitive on detection quality — is the problem. The contribution being sought is the
*scoring rule* itself: how to model normal structure from unlabeled features and turn a feature
vector into a single anomaly score.

## Background

Unsupervised anomaly detection partitions into a few families, each resting on a different
notion of "normal." **Proximity / nearest-neighbor** methods (kNN-distance, and the local
variant LOF) call a point anomalous if it sits far from its neighbors or in a region of lower
density than its neighbors enjoy; they are among the most accurate in low dimensions but the
nearest-neighbor search is `O(n^2)` and distance concentration erodes their signal as `d`
grows. **Clustering-based** methods (CBLOF and kin) cluster the data and score points by their
distance to the nearest large cluster; faster than full neighbor search but dependent on the
clustering and its `k`. **Linear / boundary** methods (one-class SVM) fit a kernel surface that
encloses the bulk of the data and score points by which side they fall on; powerful but kernel-
and `nu`-dependent and not cheap. **Statistical** methods estimate a density and score by its
inverse: parametric ones (Gaussian mixtures) are compute-heavy and assume a family, while
nonparametric ones (histograms, kernel density estimates) trade the family assumption for
bandwidth/bin choices.

Two pieces of standing theory are load-bearing for the statistical route.

The first is the **probability integral transform** and its inverse. If a scalar random
variable `X` has a continuous cumulative distribution function `F`, then `U = F(X)` is
distributed Uniform(0,1): `P[F(X) <= u] = u`. Conversely, `F^{-1}(U) ~ F` when `U` is uniform.
So `F(x)` reads off *where `x` sits in its own distribution* — `F(x)` near 0 means `x` is far
into the lower tail, near 1 means far into the upper tail. The empirical version, available
without any distributional assumption, is the **empirical cumulative distribution function**

```
F_hat(x) = (1/n) * sum_{i=1}^{n} I(X_i <= x),
```

a step function supported on the values `{1/n, 2/n, ..., 1}`: it just counts the fraction of
observed points that are `<= x`.

The second is **copula theory** (Sklar 1959; Nelsen, *An Introduction to Copulas*, 2006). A
`d`-variate copula `C: [0,1]^d -> [0,1]` is the joint CDF of a random vector with Uniform(0,1)
margins,

```
C(u_1, ..., u_d) = P[U_1 <= u_1, ..., U_d <= u_d],   with  P[U_j <= u_j] = u_j.
```

**Sklar's theorem** says that for any joint CDF `F` with margins `F_1, ..., F_d` there is a
copula `C` with

```
F(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d)),
```

and `C` is unique when the margins are continuous, given by
`C(u) = F(F_1^{-1}(u_1), ..., F_d^{-1}(u_d))`. The point of the decomposition is that it
*separates the marginals from the dependence*: you may model each one-dimensional margin on its
own and link them through the copula. Two special copulas matter here. The **independence
copula** is the product `C(u_1, ..., u_d) = u_1 * u_2 * ... * u_d` — the joint CDF when the
coordinates are independent. The **survival copula** links the joint *upper*-tail (survival)
function to the marginal survival functions, `P[X_1 > x_1, ..., X_d > x_d] =
C_bar(F_bar_1(x_1), ..., F_bar_d(x_d))` with `F_bar_j(x) = 1 - F_j(x)`. The **empirical copula**
substitutes the empirical CDFs into the construction and converges to the true copula as `n`
grows; with margins supported on `{1/n, ..., 1}` it is itself a nonparametric, parameter-free
object.

A few diagnostic facts about existing detectors set up the problem. It is well documented that
in high-dimensional spaces the joint probability of lying below a fixed threshold in *every*
coordinate at once falls off exponentially with the number of coordinates — the curse of
dimensionality — so any joint-probability estimate built by multiplying many per-coordinate
factors will underflow toward zero in high `d` unless something is done about the arithmetic.
It is also documented that purely density-based scoring confuses two different reasons a point
can sit in a low-probability region: it can be *extreme* (far out in a tail) or merely *between
modes* (in a low-density valley that is not extreme at all), and these deserve different
treatment for outlier detection. And the choice of which way a one-sided rare event lies — far
into the lower tail versus far into the upper tail — depends on the *shape* of each marginal:
when a feature's distribution has a long tail on one side, the rare points concentrate on that
side, and a detector that looks the wrong way, or symmetrically in both directions at once,
will see the rare points diluted or missed.

## Baselines

These are the prior detectors a new scoring rule would be measured against and would react to.

**Histogram-based Outlier Score, HBOS (Goldstein & Dengel, KI 2012).** The closest structural
relative. For each feature independently, build a univariate histogram (fixed-width, or a
dynamic-width variant that puts an equal count of points in each bin to cope with long tails),
normalize each histogram so its maximum height is 1 (so every feature contributes equally), and
read off the bin height `hist_i(p)` where point `p` falls in feature `i`. The score multiplies
the inverse per-feature densities assuming feature independence and takes logs to keep the
arithmetic stable on extremely unbalanced distributions,

```
HBOS(p) = sum_{i=1}^{d} log( 1 / hist_i(p) ),
```

which is exactly a discrete naive-Bayes-style independence model: a sum, over features, of a
per-feature anomaly contribution. It is `O(n)` (after sorting), parameter-light, and very fast.
**Gaps:** (1) it scores by *estimated density* — bin height — so a point in a low-density valley
*between* two modes can score as anomalous even though it is not extreme, and the density
estimate is bumpy and depends on how the bins are drawn. (2) It still carries a bin-count knob
`k` (a common rule of thumb is `sqrt(N)`) and a fixed-vs-dynamic-bin decision, so it is not
genuinely parameter-free, and the score moves with those choices. (3) Built on per-feature
densities, it cannot represent dependence between features and admittedly misses local outliers.
What it leaves open: a per-feature anomaly contribution that measures *extremeness* rather than
density, and that needs no binning.

**Proximity methods: kNN-distance and LOF (Breunig et al., SIGMOD 2000).** Score a point by the
distance to its `k`-th nearest neighbor (kNN), or, for LOF, by the ratio of the point's local
reachability density to that of its `k` neighbors, so that a point markedly less dense than its
surroundings scores high. Accurate in low dimensions and able to find *local* outliers. **Gaps:**
the neighbor search is `O(n^2)`, infeasible at large `n`; the result depends on `k` and on a
distance metric whose discriminative power concentrates and degrades as `d` grows.

**One-class SVM (Schölkopf et al., 2001).** Map the data through an RBF kernel and fit the
maximum-margin hyperplane separating the bulk from the origin; score by signed distance to that
boundary, with `nu` controlling the allowed outlier fraction. **Gaps:** kernel choice, bandwidth,
and `nu` must be set with no labels; training is not cheap; sensitive to scaling and to `d`.

**Clustering- and ensemble-based detectors (CBLOF; Feature Bagging; LSCP).** Cluster the data
and score by distance to large clusters (CBLOF), or build many sub-detectors on feature subsets
or data subsamples and combine them (Feature Bagging, LSCP). Robustness through aggregation.
**Gaps:** depend on the base clustering / base detectors and their hyperparameters, on a
combination rule, and inherit the cost of the components; results are not deterministic.

**Lightweight projection / neural detectors (LODA; SO/MO-GAAL).** LODA aggregates many cheap
one-dimensional random projections with histograms; the GAAL family trains generative
adversarial nets to synthesize boundary examples. **Gaps:** projection count / stochastic
training introduce randomness and architecture choices; the neural variants are expensive and
sensitive to training dynamics.

The common thread: the fast, high-dimensional-friendly options (HBOS, LODA) rest on per-feature
*density* and still carry binning/projection knobs; the accurate ones (LOF, OCSVM) are `O(n^2)`
or kernel-tuned and fade in high `d`. None is at once deterministic, knob-free, cheap,
high-dimensional, and interpretable.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Benchmark tabular datasets** from the ODDS repository and the DAMI collection — real-world
  outlier-detection tables spanning a wide range of sizes, dimensionalities, and anomaly rates.
  Representative members include Cardiotocography (~1.8k rows, 21 features, ~10% anomalies),
  Thyroid (~3.8k rows, 6 features, ~2.5%), Satellite/Landsat (~6.4k rows, 36 features, ~32%),
  and the NASA Shuttle set (~49k rows, 9 features, ~7%), alongside higher-dimensional and much
  larger sets.
- **Protocol:** a stratified split of each dataset into a fit portion and a held-out portion
  (a 60/40 split is standard); the detector is fit on the fit-portion features with no labels,
  and scores are computed for the held-out features. Features are typically standardized to zero
  mean and unit variance before fitting. Results are averaged over several independent random
  splits.
- **Metrics:** area under the ROC curve (ROC-AUC / AUROC), measuring ranking quality
  independent of any threshold; average precision (AP); and the F1 score at a chosen
  contamination threshold, measuring decision quality once a cut is applied.
- **Efficiency** is a first-class axis: wall-clock fit+score time across growing `n` and `d`
  (e.g. sweeping dimensionality `10 -> 100 -> 1000 -> 10000` and size `1k -> 1M`).

## Code framework

The detector plugs into the standard scikit-learn-style detector harness used by all the
baselines: a class with a `fit(X)` that learns whatever per-data structure the scoring rule
needs from the unlabeled, already-standardized feature matrix, and a `decision_function(X)`
that returns one score per row, larger meaning more anomalous. Nothing about *how* a feature
vector becomes a score is settled — that scoring rule is exactly what is to be designed — so the
substrate is only the generic machinery that already exists: the array library for the per-
column statistics, a place to stash whatever the rule fits on the training data, and the
contract that scores are comparative (higher = more anomalous). The single empty slot is the
scoring rule.

```python
import numpy as np


class Detector:
    """Generic unsupervised tabular outlier detector. fit() learns whatever
    per-data structure the scoring rule needs from unlabeled, standardized
    features; decision_function() returns one score per row (higher = more
    anomalous). Should stay cheap (near-linear in n and d) and ideally free of
    hyperparameters that need label-free tuning."""

    def __init__(self, contamination=0.1):
        # contamination only sets the label threshold (top fraction flagged);
        # it does not enter the score ranking.
        self.contamination = contamination

    def fit(self, X):
        # X: (n_samples, n_features), standardized, no labels.
        # TODO: learn whatever per-data structure the scoring rule needs.
        self.decision_scores_ = self.decision_function(X)
        self._set_threshold(self.decision_scores_)
        return self

    def decision_function(self, X):
        # X: (n_samples, n_features). Return scores: (n_samples,), higher = more anomalous.
        # TODO: the scoring rule we will design -- turn each row into one
        #       anomaly score using only the unlabeled feature values.
        scores = np.zeros(X.shape[0])
        return scores

    def _set_threshold(self, scores):
        # flag the top-`contamination` fraction as outliers (labels only)
        self.threshold_ = np.quantile(scores, 1 - self.contamination)
        self.labels_ = (scores > self.threshold_).astype(int)
```

The harness supplies the standardized feature matrix and the scoring contract;
`decision_function` is where the scoring rule will live.
