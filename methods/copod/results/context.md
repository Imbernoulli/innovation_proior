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

The contribution being sought is the *scoring rule* itself: how to model normal structure from
unlabeled features and turn a feature vector into a single anomaly score, on real tables that
range from a handful of columns to thousands and from thousands of rows to millions.

## Background

Unsupervised anomaly detection partitions into a few families, each resting on a different
notion of "normal." **Proximity / nearest-neighbor** methods (kNN-distance, and the local
variant LOF) call a point anomalous if it sits far from its neighbors or in a region of lower
density than its neighbors enjoy; the nearest-neighbor search is `O(n^2)`. **Clustering-based**
methods (CBLOF and kin) cluster the data and score points by their distance to the nearest large
cluster. **Linear / boundary** methods (one-class SVM) fit a kernel surface that encloses the
bulk of the data and score points by which side they fall on, with `nu` setting the allowed
outlier fraction. **Statistical** methods estimate a density and score by its inverse:
parametric ones (Gaussian mixtures) assume a family, while nonparametric ones (histograms,
kernel density estimates) replace the family assumption with bandwidth/bin choices.

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
own and link them through the copula. Two standard copulas recur in the literature. The
**independence copula** is the product `C(u_1, ..., u_d) = u_1 * u_2 * ... * u_d` — the joint CDF when the
coordinates are independent. The **survival copula** links the joint *upper*-tail (survival)
function to the marginal survival functions, `P[X_1 > x_1, ..., X_d > x_d] =
C_bar(F_bar_1(x_1), ..., F_bar_d(x_d))` with `F_bar_j(x) = 1 - F_j(x)`. The **empirical copula**
substitutes the empirical CDFs into the construction and converges to the true copula as `n`
grows; with margins supported on `{1/n, ..., 1}` it is itself a nonparametric, parameter-free
object.

## Baselines

These are the prior detectors a new scoring rule would be measured against.

**Histogram-based Outlier Score, HBOS (Goldstein & Dengel, KI 2012).** For each feature
independently, build a univariate histogram (fixed-width, or a dynamic-width variant that puts
an equal count of points in each bin to cope with long tails), normalize each histogram so its
maximum height is 1 (so every feature contributes equally), and read off the bin height
`hist_i(p)` where point `p` falls in feature `i`. The score multiplies the inverse per-feature
densities assuming feature independence and takes logs to keep the arithmetic stable on
extremely unbalanced distributions,

```
HBOS(p) = sum_{i=1}^{d} log( 1 / hist_i(p) ),
```

which is a discrete naive-Bayes-style independence model: a sum, over features, of a per-feature
anomaly contribution. It is `O(n)` (after sorting) and carries a bin-count `k` (a common rule of
thumb is `sqrt(N)`) and a fixed-vs-dynamic-bin choice.

**Proximity methods: kNN-distance and LOF (Breunig et al., SIGMOD 2000).** Score a point by the
distance to its `k`-th nearest neighbor (kNN), or, for LOF, by the ratio of the point's local
reachability density to that of its `k` neighbors, so that a point markedly less dense than its
surroundings scores high. Find *local* outliers; the neighbor search is `O(n^2)` and uses a
distance metric and a neighbor count `k`.

**One-class SVM (Schölkopf et al., 2001).** Map the data through an RBF kernel and fit the
maximum-margin hyperplane separating the bulk from the origin; score by signed distance to that
boundary, with `nu` controlling the allowed outlier fraction. Set by kernel choice, bandwidth,
and `nu`.

**Clustering- and ensemble-based detectors (CBLOF; Feature Bagging; LSCP).** Cluster the data
and score by distance to large clusters (CBLOF), or build many sub-detectors on feature subsets
or data subsamples and combine them (Feature Bagging, LSCP), aggregating their outputs through a
combination rule.

**Lightweight projection / neural detectors (LODA; SO/MO-GAAL).** LODA aggregates many cheap
one-dimensional random projections with histograms; the GAAL family trains generative
adversarial nets to synthesize boundary examples.

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
    anomalous)."""

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
