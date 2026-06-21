## Research question

We are handed a matrix of `n` data points in `d` dimensions, with no labels at all, and asked
to assign every point an outlier score — higher meaning "more likely a rare, anomalous
observation". The points are assumed sampled i.i.d. from some unknown distribution; outliers
are the rare events of that distribution. The setting is fully unsupervised: at no time do we
see which points are anomalies, so there is no validation signal to tune against.

The setting is shaped by three considerations. **Scalability and dimensionality**: the detector
should stay fast and accurate as both `n` and `d` grow, including the regime where `d` is
comparable to or larger than `n`. Density estimation in `d` dimensions requires the number of
samples to grow exponentially in `d` (Stone 1980), and pairwise-distance structure becomes less
informative as dimension rises. **Hyperparameters**: with no labels, any knob — `k` for a
neighbor method, a kernel bandwidth, a histogram bin width, the number of trees in an ensemble —
must be set without a held-out score to optimize against. **Interpretability**: in fraud,
intrusion, and medical screening, practitioners often want to know *which features* made a point
look anomalous, both to act on it and to trust it. The question is how to build an unsupervised
tabular outlier scoring rule given these considerations.

## Background

The working definition of an outlier that organizes this whole area is the **rare-event**
view: an outlier is a point that occurs in a low-probability region of the data distribution
(Lazarevic & Kumar 2005; Pokrajac et al. 2007). For a unimodal distribution, the
low-probability regions are the tails. This is the intuition behind the oldest heuristics: the
**three-sigma rule** flags any point more than three standard deviations from the mean as an
outlier, and its robust cousin the **1.5·IQR rule** flags points outside
`[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` (Leys et al. 2013; Dekking et al. 2005). Both are tail rules that
summarize the distribution by a couple of numbers — mean and standard deviation, or the
quartiles.

The relevant statistical machinery for tail mass is the **cumulative distribution function**
and its nonparametric estimator. For a univariate random variable, the CDF
`F(z) = P(X ≤ z)` records the entire left-tail mass below every threshold. A mirror right-tail
event asks for `P(X ≥ z)`; the complement `1 − F(z)` instead gives `P(X > z)`, which differs at
ties. The CDF can be estimated, with no distributional assumption and nothing to tune, by the
**empirical CDF (ECDF)**:
`F_n(z) = (1/n) Σ_i 1{X_i ≤ z}`, the fraction of samples at or below `z`. Two classical facts
characterize the ECDF here. The **Glivenko–Cantelli theorem** guarantees uniform
almost-sure consistency, `sup_z |F_n(z) − F(z)| → 0`. The **Dvoretzky–Kiefer–Wolfowitz
inequality** sharpens this to a finite-sample rate (with Massart's sharp constant),
`P(sup_z |F_n(z) − F(z)| > ε) ≤ 2 exp(−2nε²)` — this bound depends only on `n` and
`ε`, not on the distribution or on any notion of dimension. In one dimension, the ECDF is a
tuning-free, dimension-free estimator of tail mass.

For the *multivariate* CDF, one could in principle estimate the joint
`F(x) = P(X^{(1)} ≤ x^{(1)}, …, X^{(d)} ≤ x^{(d)})` by a joint ECDF, but the convergence rate
of the joint ECDF to the true joint CDF degrades as the number of dimensions grows (Naaman
2021). The dimension-free guarantee of the ECDF holds cleanly per dimension.

A separate and relevant framework is **copula theory**. Sklar's theorem (Sklar 1959; Nelsen
2007) states that any joint CDF factors as `F(x) = C(F_1(x^{(1)}), …, F_d(x^{(d)}))` for a
copula `C` that carries all the dependence structure, with the marginals `F_j` carried
separately; for continuous marginals `C` is unique. The empirical copula — built from the
marginal ECDFs — converges to the true copula. Copulas thus offer a principled way to think
about modeling marginals separately from the dependence between them.

A recurring structural idea across fast detectors is to trade full dependence modeling for
per-feature summaries. Under an independence assumption, a joint distribution factors into
marginals and log-space scoring can be accumulated feature by feature. Empirically, this
assumption is treated as "wrong but useful" in outlier detection: detectors that make it run in
linear time and, for global outliers, perform comparably to far more expensive multivariate
methods, with their main sensitivity on *local* outliers that only stand out in the joint
structure.

A phenomenon any tail-based scorer encounters: which tail is the outlying one can differ across
features and across datasets. In a two-dimensional toy setting, if the bulk of the data sits
near one corner and rare points fall toward the opposite corner, a one-sided rule aimed at one
tail ranks the corners one way, and aimed at the other tail ranks them the opposite way. A rule
that averages both tails blends the two tail signals. Flip the construction and the preferred
one-sided rule flips as well. So tail information is per-feature: the outlying side can vary
from feature to feature.

## Baselines

These are the prior detectors a new method would be measured against and would react to. For
each: its core idea and its actual mechanism.

**Local Outlier Factor — LOF (Breunig et al. 2000).** Density-based and local: for each point,
estimate a local reachability density from the distances to its `k` nearest neighbors, then
score the point by the ratio of its neighbors' average local density to its own. A point in a
region much sparser than its neighbors' regions gets a high ratio and is called an outlier. It
handles outliers that are only anomalous *relative to a local cluster*. It rests on
`k`-nearest-neighbor search and pairwise distances, with `k` and the distance metric as inputs.

**Distance-to-`k`-th-neighbor detection — kNN (Ramaswamy et al. 2000).** Score each point by
its distance to its `k`-th nearest neighbor (or the average over its `k` neighbors); points far
from everything are outliers. No density model, just distances, with `k` as an input.

**One-Class SVM — OCSVM (Schölkopf et al. 2001).** Fit a boundary that encloses most of the
data in an RBF-kernel feature space, with a parameter `nu` that bounds the fraction of points
left outside (treated as outliers); the signed distance to the boundary is the score. Kernel
choice and bandwidth plus `nu` are inputs; the decision is a single distance.

**Isolation Forest — iForest (Liu et al. 2008).** Build an ensemble of random binary trees,
each splitting on a random feature at a random threshold; a point's score is its average
path length to isolation across trees — outliers, being few and different, are isolated by
shorter random paths. Fast and effective, with default 100 trees and subsample size 256. The
number of trees and the subsample size are inputs; the score is an average path length over
random partitions.

**Histogram-Based Outlier Score — HBOS (Goldstein & Dengel 2012).** Assume the features are
independent. For each feature build a univariate histogram density estimate `hist_j`, and score
a point by summing per-feature negative log densities, `Σ_j −log hist_j(x^{(j)})`: a point that
falls in low-density bins across many features scores high. Linear time, and the independence
assumption is what makes the per-feature histograms combine into a single multivariate score.
The histogram is a *density* estimate with a bin width (or number of bins) as an input.

**Three-sigma and 1.5·IQR rules (Leys et al. 2013; Dekking et al. 2005).** Per-feature tail
rules: flag points beyond three standard deviations of the mean, or outside the
quartile-based fences. Tuning-free and interpretable per feature. They summarize each feature by
the mean and standard deviation (or the quartiles).

## Evaluation settings

The natural yardsticks already in use for tabular unsupervised outlier detection:

- **Benchmark datasets** from the ODDS library and DAMI benchmark collections — real tabular
  datasets with known outlier labels held out only for evaluation, spanning a wide range of
  sample counts, dimensionality, and anomaly rates. Representative examples used in the field:
  Cardiotocography (~1,800 samples, 21 features, ~10% anomalies), Thyroid (~3,800 × 6, ~2.5%),
  Satellite/Landsat (~6,400 × 36, ~32%), and the NASA Shuttle dataset (~49,000 × 9, ~7%).
- **Protocol**: a stratified train/test split (commonly 60% train / 40% test); the detector is
  fit on the training features *without labels*, and scores are computed for the test features;
  results are averaged over several independent trials. Features are standardized (zero mean,
  unit variance) before fitting.
- **Metrics**: area under the ROC curve (AUROC), measuring ranking quality independent of any
  threshold; and average precision / F1 at a contamination-based threshold, measuring decision
  quality once the scores are thresholded. Both are "higher is better".
- **Scalability probes**: runtime on synthetically generated matrices swept over dimensions
  (e.g. 10, 100, 1,000, 10,000) and sample counts (e.g. 1,000 up to 1,000,000), used to chart
  how the method scales with `n` and `d`.

## Code framework

A detector plugs into a fixed, pre-existing scoring harness with a `fit`/`decision_function`
contract: `fit(X)` consumes the unlabeled, standardized feature matrix and stores whatever
reference data the scoring rule may need; `decision_function(X)` returns one score per row,
higher meaning more anomalous. The substrate is only the generic array machinery that already
exists: NumPy/SciPy column operations, sorting/ranking, and elementwise reductions. The single
empty slot is the scoring rule.

```python
import numpy as np


class AnomalyDetector:
    """Generic unsupervised outlier detector. fit() learns per-column state from the
    unlabeled, standardized feature matrix X (n_samples x n_features); decision_function()
    returns one score per row, higher = more anomalous. The scoring rule is what we design."""

    def __init__(self):
        self.reference_ = None

    def fit(self, X):
        self.reference_ = np.asarray(X, dtype=float)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        # TODO: design the unsupervised scoring rule that maps rows to higher-is-more-anomalous scores.
        pass
```

The harness supplies the standardized matrix and expects a per-row score; `decision_function`
is where the scoring rule will live.
