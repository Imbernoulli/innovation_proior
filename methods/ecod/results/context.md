## Research question

We are handed a matrix of `n` data points in `d` dimensions, with no labels at all, and asked
to assign every point an outlier score — higher meaning "more likely a rare, anomalous
observation". The points are assumed sampled i.i.d. from some unknown distribution; outliers
are the rare events of that distribution. The setting is fully unsupervised: at no time do we
see which points are anomalies, so we cannot tune anything against a validation signal.

Three pressures define the problem, and no method on the table satisfies all three at once.
First, **scalability and the curse of dimensionality**: the detector must stay fast and
accurate as both `n` and `d` grow, even when `d` is comparable to or larger than `n`. Anything
that estimates a multivariate density, or computes pairwise distances, degrades badly here —
both in runtime and in accuracy — because density estimation in `d` dimensions requires the
number of samples to grow exponentially in `d` (Stone 1980), and pairwise-distance structure
becomes uninformative as dimension rises. Second, **no hyperparameters**: because there are no
labels, choosing `k` for a neighbor method, a kernel bandwidth, a histogram bin width, or the
number of trees in an ensemble is guesswork — there is no held-out score to optimize against,
so model selection is itself an open problem in this regime. A method with nothing to tune
sidesteps the whole difficulty. Third, **interpretability**: in fraud, intrusion, and medical
screening, a bare score is not enough; a practitioner needs to know *which features* made a
point look anomalous, both to act on it and to trust it. Most detectors return one opaque
number. The goal is a scoring rule that is simultaneously scalable to large `n` and `d`,
free of any hyperparameter, and able to attribute its score to individual dimensions.

## Background

The working definition of an outlier that organizes this whole area is the **rare-event**
view: an outlier is a point that occurs in a low-probability region of the data distribution
(Lazarevic & Kumar 2005; Pokrajac et al. 2007). For a unimodal distribution, the
low-probability regions are the tails. This is the intuition behind the oldest heuristics: the
**three-sigma rule** flags any point more than three standard deviations from the mean as an
outlier, and its robust cousin the **1.5·IQR rule** flags points outside
`[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` (Leys et al. 2013; Dekking et al. 2005). Both are tail rules,
but both summarize the distribution by only a couple of numbers — mean and standard deviation,
or the quartiles — and so implicitly assume a roughly symmetric, unimodal, Gaussian-ish shape,
throwing away everything else the data say about how the tail actually behaves.

The relevant statistical machinery for doing better is the **cumulative distribution function**
and its nonparametric estimator. For a univariate random variable, the CDF
`F(z) = P(X ≤ z)` records the entire left-tail mass below every threshold. A mirror right-tail
event asks for `P(X ≥ z)`; the complement `1 − F(z)` instead gives `P(X > z)`, which differs at
ties. The CDF can be estimated, with no distributional assumption and nothing to tune, by the
**empirical CDF (ECDF)**:
`F_n(z) = (1/n) Σ_i 1{X_i ≤ z}`, the fraction of samples at or below `z`. Two classical facts
make the ECDF a uniquely good object here. The **Glivenko–Cantelli theorem** guarantees uniform
almost-sure consistency, `sup_z |F_n(z) − F(z)| → 0`. The **Dvoretzky–Kiefer–Wolfowitz
inequality** sharpens this to a finite-sample rate (with Massart's sharp constant),
`P(sup_z |F_n(z) − F(z)| > ε) ≤ 2 exp(−2nε²)` — crucially this bound depends only on `n` and
`ε`, not on the distribution or on any notion of dimension. In one dimension, the ECDF is an
excellent, tuning-free, dimension-free estimator of tail mass.

The trouble is the *multivariate* CDF. One could in principle estimate the joint
`F(x) = P(X^{(1)} ≤ x^{(1)}, …, X^{(d)} ≤ x^{(d)})` by a joint ECDF, but the convergence rate
of the joint ECDF to the true joint CDF degrades as the number of dimensions grows (Naaman
2021) — the same curse of dimensionality that afflicts density estimation. So the
dimension-free guarantee of the ECDF holds cleanly only per dimension, and there is a tension
between the per-dimension estimator we trust and the joint object we actually want to score.

A separate and relevant framework is **copula theory**. Sklar's theorem (Sklar 1959; Nelsen
2007) states that any joint CDF factors as `F(x) = C(F_1(x^{(1)}), …, F_d(x^{(d)}))` for a
copula `C` that carries all the dependence structure, with the marginals `F_j` carried
separately; for continuous marginals `C` is unique. The empirical copula — built from the
marginal ECDFs — converges to the true copula. Copulas thus offer a principled way to think
about modeling marginals separately from the dependence between them.

A recurring structural idea across fast detectors is to trade full dependence modeling for
per-feature summaries. Under an independence assumption, a joint distribution factors into
marginals and log-space scoring can be accumulated feature by feature. Empirically, this
assumption is known to be "wrong but useful" in outlier detection: detectors that make it run in
linear time and, for global outliers, perform comparably to far more expensive multivariate
methods, paying mainly on *local* outliers that only stand out in the joint structure.

A diagnostic phenomenon that any tail-based scorer must confront: which tail is the outlying
one can differ across features and across datasets. In a two-dimensional toy setting, if the
bulk of the data sits near one corner and rare points fall toward the opposite corner, a
one-sided rule aimed at the wrong tail reverses the intended ranking and flags ordinary points
on the bulk side instead. A rule that simply averages both tails can dilute the very tail signal
it is trying to preserve. Flip the construction and the preferred one-sided rule flips as well.
The pre-method problem is to use tail information without hard-coding the same side for every
feature and without introducing an exponential set of per-feature choices.

## Baselines

These are the prior detectors a new method would be measured against and would react to. For
each: its core idea, its actual mechanism, and the specific limitation it leaves open.

**Local Outlier Factor — LOF (Breunig et al. 2000).** Density-based and local: for each point,
estimate a local reachability density from the distances to its `k` nearest neighbors, then
score the point by the ratio of its neighbors' average local density to its own. A point in a
region much sparser than its neighbors' regions gets a high ratio and is called an outlier.
This elegantly handles outliers that are only anomalous *relative to a local cluster*.
**Limitation:** it rests on `k`-nearest-neighbor search and pairwise distances, so it scales
poorly with `n` and degrades under the curse of dimensionality; and `k` (plus the distance
metric) must be chosen with no labeled signal to guide the choice.

**Distance-to-`k`-th-neighbor detection — kNN (Ramaswamy et al. 2000).** Score each point by
its distance to its `k`-th nearest neighbor (or the average over its `k` neighbors); points far
from everything are outliers. No density model, just distances. **Limitation:** the same
pairwise-distance cost and curse of dimensionality as LOF, and the same untunable `k`.

**One-Class SVM — OCSVM (Schölkopf et al. 2001).** Fit a boundary that encloses most of the
data in an RBF-kernel feature space, with a parameter `nu` that bounds the fraction of points
left outside (treated as outliers); the signed distance to the boundary is the score.
**Limitation:** kernel choice and bandwidth plus `nu` are hyperparameters with no unsupervised
way to set them; training scales superlinearly in `n`; and the decision is a single opaque
distance with no per-feature attribution.

**Isolation Forest — iForest (Liu et al. 2008).** Build an ensemble of random binary trees,
each splitting on a random feature at a random threshold; a point's score is its average
path length to isolation across trees — outliers, being few and different, are isolated by
shorter random paths. Fast and effective, with default 100 trees and subsample size 256.
**Limitation:** the number of trees and the subsample size are knobs, and the ensemble of
random partitions yields a score with no direct interpretation in terms of which features
mattered.

**Histogram-Based Outlier Score — HBOS (Goldstein & Dengel 2012).** Assume the features are
independent. For each feature build a univariate histogram density estimate `hist_j`, and score
a point by summing per-feature negative log densities, `Σ_j −log hist_j(x^{(j)})`: a point that
falls in low-density bins across many features scores high. Linear time, and the independence
assumption is what makes the per-feature histograms combine into a single multivariate score.
**Limitation:** the histogram is a *density* estimate that requires choosing a bin width (or
number of bins) — a hyperparameter with no unsupervised criterion — and the discretization
introduces artifacts at bin edges and is sensitive to how the bins are placed.

**Three-sigma and 1.5·IQR rules (Leys et al. 2013; Dekking et al. 2005).** Per-feature tail
rules: flag points beyond three standard deviations of the mean, or outside the
quartile-based fences. Tuning-free and interpretable per feature. **Limitation:** they
summarize each feature by only the mean and standard deviation (or the quartiles), which
implicitly assumes a symmetric, unimodal, near-Gaussian shape and discards the rest of the
distribution's structure, so they mis-rank tail extremeness whenever the marginal is skewed or
heavy-tailed.

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
