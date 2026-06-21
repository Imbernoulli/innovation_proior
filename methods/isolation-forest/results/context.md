## Research question

Given a set of tabular feature vectors with no labels, find the rare, unusual points — the
"anomalies" — and rank every point by how anomalous it is. Anomalies are *rare* (a small
minority of the data) and *different* (their attribute values stand apart from the bulk), and
the consequences of catching or missing them are large — fraudulent transactions, intrusions,
equipment failures, a new astronomical object. The detector must produce a faithful *ranking*
(so that the truly unusual points float to the top) and operate on data sets with very many
instances, with high dimensionality, and with many attributes that may be irrelevant to whether
a point is anomalous. The question is how to produce such a ranking from unlabelled feature
vectors alone.

## Background

Anomaly detection at this time is dominated by what can be called *model-based profiling*: you
first build a model of what "normal" looks like, then flag whatever deviates from it. The
profile can be statistical (fit a distribution, flag low-likelihood points), classification-
based (train a one-class or multi-class boundary around normal, flag the outside),
clustering-based (cluster the data, flag points far from any cluster), distance-based (a point
is anomalous if it is far from its neighbours), or density-based (a point is anomalous if it
sits in a low-density region). The comprehensive survey of the field (Chandola, Banerjee &
Kumar 2009) catalogues exactly these families.

Two pathologies of anomaly detection are well documented and are load-bearing for what a
scalable detector has to survive (Murphy 1951; studied extensively since). **Swamping**:
normal instances that lie close to anomalies (or that are themselves spread out) get wrongly
flagged as anomalies, because the boundary between ordinary variation and true abnormality
blurs. **Masking**: when anomalies arrive in a large, dense cluster, the cluster conceals its
own members; there are enough of them, packed tightly enough, that they resemble a legitimate
dense region rather than a clump of outliers.

Tree ensembles are also part of the standing background, independent of anomaly detection.
Random Forests show that many high-variance randomized trees can be averaged into a stable
predictor, with cheap parallel construction and useful robustness to irrelevant attributes.
Separately, classical analysis of random binary trees gives harmonic-number relationships for
expected tree depth and search cost (Preiss 1999; Knuth, TAOCP vol. 3).

## Baselines

**Distance-based outliers / ORCA (Bay & Schwabacher, KDD 2003; built on Knorr & Ng 1998 and
Ramaswamy, Rastogi & Shim 2000).** A point's anomaly score is its distance to its k-th nearest
neighbour, or the sum of distances to its k nearest neighbours — far-from-everything means
anomalous. A naive nested-loop computation is O(n^2). ORCA makes it near-linear *in practice*
by processing points in randomized order and maintaining a running cutoff — the score of the
n-th most anomalous point found so far; once a candidate's partial k-NN score drops below the
cutoff it can be pruned.

**LOF — Local Outlier Factor (Breunig, Kriegel, Ng & Sander, SIGMOD 2000).** Density-based and
*local*. For a point A, with k-distance(B) the distance to B's k-th neighbour,

```
reach-dist_k(A,B) = max( k-distance(B), d(A,B) ),
lrd_k(A) = |N_k(A)| / sum_{B in N_k(A)} reach-dist_k(A,B),       (local reachability density)
LOF_k(A) = (1/|N_k(A)|) sum_{B in N_k(A)} lrd_k(B) / lrd_k(A).
```

LOF ~ 1 means A is as dense as its neighbours (normal); LOF >> 1 means A is much sparser than
its neighbours (an outlier). The locality lets LOF handle clusters of differing density, which
plain distance methods cannot.

**Random Forests (Breiman 2001).** An ensemble of decision trees, each grown on a bootstrap
sample with random feature subsets at each split, aggregated across trees — accurate, robust
to irrelevant features, and embarrassingly parallel. Each tree is grown by an impurity
criterion to separate labelled *classes*, so it produces a class prediction.

**One-Class SVM (Schölkopf, Platt, Shawe-Taylor, Smola & Williamson 2001).** Fits a boundary
(an RBF-kernel hyperplane separating the data from the origin) enclosing the bulk of the data,
with a parameter nu controlling the fraction allowed outside; points outside score as
anomalous.

**Statistical / classification / clustering profiling (Chandola et al. 2009).** Fit a model of
normality (a parametric distribution, a one-class classifier, a clustering) and flag
low-likelihood / outside-the-boundary / far-from-cluster points.

## Evaluation settings

The natural yardsticks are unlabelled tabular training features with held-out labels used only
after scoring. Common benchmark families include network traffic, remote sensing, shuttle
telemetry, medical, image-derived, and synthetic data sets, spanning small to very large sample
counts and low to high dimensionality. Representative names in the anomaly-detection literature
include Http, ForestCover, Shuttle, Mammography, Annthyroid, Satellite, Pima, Breastw,
Arrhythmia, Ionosphere, and controlled synthetic data for dense anomaly clusters, swamping, and
irrelevant attributes.

The detector is fit on feature vectors only, without using labels. Evaluation then ranks test
instances by the detector's real-valued score and compares that ranking with held-out anomaly
labels. The central metrics are area under the ROC curve for ranking quality and wall-clock CPU
time / memory for scalability; thresholded precision-recall or F-measure can be reported when a
binary decision threshold is part of the benchmark protocol.

## Code framework

The detector slots into a generic unsupervised-scoring harness. The data pipeline hands over a
numeric feature matrix; the detector exposes `fit(X)` (features only, no labels) and
`decision_function(X)` (one real score per row, higher means more anomalous). The array and
random-number primitives already exist. What does not exist is the rule that learns from the
unlabelled feature matrix and turns each row into a score. That rule is the one empty slot.

```python
import numpy as np


class CustomAnomalyDetector:
    """Unsupervised anomaly scorer over standardized tabular features.

    fit() sees features only (no labels); decision_function() returns one score
    per row, higher = more anomalous.
    """

    def __init__(self):
        # TODO: any hyperparameters and internal state the scorer needs.
        pass

    def fit(self, X):
        # X: numpy array (n_samples, n_features), no labels.
        # TODO: learn whatever summary of the unlabelled features the scorer needs.
        return self

    def decision_function(self, X):
        # X: numpy array (n_samples, n_features)
        # return: numpy array (n_samples,), higher = more anomalous.
        # TODO: compute the per-row anomaly score.
        pass
```

The outer protocol fits on training features, scores the test features, and evaluates the
ranking against held-out labels; `fit` and `decision_function` are where the scoring rule will
live.
