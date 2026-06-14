## Research question

Given a set of tabular feature vectors with no labels, find the rare, unusual points — the
"anomalies" — and rank every point by how anomalous it is. The application demands are
specific: anomalies are *rare* (a small minority of the data) and *different* (their attribute
values stand apart from the bulk), and the consequences of catching or missing them are large
— fraudulent transactions, intrusions, equipment failures, a new astronomical object. So a
detector has to do two things well at once. It must produce a faithful *ranking* (so that the
truly unusual points float to the top), and it must do so *cheaply enough to scale* — to data
sets with very many instances, with high dimensionality, and with many attributes that are
simply irrelevant to whether a point is anomalous.

The pain is that the methods of the day buy ranking quality at a cost that does not scale, or
scale at a cost in ranking quality. Concretely, a usable detector would need: (1) detection
accuracy competitive with the best distance/density methods; (2) (near-)linear time in the
number of instances with a small constant, and low, bounded memory; (3) graceful behaviour as
dimensionality and the number of irrelevant attributes grow; (4) usefulness even when the
training data contains no labelled anomalies (or none at all); and (5) few knobs to tune. No
existing method achieves all of this simultaneously. Closing that gap is the problem.

## Background

Anomaly detection at this time is dominated by what can be called *model-based profiling*: you
first build a model of what "normal" looks like, then flag whatever deviates from it. The
profile can be statistical (fit a distribution, flag low-likelihood points), classification-
based (train a one-class or multi-class boundary around normal, flag the outside),
clustering-based (cluster the data, flag points far from any cluster), distance-based (a point
is anomalous if it is far from its neighbours), or density-based (a point is anomalous if it
sits in a low-density region). The comprehensive survey of the field (Chandola, Banerjee &
Kumar 2009) catalogues exactly these families.

Two pain points run through the profiling approach. First, an objective mismatch: the detector
is *optimized to describe normal instances*, not to *separate the anomalies*. A model that
fits the bulk of the data well can still be poorly tuned for the rare points at the
extremes — the result is too many false alarms (normal points flagged) or too few detections
(anomalies missed), because the model's effort went into the part of the data we care least
about. Second, a cost barrier: the distance- and density-based methods, which tend to give the
best rankings, rest on nearest-neighbour or pairwise-distance computations. Those scale poorly
— quadratically in the worst case — and the cost grows with dimensionality, so these methods
are in practice confined to small, low-dimensional data sets.

Two pathologies of anomaly detection are well documented and are load-bearing for what a
scalable detector has to survive (Murphy 1951; studied extensively since). **Swamping**:
normal instances that lie close to anomalies (or that are themselves spread out) get wrongly
flagged as anomalies, because the boundary between ordinary variation and true abnormality
blurs. **Masking**: when anomalies arrive in a large, dense cluster, the cluster conceals its
own members; there are enough of them, packed tightly enough, that they resemble a legitimate
dense region rather than a clump of outliers. These failures warn that both local
neighbourhoods and global data composition can distort a score.

Tree ensembles are also part of the standing background, independent of anomaly detection.
Random Forests show that many high-variance randomized trees can be averaged into a stable
predictor, with cheap parallel construction and useful robustness to irrelevant attributes.
Separately, classical analysis of random binary trees gives harmonic-number relationships for
expected tree depth and search cost (Preiss 1999; Knuth, TAOCP vol. 3). Those facts are
available mathematical tools, but they do not by themselves define a detector.

## Baselines

The prior methods a new detector would be measured against and would react to.

**Distance-based outliers / ORCA (Bay & Schwabacher, KDD 2003; built on Knorr & Ng 1998 and
Ramaswamy, Rastogi & Shim 2000).** A point's anomaly score is its distance to its k-th nearest
neighbour, or the sum of distances to its k nearest neighbours — far-from-everything means
anomalous. A naive nested-loop computation is O(n^2). ORCA makes it near-linear *in practice*
by processing points in randomized order and maintaining a running cutoff — the score of the
n-th most anomalous point found so far; once a candidate's partial k-NN score drops below the
cutoff it can be pruned. **Limitation:** the cost is the distance computation itself, and the
worst case remains quadratic; the score is purely local-geometric so it reliably surfaces
*scattered* outliers but not anomalies packed into a dense clump; and to find a dense anomaly
cluster k must be set larger than that cluster, while a large k drives the cost up sharply.

**LOF — Local Outlier Factor (Breunig, Kriegel, Ng & Sander, SIGMOD 2000).** Density-based and
*local*. For a point A, with k-distance(B) the distance to B's k-th neighbour,

```
reach-dist_k(A,B) = max( k-distance(B), d(A,B) ),
lrd_k(A) = |N_k(A)| / sum_{B in N_k(A)} reach-dist_k(A,B),       (local reachability density)
LOF_k(A) = (1/|N_k(A)|) sum_{B in N_k(A)} lrd_k(B) / lrd_k(A).
```

LOF ~ 1 means A is as dense as its neighbours (normal); LOF >> 1 means A is much sparser than
its neighbours (an outlier). The locality lets LOF handle clusters of differing density, which
plain distance methods cannot. **Limitation:** it computes a k-NN neighbourhood for every
point — the same distance cost — and being *local* it can be fooled: a point sitting in a
locally dense little group can have LOF ~ 1 yet be a genuine *global* anomaly relative to the
whole data set. There is also no clean rule for the LOF threshold, and the choice of k matters.

**Random Forests (Breiman 2001).** An ensemble of decision trees, each grown on a bootstrap
sample with random feature subsets at each split, aggregated across trees — accurate, robust
to irrelevant features, and embarrassingly parallel. **Limitation:** it is *supervised*: each
tree is grown by an impurity criterion to separate labelled *classes*, so it needs labels and
produces a class prediction, not an unlabelled anomaly ranking.

**One-Class SVM (Schölkopf, Platt, Shawe-Taylor, Smola & Williamson 2001).** Fits a boundary
(an RBF-kernel hyperplane separating the data from the origin) enclosing the bulk of the data,
with a parameter nu controlling the fraction allowed outside; points outside score as
anomalous. **Limitation:** kernel methods scale poorly in n, and nu plus the kernel bandwidth
must be tuned to the data.

**Statistical / classification / clustering profiling (Chandola et al. 2009).** Fit a model of
normality (a parametric distribution, a one-class classifier, a clustering) and flag
low-likelihood / outside-the-boundary / far-from-cluster points. **Limitation:** the objective
mismatch above — effort spent describing the normal bulk — plus brittle assumptions
(Gaussianity, cluster shape) and, again, cost on large/high-dim data.

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
