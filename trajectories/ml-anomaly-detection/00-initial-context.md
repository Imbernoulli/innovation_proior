## Research question

Unsupervised anomaly detection on tabular data: given an unlabeled matrix of `n` rows and `d`
features, hand back one number per row that says how anomalous it is. The single thing being
designed is the **scoring rule** — how to model "normal structure" on standardized tabular
features and assign higher scores to deviating points — using only the unlabeled features at fit
time. Everything around it (the split, the standardization, the metrics) is frozen. The rule must
generalize across datasets that differ wildly in size (~1.8k to ~49k rows), dimensionality (6 to
36 features), and anomaly rate (~2.5% to ~32%), because there is no per-dataset label signal to
tune against.

## Prior art before the first rung (the unsupervised-detection lineage)

The ladder's first rung reacts to a line of classical unsupervised detectors. These precede the
ladder; each is a way to answer "what is normal, and how far is this point from it," and each has
a gap the rung below it inherits or tries to escape.

- **Statistical tail rules (3σ, 1.5·IQR; Hawkins 1980).** Flag a value if it lies beyond a fixed
  multiple of the spread — three standard deviations, or 1.5 interquartile ranges past the
  quartiles. Tuning-free and per-feature, but they describe each feature by two numbers (mean/sd
  or the quartiles) and so implicitly assume a symmetric, roughly Gaussian shape; on a skewed or
  heavy-tailed marginal the yardstick is wrong. Gap: a two-number caricature of each feature's
  distribution.
- **Distance-based detection (DB(p,D), Knorr & Ng 1998; D^k, Ramaswamy et al. 2000).** Score a
  point by its distance to its k-th nearest neighbor; far-from-everything is anomalous.
  Distribution-free and rankable, but it compares every point to a *single global distance scale*,
  so it cannot tell a genuine anomaly beside a dense cluster from an ordinary member of a sparse
  cluster — their absolute distances coincide. And the neighbor search is near-/super-linear with
  heavy constants. Gap: one global scale, and a distance bill that does not scale.
- **Density estimation (histograms, KDE, HBOS; Goldstein & Dengel 2012).** Estimate a per-feature
  density and score by how low the density is where a point lands. Closer to the right object, but
  it needs a bin count or a bandwidth — exactly the knob there is no label-free way to set — and
  it conflates "rare because far out in a tail" with "rare because it fell in a low-density valley
  between two clumps." Gap: a tuning knob, and density confused with extremeness.
- **Profiling / boundary models (clustering, one-class boundaries).** Build a model of the normal
  bulk and flag whatever falls outside. The objective is aimed at the wrong target: capacity goes
  into describing the majority, while the thing of interest is the rare minority at the edges.
  Gap: optimizes a description of normality rather than a separation of anomalies.

The ladder below is a sequence of fills of the same scaffold slot, each reacting to the measured
failure of the one before it.

## The fixed substrate

A single-file benchmark harness is frozen and must not be touched. It loads one ADBench/ODDS
dataset (Cardio, Thyroid, Satellite, Shuttle) as a feature matrix `X` and binary labels `y`
(0 normal, 1 anomaly), makes a **60/40 stratified train/test split** at the run's seed, fits a
`StandardScaler` on the training rows (so the detector always sees zero-mean, unit-variance
features), transforms both splits, then fits the detector on the *unlabeled* training features and
scores the test features. `roc_auc_score` and an F1 at the contamination-percentile threshold are
read off the test scores. The harness catches exceptions and falls back to AUROC 0.5 / F1 0.0, so
a detector that throws scores at the floor. The labels are never visible to the detector — `fit(X)`
receives features only.

## The editable interface

Exactly one region is editable: the `CustomAnomalyDetector` class in
`scikit-learn/custom_anomaly.py` (lines 160–212). Every method on the ladder is a fill of this
same three-method contract:

- `__init__(self)` — set hyperparameters and build the internal model; no labels, no data yet.
- `fit(self, X)` — train on the unlabeled, standardized training matrix `X` of shape
  `(n_samples, n_features)`; return `self`.
- `decision_function(self, X)` — return anomaly scores of shape `(n_samples,)` for the
  standardized rows `X`, **higher = more anomalous**.

Available, pre-installed: `numpy`, `scipy`, `scikit-learn` (PCA, KernelDensity, NearestNeighbors,
GaussianMixture, …), and `pyod` (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, …). The
module-level `SEED` is in scope inside the class. The starting point is the scaffold default — a
plain Isolation-Forest wrapper at PyOD defaults — and each later method replaces exactly this
class body and nothing else.

```python
# EDITABLE region of scikit-learn/custom_anomaly.py (lines 160-212) — default fill
class CustomAnomalyDetector:
    """Custom unsupervised anomaly detection algorithm.

    fit(X): train on unlabeled, standardized X (n_samples, n_features).
    decision_function(X): return anomaly scores (n_samples,), higher = more anomalous.
    """

    def __init__(self):
        # Default: simple Isolation Forest wrapper
        from pyod.models.iforest import IForest

        self.model = IForest(random_state=SEED)

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)
```

## Evaluation settings

Four tabular benchmarks spanning the regime ranges: **Cardio** (1,831 rows, 21 features, ~9.6%
anomalies), **Thyroid** (3,772 rows, 6 features, ~2.5% anomalies), **Satellite** (6,435 rows, 36
features, ~31.6% anomalies), **Shuttle** (49,097 rows, 9 features, ~7.2% anomalies). Each is run
over three seeds {42, 123, 456}, each seed a fresh 60/40 stratified split with its own fitted
standardizer. Two metrics per dataset, higher is better on both: **AUROC** (area under the ROC
curve — pure ranking quality, threshold-free) and **F1** (at the threshold set by the test-set
contamination percentile — decision quality after thresholding). A method is judged on all eight
columns (four datasets × two metrics) across the three seeds.
