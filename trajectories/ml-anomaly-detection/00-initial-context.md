## Research question

Unsupervised anomaly detection on tabular data: given an unlabeled matrix of `n` rows and `d`
features, hand back one number per row that says how anomalous it is. The single thing being
designed is the **scoring rule** — how to model "normal structure" on standardized tabular
features and assign higher scores to deviating points — using only the unlabeled features at fit
time. Everything around it (the split, the standardization, the metrics) is frozen. The rule must
generalize across datasets that differ wildly in size (~1.8k to ~49k rows), dimensionality (6 to
36 features), and anomaly rate (~2.5% to ~32%), because there is no per-dataset label signal to
tune against.

## Prior art / Background / Baselines

- **Statistical tail rules (3σ, 1.5·IQR).** Core idea: flag a value when it exceeds a fixed
  multiple of the feature's spread. Observed gap: a two-number caricature (mean/sd or quartiles)
  assumes a roughly symmetric, Gaussian marginal; the yardstick is wrong on skewed or heavy-tailed
  features.
- **Distance-based detection (DB(p,D); D^k).** Core idea: score a point by its distance to its
  k-th nearest neighbor, treating far-from-everything as anomalous. Observed gap: a single global
  distance scale cannot separate a genuine anomaly beside a dense cluster from an ordinary member
  of a sparse cluster, and the neighbor search is heavy and near-superlinear.
- **Density estimation (histograms, KDE, HBOS).** Core idea: estimate a per-feature density and
  score points by how low the density is where they land. Observed gap: it requires a bin count or
  bandwidth that has no label-free way to be set, and it conflates tail rarity with low-density
  valleys between clumps.
- **Profiling / boundary models (clustering, one-class boundaries).** Core idea: build a model of
  the normal bulk and flag whatever falls outside it. Observed gap: capacity goes into describing
  the majority, while the quantity of interest is the rare minority at the edges.

## Fixed substrate / Code framework

A single-file benchmark harness is frozen and must not be touched. It loads one ADBench/ODDS
dataset (Cardio, Thyroid, Satellite, Shuttle) as a feature matrix `X` and binary labels `y`
(0 normal, 1 anomaly), makes a **60/40 stratified train/test split** at the run's seed, fits a
`StandardScaler` on the training rows (so the detector always sees zero-mean, unit-variance
features), transforms both splits, then fits the detector on the *unlabeled* training features and
scores the test features. `roc_auc_score` and an F1 at the contamination-percentile threshold are
read off the test scores. The harness catches exceptions and falls back to AUROC 0.5 / F1 0.0, so
a detector that throws scores at the floor. The labels are never visible to the detector — `fit(X)`
receives features only.

## Editable interface

Exactly one region is editable: the `CustomAnomalyDetector` class in
`scikit-learn/custom_anomaly.py` (lines 160–212). Every method is a fill of this same three-method
contract:

- `__init__(self)` — set hyperparameters and build the internal model; no labels, no data yet.
- `fit(self, X)` — train on the unlabeled, standardized training matrix `X` of shape
  `(n_samples, n_features)`; return `self`.
- `decision_function(self, X)` — return anomaly scores of shape `(n_samples,)` for the
  standardized rows `X`, **higher = more anomalous**.

Available, pre-installed: `numpy`, `scipy`, `scikit-learn` (PCA, KernelDensity, NearestNeighbors,
GaussianMixture, …), and `pyod` (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, …). The
module-level `SEED` is in scope inside the class. The starting point is the scaffold default — a
plain Isolation-Forest wrapper at PyOD defaults — and each method replaces exactly this class body
and nothing else.

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
