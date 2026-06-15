**Problem.** Fill the scaffold's one editable slot with a first detector that encodes a real
property of anomalies, so its measured strengths and failures set the agenda for the ladder. Plain
distance-based detection compares every point to one global distance scale and so cannot tell a
genuine anomaly beside a dense cluster from an ordinary member of a sparse cluster.

**Key idea.** Make the verdict *local* and *graded*. Estimate each point's density from its
`MinPts`-nearest-neighbor distances (no distribution, no global volume), stabilize it by flooring
every distance to a neighbor `o` at `o`'s own k-distance — the reachability distance
`reach-dist(p,o) = max{k-distance(o), d(p,o)}` — invert the average into a local reachability
density `lrd(p)`, then score each point by the *ratio* of its neighbors' densities to its own,
`LOF(p) = mean_o lrd(o)/lrd(p)`. The absolute density cancels, so interior points of any cluster
score ≈ 1 and a point in a sparse pocket surrounded by dense neighbors scores well above 1.

**Why it works (and why it is the floor).** The ratio is the minimal fix to global-scale distance
detection — it is local relative-density. But it is a neighbor method (a k-NN bill, and distances
concentrate as `d` grows), it is blind to global outliers that hide inside a locally-coherent
clump, and it carries no notion of per-feature tail extremeness — exactly the gaps the later rungs
exist to close.

**Scaffold detail (task-specific).** LOF's neighbor graph is decided by feature scaling *before*
any reachability is computed, and the reference LOF numbers on these datasets come from the
ADBench protocol, which min-max normalizes features to [0,1]. The harness instead pre-applies a
StandardScaler, a different geometry. So the rung re-normalizes internally: fit a `MinMaxScaler` on
the training rows, fit `LOF` on the [0,1] features, and apply the same fitted scaler at score time.
This restores LOF's calibrated geometry without changing the algorithm.

**Hyperparameters.** PyOD `LOF()` defaults (no label-free tuning): `n_neighbors=20`,
`algorithm='auto'`, `metric='minkowski'`, `p=2`, `contamination=0.1`; internal `MinMaxScaler`.
PyOD's `decision_function` is already oriented higher = more anomalous.

```python
class CustomAnomalyDetector:
    """Local Outlier Factor anomaly detector (ADBench protocol).

    Applies MinMax normalization internally to match the preprocessing
    used by ADBench (data_generator.py: MinMaxScaler().fit(X_train)).
    LOF is density-based and extremely sensitive to feature scaling,
    so this is required to reproduce the Table D4 numbers.
    """

    def __init__(self):
        from pyod.models.lof import LOF

        # PyOD defaults (matches ADBench with no hyperparameter tuning):
        # n_neighbors=20, algorithm='auto', metric='minkowski', p=2,
        # contamination=0.1.
        self.model = LOF()
        self._scaler = None

    def fit(self, X):
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler()
        Xs = self._scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def decision_function(self, X):
        Xs = self._scaler.transform(X)
        return self.model.decision_function(Xs)
```
