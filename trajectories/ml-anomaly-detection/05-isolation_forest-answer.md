**Problem (from step 4).** The whole tail-probability family (COPOD, ECOD) plateaued on Satellite
(0.634, 0.566) because both judge each feature on its own marginal and multiply under an
*independence* assumption — invisible to an anomaly that is ordinary on every axis but lands in a
hole of the *joint* distribution. ECOD removing the contamination knob did not move Satellite up, so
the gap is the independence assumption, not the knob. I need a method that responds to joint
placement, cheaply, without the density/distance bill that killed OCSVM at `d=36`.

**Key idea.** Isolate, don't profile. Recursively partition the feature space with random
axis-aligned cuts (random attribute, random split in its current range) until points are alone.
"Few and different" means anomalies sit in sparse regions of the *joint* space, so a random cut
peels them off early — short root-to-leaf path length — while normal points in dense regions take
many cuts. Average the path length over an ensemble of random trees, normalize by the average
unsuccessful-search depth of a random BST `c(ψ) = 2H(ψ−1) − 2(ψ−1)/ψ`, and score
`s(x) = 2^(−E(h(x))/c(ψ))` (higher = more anomalous). The path length responds to joint emptiness,
not to any single marginal — exactly what the tail methods missed.

**Why it works (and why it tops the ladder).** Isolation is the first mechanism on the ladder
sensitive to joint structure. Crucially, each tree is built from a small sub-sample (`ψ = 256`)
without replacement, which relieves *swamping* (normals crowding an anomaly inflate its path) and
*masking* (a large dense anomaly cluster shields its own members) — both are too-much-data
pathologies, and masking is exactly what Satellite's ~32% anomaly rate inflicts. Sub-sampling also
caps cost at `ψ`, not `n`, so it scales to Shuttle's 49k rows.

**Scaffold detail (task-specific).** Like COPOD/ECOD and unlike LOF/OCSVM, no internal
re-normalization is needed (random axis-aligned cuts use each feature's own min/max, so the
harness's StandardScaler is harmless). `contamination=0.1` here sets *only* the predict threshold,
not the ranking score, so it does not affect AUROC. The fill is the PyOD `IForest` library
realization of the path-length score (sklearn `IsolationForest` backend), instantiated explicitly.

**Hyperparameters.** PyOD `IForest(n_estimators=100, max_samples="auto"` (=`min(256,n)`)`,
contamination=0.1, random_state=SEED)`. `decision_function` returns higher = more anomalous.

```python
class CustomAnomalyDetector:
    """Isolation Forest anomaly detector.

    Ensemble of random isolation trees. Anomaly score is based on the
    average path length to isolate each sample.
    """

    def __init__(self):
        from pyod.models.iforest import IForest

        self.model = IForest(
            n_estimators=100,
            max_samples="auto",
            contamination=0.1,
            random_state=SEED,
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)
```
