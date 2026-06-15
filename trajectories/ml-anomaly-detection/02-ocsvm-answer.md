**Problem (from step 1).** LOF ranked these datasets at chance (AUROC 0.53–0.55 on three of
four), because its *local* density contrast is blind to the global tail events that dominate these
anomalies and washes out under distance concentration. I need a *global* model of the region the
normal mass occupies, so a point outside it scores high regardless of local neighborhood coherence.

**Key idea.** Estimate the *support* of the data, not its density. Map the points into an RBF
feature space — where every point has unit norm and all pairwise angles are acute, so the data sit
on a spherical patch separable from the origin — and find the maximum-margin hyperplane separating
the data from the origin. The accepted region is the spherical cap on the data side; a test point's
signed distance to the hyperplane is its (inlier) score, negated for anomaly scoring. The dual is a
box-plus-equality QP, `min ½ Σ α_iα_j k(x_i,x_j)` s.t. `0 ≤ α_i ≤ 1/(νℓ)`, `Σ α_i = 1`.

**Why it works.** The `½‖w‖²` regularizer controls flatness in feature space (a computable proxy
for the uncomputable input-space minimum-volume set), and the kernel makes the boundary nonlinear.
The KKT conditions give `ν` a counting meaning: outliers are pinned at the box ceiling `1/(νℓ)`, so
the training outlier fraction is ≤ `ν` and the support-vector fraction is ≥ `ν` — one interpretable
contamination knob, unlike LOF's geometric `n_neighbors`. This is global rarity: a point is
anomalous because it falls outside the whole cloud's support.

**Scaffold detail (task-specific).** The RBF exponent is an input-space distance, so OCSVM is
scale-sensitive; the default `gamma='auto'` (= `1/n_features`) and the reference numbers assume the
ADBench min-max-to-[0,1] geometry, but the harness pre-applies StandardScaler. So the rung
re-normalizes internally with a re-applied `MinMaxScaler`, same as the LOF rung — the algorithm is
unchanged, only the geometry is restored.

**Hyperparameters.** PyOD `OCSVM()` defaults: `kernel='rbf'`, `nu=0.5`, `gamma='auto'`
(=`1/n_features`); internal `MinMaxScaler`. PyOD's `decision_function` already returns higher = more
anomalous.

```python
class CustomAnomalyDetector:
    """One-Class SVM anomaly detector (ADBench protocol).

    Applies MinMax normalization internally to match ADBench's
    preprocessing. Uses PyOD defaults: kernel='rbf', nu=0.5,
    gamma='auto' (= 1/n_features).
    """

    def __init__(self):
        from pyod.models.ocsvm import OCSVM

        # PyOD default: kernel='rbf', nu=0.5, gamma='auto'.
        self.model = OCSVM()
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
