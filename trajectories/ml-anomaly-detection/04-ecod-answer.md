**Problem (from step 3).** COPOD confirmed the per-feature tail family — Satellite stabilized
(0.634, zero seed variance) where OCSVM's kernel had wobbled — but two residuals remained: a
baked-in `contamination=0.1` that crushed Thyroid's F1 to 0.180 against its ~2.5% true rate, and a
copula framing whose aggregation I wanted to clean. I want the same tail object with *no* knob at
all and a tighter, derivation-grounded aggregation.

**Key idea.** Score each row by per-dimension empirical-CDF tail probabilities, parameter-free. Per
feature, `F̂_left(z) = (1/n) Σ_i 1{X_i ≤ z}` (left tail) and `F̂_right(z) = (1/n) Σ_i 1{X_i ≥ z}`
(right tail, via the ECDF of the negated column to dodge the `1 − F̂(max) = 0 → log 0 = ∞`
boundary). The ECDF is Glivenko–Cantelli/DKW-certified — tuning-free and *dimension-free in one
dimension*. Build three views — left-only `O_left`, right-only `O_right`, and skewness-corrected
`O_auto` (pick the tail by the sign of each feature's skewness) — and take the most extreme,
`O(x) = max{O_left, O_right, O_auto}` (canonical code: per-feature max of `{U_left, U_right,
U_skew}`, then sum). No contamination argument enters the model.

**Why it works.** Same dimension-immune, deterministic, interpretable per-feature object as COPOD,
but parameter-free: the score is fixed entirely by the data, so the only rate in the pipeline is the
harness's own thresholding against the true test contamination — which is what should fix Thyroid's
F1. The `max`-of-three-views aggregation never loses a one-sided outlier to a bad skew call.

**Scaffold detail (task-specific).** Like COPOD and unlike LOF/OCSVM, ECOD needs no internal
re-normalization (the ECDF rank is invariant to StandardScaler). The canonical ECOD
`decision_function` is **transductive** — it pools the stored training rows with the rows being
scored, computes the ECDFs on the pooled matrix, and returns only the new rows' scores, ranking
test points against the established training distribution. PyOD's default ECOD does exactly this,
and it is the protocol the reference numbers use, so the rung is a direct PyOD `ECOD()` wrap.

**Hyperparameters.** None. PyOD `ECOD()` at defaults; `decision_function` returns higher = more
anomalous.

```python
class CustomAnomalyDetector:
    """ECOD anomaly detector (PyOD default, matches ADBench)."""

    def __init__(self):
        from pyod.models.ecod import ECOD

        self.model = ECOD()

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)
```
