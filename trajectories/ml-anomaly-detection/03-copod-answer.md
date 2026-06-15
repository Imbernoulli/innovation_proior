**Problem (from step 2).** OCSVM lifted the easy global-tail datasets but stalled on Satellite
(AUROC 0.591, swinging 0.55–0.67 across seeds) because its fixed-bandwidth RBF region degrades at
high `d` around a heavily-contaminated cloud, and its `nu=0.5` contamination knob mis-set the F1
threshold on low-rate datasets. Both failures are hyperparameters tuned blind. I need a scoring
rule with no knobs in the ranking, immune to dimension, and deterministic.

**Key idea.** Keep HBOS's skeleton — a sum over features of a per-feature anomaly contribution
under an independence assumption, with a log for numerical stability — but swap the per-feature
quantity from histogram *density* to empirical-CDF *tail probability*. For feature `j`,
`u_j = F̂_j(x_j) = (1/n) Σ_i 1{X_{j,i} ≤ x_j}` is the canonical position-in-distribution (probability
integral transform), so `u_j` near 0 = far in the lower tail, with no bins. By Sklar's theorem the
joint left tail is a copula of the marginals; the independence (product) copula gives
`−log Π_j u_j = −Σ_j log u_j`. Add the right tail via the negated-data ECDF (avoiding the
`F̂(max)=1 → −log 0 = ∞` boundary), select the informative tail per feature by skewness sign, and
guard the sign with a two-tail-average safety net by taking, per feature,
`max(U_skew, (U_l+U_r)/2)`, then summing.

**Why it works.** It never computes a `d`-dimensional distance or region, so the curse of
dimensionality that crippled OCSVM on Satellite cannot bite; the log-sum tames the product
underflow; the ECDF rank is deterministic and parameter-free. The only external number,
`contamination`, sets the binary label threshold, not the score ranking. The additive structure
makes the score interpretable per feature for free.

**Scaffold detail (task-specific).** Unlike the LOF and OCSVM rungs, COPOD needs **no** internal
re-normalization: its score is the ECDF *rank* of each value, invariant to any monotone per-feature
transform, so the harness's StandardScaler is harmless. The rung is a direct PyOD `COPOD` wrap.

**Hyperparameters.** PyOD `COPOD(contamination=0.1)` (parameter-free score; `contamination` only
sets the label cut). PyOD's `decision_function` returns higher = more anomalous.

```python
class CustomAnomalyDetector:
    """COPOD: Copula-Based Outlier Detection.

    Parameter-free method using empirical copula functions to model
    the joint tail probability of observations across features.
    """

    def __init__(self):
        from pyod.models.copod import COPOD

        self.model = COPOD(contamination=0.1)

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)
```
