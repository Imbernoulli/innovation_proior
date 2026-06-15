**Problem.** The subgroup-aware run left the *worst* subgroup barely corrected (worst-group ECE mean
0.435; COMPAS 0.497), because empirical-Bayes shrinkage routes the small worst group through the global
temperature anyway. I need the lowest-variance control — a single global scalar with no per-group
parameters — to isolate how much the subgroup machinery actually bought, and to establish the honest
floor under this shift.

**Key idea.** Plain temperature scaling: `q = σ(z/T)`, `z = logit(p)`, one positive scalar fit by
minimizing the calibration-split NLL. It softens (`T>1`) the dominant "logits uniformly too large"
over-confidence, it is the maximum-entropy correction of exactly that scale error, and being monotone in
`z` it leaves ranking, accuracy, and `subgroup_auroc` untouched. `groups` is accepted and ignored — this
is the group-agnostic baseline by construction.

**Why.** A single scalar estimated from the whole calibration set has the least variance of anything on
this ladder, so it should transfer to the shifted test tail better than the noisy per-group fits did. It
is the one-parameter Platt special case with the intercept dropped (a nonzero intercept would move the
`z=0` boundary and change predictions). NLL is a proper scoring rule; ECE is non-differentiable and only
measured.

**Hyperparameters.** `eps = 1e-6` (clip `p` so the logit is finite); fit `log T ∈ [−3, 3]` (`T ≈
[0.05, 20]`) by a bounded 1-D scalar search — `log T` because `T>0` is multiplicative; fall back to
`T = 1.0` if the search does not converge.

```python
class CalibrationMethod:
    """Global temperature scaling on positive-class probabilities."""

    def __init__(self):
        self.eps = 1e-6
        self.temperature_ = 1.0

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))

        def objective(log_t):
            t = float(np.exp(log_t))
            cal = special.expit(logits / t)
            p = np.clip(cal, self.eps, 1.0 - self.eps)
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        result = optimize.minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        self.temperature_ = float(np.exp(result.x)) if result.success else 1.0
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))
        return np.clip(special.expit(logits / self.temperature_), self.eps, 1.0 - self.eps)
```
