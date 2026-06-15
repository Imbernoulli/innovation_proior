**Problem.** A frozen binary classifier emits over-confident positive-class probabilities, and the
miscalibration differs by subgroup; the metric is the *worst* subgroup's ECE under a calibration→test
distribution shift. I may only learn a map `p ↦ q` (optionally using the subgroup id), on a small,
shifted calibration split. The task is about subgroups, so the first move attacks subgroups directly.

**Key idea.** Temperature scaling on logits: `q = σ(z/T)`, `z = logit(p)`, a single positive scalar
that softens (`T>1`) over-confidence, is monotone in `z` so it preserves ranking and accuracy, and is
the maximum-entropy correction of the dominant "logits uniformly too large" failure. One global `T`
cannot serve heterogeneous subgroups, so fit a temperature per group — but fitting each independently
from its own small sample is the inadmissible coordinatewise MLE (variance `∝ 1/n_g`), so **shrink each
group's temperature toward the pooled global temperature**, in log-space, by the empirical-Bayes weight
`α_g = n_g/(n_g + k)`. Small or label-degenerate groups fall back to global; large groups keep their own.

**Why.** NLL (a proper scoring rule) is the fit objective; ECE is non-differentiable and only measured.
A single positive scalar per group never moves the `z=0` boundary, so accuracy and `subgroup_auroc` are
unchanged. The log-space shrink respects `T`'s multiplicative scale; the two-level Gaussian model forces
`α_g = n_g/(n_g+k)` with `k` the crossover pseudo-count. The hard floor guards the unidentified case
(too few points or one class) where the local 1-D NLL hits a box boundary.

**Hyperparameters.** `eps = 1e-6` (clip `p,q` so `logit`/`log` stay finite); `k_shrink = 200` (50/50 at
200 points; fixed, not estimated, because too few groups to estimate the between-group variance);
local-fit floor `n_g ≥ 20` and `≥ 2` distinct labels; `log T ∈ [−3, 3]` bounded 1-D search. Degenerates
to plain global temperature scaling when `groups is None`.

```python
class CalibrationMethod:
    """Group temperature scaling with James-Stein shrinkage to global T."""

    def __init__(self):
        self.eps = 1e-6
        self.k_shrink = 200.0
        self.group_temperatures_ = {}
        self.global_temperature_ = 1.0

    def _fit_temperature(self, probs, labels):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))

        def objective(log_t):
            t = float(np.exp(log_t))
            cal = special.expit(logits / t)
            p = np.clip(cal, self.eps, 1.0 - self.eps)
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        result = optimize.minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        return float(np.exp(result.x)) if result.success else 1.0

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self.global_temperature_ = self._fit_temperature(probs, labels)
        log_T_global = float(np.log(self.global_temperature_))
        self.group_temperatures_ = {}
        if groups is None:
            return self
        groups = np.asarray(groups).reshape(-1)
        for g in np.unique(groups):
            mask = groups == g
            n_g = int(mask.sum())
            if n_g < 20 or np.unique(labels[mask]).size < 2:
                self.group_temperatures_[int(g)] = self.global_temperature_
                continue
            T_local = self._fit_temperature(probs[mask], labels[mask])
            log_T_local = float(np.log(T_local))
            alpha = n_g / (n_g + self.k_shrink)
            log_T_g = alpha * log_T_local + (1.0 - alpha) * log_T_global
            self.group_temperatures_[int(g)] = float(np.exp(log_T_g))
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))
        if groups is None:
            temp = self.global_temperature_
            return np.clip(special.expit(logits / temp), self.eps, 1.0 - self.eps)
        groups = np.asarray(groups).reshape(-1)
        out = np.empty_like(probs)
        for g in np.unique(groups):
            mask = groups == g
            temp = self.group_temperatures_.get(int(g), self.global_temperature_)
            out[mask] = special.expit(logits[mask] / temp)
        return np.clip(out, self.eps, 1.0 - self.eps)
```
