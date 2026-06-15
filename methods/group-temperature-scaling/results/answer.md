# Group-wise temperature scaling with shrinkage, distilled

Group-wise (subgroup-wise) temperature scaling is post-hoc calibration of a frozen
binary classifier in which **each subgroup gets its own temperature**, but the per-group
temperatures are **regularized by empirical-Bayes (James-Stein-style) shrinkage toward a
single pooled temperature**, blending in log-space with the partial-pooling weight
`alpha = n_g/(n_g + k)`. Small or label-degenerate groups fall back to the global temperature;
large groups keep their own. It improves worst-subgroup calibration without the small-sample
overfitting that independent per-group calibration suffers under distribution shift, and —
being a single positive scalar per group — never changes the predicted class, so accuracy is
unchanged.

## Problem it solves

A trained binary classifier emits positive-class probabilities `p` that are miscalibrated, and
the miscalibration differs by subgroup. We want low *worst-subgroup* calibration error under a
shift between the (small) calibration sample and the test distribution, learning only a map
`p -> q` (optionally using the subgroup id), with the model and splits fixed.

## Key idea

1. **Temperature scaling on logits.** With `z = logit(p)` and `q = sigma(z/T)`, a single
   positive scalar `T` softens (`T>1`) or sharpens (`T<1`) the confidences. It is monotone in
   `z`, so it preserves ranking and the `z=0` decision boundary — accuracy is unchanged — and
   it is the maximum-entropy correction of the dominant "logits are uniformly too large"
   failure mode. `T` is fit by minimizing validation NLL (a proper scoring rule; ECE is the
   non-differentiable evaluation metric, not the loss).

2. **Per-group temperatures, but coupled.** One global `T` cannot satisfy heterogeneous
   subgroups, so fit a temperature per group. Fitting each group's `T` *independently* from its
   own small sample is the analogue of the per-coordinate MLE: its variance scales as `1/n_g`,
   and in the equal-variance normal model Stein (1956) / James & Stein (1961) proved that the
   coordinatewise MLE is **inadmissible**. Under test-time shift, unshrunk small-group
   temperatures are noisier than — and often worse than — the global one.

3. **Shrink toward the pooled temperature, in log-space, by the EB weight.** The common center
   is the global temperature `T_global` (fit on all calibration data — the low-variance "grand
   mean"). In the two-level model `m_g = log T_local,g ~ N(theta_g, sigma_w^2/n_g)` and
   `theta_g ~ N(mu = log T_global, sigma_b^2)`, the posterior mean is the precision-weighted
   blend

   ```
   theta_hat_g = [ (n_g/sigma_w^2) m_g + (1/sigma_b^2) mu ] / [ n_g/sigma_w^2 + 1/sigma_b^2 ]
               = alpha_g * m_g + (1 - alpha_g) * mu,
   alpha_g     = n_g / (n_g + k),   k = sigma_w^2 / sigma_b^2.
   ```

   So `log T_g = alpha_g log T_local,g + (1 - alpha_g) log T_global`. Log-space because `T>0`
   lives on a multiplicative scale and a convex combination of logs stays positive and treats
   `T` and `1/T` symmetrically. `alpha_g` rises monotonically from 0 to 1 with group size: tiny
   groups are pulled fully to global, huge groups keep their own; `k` is the crossover group
   size (50/50 at `n_g = k`), equivalently the prior pseudo-count of the beta-binomial EB form
   `(successes + a0)/(total + a0 + b0)`.

## Defaults and why

- `k_shrink = 200`: subgroup samples here run from dozens to a couple thousand, so a group
  needs ~200 points to half-trust its own temperature; this anchors genuinely small groups to
  global while letting large groups individuate. `k` is fixed rather than estimated because the
  number of groups is too small to estimate the between-group variance reliably.
- Hard floor `n_g >= 20` and `>= 2` distinct labels before attempting a local fit: below this
  the 1-parameter NLL is essentially unidentified and the minimizer hits a boundary; default to
  global. (Below ~20 points `alpha` would be near 0 anyway, so this is a degenerate-case guard,
  not a tuning knob.)
- `eps = 1e-6`: clip `p` and `q` into `[eps, 1-eps]` to keep `logit`/`log` finite and outputs
  valid; well below real probability mass.
- NLL fit over `log_t in [-3, 3]` (`T` in ~`[0.05, 20]`): optimize the positive multiplicative
  parameter on its log scale, bounded so the 1-D search stays well conditioned and `T` cannot
  run away on a flat objective.

## Algorithm

```
fit(probs, labels, groups):
    T_global  <- argmin_T  NLL( sigma(logit(p)/T) )         # over all calibration data
    for each group g:
        if n_g < 20 or group has < 2 labels:
            T_g <- T_global                                 # unidentified -> full pooling
        else:
            T_local <- argmin_T NLL on group g's points
            alpha   <- n_g / (n_g + k)
            log T_g <- alpha * log T_local + (1 - alpha) * log T_global
predict_proba(probs, groups):
    per group g:  q <- sigma( logit(p) / T_g ),  T_g defaulting to T_global for unseen g
    return clip(q, eps, 1-eps)
```

## Code

```python
import numpy as np
from scipy import optimize, special


class CalibrationMethod:
    """Group temperature scaling with empirical-Bayes (James-Stein-style)
    shrinkage to a global temperature. Per-group temperatures are shrunk toward
    the pooled temperature in log-space by alpha = n_g / (n_g + k_shrink):
    small/degenerate groups fall back to the global fit, large groups keep their
    own. A single positive scalar per group is monotone in the logit, so the
    predicted class -- and accuracy -- is unchanged."""

    def __init__(self):
        self.eps = 1e-6
        self.k_shrink = 200.0
        self.group_temperatures_ = {}
        self.global_temperature_ = 1.0

    def _fit_temperature(self, probs, labels):
        probs = np.asarray(probs, dtype=float).reshape(-1)
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
        probs = np.asarray(probs, dtype=float).reshape(-1)
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
        probs = np.asarray(probs, dtype=float).reshape(-1)
        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))
        if groups is None:
            temp = self.global_temperature_
            return np.clip(special.expit(logits / temp), self.eps, 1.0 - self.eps)
        groups = np.asarray(groups).reshape(-1)
        out = np.empty_like(probs, dtype=float)
        for g in np.unique(groups):
            mask = groups == g
            temp = self.group_temperatures_.get(int(g), self.global_temperature_)
            out[mask] = special.expit(logits[mask] / temp)
        return np.clip(out, self.eps, 1.0 - self.eps)
```

## Lineage

- **Temperature scaling** — Guo, Pleiss, Sun & Weinberger, ICML 2017 (binary form = single-
  scalar Platt scaling, Platt 1999): the per-group calibration map `q = sigma(z/T)`.
- **Empirical-Bayes / James-Stein shrinkage** — Stein 1956; James & Stein 1961; Efron & Morris
  1973, 1975 (shrink toward the grand mean): the partial-pooling regularizer
  `alpha = n_g/(n_g+k)` applied to the per-group log-temperatures.
