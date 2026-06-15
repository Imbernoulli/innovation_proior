# Platt scaling, distilled

Platt scaling turns a frozen classifier's uncalibrated real-valued scores into calibrated
positive-class probabilities by fitting a two-parameter sigmoid
`P(y=1|f) = 1/(1 + exp(A f + B))` on a held-out calibration set. It leaves the classifier
untouched, preserving the sparseness and accuracy of a margin-trained SVM, and adds only a
cheap trainable post-processor. The sigmoid form is justified by the observed exponential
class-conditional score densities between the SVM margins; Bayes' rule on those exponentials
gives an affine log-odds model. Two regularizing choices make the fit usable on small data:
fit on scores from examples the classifier did not train on, and use Bayesian soft targets so
a separable calibration set cannot drive the slope to `A -> -inf`.

## Problem it solves

Post-hoc probability calibration: map a fixed classifier's confidence score `f(x)` to a
calibrated `P(y=1 | x)` without retraining the classifier, using only a limited calibration set.

## Key idea

Model the posterior directly with a sigmoid in the score:

```
P(y = 1 | f) = 1 / (1 + exp(A f + B)).
```

The sigmoid decreases as `Af+B` increases, so it is monotone increasing in the original score
only when `A < 0`. If the supplied score is already a calibrated log-odds `f = log(q/(1-q))`,
the identity special point is `A = -1, B = 0`, because `1/(1+exp(-f)) = q`.

Why this form: between the SVM margins, the positive-class density rises approximately like
`exp(γ_1 f)` and the negative-class density falls approximately like `exp(-γ_0 f)`. Bayes'
rule gives

```
P(y=1|f) = 1 / (1 + exp(-(γ_0+γ_1) f + B)),
```

with normalizing constants and priors absorbed into `B`; hence `A = -(γ_0+γ_1) < 0`. The
equivalent statement is `log(P/(1-P)) = -(A f + B)`: the classifier score is affine in the
positive-class log-odds. The free offset `B` lets the `P=0.5` crossing move away from `f=0`
when class priors are skewed.

## Fit

Given calibration scores `f_i` and labels `y_i ∈ {-1,+1}`, set
`p_i = 1/(1+exp(A f_i + B))` and minimize the Bernoulli negative log likelihood

```
F(A,B) = -Σ_i [t_i log p_i + (1-t_i) log(1-p_i)].
```

Do not use the classifier's training scores: support vectors and margin violators have biased
scores. Use held-out scores or out-of-fold scores from cross-validation.
If calibration examples are weighted, use those same weights in the class priors, the loss sum,
and the gradient.

Use Laplace-rule Bayesian targets instead of `{0,1}`:

```
t_+ = (N_+ + 1)/(N_+ + 2)   for a positive example,
t_- = 1/(N_- + 2)           for a negative example.
```

These targets stay inside `(0,1)`, making the optimum finite on separable calibration sets, and
converge to `{0,1}` as the calibration set grows.

## Convexity and numerics

With `s_i = A f_i + B` and `p_i = 1/(1+e^{s_i})`,

```
∇F = [Σ f_i(t_i-p_i),  Σ (t_i-p_i)],
H  = [[Σ f_i^2 p_i(1-p_i),  Σ f_i p_i(1-p_i)],
      [Σ f_i p_i(1-p_i),    Σ p_i(1-p_i)    ]].
```

The Hessian is the 2-by-2 Gram matrix built from `u_i = f_i sqrt(p_i(1-p_i))` and
`v_i = sqrt(p_i(1-p_i))`, so it is positive semidefinite by Cauchy-Schwarz and
positive definite unless all scores are identical. The problem is convex, usually strictly convex.

For stable evaluation, use the equivalent NLL forms

```
-(t log p + (1-t) log(1-p)) = (t-1)s + log(1+e^s) = t s + log(1+e^{-s}),
```

choosing `t s + log(1+e^{-s})` when `s >= 0` and `(t-1)s + log(1+e^s)` when `s < 0`. This
keeps the exponential argument non-positive, avoids overflow, and never forms `1-p`.
Initialize `A = 0` and `B = log((N_-+1)/(N_++1))`, the flat sigmoid at the smoothed base rate.

## Working code

```python
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator


class CalibrationMethod(BaseEstimator):
    """Score-level sigmoid calibrator: fit 1/(1+exp(A*f+B))."""

    def __init__(self, max_abs_prediction_threshold=30.0):
        self.max_abs_prediction_threshold = max_abs_prediction_threshold
        self.a_ = None
        self.b_ = None

    @staticmethod
    def _stable_sigmoid(s):
        out = np.empty_like(s, dtype=float)
        pos = s >= 0
        z = np.exp(-s[pos])
        out[pos] = z / (1.0 + z)
        z = np.exp(s[~pos])
        out[~pos] = 1.0 / (1.0 + z)
        return out

    def fit(self, scores, labels, sample_weight=None):
        f = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(labels).ravel()
        if sample_weight is None:
            w = None
        else:
            w = np.asarray(sample_weight, dtype=float).ravel()
            if w.shape[0] != y.shape[0]:
                raise ValueError("sample_weight must have one entry per label")

        scale = 1.0
        max_abs = np.max(np.abs(f)) if f.size else 0.0
        if max_abs >= self.max_abs_prediction_threshold:
            scale = max_abs
            f = f / scale

        neg = y <= 0
        if w is None:
            prior0 = float(np.sum(neg))
            prior1 = float(y.shape[0] - prior0)
        else:
            prior0 = float(w[neg].sum())
            prior1 = float(w[~neg].sum())
        t = np.empty_like(f, dtype=float)
        t[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
        t[neg] = 1.0 / (prior0 + 2.0)

        def loss_grad(AB):
            A, B = AB
            s = A * f + B
            pos = s >= 0
            loss = np.empty_like(s)
            loss[pos] = t[pos] * s[pos] + np.log1p(np.exp(-s[pos]))
            loss[~pos] = (t[~pos] - 1.0) * s[~pos] + np.log1p(np.exp(s[~pos]))
            p = self._stable_sigmoid(s)
            d = t - p
            if w is None:
                weighted_d = d
                total_loss = loss.sum()
            else:
                weighted_d = w * d
                total_loss = np.dot(w, loss)
            grad = np.array([np.dot(weighted_d, f), weighted_d.sum()], dtype=float)
            return total_loss, grad

        AB0 = np.array([0.0, np.log((prior0 + 1.0) / (prior1 + 1.0))])
        result = optimize.minimize(
            loss_grad,
            AB0,
            jac=True,
            method="L-BFGS-B",
            options={"gtol": 1e-6, "ftol": 64 * np.finfo(float).eps},
        )
        self.a_ = float(result.x[0] / scale)
        self.b_ = float(result.x[1])
        return self

    def predict_proba(self, scores):
        f = np.asarray(scores, dtype=float)
        return self._stable_sigmoid(self.a_ * f + self.b_)
```
