**Problem (from step 2).** Platt's two-parameter sigmoid fixed isotonic's proper-score bleed (RF-MNIST
NLL 0.237 → 0.163, Madelon NLL 0.582 → 0.451) but **regressed ECE** where isotonic was strongest — SVM
ECE tripled to 0.049, RF ECE rose to 0.025 — because the per-column sigmoid is one wrong shape and the
independent per-class fitting ignores the joint softmax. The fix: spend even less capacity and respect the
joint distribution — hypothesize the dominant error is a single shared *scale*.

**Key idea.** Divide every logit by one shared positive scalar `T` and re-softmax: `softmax(z/T)`. Since
`T` is identical across classes, the argmax — and therefore the **accuracy** — is exactly preserved
(unlike Platt's renormalized per-class fit); raising `T` softens an overconfident model toward uniform.
One parameter. The harness hands probabilities, so rebuild logits: `z = log(p)` for multiclass (softmax
is shift-invariant), `z = [log(1−p), log(p)]` for binary. Fit `T` by minimizing calibration NLL.

**Why it works.** Temperature scaling is the unique **maximum-entropy** distribution that matches the
average true-class logit (the lone Lagrange multiplier `λ = 1/T` of the lone moment constraint), so the
single scalar is the NLL-optimal global correction — the minimal fix when the error is overconfidence-as-scale.

**Scaffold edit / hyperparameters.** Build logits from probs, stable softmax of `z/T` (subtract row max).
Fit with `optimize.minimize(nll, x0=[1.5], bounds=[(0.01, 20.0)], method="L-BFGS-B")`, clamp `T ≥ 0.01`;
start above 1 (expect overconfidence). Binary returns the positive-class column.

**What to watch.** Expect the ECE regressions to reverse where the error is scale — RF ECE back below 0.025
(toward isotonic's 0.016), SVM ECE well below Platt's 0.049 — while proper scores hold or improve (RF NLL
≤ 0.163, SVM NLL ≤ 0.101) and accuracy is preserved by construction. Open gap: a column whose distortion
needs a *location offset* or a two-way-bending *shape* (bounded scores, inverse-sigmoid) — one scalar can
soften but not reshape — e.g. Madelon's ECE could sit near Platt's. That gap is the opening for a richer
parametric family.

```python
# EDITABLE region of custom_calibration.py (lines 45-102) - step 3: temperature scaling
class CalibrationMethod(BaseEstimator):
    """Temperature Scaling calibration.

    Learns a single temperature T that scales all logits: softmax(z/T).
    Optimized by minimizing NLL on the calibration set.
    """

    def __init__(self):
        self.is_binary = None
        self.temperature_ = 1.0

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            # Convert to 2-class logits
            eps = 1e-15
            p = np.clip(probs, eps, 1 - eps)
            logits = np.column_stack([np.log(1 - p), np.log(p)])
        else:
            self.is_binary = False
            eps = 1e-15
            logits = np.log(np.clip(probs, eps, 1.0))

        def nll(T):
            T_val = max(T[0], 0.01)
            scaled = logits / T_val
            # Numerically stable softmax
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            log_probs = scaled - np.log(exp_scaled.sum(axis=1, keepdims=True))
            return -log_probs[np.arange(len(labels)), labels.astype(int)].mean()

        result = optimize.minimize(nll, x0=[1.5], bounds=[(0.01, 20.0)],
                                   method="L-BFGS-B")
        self.temperature_ = max(result.x[0], 0.01)
        return self

    def predict_proba(self, probs):
        eps = 1e-15
        if self.is_binary:
            p = np.clip(probs, eps, 1 - eps)
            logits = np.column_stack([np.log(1 - p), np.log(p)])
        else:
            logits = np.log(np.clip(probs, eps, 1.0))

        scaled = logits / self.temperature_
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        calibrated = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        if self.is_binary:
            return calibrated[:, 1]
        return calibrated
```
