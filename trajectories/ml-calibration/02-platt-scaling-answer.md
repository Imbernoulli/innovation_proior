**Problem (from step 1).** Isotonic's non-parametric map nailed ECE but bled on the proper scores —
RF-MNIST NLL 0.237 with a >0.05 seed spread, GBM-Madelon NLL 0.582 — because its coarse, few-example
blocks hand the same flat probability to test points that deserve different ones. The fix is to spend
**fewer** degrees of freedom: a two-parameter parametric map, fit by likelihood, with too little capacity
for that variance.

**Key idea.** Two exponential class-conditional tails through Bayes' rule give a **sigmoid** in a
real-line score: `P(+|f) = 1/(1+exp(A f + B))`, with `A < 0` (monotone, honoring the prior isotonic
validated) and `B` absorbing the class-prior offset — equivalently, "`f` is the log-odds up to an affine
transform." Since the harness hands a probability `p ∈ [0,1]`, map it to the space the sigmoid lives in
first: the **log-odds** `f = log(p/(1−p))`. Fit `(A, B)` by cross-entropy (logistic regression on the
single feature `f`).

**Why it works.** One smooth graded map replaces the coarse blocks, so every score gets a distinct
probability — the proper-score bleed disappears. Separable-data runaway (`A→−∞`) is killed without a
hyperparameter by Platt's Laplace-smoothed targets `t₊ = (N₊+1)/(N₊+2)`, `t₋ = 1/(N₋+2)`, which sit
strictly inside `(0,1)` and converge to `{0,1}` as data grows.

**Scaffold edit / hyperparameters.** Per binary column: `f = log(p/(1−p))` (clip `p` to `(eps, 1−eps)`),
soft targets, minimize mean cross-entropy with `optimize.minimize(method="L-BFGS-B")` from `x0=[1.0, 0.0]`.
Multiclass: one-against-all per class (separate `A_c, B_c`), then renormalize rows to sum to 1.

**What to watch.** The sigmoid is **one** shape. Expect proper scores (NLL/Brier) to drop where isotonic
bled — Madelon NLL well below 0.582, RF-MNIST NLL below 0.237 with a tighter spread — but ECE may **rise**
where isotonic's data-chosen binning matched accuracy best (the SVM column especially, if its log-odds
curve isn't quite sigmoidal). Residual ECE = the shape flexibility traded away, which forces a richer
parametric family next.

```python
# EDITABLE region of custom_calibration.py (lines 45-102) - step 2: Platt scaling
class CalibrationMethod(BaseEstimator):
    """Platt Scaling (logistic/sigmoid calibration).

    Fits A*f + B through a sigmoid for each class, where f is the
    uncalibrated probability (log-odds transformed).
    """

    def __init__(self):
        self.is_binary = None
        self.a_ = None
        self.b_ = None

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            self.a_, self.b_ = self._fit_sigmoid(probs, labels)
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.a_ = np.zeros(n_classes)
            self.b_ = np.zeros(n_classes)
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)       # one-against-all per class
                self.a_[c], self.b_[c] = self._fit_sigmoid(probs[:, c], binary_labels)
        return self

    def _fit_sigmoid(self, probs, labels):
        """Fit sigmoid parameters A, B: calibrated = 1 / (1 + exp(A*f + B))."""
        # Transform to log-odds space, clip to avoid inf
        eps = 1e-12
        f = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))

        # Target probabilities (Platt's Laplace-smoothed target encoding)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        t_pos = (n_pos + 1) / (n_pos + 2) if n_pos > 0 else 0.5
        t_neg = 1 / (n_neg + 2) if n_neg > 0 else 0.5
        target = np.where(labels > 0.5, t_pos, t_neg)

        def objective(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * f + b))
            p = np.clip(p, eps, 1 - eps)
            loss = -(target * np.log(p) + (1 - target) * np.log(1 - p)).mean()
            return loss

        result = optimize.minimize(objective, x0=[1.0, 0.0], method="L-BFGS-B")
        return result.x[0], result.x[1]

    def predict_proba(self, probs):
        eps = 1e-12
        if self.is_binary:
            f = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
            calibrated = 1.0 / (1.0 + np.exp(self.a_ * f + self.b_))
            return np.clip(calibrated, 0, 1)
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                f = np.log(np.clip(probs[:, c], eps, 1 - eps) /
                           np.clip(1 - probs[:, c], eps, 1 - eps))
                calibrated[:, c] = 1.0 / (1.0 + np.exp(self.a_[c] * f + self.b_[c]))
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)     # renormalize
            return calibrated
```
