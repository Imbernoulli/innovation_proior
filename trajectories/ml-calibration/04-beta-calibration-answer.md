**Problem (from step 3).** Temperature scaling made the minimal scale correction and won where the error
was overconfidence-as-scale (RF ECE 0.0101, SVM ECE 0.0305), but a single shared scalar has no location
offset and no second shape degree of freedom, so it left **GBM on Madelon** ECE at 0.0305 — essentially
Platt's, worse than isotonic's 0.0161 — a column whose curve needs both a location shift and a two-way
bend. Isotonic could bend but bled the proper scores; Platt/temperature kept the proper scores but couldn't
bend. The next family must do both.

**Key idea.** Replace the equal-variance Gaussian behind the sigmoid with the **beta distribution**, the
natural density on `[0,1]`. The likelihood ratio of two betas is a power law on the odds,
`LR(s) = e^c · s^a/(1−s)^b`, whose calibrated posterior is the three-parameter family
`μ_beta(s; a, b, c) = 1/(1 + 1/(e^c s^a/(1−s)^b))`. It is monotone for `a, b ≥ 0`; it **contains** the
sigmoid (`a = b > 1`), the **inverse** sigmoid that gathers extreme scores (`a = b < 1`), the **identity**
(`a = b = 1, c = 0`), and asymmetric/location-shifted maps (`a ≠ b`) — exactly the directions Platt and
temperature both miss.

**Why it works.** `ln LR = a ln s − b ln(1−s) + c` is linear in `log s` and `−log(1−s)`, so the whole
family is fitted as cheaply as a sigmoid (one bivariate logistic / log-loss fit). Because it contains every
prior rung as a special case, it can fall back to what they found and only add capacity where the data
support it.

**Scaffold edit / hyperparameters.** Per binary column build `f1 = log(p/(1−p))`, `f2 = log(1−p)` (clip
`p` to `(eps, 1−eps)`), minimize mean cross-entropy of `1/(1+exp(−(a f1 + b f2 + c)))` with
`optimize.minimize(method="L-BFGS-B")` from `x0=[1, 0, 0]` (log-odds-identity start), essentially
unregularized (beta calibration *is* the MLE). Multiclass: one-against-all per class, then renormalize rows
to sum to 1.

**Reference.** Kull, Silva Filho & Flach, "Beta calibration: a well-founded and easily implemented
improvement on logistic calibration for binary classifiers," AISTATS 2017
([PMLR v54](https://proceedings.mlr.press/v54/kull17a.html)).

**Harness vs canonical (noted).** This task's edit fits all three coefficients **unconstrained** via scipy
and uses them as-is; the canonical `betacal` recipe adds a **drop-and-refit monotonicity guard** (if `a`
or `b` comes out negative, fix it to 0 and refit the reduced model). The guard rarely fires on
well-behaved classifiers, but it is the one place the harness implementation is looser than the reference.
The feature build `a·log(p/(1−p)) + b·log(1−p) + c` is the same linear-in-`(log s, log(1−s))` beta map up
to a relabeling of the two coefficients.

**Bar to clear (no feedback — endpoint).** Beat temperature scaling on the Madelon ECE it couldn't move
(below 0.0305, toward isotonic's 0.0161) without isotonic's proper-score bleed; hold RF (ECE 0.0101, NLL
0.1553) and SVM (ECE 0.0305, NLL 0.0864) within noise; improve or hold NLL/Brier everywhere. Risk: three
parameters per column on the smallest (SVM) split could overfit and give back ECE — if so, the symmetric
`a = b` restriction (univariate logistic on the log-odds) is the two-parameter fallback.

```python
# EDITABLE region of custom_calibration.py (lines 45-102) - finale: beta calibration
class CalibrationMethod(BaseEstimator):
    """Beta Calibration.

    3-parameter model per class: uses log-odds and log(1-p) as features
    in a logistic regression, giving a richer calibration curve than Platt.
    """

    def __init__(self):
        self.is_binary = None
        self.params_ = None

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            self.params_ = [self._fit_beta(probs, labels)]
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.params_ = []
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)       # one-against-all per class
                self.params_.append(self._fit_beta(probs[:, c], binary_labels))
        return self

    def _fit_beta(self, probs, labels):
        """Fit beta calibration: logit(q) = a*log(p/(1-p)) + b*log(1-p) + c."""
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        # Features: log(p/(1-p)) and log(1-p)
        f1 = np.log(p / (1 - p))  # log-odds
        f2 = np.log(1 - p)         # log(1-p)

        def objective(params):
            a, b, c = params
            logit_q = a * f1 + b * f2 + c
            q = 1.0 / (1.0 + np.exp(-logit_q))
            q = np.clip(q, eps, 1 - eps)
            loss = -(labels * np.log(q) + (1 - labels) * np.log(1 - q)).mean()
            return loss

        result = optimize.minimize(objective, x0=[1.0, 0.0, 0.0],
                                   method="L-BFGS-B")
        return result.x

    def _predict_beta(self, probs, params):
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        a, b, c = params
        f1 = np.log(p / (1 - p))
        f2 = np.log(1 - p)
        logit_q = a * f1 + b * f2 + c
        q = 1.0 / (1.0 + np.exp(-logit_q))
        return np.clip(q, 0, 1)

    def predict_proba(self, probs):
        if self.is_binary:
            return self._predict_beta(probs, self.params_[0])
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                calibrated[:, c] = self._predict_beta(probs[:, c], self.params_[c])
            calibrated = np.clip(calibrated, 1e-15, None)
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)     # renormalize
            return calibrated
```
