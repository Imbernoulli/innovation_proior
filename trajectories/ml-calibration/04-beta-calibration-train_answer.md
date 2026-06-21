Temperature scaling won the ladder by making the *minimal* correction, and its numbers show both why it won and the one thing it still cannot do. Where the error was scale the wins are clean — RF on MNIST ECE fell to 0.0101 (below isotonic's 0.0156, far below Platt's 0.0254) with NLL 0.1553 holding the gain, and SVM on Breast Cancer ECE dropped from Platt's blown-up 0.0493 to 0.0305 with NLL improving to 0.0864 — one shared scalar fixed the overconfidence Platt's per-class sigmoid had overfit. But GBM on Madelon ECE sits at 0.0305, essentially Platt's 0.0288 and *worse* than isotonic's 0.0161: temperature scaling left Madelon almost untouched, because one shared $T$ can soften a confidence vector but cannot move a *location* (it has no offset to shift where $P=\tfrac12$ sits) and cannot *reshape* a curve that bends the wrong way (no second degree of freedom to gather extreme scores back toward the middle). So the verdict across three rungs is sharp — isotonic too flexible, Platt one rigid shape per column, temperature one scalar that misses location and two-way shape — and what I want now is a family that keeps Platt's cheapness and data efficiency but bends in the directions both miss: it must contain the sigmoid (so it never loses where Platt won), contain the *inverse* sigmoid (so it can gather extremes, which neither prior rung can), place its midpoint freely (the location offset temperature lacks), and contain the *identity* (so on an already-calibrated column it leaves it alone instead of being forced to move it).

I propose **beta calibration**, and I derive it by fixing the one modelling choice that makes the sigmoid rigid. Platt's sigmoid drops out of a generative story: posit per-class scores normally distributed with the *same* variance $\sigma^2$, means $s_+, s_-$; the likelihood ratio $\mathrm{LR}(s) = p(s\mid+)/p(s\mid-)$ has its $s^2$ terms cancel — that cancellation is what makes the exponent linear in $s$ — leaving $\mathrm{LR}(s) = \exp[\gamma(s-m)]$ with $\gamma = (s_+ - s_-)/\sigma^2$, $m = (s_+ + s_-)/2$, and under a uniform prior the posterior is $\mu(s) = 1/(1+\exp(-\gamma(s-m)))$, the sigmoid exactly. So the sigmoid family is precisely "I believe the per-class scores are equal-variance Gaussians," and that single assumption is too rigid in three concrete ways: my scores live in $[0,1]$ but a Gaussian puts mass on the whole line, so modelling a bounded score as Gaussian is incoherent; equal-variance Gaussians force $\gamma \ge 0$, an S-curve that only *spreads* scores toward 0 and 1 and can never *gather* them; and the identity $\mu(s)=s$ is a straight line that no bounded S-curve equals for any finite $\gamma, m$, so Platt applied to an already-calibrated column necessarily *un*calibrates it. All three trace to one place — equal-variance Gaussians on a bounded score — so I fix the assumption and re-run the derivation.

The natural density on $[0,1]$ is the **beta distribution**, $p(s; \alpha, \beta) = s^{\alpha-1}(1-s)^{\beta-1}/B(\alpha,\beta)$, which lives exactly on $[0,1]$, can be unimodal, U-shaped, J-shaped, or flat, and has *two* shape parameters per class rather than one shared variance. Putting positives $\sim \mathrm{Beta}(\alpha_1, \beta_1)$ and negatives $\sim \mathrm{Beta}(\alpha_0, \beta_0)$ through the same likelihood-ratio machine, the powers of $s$ subtract, the powers of $(1-s)$ subtract, and the beta-function constants collect into one factor, giving a clean power law on the odds,

$$\mathrm{LR}(s) = e^c \cdot \frac{s^a}{(1-s)^b}, \qquad a = \alpha_1 - \alpha_0,\; b = \beta_0 - \beta_1,$$

whose calibrated posterior $\mu(s) = 1/(1 + \mathrm{LR}(s)^{-1})$ is the three-parameter family

$$\mu_\text{beta}(s; a, b, c) = \frac{1}{1 + 1/\!\left(e^c\, s^a/(1-s)^b\right)},$$

with $a, b$ shapes and $c$ location. It does exactly what the sigmoid and the scalar could not. It is monotone — $\mu$ increases in $s$ iff $\mathrm{LR}$ does, and $\tfrac{d}{ds}\ln \mathrm{LR} = a/s + b/(1-s) \ge 0$ on $(0,1)$ exactly when $a, b \ge 0$, the analogue of $\gamma \ge 0$. The curves are *not* translation-invariant in $s$ (unlike the sigmoid, where moving $m$ just slid the same S-curve), because $s^a$ and $(1-s)^b$ are asymmetric unless $a=b$ — and that asymmetry is the location freedom temperature lacked, with the midpoint at $c = b\ln(1-m) - a\ln m$. The decisive test, the gathering shape no prior rung could make: $a = b < 1$ gives $\mathrm{LR}(s) = e^c (s/(1-s))^a$ with exponent $<1$, which *damps* the log-odds and pulls extreme scores toward the middle — an inverse sigmoid — while the sigmoid is $a=b>1$. And the identity, which the sigmoid family flatly lacked: $a=b=1, c=0$ gives $\mathrm{LR}(s) = s/(1-s)$ and $\mu(s) = s$ exactly. One move — swapping Gaussians for betas — repairs all three sigmoid failures and adds the location offset temperature lacked. A concrete sanity check on the gathering case: naive-Bayes fed $k$ identical copies of a calibrated feature outputs $s = x^k/(x^k+(1-x)^k)$, and the exact map recovering $x$ is $\mu_\text{beta}$ with $a=b=1/k, c=0$ — double-counting is a power on the odds, and the beta family is precisely powers on the odds.

Made concrete on the task surface, the family fits as cheaply as a sigmoid because $\ln \mathrm{LR} = a\ln s - b\ln(1-s) + c$ is linear in $\ln s$ and $-\ln(1-s)$ — a bivariate logistic regression. The literal fill builds two features directly, $f_1 = \log(p/(1-p))$ (the log-odds) and $f_2 = \log(1-p)$, noting that $a\log(p/(1-p)) + b\log(1-p) + c = a\ln s - a\ln(1-s) + b\ln(1-s) + c$ — the same linear-in-$(\ln s, \ln(1-s))$ object up to a relabeling of the two coefficients, so this featurization fits the identical three-parameter beta map. I minimize the mean cross-entropy of $q = 1/(1+\exp(-(a f_1 + b f_2 + c)))$ against the labels with L-BFGS-B from $x_0 = [1, 0, 0]$, the log-odds-identity start ($a=1, b=0, c=0$), the natural neutral point; for multiclass I run the same one-against-all per-class fit and renormalize rows to sum to one, exactly as the earlier rungs did. The fit is essentially unregularized, which is correct — beta calibration *is* the maximum-likelihood / log-loss fit, and strong shrinkage would pull $a, b$ toward zero and collapse the map toward an intercept-only constant. One difference from the canonical recipe I am explicit about: the unconstrained `optimize.minimize` can in principle return $a<0$ or $b<0$, making the map non-monotone over part of the range; the reference (Kull, Silva Filho & Flach, AISTATS 2017) guards this with a drop-and-refit step that fixes a negative coefficient to zero and refits the reduced model, and this task's edit omits that guard, fitting all three coefficients unconstrained. The guard rarely fires on well-behaved classifiers, where real distortions are monotone and the data pull both coefficients positive, but it is the one place this implementation is looser than the reference.

This rung carries no feedback, so the bar I have to clear is stated against temperature scaling's real numbers. The clearest target is the column it left untouched, **GBM on Madelon** at ECE 0.0305: beta calibration's whole reason to exist is the location offset and two-way shape a single scalar cannot supply, so the falsifiable claim is that Madelon's ECE should fall below 0.0305 — toward, and possibly below, isotonic's 0.0161 — *without* isotonic's proper-score bleed, because beta is a smooth three-parameter map rather than a coarse block fit. On the columns temperature already calibrated by scale — RF on MNIST (ECE 0.0101, NLL 0.1553) and SVM on Breast Cancer (ECE 0.0305, NLL 0.0864) — the bar is to *not regress*: since the sigmoid is a strict sub-case ($a=b>1$) and the identity is reachable ($a=b=1, c=0$), the richer family can always fall back to what Platt and temperature found and should match or modestly beat them rather than overfit three parameters where two sufficed. The real risk is exactly that overfit — three parameters per column on the smallest (SVM) split is more capacity than two — and if it gives back some ECE there, the diagnosis would be that the binary columns want the symmetric $a=b$ restriction (univariate logistic on the log-odds, two effective parameters). The success condition is concrete: beat temperature on the Madelon ECE it could not move, hold its RF and SVM wins within noise, and improve or hold NLL and Brier everywhere — which is what a family that *contains* every rung below it as a special case should be able to do.

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
