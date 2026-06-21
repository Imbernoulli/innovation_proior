Platt did what I went parametric to get, and also the thing I feared, and both are legible in the numbers. The proper-score bleed is gone: RF on MNIST NLL fell from 0.237 to 0.163 mean with the seed spread tightened under 0.016 (where isotonic was over 0.05), GBM on Madelon NLL dropped from 0.582 to 0.451, and Brier improved across the board. But ECE regressed where isotonic had been strongest — SVM on Breast Cancer nearly tripled, 0.0188 to 0.0493 with a wild seed spread, RF on MNIST rose to 0.0254, GBM on Madelon to 0.0288. That is exactly the predicted failure: one sigmoid shape per column fits the proper scores well but cannot bend to a reliability curve that is not sigmoidal, and on the multiclass tasks fitting $C$ separate sigmoids on $C$ separate log-odds columns and renormalizing both multiplies the chance of a bad per-column fit and throws away the joint structure of the softmax. The diagnosis points two ways at once: keep the parametric data efficiency, but cost *even less* capacity and *respect the joint distribution* rather than calibrate columns independently. The cleanest hypothesis that does both is that the dominant error is not a per-class shape problem but a single shared *scale* problem — the confidence vector has the right direction and the wrong magnitude.

I propose **temperature scaling** — divide every logit by one shared positive scalar $T$ and re-softmax,

$$\text{calibrated} = \mathrm{softmax}(z/T), \qquad T > 0 \text{ shared across all classes and examples},$$

borrowing the name from where this operation already lives in statistical mechanics and distillation. Before committing I check it has the properties the diagnosis demands. Because $T$ is the *same* for every class, dividing by a positive $T$ is a monotone transform of the logits that does not reorder them: $\arg\max_k z_k/T = \arg\max_k z_k$, so the predicted class never changes and the accuracy is preserved *exactly* — a guarantee Platt's per-class renormalized fit could not make, since different $(A_c, B_c)$ can reorder the argmax. The effect on the probabilities is precisely a softening: as $T \to \infty$ the scaled logits flatten toward uniform $1/C$, at $T=1$ I recover the original, as $T \to 0^+$ mass collapses onto the argmax. Sweeping $T$ upward does exactly one thing — bleed confidence out of an overconfident model toward uniform — which is the single knob the shared-scale hypothesis says I need, and with *one* parameter there is even less capacity to overfit than Platt's two, the direction the variance diagnosis demanded.

The task surface forces a subtlety, and getting it right is the whole ballgame: temperature scaling is defined on *logits* but the harness hands me *probabilities*. I have to recover logits to divide. For multiclass the softmax is invariant to an additive shift in the logits, so $z = \log(p)$ is a valid logit vector — any constant offset washes out in the next softmax — and dividing $\log(p)$ by $T$ and re-softmaxing is exactly temperature scaling on the implied logits. For binary the harness gives only the positive-class probability $p$, so I reconstruct a two-class pair $z = [\log(1-p), \log(p)]$, run the same scalar division and two-class softmax, and peel off the positive column. Numerically I subtract the row max before exponentiating so the softmax is stable.

I pick $T$ by the calibration negative log-likelihood. NLL is a strictly proper scoring rule, minimized in expectation when the predicted distribution matches the truth, the same proper score where Platt did well and isotonic bled, and crucially it is differentiable — the binned ECE I also care about is not, a bad thing to optimize directly, and NLL is its smooth cousin. So I freeze everything, compute the implied logits on the calibration split once, and minimize the NLL over the single scalar $T$. Since the argmax-preservation argument needs $T>0$, I bound the search to $T \in [0.01, 20.0]$, clamp $\max(T, 0.01)$ inside the objective, and run L-BFGS-B from $x_0 = 1.5$ — above 1, because I expect an overconfident model and want to soften rather than sharpen. One parameter, a smooth one-dimensional problem, done in a couple hundred iterations.

What convinces me this is principled and not a lucky hack matching my story is that it falls out of a clean optimization principle. Calibration is about not being more confident than the evidence warrants — as high-entropy as possible while consistent with the data. Maximize the entropy of the predicted distributions subject to one coupling constraint, that the average logit assigned to the *true* class equals the average logit under the predicted distribution (a moment-matching condition), plus per-example normalization. Maximum entropy under a moment constraint gives an exponential-family form: turning the Lagrangian crank, each $q_i^{(k)} = \exp(\lambda z_i^{(k)} + \beta_i - 1)$, and imposing $\sum_k q_i^{(k)} = 1$ divides out the per-example factor to leave $q_i^{(k)} = \mathrm{softmax}(\lambda z_i)^{(k)}$. Identify $\lambda = 1/T$ and that is temperature scaling exactly. So the single scalar is not a convenient small choice — it is the lone Lagrange multiplier of the lone coupling constraint, and raising $T$ (lowering $\lambda$) raises the entropy of an overconfident model back to what the calibration data support.

I am honest about where this can lose, because its hypothesis is strong. If a classifier's miscalibration is *not* a shared scale error — different classes, or the two extremes of a binary score, miscalibrated in different directions — one global $T$ cannot fix it. A binary curve that needs a *location* shift has no $B$-like offset to move its midpoint, and an inverse-sigmoid column that needs *gathering* can be softened but not reshaped by one scalar. So against Platt's measured numbers I expect, falsifiably: the ECE regressions should *reverse* where a single scale is the right correction — RF on MNIST ECE back well below 0.0254 toward isotonic's 0.0156, SVM on Breast Cancer ECE well below Platt's blown-up 0.0493 — while the proper scores hold or improve (RF NLL at or below 0.163, SVM NLL at or below 0.101) and accuracy is preserved by construction. Where I am not confident is GBM on Madelon: if its error has a location component, temperature scaling's lack of an offset could leave its ECE near or even slightly above Platt's. If the pattern is ECE recovered where the error was scale and proper scores held, temperature scaling is the strongest of the three by making the *minimal* correction the data support — and the one gap it leaves open, by construction, is the column whose distortion needs both a two-way *shape* and a *location* offset on bounded scores, exactly the opening a richer parametric family would fill.

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
