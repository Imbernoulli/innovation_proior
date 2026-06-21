Isotonic's numbers tell me exactly what I bought and exactly what I paid. On ECE it did just what it was built to do — RF on MNIST 0.0156, MLP on Fashion-MNIST 0.0122, GBM on Madelon 0.0161, SVM on Breast Cancer 0.0188 — matching confidence to accuracy bin by bin. But the proper scores bled. RF on MNIST sits at 0.237 mean NLL and the seeds wobble across more than 0.05 (0.223, 0.272, 0.215), the non-parametric variance I warned myself about showing up directly; and GBM on Madelon's NLL is 0.582, by far the worst of the four, on exactly the dataset with the steep curve and the smaller calibration split, where the coarse few-example blocks hand the same flat probability to test points that deserve different ones. The diagnosis is precise: isotonic spends too many degrees of freedom and the proper scores bleed where the calibration set is small. The fix is not a different non-parametric scheme — it is to spend *fewer* degrees of freedom, go parametric, buy back data efficiency, and accept a little shape rigidity in exchange for stable proper scores.

I propose **Platt scaling** — a two-parameter sigmoid fit by likelihood, derived rather than guessed. The textbook generative route is the place to derive the form from, not to implement: I do not want to estimate two whole class-conditional densities when I only care about the one posterior. For a margin classifier the class-conditional score histograms are nowhere near Gaussian; they have kinks and decay roughly exponentially between the class clusters, so write the two conditionals as exponential tails, $p(f\mid+) \propto e^{\gamma_1 f}$ and $p(f\mid-) \propto e^{-\gamma_0 f}$ with $\gamma_0, \gamma_1 > 0$, on a score $f$ ranging over the real line. Putting priors $\pi_1, \pi_0$ into Bayes' rule and dividing top and bottom by the numerator gives

$$P(+\mid f) = \frac{\pi_1 e^{\gamma_1 f}}{\pi_1 e^{\gamma_1 f} + \pi_0 e^{-\gamma_0 f}} = \frac{1}{1 + \exp(A f + B)},$$

with $A = -(\gamma_0 + \gamma_1) < 0$ the slope and $B = \log(\pi_0/\pi_1)$ the offset that absorbs the class-prior ratio. Two exponential conditionals give a *sigmoid* in $f$. The reading I like even more is that $1/(1+\exp(Af+B))$ is exactly the model "$f$ is, up to an affine transform, the log-odds of the positive class": the probability rises with $f$ precisely when $A < 0$, which honors the monotone prior isotonic just validated, and the output is automatically in $[0,1]$. Two free parameters — $A$ how fast confidence grows with the score, $B$ where $P=\tfrac12$ sits — fit to data, not one parameter read off a Gaussian variance.

The task surface forces one specific choice. The sigmoid is naturally a function of a real-line score, but the harness hands me a *probability* $p \in [0,1]$; feeding $p$ straight into $1/(1+\exp(Ap+B))$ puts a bounded variable through a function designed for an unbounded one, and the affine $Ap+B$ cannot reach the log-odds the sigmoid wants. So I map the probability back to the space where the derivation lives, the **log-odds** $f = \log(p/(1-p))$ — it ranges over the whole real line, is monotone in $p$, and is exactly the input the "score is proportional to log-odds" reading wants. The calibrator is then logistic regression of the label on the single transformed feature $f$: take $p$, clip it into $(\epsilon, 1-\epsilon)$ so the log is finite, transform to $f$, and fit $\text{calibrated} = 1/(1+\exp(Af+B))$.

I fit $A, B$ by the negative log-likelihood of the probability model, $-\sum_i [t_i \log p_i + (1-t_i)\log(1-p_i)]$ with $p_i = 1/(1+\exp(Af_i+B))$ — two parameters, smooth and convex (the Hessian is a positive-definite Gram matrix unless every $f_i$ coincides), so L-BFGS-B from a flat start $[1.0, 0.0]$ finds the unique optimum. Plain $\{0,1\}$ targets have a failure mode I have to head off: if the calibration scores are linearly separable in $f$, maximum likelihood tries to drive every positive's $p_i$ to exactly 1 and every negative's to 0, which the sigmoid can only do by becoming infinitely steep — $A \to -\infty$, no finite optimum. I need regularization but I refuse to add a hyperparameter to tune, because that reopens the cross-validation problem isotonic suffered. The hyperparameter-free fix is Platt's: model a positive example as positive with a small residual chance of being negative out of sample. The rule of succession pins the numbers — with a uniform prior over the true positive probability, $N_+$ positives give a positive example the soft target

$$t_+ = \frac{N_+ + 1}{N_+ + 2}, \qquad t_- = \frac{1}{N_- + 2},$$

both strictly inside $(0,1)$, so pushing $A \to -\infty$ would overshoot $t_+ < 1$ and *increase* the loss — the optimum becomes finite — while as $N_+, N_- \to \infty$ they converge to $\{0,1\}$ and recover plain MLE in the large-data limit. No knob, exactly the regularizer I want.

For multiclass I again land *this task's* choice, not a generic one. I do not calibrate the softmax jointly — same reason isotonic did not — but the parametric per-class fit makes one-against-all cheap: for each class $c$, treat "is the true label $c$?" against the classifier's probability for class $c$, fit a separate $(A_c, B_c)$ on the log-odds of that column with its own Laplace-smoothed targets, independently for all $C$ classes. Independently fit, the outputs do not sum to one, so I renormalize each by the row sum — each one-against-all output already estimates $P(c\mid x)$, so dividing by the row sum is the natural combiner and gives the harness the valid distribution its row-sum assertion requires.

The honest worry is the flip side of the data efficiency I am buying: the sigmoid is *one* shape. On a column whose true distortion is an inverse-sigmoid — a tree-style classifier that slams probabilities to the extremes, so the correct map must *gather* them back — the sigmoid with $A<0$ can only push toward the extremes, never gather, so the best fit can be a bad fit; and on an already-calibrated column the sigmoid cannot represent the identity, so it moves probabilities it should leave alone. That rigidity is the price of two-parameter stability, and the multiclass tasks are where I expect it to show. So against isotonic's measured numbers I expect, falsifiably: the proper-score bleed should *fall* — Madelon NLL well below 0.582, RF NLL below 0.237 with a tighter seed spread — because a graded sigmoid replaces the coarse flat blocks; but ECE may *rise* where isotonic's data-chosen binning matched accuracy best, the SVM column especially if its log-odds curve is not quite sigmoidal. If the pattern is proper scores down, ECE up where the curve fights the single shape, the verdict points straight at the next rung: the residual ECE is the shape flexibility I gave up, which a richer-but-still-parametric family would buy back without reopening the variance problem.

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
