The subgroup-aware run paid off where I least needed it and failed where I am graded. Group temperature scaling posted worst-group ECE of 0.4938 on Adult, 0.4972 on COMPAS, 0.3138 on Law School (mean 0.4349), while its Brier (mean 0.370) was respectable and its `subgroup_auroc` was untouched at 0.859/0.882/0.749 exactly as the monotone-map argument promised. So the per-group temperatures did something sensible on average but the *worst* subgroup barely moved — precisely the structural failure I had flagged, because the worst subgroup is the small one, and the empirical-Bayes weight $\alpha_g = n_g/(n_g+200)$ pulls that group hard toward the global temperature, so on the group that decides my metric I was effectively already doing global temperature scaling. The per-group degrees of freedom bought Brier and almost nothing in worst-group ECE under shift. Before I climb to a richer map I need a number I do not yet have: how much of that 0.4349 is the subgroup machinery and how much is just what a single global scalar does here. So this rung deliberately steps *back*.

I propose **plain (global) temperature scaling** — strip out every per-group parameter and fit one temperature for everyone. It is the bare version of what was already running globally inside the group method, with the group loop removed, and the point of it is to be the floor against which the subgroup machinery either justifies itself or does not. Let me re-derive it airtight, because everything downstream hangs on this control being clean. The raw $p$ is over-confident in the standard way: a log-loss-trained classifier, once it is classifying almost everything correctly, keeps lowering its loss by pushing probabilities toward 0 and 1, overfitting NLL long after the 0/1 error flattened, and the excess goes into confidence. The failure is *scale*, not order — the ranking is intact, which `subgroup_auroc` keeps confirming at 0.86/0.88/0.75 on every method I run. Scale does not live in $p$, squashed into $[0,1]$; it lives in the logit $z = \mathrm{logit}(p) = \log\frac{p}{1-p}$, where $\sigma$ and $\mathrm{logit}$ are the monotone bijection between $[0,1]$ and the real line. The minimal scale correction divides the logit by a positive number,

$$q = \sigma(z/T).$$

$T=1$ is the identity floor; $T>1$ shrinks every logit toward zero and pulls every $q$ toward $1/2$, softening over-confidence and raising entropy; $T<1$ sharpens; $T\to\infty$ collapses to $1/2$, $T\to 0$ snaps to hard 0/1. The property that makes it safe here is that $z/T$ is monotone increasing in $z$ for any $T>0$: it never reorders examples and never moves the $z=0 \leftrightarrow p=0.5$ boundary, so the predicted class, accuracy, and AUROC are exactly preserved and only the confidences soften. This is the one-parameter special case of Platt's $\sigma(a z + b)$ with $a = 1/T$ and the intercept $b$ dropped. Dropping $b$ is deliberate: a nonzero intercept moves the boundary $az+b=0$ off $z=0$ and would let recalibration change predictions, and an extra parameter is extra variance I refuse to pay when the whole lesson of the last run is that variance under shift is what hurts.

I want to be sure dividing the logit by a scalar is the *right* correction and not merely convenient, so I pin down what it optimizes. I fit $T$ by minimizing the calibration-split binary NLL $-\mathrm{mean}\big(y\log q + (1-y)\log(1-q)\big)$, because NLL is a proper scoring rule — minimized in expectation exactly when the reported probability is the true conditional — so descending it pushes $q$ toward calibration; ECE bins and is non-differentiable, so I fit NLL and *measure* ECE, never the reverse. And the family "scale the logits" is not arbitrary: among all per-example valid distributions that match one moment — the average true-class logit equals the average expected logit under $q$ — the maximum-entropy one is the softmax of $\lambda z$. Setting up the Lagrangian and taking stationarity gives $q_i^{(k)} = \mathrm{softmax}(\lambda z_i)^{(k)}$; writing $\lambda = 1/T$ and specializing to binary, $q = \sigma(z/T)$. So the single scalar is the lone Lagrange multiplier of the lone moment constraint — the honest minimal fix for exactly the scale error I diagnosed. There is essentially no capacity here to overfit the calibration split, which is precisely why I expect it to *survive the shift* better than the per-group fits did: a single number estimated from the whole calibration set has the lowest variance of anything on this ladder, and low variance is what transfers when the test tail is shifted.

The implementation is therefore minimal. Clip $p$ into $[\varepsilon, 1-\varepsilon]$ with $\varepsilon = 10^{-6}$ so the logit is finite; take $z = \mathrm{logit}(p)$; minimize the NLL of $\sigma(z/T)$ over $\log T \in [-3, 3]$ (so $T \approx [0.05, 20]$) by a bounded 1-D scalar search — fit in $\log T$ because $T>0$ is a positivity constraint I would rather not babysit and $T$ lives on a multiplicative scale where $\log T$ is the natural unconstrained coordinate, and bound the box so the search stays well conditioned and $T$ cannot run away on a flat objective; fall back to $T=1$ if the search fails to converge. At predict time take the logit of the clipped input, divide by the fitted $T$, apply $\sigma$, and clip back into $[\varepsilon, 1-\varepsilon]$. `groups` is accepted and ignored — this is group-agnostic by construction, which is exactly the control I want. If even this lowest-variance map cannot get Adult and COMPAS worst-group ECE much below the mid-0.4s, then the bottleneck is not the parameter count or the per-group split but the *shape* of the global map, and the next move is to keep this low-variance posture while giving the map a richer, still-monotone shape than a single division can produce.

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
