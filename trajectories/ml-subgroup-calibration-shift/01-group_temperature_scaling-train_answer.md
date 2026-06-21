The floor I build from is the identity fill: clip the raw positive-class probability $p$ into $[\varepsilon, 1-\varepsilon]$ and hand it back. That is not a method, it is the absence of one — it reports the frozen classifier's over-confident numbers and lets the worst-group ECE measure how wrong they are. The task is graded on the *worst* subgroup, so my first move attacks subgroups head on: give each subgroup its own correction. The whole risk lives there, because the calibration split is small and the test tail is shifted, so I derive the per-group machinery carefully and build in the defense against its failure mode from the start.

I propose **group temperature scaling with James–Stein shrinkage toward a global temperature**. It rests on three pieces, and the order matters. The first piece is plain temperature scaling. The raw $p$ is wrong in a specific, diagnosable way: a log-loss-trained classifier, once it is classifying almost everything correctly, keeps lowering its loss by pushing probabilities toward 0 and 1, overfitting NLL long after the 0/1 error has flattened, and all the excess goes into confidence. The failure is *scale*, not order — the ranking is intact, which the diagnostic `subgroup_auroc` confirms, and a monotone map cannot touch it anyway. Scale does not live in $p$, which is squashed into $[0,1]$; it lives in the logit $z = \mathrm{logit}(p) = \log\frac{p}{1-p}$, where $\sigma$ and $\mathrm{logit}$ are the monotone bijection between $[0,1]$ and the real line. So the minimal scale correction divides the logit by a positive number,

$$q = \sigma(z/T).$$

Here $T=1$ is the identity floor; $T>1$ shrinks every logit toward zero and pulls every $q$ toward $1/2$, softening over-confidence and raising entropy; $T<1$ sharpens. The property that makes it safe on this benchmark is that $z/T$ is monotone increasing in $z$ for any $T>0$, so it never reorders examples and never moves the $z=0 \leftrightarrow p=0.5$ boundary — accuracy and `subgroup_auroc` are untouched, only the confidences soften. This is the one-parameter special case of Platt's $\sigma(a z + b)$ with $a = 1/T$ and the intercept $b$ dropped; I drop $b$ deliberately, because a nonzero intercept moves the boundary off $z=0$ and could change predictions, and because every extra parameter is variance I will have to pay once I split by subgroup. I fit $T$ by minimizing the calibration-split binary NLL $-\mathrm{mean}\big(y\log q + (1-y)\log(1-q)\big)$, because NLL is a proper scoring rule — minimized in expectation exactly when the reported probability is the true conditional — whereas the binned ECE I am graded on is non-differentiable, so I fit NLL and only *measure* ECE. And the scalar is not an arbitrary squash: among all per-example valid distributions that match one moment (the average true-class logit equals the average expected logit under $q$), the maximum-entropy one is the softmax of $\lambda z$, and in binary form that is $\sigma(z/T)$ with $\lambda = 1/T$. So a single temperature is the honest maximum-entropy correction of exactly the scale error I diagnosed.

The second piece is per-group fitting, and the third is the shrinkage that keeps it from backfiring. A global $T$ minimizes pooled NLL — a population-weighted compromise — but my objective is the worst subgroup, and subgroups can genuinely need different amounts of softening. So I would fit a separate $T_g$ per subgroup by the same NLL minimization. The trouble is statistical and I can see it before running: how well is $T_g$ pinned down by $n_g$ calibration points? Working in $r_g = \log T_g$, the per-example NLL has Fisher curvature of order one, so a group loss has curvature $O(n_g)$ and the fitted log-temperature has variance $\mathrm{Var}(\log T_g) \approx c/n_g$. A group with thousands of points gets a trustworthy $T$; a group with thirty gets a temperature read off an almost-flat likelihood, and on a tiny or lopsided sample the NLL can be monotone across the whole search box so the fit just slams into a boundary — a $T$ of 20 or 0.05 that means nothing. Apply that noisy $T_g$ to the group's *shifted* test points and the noise does not average out; it becomes a per-group systematic distortion on a distribution the calibration sample never represented, routinely making a small subgroup *worse* than the global $T$ would have. And the worst-group metric reads off exactly the group where that garbage is worst.

This is the shape Stein exposed: many related-but-not-identical parameters (one temperature per group, all answering "how over-confident is the model here"), each estimated in isolation from its own small sample, is the coordinatewise MLE, which is inadmissible in three or more dimensions — there is an estimator with strictly smaller total risk that pulls every coordinate toward a common center, harder for the noisier ones (James & Stein 1961; Efron & Morris 1973 shrank toward the grand mean). The common center is sitting right there: the global temperature $T_{\text{global}}$ fit on all the data, the low-variance pooled estimate. I blend in $\log$-space, not on $T$ directly, because $T$ is positive and multiplicative — a convex combination of log-temperatures exponentiates to a positive temperature automatically, and the two mirror softenings sit symmetrically about zero:

$$\log T_g = \alpha_g \log T_{\text{local},g} + (1-\alpha_g)\log T_{\text{global}}.$$

The two-level Gaussian model hands me $\alpha_g$ rather than my guessing it. Each local fit $m_g = \log T_{\text{local},g}$ estimates the truth $\theta_g$ with sampling variance $\sigma_w^2/n_g$, and the truths scatter about the center as $\theta_g \sim \mathcal{N}(\mu = \log T_{\text{global}}, \sigma_b^2)$. The posterior mean is the precision-weighted blend, and writing $k = \sigma_w^2/\sigma_b^2$ it collapses to

$$\alpha_g = \frac{n_g}{n_g + k}.$$

This is monotone in group size and lives in $(0,1)$: $n_g \to 0$ gives $\alpha_g \to 0$ (full pooling to global, correct because the local fit told me nothing), $n_g \to \infty$ gives $\alpha_g \to 1$ (no pooling, correct because the local fit is now sharp). $k$ is the crossover group size — at $n_g = k$ the group is half local, half global — with a clean beta-binomial reading as the prior pseudo-count of evidence a group must accumulate before it earns its own estimate. I fix $k = 200$ rather than estimating it, because with only a handful of groups (the cross-product of two protected attributes), estimating the between-group variance is itself the high-variance disease I am treating; $k=200$ weights a 200-point group 50/50, a 50-point group at $\alpha=0.2$ (mostly global), a 2000-point group at $\alpha \approx 0.91$ (mostly local). The degenerate tail gets a hard guard: if a group has fewer than 20 points or all-one-class labels, its local NLL is unidentified and the minimizer wanders to a box boundary, so I refuse to fit it and set $T_g = T_{\text{global}}$ outright. Throughout I clip $p$ and $q$ into $[\varepsilon, 1-\varepsilon]$ with $\varepsilon = 10^{-6}$ so the logit and log stay finite, and optimize $\log T$ over $[-3, 3]$ (so $T \approx [0.05, 20]$) so the 1-D search stays well conditioned and $T$ cannot run away on a flat objective. With no group ids supplied, the whole thing degenerates cleanly to global temperature scaling.

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
