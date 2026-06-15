Let me start from what actually goes wrong. I have a binary classifier I'm not allowed to retouch; it hands me a positive-class probability `p` for each example, and I want those numbers to mean what they say — out of all the examples it labels "0.7", roughly 70% should really be positive. And I don't just want this on average over everyone; I want it to hold inside each subgroup, because a calibrator that looks perfect pooled across the whole population can still be badly over-confident on one subgroup and under-confident on another, and the moment someone thresholds the confidence to make a decision, that mismatch becomes a per-group bias. So the thing I actually care about is the *worst* subgroup's calibration error, not the mean. On top of that the calibration data and the test data don't come from the same place — a domain score peels off a held-out tail for test — so whatever I learn on the calibration split has to survive a shift. I can only learn a map from `p` (and maybe the subgroup id `g`) to a corrected `q`; the model and the splits are frozen.

Why is `p` wrong at all? It's the standard story for a model trained to minimize log-loss: once it's classifying almost everything correctly, the loss can still go down by shoving the probabilities harder toward 0 and 1, so the model keeps sharpening past the point where the numbers reflect real frequencies. It overfits the log-loss long after the 0/1 error has flattened, and all that excess goes into confidence. The useful part of that observation is *which* way it's wrong: the cheap, dominant failure is that the overall *scale* of the score is too big — the model is uniformly too sure — not that the ranking is scrambled. If the ordering were broken I'd need something that can reorder examples; but it isn't, so I should look for the smallest knob that fixes a scale.

Where does "scale" live? Not in `p` directly — `p` is squashed into `[0,1]` and a "scale" there isn't even well defined. It lives in the logit. For a binary model `p = sigma(z)` with `z = log(p/(1-p))`, and `sigma` and `logit` are the canonical monotone bijection between `[0,1]` and the whole real line. So any monotone reshaping of `p` is a reshaping of `z`, and the cleanest scale knob is to divide the logit by a positive number, `q = sigma(z/T)`. Stare at what that does. `T = 1` is the identity. `T > 1` shrinks every logit toward zero, so every `q` moves toward `1/2` — it *softens*, raises the entropy, which is exactly the cure for over-confidence. `T < 1` sharpens. As `T -> infinity` everything collapses to `1/2`; as `T -> 0` everything snaps to a hard 0 or 1. And here's the property I really need: `z/T` is a monotone increasing function of `z` for any `T > 0`, so it never changes the order of examples and never moves the `z = 0 <-> p = 0.5` decision boundary. The predicted class — and therefore the accuracy — is untouched. One positive scalar, fit to soften the model just enough, with no risk to accuracy. That's temperature scaling, and it's the one-parameter special case of the logistic recalibration `q = sigma(a z + b)` from Platt — set `a = 1/T` and drop the intercept `b`. I deliberately drop `b`, because a nonzero intercept makes the boundary `a z + b = 0` no longer `z = 0`, which would let the recalibration *change* predictions, and I want to leave accuracy alone; and because every extra parameter is capacity I'll have to pay for in variance later, which is going to matter a lot here.

I want to be sure dividing the logit by a scalar is the *right* correction and not just a convenient one, so let me pin down what it optimizes. I'm fitting `T` by minimizing the validation NLL — binary cross-entropy `-mean(y log q + (1-y) log(1-q))` — because NLL is a proper scoring rule: in expectation it's minimized exactly when the reported probability is the true conditional probability, so descending it literally pushes `q` toward calibration. (ECE, the binned accuracy-minus-confidence gap, is what I'll *measure*, but it bins and isn't differentiable, so it can't be the thing I fit.) Now, what family of recalibrations does "scale the logits" correspond to? Ask the dual question: among all per-example distributions `q_i` that are valid probabilities and that match one moment of the data — the average true-class logit equals the average expected logit under `q` — which has maximum entropy? Set up the Lagrangian, `L = -sum_i sum_k q_i^{(k)} log q_i^{(k)} + lambda sum_i [ sum_k z_i^{(k)} q_i^{(k)} - z_i^{(y_i)} ] + sum_i beta_i ( sum_k q_i^{(k)} - 1 )`. Differentiate in `q_i^{(k)}`: `-log q_i^{(k)} - 1 + lambda z_i^{(k)} + beta_i = 0`, so `q_i^{(k)} = exp(lambda z_i^{(k)} + beta_i - 1)`, and normalizing over `k` kills `beta_i` and leaves `q_i^{(k)} = softmax(lambda z_i)^{(k)}`. Write `lambda = 1/T` and that's exactly logits-over-`T`. So softening by a single temperature isn't an arbitrary squash — it's the maximum-entropy distribution that corrects nothing but the average logit scale, which is precisely the failure I diagnosed. Good. The scalar is the honest minimal fix. In binary form it's just `q = sigma(z/T)`, `z = logit(p)`.

So my first concrete calibrator: clip `p` into `[eps, 1-eps]` so the logit is finite, take `z = logit(p)`, and find `T` minimizing the NLL of `sigma(z/T)` on the calibration set. One scalar, one-dimensional optimization, accuracy preserved. And I should fit it in `log T` rather than `T`, both because `T > 0` is a positivity constraint I'd rather not babysit and because `T` lives on a multiplicative scale — `T = 2` and `T = 1/2` are inverse softenings — so the natural unconstrained coordinate is `log T`. Optimize `log_t` over a bounded box, say `[-3, 3]`, i.e. `T` from about `0.05` to `20`, which is generous but keeps the 1-D search well conditioned and stops `T` running off to infinity on a degenerate objective. That last worry isn't idle; I'll come back to it.

Now the actual problem: this one `T` is shared by everyone, and my objective is the *worst subgroup*, not the average. Different subgroups can genuinely need different amounts of softening — their score distributions and base rates differ, so the model is over-confident by different amounts on each. A single `T` minimizes pooled NLL, which is a population-weighted compromise; it can leave a big well-behaved subgroup nicely calibrated while one minority subgroup stays over-confident and another ends up under-confident, and the worst-group ECE barely moves. The obvious response is to stop sharing: fit a separate temperature `T_g` per subgroup, each on its own calibration points, each by the same NLL minimization. Per-group softening, each group fixed on its own terms. Let me just do that and see if it holds up.

It doesn't, and I can see why before I even run it. How well is `T_g` pinned down by `n_g` calibration points? Use the parameter I actually optimize, `r_g = log T_g`, and write each fitted probability as `q_i(r_g) = sigma(z_i e^{-r_g})`. For one example the NLL derivative is `(y_i - q_i) z_i e^{-r_g}`; near the population optimum the expected second derivative is the Fisher term `E[q_i(1-q_i)(z_i e^{-r_g})^2]`, a per-example constant as long as the group's score distribution is not degenerate. A group loss is a sum of `n_g` such terms, so its curvature is `O(n_g)`, and the variance of the fitted `r_g` is the inverse curvature: `Var(log T_g) ~ c / n_g`. A subgroup with a few thousand points gets a sharp, trustworthy temperature; a subgroup with thirty points gets a temperature read off an almost-flat likelihood, and worse, on a small or lopsided sample the NLL can be monotone in `log_t` over my whole box, so the "fit" just slams into a boundary — a `T` of 20 or 0.05 that means nothing except "this group's likelihood didn't constrain me." Then I take that noisy, possibly boundary-pinned `T_g`, and I apply it to that group's *test* points, which are drawn from a shifted distribution. The noise doesn't average out; it's a per-group systematic distortion. I'd routinely make a small subgroup's calibration *worse* than if I'd just used the single global `T`. So independent per-group fitting trades a real bias problem (one `T` can't fit everyone) for a worse variance problem (each small `T_g` is garbage). Wall.

Let me name what kind of mistake "fit each group's parameter from its own sample" is, because it feels familiar. I have many parameters — one temperature per group — that are *related but not identical*: they're all "how over-confident is the model here," they should be similar, but not equal. And I'm estimating each one in isolation from its own small, noisy sample. That's the statistical shape Stein exposed. For `X ~ N(theta, sigma^2 I)` with three or more coordinates, the estimator "report each coordinate's own observation" — the per-coordinate MLE, the clean normal-means analogue of my independent per-group fits — is *inadmissible*: there is another estimator with strictly smaller total squared-error risk for **every** `theta`, no exceptions, and it works by pulling every coordinate toward a common center, even when the coordinates are unrelated. James & Stein wrote it down, `(1 - (p-2)sigma^2/||X||^2) X`, shrinking toward zero; Efron & Morris read it as empirical Bayes and made the punchline concrete: shrink each estimate **toward the grand mean**, by an amount that depends on how noisy that estimate is, and on their baseball data the shrunk batting averages beat the raw per-player averages for 16 of 18 players. The lesson transfers directly: I should not estimate `K` temperatures independently. I should pull each group's temperature toward a common center, and pull the noisy ones harder.

What's the common center? I already have a low-variance, well-identified one sitting right there: the global temperature `T_global` fit on *all* the calibration data. It's the pooled estimate, the analogue of the grand mean — it has the most data behind it and the least variance. So the shape of the fix is: fit `T_global` on everyone; fit `T_local,g` on each group; and report, for each group, something between the two, leaning on `T_local,g` when the group is big and well-determined and on `T_global` when the group is small and its local fit is noise. A group that barely constrains its own temperature should mostly inherit the global one; a group with thousands of points should mostly keep its own.

In what space do I blend? Not on `T` directly. `T` is a multiplicative, positive quantity, and a plain linear average of two temperatures doesn't respect that — it weights `T = 4` and `T = 1` asymmetrically against `T = 1` and `T = 1/4`, even though those are mirror-image softenings, and I'd have to keep checking the result stays positive. The natural coordinate is the one I'm already optimizing in, `theta_g = log T_g`. It's unconstrained, a convex combination of two log-temperatures is automatically a positive temperature when I exponentiate, and on the log scale the two mirror softenings sit symmetrically about zero. Shrinking the *unconstrained* parameter toward the common value is also just the standard empirical-Bayes move. So:

  `log T_g = alpha_g * log T_local,g + (1 - alpha_g) * log T_global`,

with `alpha_g` near 1 for big groups and near 0 for small ones. The only thing left is the exact form of `alpha_g`, and I don't want to guess it — the hierarchical model hands it to me.

Set up the two-level picture. Each group's own fitted parameter `m_g = log T_local,g` is an estimate of the group's true `theta_g`, with sampling variance `Var(m_g) ~ sigma_w^2 / n_g` — that's the `1/n_g` scaling I derived from the likelihood curvature, with `sigma_w^2` collecting the per-point Fisher constant ("within-group" noise). And the group truths themselves are spread around a common value `mu = log T_global` with some "between-group" variance `sigma_b^2`: `theta_g ~ N(mu, sigma_b^2)`. Now I want the estimate of `theta_g` that combines the local observation `m_g` and the prior mean `mu`. Each carries information in proportion to its precision (inverse variance): the local observation has precision `n_g/sigma_w^2`, the prior has precision `1/sigma_b^2`. The precision-weighted combination — which is the posterior mean in this Gaussian model — is

  `theta_hat_g = [ (n_g/sigma_w^2) m_g + (1/sigma_b^2) mu ] / [ n_g/sigma_w^2 + 1/sigma_b^2 ]`.

Factor out to read off the weight on the local estimate. Multiply top and bottom by `sigma_w^2`:

  `theta_hat_g = [ n_g * m_g + (sigma_w^2/sigma_b^2) mu ] / [ n_g + sigma_w^2/sigma_b^2 ]`.

Call `k = sigma_w^2/sigma_b^2`. Then

  `theta_hat_g = (n_g/(n_g + k)) m_g + (k/(n_g + k)) mu = alpha_g m_g + (1 - alpha_g) mu`,
  `alpha_g = n_g / (n_g + k)`.

The shrinkage weight is `alpha_g = n_g/(n_g + k)`. It's monotone increasing in the group size, it lives in `(0,1)`, and the limits are exactly the behavior I argued for: as `n_g -> 0`, `alpha_g -> 0`, the group is pulled entirely to the global temperature (full pooling — correct, because its local fit told me nothing); as `n_g -> infinity`, `alpha_g -> 1`, the group keeps its own temperature (no pooling — correct, because its local fit is now sharp). And `k` has a clean meaning: it's the ratio of within-group sampling noise to between-group spread, and it's the *crossover* group size — at `n_g = k` the group is weighted exactly half local, half global. The same law shows up if I phrase calibration of a rate as a beta-binomial: the empirical-Bayes estimate of a success rate is `(successes + a0)/(total + a0 + b0)`, which is a weighted average of the observed rate and the prior mean with weight `n/(n + (a0+b0))` on the data — the prior pseudo-count `a0+b0` *is* `k`. So `k` is a pseudo-count, the amount of evidence a group must accumulate before it earns its own estimate.

So what should `k` be? In principle I could estimate `sigma_w^2` and `sigma_b^2` from the data and let the model set its own shrinkage — that's the fully empirical-Bayes route. But here the number of groups is tiny (the cross-product of a couple of protected attributes — a handful of groups), so a data-driven estimate of the *between-group* variance `sigma_b^2` from those few groups would itself be a high-variance estimate, and I'd be back to estimating a noisy quantity from a small sample, the very disease I'm treating. Better to fix `k` to a conservative constant matched to the regime. The subgroup calibration samples here run from dozens to maybe a couple thousand points. I want a group to need real evidence — on the order of a couple hundred points — before it gets to half-trust its own temperature, so that the genuinely small groups stay firmly anchored to the global fit while the large groups are still allowed to individuate. `k = 200` does that: a 200-point group is weighted 50/50, a 50-point group gets `alpha = 50/250 = 0.2` (mostly global), a 2000-point group gets `alpha = 2000/2200 ≈ 0.91` (mostly local). That's exactly the conservative profile I want on data where unshrunk per-group calibration was overfitting.

There's still the degenerate tail to handle cleanly. If a group has only a handful of points, or all its calibration labels are the same class, the local NLL isn't just noisy — it's essentially unidentified; the minimizer will wander to a boundary of my `[-3,3]` box and the resulting `T_local` is meaningless. I could let `alpha_g = n_g/(n_g + k)` handle it — for a 5-point group `alpha = 5/205 ≈ 0.024`, almost pure global anyway — but multiplying even a tiny weight onto a boundary-pinned garbage `log T_local` is asking for trouble, and fitting a 1-D optimizer on a one-class group is wasted work that can throw. Cleaner to refuse outright: if `n_g` is below a small floor (say 20) or the group has fewer than two distinct labels, don't even attempt a local fit — set `T_g = T_global` and move on. Above the floor, do the local fit and the log-space shrink. This isn't a second tuning knob; it's a guard on the unidentified case, and it sits below the point where `alpha` would have given the group any real weight anyway.

Let me also keep the numerics honest throughout. Probabilities of exactly 0 or 1 send the logit to `±infinity`, so I clip `p` into `[eps, 1-eps]` with `eps = 1e-6` before taking `logit`, and clip the calibrated output back into `[eps, 1-eps]` so I always return a valid probability. `eps` that small is well below any real probability mass, so it floors the singularities without perturbing the calibration. And the local NLL objective needs the same clip inside it, since `sigma(z/T)` can graze 0 or 1 for large `|z/T|` and `log` would blow up.

Now assemble it. Fitting: clip and take logits; fit the global temperature by minimizing the NLL of `sigma(z/T)` over `log_t in [-3,3]`. Then for each group: if it's too small or single-class, record `T_global`; otherwise fit a local temperature the same way on the group's points, compute `alpha = n_g/(n_g+k)`, blend in log space `log T_g = alpha log T_local + (1-alpha) log T_global`, exponentiate, store. Prediction: take logits of the (clipped) input; for each group apply its stored `T_g` (falling back to `T_global` for an unseen group id), `q = sigma(z/T_g)`, and clip the output. If no group ids are supplied at all, it degenerates to plain global temperature scaling — exactly the right behavior, since with no group structure there's nothing to shrink.

```python
import numpy as np
from scipy import optimize, special


class CalibrationMethod:
    """Group-wise temperature scaling with empirical-Bayes shrinkage toward
    a single pooled temperature. Per-group temperatures are blended in
    log-space with the global temperature by alpha = n_g / (n_g + k_shrink):
    small/degenerate groups fall back to the global fit, large groups keep
    their own. Accuracy is unchanged (a single positive scalar per group is
    monotone in the logit)."""

    def __init__(self):
        self.eps = 1e-6
        self.k_shrink = 200.0          # crossover group size: alpha = 1/2 at n_g = k
        self.group_temperatures_ = {}
        self.global_temperature_ = 1.0

    def _fit_temperature(self, probs, labels):
        # fit one T by minimizing binary NLL of sigma(z/T), z = logit(p)
        probs = np.asarray(probs, dtype=float).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))

        def objective(log_t):
            t = float(np.exp(log_t))                       # optimize in log T (positive, multiplicative scale)
            cal = special.expit(logits / t)                # q = sigma(z / T)
            p = np.clip(cal, self.eps, 1.0 - self.eps)     # keep log finite
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        result = optimize.minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        return float(np.exp(result.x)) if result.success else 1.0

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs, dtype=float).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        # pooled global temperature = the common center (the "grand mean")
        self.global_temperature_ = self._fit_temperature(probs, labels)
        log_T_global = float(np.log(self.global_temperature_))
        self.group_temperatures_ = {}
        if groups is None:                                 # no subgroups -> plain global TS
            return self
        groups = np.asarray(groups).reshape(-1)
        for g in np.unique(groups):
            mask = groups == g
            n_g = int(mask.sum())
            # unidentified local fit (too few points, or one class) -> fall back to global
            if n_g < 20 or np.unique(labels[mask]).size < 2:
                self.group_temperatures_[int(g)] = self.global_temperature_
                continue
            T_local = self._fit_temperature(probs[mask], labels[mask])
            log_T_local = float(np.log(T_local))
            alpha = n_g / (n_g + self.k_shrink)            # n_g/(n_g+k): EB / partial-pooling weight
            log_T_g = alpha * log_T_local + (1.0 - alpha) * log_T_global   # shrink in log-space
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
            temp = self.group_temperatures_.get(int(g), self.global_temperature_)  # unseen group -> global
            out[mask] = special.expit(logits[mask] / temp)
        return np.clip(out, self.eps, 1.0 - self.eps)
```

The chain is: the model's over-confidence is dominantly a logit-scale error, so the minimal accuracy-preserving fix is to divide logits by a single temperature, fit by NLL — that's the maximum-entropy correction of exactly that error. One global temperature can't serve heterogeneous subgroups, so I want one per group; but fitting each group's temperature independently from its own small sample has the variance profile of the coordinatewise MLE that Stein showed how to improve, and under test-time shift the small-group temperatures can be worse than the global one. The cure is to shrink each group's temperature toward the pooled global temperature, in log-space because that's the natural scale of a positive multiplicative parameter, with the precision-weighted empirical-Bayes weight `alpha = n_g/(n_g + k)` that the two-level Gaussian model forces — full pooling for tiny groups, no pooling for huge ones, `k` the crossover pseudo-count fixed conservatively at 200 — plus a hard floor that refuses to fit a local temperature on an unidentified (too-small or single-class) group.
