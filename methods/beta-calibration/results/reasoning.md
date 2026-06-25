Let me start from what actually goes wrong when I take a classifier's score and treat it as a probability. I have a scoring model `s = f(x)` that hands me a number in `[0, 1]`, and I want `s` to mean what it says: among all the instances that got score `s`, a fraction `s` of them should really be positive. That is the whole definition of calibration — for every output value `s_i = f(x_i)`, I want `s_i = E[Y | f(X) = s_i]`. There is in fact a unique map on any fixed dataset that makes this exactly true, namely `mu(s_i) = E[Y | f(X) = s_i]`, the empirical positive rate at each score. But I can't just fit that. If `f` gives a distinct score to every training instance, then the empirical rate at each score is just that instance's own label, so the "perfect" map is `mu(s_i) = y_i`, a pile of 0s and 1s — maximally overconfident, and nowhere near calibrated on data I haven't seen. So I'm forced to choose a *family* of maps with some inductive bias, fit it on a held-out calibration split, and hope the bias is the right one. The entire game is choosing that family.

The family everyone reaches for is the sigmoid — logistic, or Platt, calibration. `mu(s) = 1/(1 + 1/exp(gamma*s + delta))` with `gamma >= 0` to keep it monotone, and if I write `m = -delta/gamma` it becomes `mu(s) = 1/(1 + 1/exp(gamma*(s - m)))`, a sigmoid whose midpoint (where the calibrated value crosses `1/2`) sits at `s = m` and whose slope there is `gamma/4`. It fits with a single call to a logistic-regression routine — feature `s`, label `y`, minimise log-loss — which is maximum likelihood, because log-loss *is* the negative log-likelihood of the labels. Cheap, low-variance, needs little data. So why am I not happy.

I want to understand *where the sigmoid comes from*, because the place it comes from is probably the place it's going to fail. It isn't an arbitrary squashing function; it drops out of a generative story. Posit that within each class the score is normally distributed with the *same* variance `sigma^2`, means `s_+` for positives and `s_-` for negatives. Form the per-class likelihood ratio and watch what happens:

```
LR(s) = p(s|+)/p(s|-)
      = exp[ ( -(s - s_+)^2 + (s - s_-)^2 ) / (2*sigma^2) ].
```

Expand the two squares. `-(s - s_+)^2 + (s - s_-)^2 = -(s^2 - 2 s s_+ + s_+^2) + (s^2 - 2 s s_- + s_-^2) = 2 s (s_+ - s_-) - (s_+^2 - s_-^2)`. The `s^2` terms cancel — that cancellation is the whole point, it's what makes the exponent *linear* in `s` — and `s_+^2 - s_-^2 = (s_+ - s_-)(s_+ + s_-)`, so

```
LR(s) = exp[ (s_+ - s_-)/sigma^2 * ( s - (s_+ + s_-)/2 ) ] = exp[ gamma (s - m) ],
```

with `gamma = (s_+ - s_-)/sigma^2` and `m = (s_+ + s_-)/2`. Under a uniform class prior the likelihood ratio is the posterior odds, so `mu(s) = 1/(1 + LR(s)^{-1}) = 1/(1 + exp(-gamma(s - m)))` — the sigmoid, exactly. And it runs both ways: pick any `gamma > 0` and `m`, set `s_+ - s_- = gamma` and `s_+ + s_- = 2m` (so `s_+ = m + gamma/2`, `s_- = m - gamma/2`) with `sigma = 1`, and those two equal-variance Gaussians reproduce that sigmoid. So the sigmoid family *is* the set of maps I get if I believe the per-class scores are equal-variance Gaussians, no more and no fewer. Fine. Now I know what assumption I'm actually making every time I Platt-scale, and I can ask whether it's true.

It is not true, and I want to be precise about *how* it's false rather than just complaining. Two structural problems. First, my scores live in `[0, 1]` — they're bounded — and a Gaussian puts mass on the whole real line. For a classifier like Naive Bayes whose outputs are genuinely confined to `[0, 1]`, modelling the per-class score as a Gaussian is incoherent on its face; the density I'm assuming says positive things outside the only region the score can occupy. Second, and this is the one that actually bites in the reliability diagrams, the *shape* the sigmoid can produce is one-directional. The equal-variance Gaussians give `gamma >= 0`, an S-curve that takes scores clustered near the middle and *spreads them toward 0 and 1*. That is exactly right for a maximum-margin classifier or for the SAMME-style boosting that pulls its scores toward `0.5` — there the empirical positive rate is more extreme than the score, and I need to push the score out. But Naive Bayes does the opposite. It multiplies many per-feature likelihood ratios, and when features are correlated it double-counts the evidence, so it slams its scores against 0 and 1 — overconfident at the extremes. Original-formulation Adaboost, read through the additive-logistic-regression lens, does the same. For those classifiers the empirical positive rate is *less* extreme than the score: a score of `0.99` corresponds to maybe `0.8` true positives. The correct map has to *pull the extreme scores back in*. That's an inverse-sigmoid shape — steep near the ends, flat in the middle — and `gamma >= 0` only spreads; it never gathers. So on Naive Bayes the best possible logistic fit is still a bad fit, and it can leave me with probabilities that are *worse* than the raw scores I started with.

There's a third concern, and I want to actually check it rather than wave at it. Suppose the classifier is *already* perfectly calibrated and I don't know it, so I run calibration anyway. The right thing to learn is the identity, `mu(s) = s`, leave it alone. Is the identity in the sigmoid family? `1/(1 + exp(-gamma(s - m))) = s` for all `s`? The left side is a bounded S-curve that only hits 0 and 1 in the limits `s -> -inf, +inf`; the right side is a straight line through `(0,0)` and `(1,1)`, so no finite `gamma, m` can make them equal at the endpoints. But "equal at the endpoints in a limit" is a weak statement — I'd believe it more if I saw how badly the *best* sigmoid does in the middle of the range. So I fit the least-squares-closest sigmoid to `mu(s) = s` over `s in (0,1)`. The optimiser lands at `gamma ~ 4.92`, `m = 0.5` (symmetric, as it should be), and the worst-case gap `max_s |sigmoid(s) - s|` is about `0.078` — nearly eight points of probability off. Concretely, at `s = 0.01` the best sigmoid returns `0.082`, eight times too large, and at `s = 0.99` it returns `0.918`. So a classifier that was already perfectly calibrated comes out of Platt scaling reading `0.082` where the truth is `0.01`. The identity really is outside the family, and the damage is not a rounding error.

So the sigmoid is too rigid in three concrete ways — wrong support, one-directional shape, no identity — and all three trace back to a single modelling choice: equal-variance Gaussians on a bounded score. The non-parametric escape is isotonic calibration: don't assume any shape, just fit the best monotone non-decreasing step function from score to label, `min over isotonic m of sum_i (y_i - m(f(x_i)))^2`, solved by pair-adjacent-violators — sort by score, and wherever the running fitted value would decrease, merge adjacent blocks into their pooled average until monotonicity is restored. That captures *any* monotone distortion, inverse-sigmoid included, so it dodges the shape problem entirely. But it trades the shape problem for a data problem. With no parametric bias, a step function has as many effective degrees of freedom as there are blocks, and the learning-curve studies are blunt about it: isotonic calibration overfits badly on small calibration sets and needs a lot more data than the sigmoid to reach the same reliability. On a small held-out split — which is the regime I'm usually in, because the calibration fold is a *fraction* of an already-limited dataset — isotonic is the wrong tool. And it gives me a jagged piecewise-constant map with no smoothness to lean on.

So I'm stuck between a parametric family that's the right *cost* (one logistic-regression call, low variance, fine on small data) but the wrong *shape* (can't do inverse sigmoids or the identity), and a non-parametric method that's the right shape but the wrong cost. What I want is a parametric family — keep the cheap, low-variance fit — that is rich enough to include inverse sigmoids and the identity. The sigmoid's poverty came from one place: the generative assumption. So fix the assumption and re-run the derivation, and see what family falls out of a better story.

Where did the trouble enter? The Gaussian: infinite support, and a single shared variance that pins the slope and forbids the gathering shape. My score lives on `[0, 1]`. What's the natural density on `[0, 1]`? The beta distribution. Its density is `p(s; alpha, beta) = s^{alpha-1} (1-s)^{beta-1} / B(alpha, beta)` with shape parameters `alpha, beta > 0` and `B` the beta function for normalisation. It lives exactly on `[0, 1]`, it can be unimodal, U-shaped, J-shaped, flat — and critically it has *two* shape parameters per density, not one variance shared across classes. Let me just plug it into the same likelihood-ratio machine and see what map comes out. Positives ~ `Beta(alpha_1, beta_1)`, negatives ~ `Beta(alpha_0, beta_0)`:

```
LR(s) = [ s^{alpha_1 - 1} (1-s)^{beta_1 - 1} / B(alpha_1, beta_1) ]
        / [ s^{alpha_0 - 1} (1-s)^{beta_0 - 1} / B(alpha_0, beta_0) ].
```

The powers of `s` subtract: `s^{(alpha_1 - 1) - (alpha_0 - 1)} = s^{alpha_1 - alpha_0}`. The powers of `(1-s)` subtract: `(1-s)^{(beta_1 - 1) - (beta_0 - 1)} = (1-s)^{beta_1 - beta_0}`. The beta-function constants collect in the opposite direction because they are in denominators. So, writing `a = alpha_1 - alpha_0`, `b = beta_0 - beta_1` (note the order on `b` — I want the exponent on `(1-s)` to come out as `-b`, so I define `b` as the *negative* of the difference, `beta_0 - beta_1`), and `K = B(alpha_1, beta_1)/B(alpha_0, beta_0)`,

```
LR(s) = s^{a} * (1-s)^{-b} / K = s^a / ((1-s)^b K).
```

It's a clean power law in `s` and `(1-s)`. Now turn the likelihood ratio into a calibrated probability the same way as before, `mu(s) = 1/(1 + LR(s)^{-1})`. Let me absorb the divisor `K` into an additive log term for later convenience — set `K = e^{-c}`, so division by `K` is multiplication by `e^c`. Then

```
mu_beta(s; a, b, c) = 1 / ( 1 + 1/( e^c * s^a / (1-s)^b ) ).
```

Three parameters: `a` and `b` are shapes, `c` is location. Before I trust this I should check it actually does the things the sigmoid couldn't, because that's the whole reason I changed the assumption — and I'd rather find out now if it doesn't.

Monotonicity first, since I can't ship a calibration map that runs backwards. `mu` increases in `s` iff `LR(s) = e^c s^a/(1-s)^b` increases in `s`. Take `ln LR = c + a ln s - b ln(1-s)`; its derivative is `a/s + b/(1-s)`, which is `>= 0` for all `s` in `(0,1)` exactly when `a >= 0` and `b >= 0`. Good — so I'll require `a, b >= 0`, the analogue of the sigmoid's `gamma >= 0`.

Now a question that worried me about the sigmoid: does *every* map of this form actually come from two real beta distributions, or have I written down a parametrisation that's partly fictional? I need to exhibit, for any `a, b > 0` and any real `c`, some valid `alpha_0, alpha_1, beta_0, beta_1 > 0`. The shape constraints `a = alpha_1 - alpha_0` and `b = beta_0 - beta_1` leave the overall location free, so let me pin most of it and keep one degree of freedom to chase `c`. Try `alpha_0 = 1`, `alpha_1 = 1 + a`, `beta_0 = M + b`, `beta_1 = M` for a free `M > 0`; that satisfies both differences for every `M`, and the only thing left to match is `B(alpha_1, beta_1)/B(alpha_0, beta_0) = e^{-c}`, i.e. the ratio `R(M) = B(1 + a, M)/B(1, M + b)` has to hit `e^{-c}`. So the question is whether `R(M)` sweeps out all of `(0, infinity)` as `M` ranges over `(0, infinity)`.

My first instinct was that `R(M)` starts near 0 for small `M` and climbs without bound as `M` grows. Let me actually compute it rather than guess, with `a = 0.7, b = 1.3`:

```
M = 0.0001   R = 1.30e+04
M = 0.01     R = 1.30e+02
M = 1        R = 1.35
M = 10       R = 1.93e-01
M = 100      R = 3.64e-02
M = 1000     R = 7.22e-03
M = 1e5      R = 2.87e-04
```

That is the *opposite* of what I expected — `R(M)` is large for small `M` and decays toward 0 as `M` grows, monotonically decreasing across the whole sweep. My guess about the direction was just wrong. But the thing I actually need survives: as `M -> 0+` the ratio runs off to `+infinity`, and as `M -> infinity` it goes to `0`, and it moves through every value in between continuously. I checked a few more `(a, b)` pairs — `(2, 0.5)`, `(0.1, 0.1)`, `(3, 3)` — and in each the sweep is monotone decreasing and spans many orders of magnitude from above 1 down toward 0. So for any real `c`, the target `e^{-c} > 0` lies in the range of `R(M)`, and continuity hands me an `M` that hits it. The family is exactly the set of monotone LR maps for two beta-distributed classes — surjective onto every `(a, b, c)` with `a, b > 0`. The direction of the sweep didn't matter; only that it covers `(0, infinity)`, which it does.

Now the shapes. The midpoint, where `mu = 1/2`, is where `LR(m) = 1`: `e^c m^a/(1-m)^b = 1`, so `c = b ln(1-m) - a ln m`. I can reparametrise the location by an interpretable midpoint `m` if I want, just like the sigmoid's `m`. But — and this is the first thing the sigmoid couldn't do — the curves are *not* translation-invariant in `s`. The sigmoid was: changing `m` just slid the same S-curve left or right. Here moving `m` reshapes the curve, because `s^a` and `(1-s)^b` are not symmetric about any point unless `a = b`. That asymmetry is exactly what I'd want for classifiers that score one class to the extreme while the other clusters in the middle.

The decisive test is whether I can get the gathering shape Naive Bayes needs. Look at `a = b < 1`: then `LR(s) = e^c (s/(1-s))^a`, and with `a < 1` the exponent damps the log-odds — so extreme scores should get pulled toward the middle rather than pushed out. Let me tabulate `mu_beta` at `a = b = 0.5, c = 0` against the raw score and against the spreading case `a = b = 3`:

```
raw s          0.01  0.10  0.30  0.50  0.70  0.90  0.99
a=b=0.5  mu =   0.09  0.25  0.40  0.50  0.60  0.75  0.91   (gathers toward 0.5)
a=b=3    mu =   0.00  0.00  0.07  0.50  0.93  1.00  1.00   (spreads toward 0/1)
```

The `a = b = 0.5` row pulls `0.99` back to `0.91` and `0.01` up to `0.09` — an inverse sigmoid, the exact gathering behaviour I needed and could not get from `gamma >= 0`. The `a = b = 3` row is the ordinary sigmoid spread. And an asymmetric `a = 2, b = 0.5` gives `[0.00, 0.01, 0.10, 0.26, 0.47, 0.72, 0.91]` — flat near 0, gentle near 1, a shape with no symmetry axis at all. So both directions and asymmetry are reachable.

The identity — the thing the sigmoid family flatly lacked, with that ~0.078 worst-case gap? Set `a = b = 1` and `c = 0`: `LR(s) = s/(1-s)`, the raw odds, so `mu(s) = 1/(1 + (1-s)/s) = 1/((s + 1 - s)/s) = s`. Algebraically the identity is a member. I evaluated `mu_beta(s; 1, 1, 0) - s` on a grid across `(0,1)` and the largest deviation is `1.1e-16` — machine zero, not an approximation. So a beta calibrator handed an already-calibrated classifier can learn `a = b = 1, c = 0` and leave it exactly alone, where Platt scaling was off by eight points.

Let me make the inverse-sigmoid case concrete in a setting I can derive end to end, because I want the family not just to *contain* a plausible shape but to *be* the correct map in a case I can compute. Take a perfectly calibrated feature with value `x` on each instance, and feed Naive Bayes `k` identical copies of it. Naive Bayes assumes independence and multiplies, so it outputs `s = x^k / ( x^k + (1-x)^k )` — it treats one piece of evidence as `k` pieces and slams the score toward 0 or 1. What map undoes that and recovers the calibrated `x`? Note `s/(1-s) = x^k/(1-x)^k = (x/(1-x))^k`, so `(s/(1-s))^{1/k} = x/(1-x)`, and solving, `x = 1/(1 + ((1-s)/s)^{1/k})`. Compare to my beta map with `a = b = 1/k, c = 0`: `mu(s) = 1/(1 + 1/( (s/(1-s))^{1/k} )) = 1/(1 + ((1-s)/s)^{1/k})`. Algebraically identical. I checked it numerically too: for `k = 2, 3, 5`, generating `s` from a grid of `x` in `(0.1, 0.9)` and pushing it through `mu_beta(s; 1/k, 1/k, 0)`, the worst `|mu(s) - x|` came back at `1e-16` to `3e-14` — recovery to machine precision. So the exact correction for Naive Bayes double-counting is a member of the beta family with `a = b = 1/k < 1`, an inverse sigmoid. That isn't luck: double-counting is a power on the odds, and the beta family is precisely powers on the odds.

Now the part that decides whether this is *practical*. A three-parameter family with a custom log-loss objective would mean writing a bespoke optimiser, supplying gradients, the whole nuisance — and the sigmoid's big selling point was that it's one library call. So let me stare at the map and ask whether it's secretly already a logistic regression in disguise. The log-likelihood-ratio is `ln LR(s) = a ln s - b ln(1-s) + c`. A logistic-regression posterior on a feature vector `phi` with weights `w` and bias is `1/(1 + exp(-(w . phi + bias)))`, whose log-odds is `w . phi + bias` — *linear in the features*. My `ln LR` is linear in the two quantities `ln s` and `-ln(1-s)` with coefficients `a` and `b` and intercept `c`. That's the same object. So define features `s' = ln s` and `s'' = -ln(1-s)`, and the beta calibration posterior

```
mu_beta(s; a, b, c) = 1 / (1 + 1/exp( a*s' + b*s'' + c ))
```

should be exactly the bivariate logistic-regression model on `(s', s'')` with weights `(a, b)` and intercept `c`. Let me verify the substitution leaves nothing behind: `exp(a ln s + b(-ln(1-s)) + c) = e^c * s^a * (1-s)^{-b} = e^c * s^a/(1-s)^b = LR_beta(s; a, b, c)`. It matches term for term.

I don't fully trust an algebraic identity until I've run it, so I fit it both ways on synthetic over-confident scores: generate calibrated `x`, draw labels `y ~ Bernoulli(x)`, push `x` through the `k = 3` Naive-Bayes squashing to get over-confident `s`, then fit a bivariate logistic regression on `(ln s, -ln(1-s))` against `y`. The solver returns `a ~ 0.32`, `b ~ 0.32`, `c ~ 0.035` — and `a, b` land right next to `1/k = 0.333`, exactly the inverse-sigmoid correction the earlier derivation predicted this distortion needs. Evaluating `mu_beta` with those fitted `(a, b, c)` directly and comparing against the solver's own `predict_proba` on the same scores, the maximum discrepancy is `2.2e-16`. So fitting the full three-parameter beta map by maximum likelihood *is* running an ordinary bivariate logistic regression on the two transformed features — same model, same objective, same optimum. One off-the-shelf call, exactly the cost of Platt scaling, and the richer family for free.

And the symmetric special case is even cheaper. If I restrict `a = b`, then `ln LR = a(ln s - ln(1-s)) + c = a ln(s/(1-s)) + c`, linear in the single feature `ln(s/(1-s))`, the log-odds of the score. So beta calibration with `a = b` is *univariate* logistic regression on the log-odds — same arithmetic as plain logistic calibration, but with the log-odds as the feature instead of the raw score. Which also tells me something retrospective: the "linear in log-odds" recalibration people had been using as a heuristic, with no model behind it, is exactly beta calibration restricted to `a = b`. It had a generative justification all along; nobody had connected it to the beta distributions.

I should be honest about one subtlety before I write code. The logistic regression is unconstrained, so in principle it could return `a < 0` or `b < 0`, which would make the map non-monotone — a calibration map that runs backwards in part of the range. Real distortions are monotone, so the data should pull both coefficients positive and this should rarely fire, but I don't want to leave the invariant to chance. The cheap guard is: fit unconstrained, check the signs, and if one coefficient comes out negative, drop that feature (fix its coefficient to zero) and refit the remaining univariate logistic regression. The result is still the same beta family, just projected back to the monotone boundary.

One more decision I shouldn't gloss over: how hard to regularise the logistic regression. Beta calibration *is* the maximum-likelihood / log-loss fit, the same as Platt scaling — that's the definition that gives the propositions their bite. So the natural choice is essentially no regularisation: let the solver find the true MLE `(a, b, c)`. Heavy L2 shrinkage would pull the feature weights toward zero, i.e. toward `a = b = 0`, which sends the map toward an intercept-only constant — the opposite of fitting the distortion. A light penalty is defensible on a very small calibration split as overfit insurance, but the canonical method is the essentially unregularised MLE. I'll make the regularisation an explicit knob rather than a silent default.

So I have everything, and every piece has been checked rather than asserted. The family is `mu_beta(s; a, b, c) = 1/(1 + 1/(e^c s^a/(1-s)^b))`, born from assuming the per-class scores are beta-distributed; it contains sigmoids (`a = b > 1`), inverse sigmoids (`a = b < 1`, verified to gather the extremes), the identity (`a = b = 1, c = 0`, verified to machine zero), and asymmetric maps (`a != b`); it is monotone when `a, b >= 0`; it covers every parameter triple, because the `M`-sweep ranges over all of `(0, infinity)`; and — the part that makes it shippable — it is fitted by one bivariate logistic regression on the features `(ln s, -ln(1-s))`, which I confirmed reproduces the map to machine precision and recovers the `1/k` correction on a known distortion. Let me write it into the calibration slot, mirroring the standard implementation: clip the score into `(0, 1)` so the logs are finite, build the two features, fit logistic regression, and at predict time run the same featurisation through the fitted model.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibrationMethod:
    """Beta calibration: maps positive-class scores in (0, 1) to calibrated
    probabilities by assuming the per-class scores are beta-distributed.

    The likelihood ratio of two betas is e^c * s^a / (1-s)^b, so the calibrated
    posterior 1/(1 + 1/LR) is a logistic-regression model on the features
    (log s, -log(1-s)) with weights (a, b) and intercept c -- fittable by a
    single off-the-shelf logistic-regression call (Proposition 2)."""

    def __init__(self, C=1e10):
        self.eps = 1e-6
        # near-zero regularization => the log-loss / MLE fit that defines the method;
        # raise the penalty (lower C) only as overfit insurance on a tiny calibration split.
        self.model_ = LogisticRegression(max_iter=2000, solver="lbfgs", C=C)
        self.active_features_ = None

    def _featurize(self, probs):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)      # keep log s, log(1-s) finite
        # features (log s, -log(1-s)); LR weights become (a, b), intercept c
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, probs, labels, groups=None):
        X = self._featurize(probs)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self.model_.fit(X, labels)                        # MLE of (a, b, c) == beta fit
        coef = self.model_.coef_[0]
        if coef[0] < 0:
            self.active_features_ = [1]
            self.model_.fit(X[:, self.active_features_], labels)
        elif coef[1] < 0:
            self.active_features_ = [0]
            self.model_.fit(X[:, self.active_features_], labels)
        else:
            self.active_features_ = [0, 1]
        return self

    def predict_proba(self, probs, groups=None):
        X = self._featurize(probs)[:, self.active_features_]
        # 1/(1 + 1/exp(a*log s + b*(-log(1-s)) + c)) == mu_beta(s; a, b, c)
        return np.clip(self.model_.predict_proba(X)[:, 1], self.eps, 1.0 - self.eps)
```

The causal chain, start to finish. I needed a calibration map and couldn't fit the exact empirical one without overfitting, so I needed a parametric family with the right inductive bias. The standard sigmoid family is cheap to fit but I traced it back to its generative root — equal-variance Gaussians on the score — and that root forces three failures, each of which I checked: it puts mass outside `[0, 1]` where bounded scores can't go, it can only spread scores toward the extremes and never gather them, so it fits over-confident scores badly, and it excludes the identity (the best sigmoid fit to `mu(s)=s` is still ~0.078 off, eight-fold wrong at `s=0.01`), so it uncalibrates an already-calibrated model. Isotonic calibration has the right shape freedom but overfits on the small calibration splits I actually have. So I changed the one assumption that caused the damage: replace the Gaussian with the beta distribution, the natural density on `[0, 1]`. Running the same likelihood-ratio derivation, the beta gives `LR(s) = e^c s^a/(1-s)^b`, a power law on the odds, whose calibrated posterior is a three-parameter family that — as the tabulations and recovery checks confirmed — contains inverse sigmoids (so Naive Bayes is fixable, and the exact `k`-fold double-counting correction is the member `a = b = 1/k`, recovered to machine precision), asymmetric maps, and the identity (`a = b = 1, c = 0`, machine zero), while staying monotone under `a, b >= 0` and covering every parameter triple. Then the implementation payoff: because `ln LR` is linear in `ln s` and `-ln(1-s)`, the whole family is a bivariate logistic regression on those two features, which I verified reproduces the map to `2e-16`, so I fit it with one library call — the same cost as the sigmoid I started from, with none of its rigidity. The symmetric restriction `a = b` collapses to univariate logistic regression on the log-odds, which retroactively explains the old linear-in-log-odds heuristic as this family with `a = b`.
