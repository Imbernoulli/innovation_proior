The classifier is already trained and I am not allowed to touch it. It takes an input and returns a class, and because its last layer is a softmax it also hands me a vector of nonnegative numbers over the classes that sum to one. In production it will be wrong some of the time, and in the settings I actually care about — an autopilot reading a scene, a model triaging a scan, a credit decision — a confident wrong answer can cost far more than no answer at all. So I want to give this fixed model the ability to say "don't know" on the inputs it is least equipped to handle, hand those off to a human or a slower backup, and keep answering on as many of the rest as I can. The classifier stays as it is; the thing I get to build is the rule that decides, per input, whether to answer or to step back.

Let me name the two things I am trading before I do anything else, because the whole problem is a trade-off and I want the axes right. The fraction of inputs I actually answer on — call it coverage. And the error rate measured *only* over the inputs I answered — call it the selective risk. These pull against each other in the obvious direction: if I abstain on more inputs, coverage drops, but the selective risk should also drop, because the inputs I am dropping ought to be the ones I was least sure of. If I abstain on the *right* inputs. That "if" is the entire game. Abstaining randomly would lower coverage and do nothing for risk. Abstaining on exactly the inputs most likely to be wrong would buy the steepest possible drop in risk per unit of coverage given up. So I am really after two separable things: a per-input quantity that orders inputs from most-trustworthy to least, and then a way to cut that ordering at a point that hits whatever coverage I am told to hit.

Formalize the abstaining classifier first so I have language. I have my fixed classifier `f`, and I am going to attach a selection function `g` that maps an input to {0,1}: 1 means answer, 0 means abstain. The pair `(f,g)` answers `f(x)` when `g(x)=1` and says don't-know when `g(x)=0`. Coverage is then just the expected value of the gate, `Φ(f,g) = E[g(X)]` — the probability mass I let through. And the selective risk is the loss averaged over the let-through region only:

  R(f,g) = E[ ℓ(f(X),Y) · g(X) ] / E[g(X)].

The `g(X)` in the numerator zeroes out every abstained input so they contribute no loss, and dividing by the coverage renormalizes so I am averaging over the answered set and not the whole domain. Sanity check: if `g ≡ 1` I answer on everything, numerator is `E[ℓ]`, denominator is 1, and this collapses to the ordinary risk — good, abstaining is a strict generalization of just predicting. Now the object I actually want is risk *as a function of* coverage: as I let `g` admit fewer inputs, how low does the risk on the survivors go? That curve — the risk-coverage curve — is the complete description of an abstaining classifier, and any acceptance rule I build is good or bad according to how low it pushes that curve.

So what should `g` be? My instinct is that it should not be some elaborate independent object. It should be downstream of a single per-input number — a confidence, a reliability score, whatever I want to call it — with the property that higher means "trust this prediction more." Suppose for a moment I had such a score `κ(x)`. Then the obvious selection function is a threshold: answer iff `κ(x) ≥ θ`, abstain otherwise. And sweeping `θ` from low to high traces the entire risk-coverage curve — high `θ` admits only the most confident handful (low coverage, hopefully low risk), low `θ` admits almost everything (high coverage, risk approaching the base error). One scalar `θ` is the dial.

I want to check whether a single global threshold is actually the *right* shape for `g` or just a convenient one, because I could imagine wanting something cleverer — different cuts in different regions of input space, say. Picture it abstractly: for a fixed classifier, the best possible selection rule at a given coverage is the one that, among all subsets of that size, admits the subset with the lowest total loss. Suppose I have a score that ranks inputs by their true expected loss, most-likely-correct first. Then "admit the lowest-loss subset of size matching my coverage" should coincide with "admit the top-ranked prefix of that size," and a prefix of a sorted list is exactly what a threshold on the score carves out. Let me not just assert this — let me try to break it on a tiny case and see if it survives. Ten inputs sorted by ascending confidence `κ = [.55,.60,.62,.70,.75,.80,.85,.90,.95,.99]`, with 0/1 losses `[1,1,0,1,0,0,0,0,0,0]` so that the wrong predictions sit at the low-confidence end. I want to accept 8. The confidence prefix — the 8 highest-`κ` inputs — drops the two lowest, indices 0 and 1, both of which have loss 1, leaving total accepted loss `1`. Now I deliberately pick a *non-*prefix set of the same size 8: keep the eight lowest-confidence inputs instead, i.e. drop the two most confident. That set carries total loss `3`. So the prefix beats the deliberately-chosen alternative, `1 < 3`, and the reason generalizes: once the ranking is fixed, the only freedom left is how deep down the list to go, and any non-prefix set of size `k` swaps a kept high-rank input for a dropped lower-rank one, which can only raise total loss when the ranking tracks loss. The thing that decides "how deep" is a single number — the cut. So *if* my score genuinely orders inputs by reliability, a single global threshold is the optimal selection function for that score, not a shortcut, and the whole problem reduces to two pieces: find a good `κ`, and find the threshold that hits my coverage. The "if" is doing real work — everything now hangs on whether I can produce a score whose ordering tracks reliability.

Now, the score. Chow worked this out sixty years ago for the case where you actually know the class posteriors `P(y|x)`. Under 0/1 loss, his Bayes-optimal reject rule is ambiguity rejection: reject an input exactly when no class is dominant enough, i.e. when `max_y P(y|x)` is below a threshold; accept when `max_y P(y|x) ≥ 1 - t`. And he showed the error and reject rates move monotonically against each other as `t` sweeps — there is one trade-off curve, which is the same picture I just drew. The lesson I take is structural: the optimal thing to threshold is the *maximum posterior*. The most-likely class's probability is exactly how reliable the prediction is. If `max_y P(y|x)` is near 1 the input is unambiguous; if it is near `1/k` the model is basically guessing.

The trouble is I do not have `P(y|x)`. I have a trained net. What I have that is *shaped* like a posterior is the softmax output vector — nonnegative, sums to one, one entry per class. So the natural move is to take Chow's rule and substitute the only posterior-surrogate the model gives me:

  κ(x) = max_j f(x | j),

the maximum softmax response. Threshold that.

I should immediately push on whether this is legitimate, because there is a well-known objection sitting right here and I do not want to wave it away. The objection is that softmax outputs are *not* calibrated probabilities. Modern nets are overconfident: the softmax max can sit at 0.9 on inputs the model gets wrong, and you can feed in pure Gaussian noise and get a 0.9 "confidence" out of an MNIST classifier. If I were claiming `max_j f(x|j)` literally equals `P(correct)`, this would sink me. But look back at what I actually need. The selection function only ever compares inputs to each other and to a cut point. I never need the absolute value of `κ` to mean a probability; I only need its *ordering* to be right — that more-reliable predictions get higher `κ` than less-reliable ones. The ideal `κ` is purely a ranking object: I would want `κ(x₁) ≤ κ(x₂)` exactly when the loss on `x₁` is at least the loss on `x₂`, so that thresholding `κ` peels off the highest-loss inputs first. Calibration is a much stronger property than ranking, and it is the wrong property to demand. A monotone but miscalibrated score ranks perfectly and selects perfectly. So the calibration critique, which kills the "softmax = probability" reading, does not touch the "softmax = reliability ranking" reading, and the second is all I am using.

Is the ranking actually any good, though? Two reasons to believe it, one empirical and one mechanistic. Empirically, across trained nets the maximum softmax probability separates correct from incorrect predictions well above chance: correctly classified inputs tend to get a higher max-softmax than misclassified or out-of-distribution ones. That is exactly the ranking property I need, and it is the documented, reproducible behavior of these models — the same statistic that fails as a calibrated probability succeeds as a discriminator of right-versus-wrong. Mechanistically, I can look one layer down. If I average the activations of the second-to-last layer separately over an output class's true positives and its false positives, the true positives drive the active neurons much higher than the false positives do, and that gap is spread across many neurons rather than riding on one — the confidence signal is the accumulation of many independently-fired pattern detectors agreeing. A prediction that lights up a broad coalition of features is more trustworthy than one that squeaks through on a couple, and the softmax max at the top is the funnel that gap pours into. So the max softmax is high precisely when many internal detectors concur, which is a sensible proxy for reliability even when its numeric value is inflated.

There is a competing source of a reliability signal I should weigh, so I am not picking max-softmax out of mere convenience. Monte-Carlo dropout: leave dropout switched on at test time and run the same input through several stochastic forward passes, then take the variance of the predicted class's response as an uncertainty, and use minus that variance as the confidence rate. It is a genuinely different signal — it probes the model's *instability* under perturbation rather than the *height* of its single deterministic output. But it costs many forward passes per input, and as a Monte-Carlo estimate its quality only improves with the number of samples, so it is heavy at inference time. Against a single forward pass that gives me `max_j f(x|j)` for free, I would need it to rank substantially better to justify the cost, and there is no a priori reason a variance-of-perturbations should out-rank the max posterior — they are both reasonable, and the max posterior is the one Chow's analysis singles out as optimal in the known-posterior limit. So I will build around the max softmax and treat MC-dropout as the more expensive alternative I can drop into the same `κ` slot if I ever want to.

I want to be honest that I am departing from the one rejection rule in this area that comes with a proof. In the noise-free, realizable setting — where some hypothesis in the class is exactly right — the provably optimal selective strategy is not to threshold a soft score at all; it is to take any classifier consistent with the training sample and reject every point on which the consistent hypotheses disagree, accept where they all agree. That has zero selective risk by construction and the largest coverage any zero-risk strategy can have. But it needs an enumerable, tractable version space to test agreement against, and it needs the noise-free assumption. A deep net trained on noisy data has neither: there is no version space I can sweep, and the data is not realizable. So that elegant guarantee simply does not reach my setting, and the practical move — threshold a confidence score — is the one left standing, exactly as the "commonly used heuristic" it was always described as. My job is to take that heuristic and pin down how to set the threshold so it does something I can state precisely.

Which brings me to the threshold, and here the problem splits into two modes depending on what the operator hands me as the target.

The first mode is target *coverage*. Say I must answer on 80% of inputs. Then I want the threshold `θ` such that the fraction of inputs with `κ(x) ≥ θ` is about 0.8. That is just a quantile. If 80% of inputs must be accepted, then 20% must be rejected, and the rejected ones are the lowest-`κ` 20%, so `θ` is the 20th percentile of the score distribution — the `(1 - coverage)` quantile. Set `θ = quantile(κ-values, 1 - target_coverage)` and accept everything at or above it. Let me trace this on the ten scores from before to make sure the arithmetic does what I think. With `c = 0.8`, the cut is `np.quantile(κ, 0.2)`, which on `[.55,.60,.62,.70,.75,.80,.85,.90,.95,.99]` lands at `0.616` (linear interpolation between the 0.60 and 0.62 order statistics). Accepting `κ ≥ 0.616` keeps `[.62,.70,.75,.80,.85,.90,.95,.99]` — eight of ten, coverage exactly `0.8`, the target. And the two it drops, `.55` and `.60`, are the two lowest-confidence inputs, which is the point. Carrying through the loss vector from before, the selective risk on the accepted eight is `1/8 = 0.125`, against a base error of `3/10 = 0.3` if I had answered on everything — so abstaining on 20% of the inputs roughly halved the error on the rest, which is the whole bargain made concrete.

Where do I compute that quantile? Not on the data I will report risk and coverage on, obviously — if I pick the cut on the test set I am peeking, and my coverage will look exactly on-target only because I forced it to on those very points. And not on the training data the classifier was fit to, because the model has seen those and is over-confident on them, so its score distribution there is not the one it will face. I need a separate held-out calibration set: data the classifier did not train on, which I use only to locate the threshold. I compute the scores on the calibration set, take the `(1 - coverage)` empirical quantile, freeze that as `θ`, and apply it unchanged at test time. The trace above shows that, up to finite-sample interpolation and score ties, the empirical accept-rate on calibration is the target coverage; and because the calibration set is an i.i.d. draw, the same `θ` gives approximately the target coverage on test. That is the whole policy in coverage mode: rank by max-softmax, cut at the coverage quantile fit on held-out data.

Before I write it down let me make sure the direction conventions are airtight, because an off-by-one in the inequality or a flipped quantile silently inverts the whole thing. `κ` is a confidence: higher means more reliable, so I *accept* high `κ` and *defer* low `κ`. The accept predicate is `κ(x) ≥ θ`. To accept a fraction `c` of inputs I must place `θ` so that a `c`-fraction lies at or above it, which means `θ` sits at the `(1-c)` quantile from the bottom: with `c = 0.8`, `θ` is the 0.2 quantile, the value below which 20% of scores fall. `np.quantile(scores, 1 - c)`. I just saw this give the right answer above — `0.616`, coverage `0.8`. To be sure the flip is the disaster I think it is rather than a wash, I run the wrong version on the same scores: `np.quantile(κ, 0.8)` is `0.91`, and accepting `κ ≥ 0.91` keeps only `[.95,.99]` — coverage `0.2`, the exact complement of what was asked. So the flip really does invert coverage, not perturb it; the sign of the `1 -` is load-bearing. So: `quantile = 1 - target_coverage`, clipped into `[0,1]` for safety in case someone passes a degenerate coverage, and `θ = np.quantile(scores, quantile)`. Accept `score >= θ`. Defer otherwise.

The second mode is the one the same machinery handles when the operator instead hands me a target *risk* — "guarantee no more than 2% error, with high probability, and give me as much coverage as you can." This is the more demanding ask and it is where the guarantee lives, so I want to derive it as the companion to coverage mode, because it is the same selection rule viewed from the other axis and it tells me exactly how trustworthy a thresholded score can be made. Now I cannot just set a quantile, because I am told the risk, not the coverage. Sort the calibration inputs by `κ`, and a candidate threshold is the score of some sorted position; testing that position means "accept everyone from here up," compute the empirical selective risk on that accepted suffix, and ask whether an upper bound on the true selective risk is below `r*`. I have to be careful about what I can prove. The search moves in the natural direction — if the bound is too high, raise the cut and accept fewer, more confident points; if the bound is safe, lower the cut and see whether more coverage is still certifiable — but the theorem I can make airtight is not a global optimality theorem for every possible noisy score. It is the simultaneous validity of the risk bounds for the threshold candidates the binary search actually tests. Once the returned candidate has a bound at or below `r*`, the risk certificate follows.

The subtlety, and the reason I need a real bound and not just the empirical risk, is that the empirical selective risk on the calibration set understates the true risk — it is an in-sample number, and if I just demanded "empirical risk ≤ r*" I would routinely ship a classifier whose true risk exceeds `r*`. I need, for each candidate threshold, a high-probability *upper bound* on the true selective risk given the empirical one. So I want an honest statement of the form "I observed `m·r̂` errors among `m` accepted calibration points; with confidence `δ`, how large can the true error rate `b` be?" The accepted points, conditioned on the threshold, are an i.i.d. sample from the accepted-region distribution, and the number of errors among `m` of them is Binomial(`m`, `b`) where `b` is the true selective risk. The largest `b` consistent with seeing at most `m·r̂` errors at confidence `δ` is the `b` for which the probability of seeing that few or fewer errors equals `δ`:

  Σ_{j=0}^{m·r̂} C(m,j) b^j (1-b)^{m-j} = δ.

Solve for `b`. This is the inverse of the Binomial CDF — the Clopper-Pearson upper confidence limit on a proportion. The left side is the Binomial CDF evaluated at `m·r̂` with success probability `b`, and it is *decreasing* in `b` (a higher true error makes "few errors observed" less likely), so I can find the root by binary search on `b` between `r̂` (the bound can't be below the observed rate) and 1. That gives me, per candidate threshold, the bound `B*(r̂, δ, m)`.

Let me put numbers on this so the bound isn't an abstraction. Take the accepted set from the coverage trace: `m = 8` accepted, `1` error observed, so `r̂ = 0.125`, and pick `δ = 0.05`. Solving `Binom.cdf(1; 8, b) = 0.05` for `b` gives `B* ≈ 0.471`. Two things to notice. First, the bound is far above the empirical `0.125` — that gap is exactly the price of honesty, and it is large here only because `m = 8` is tiny; the bound tightens toward `r̂` as `m` grows. Second, the bound is genuinely the level at which "seeing ≤ 1 error in 8 draws" becomes a 5%-tail event: plugging `b = 0.471` back in gives `Binom.cdf(1; 8, 0.471) = 0.050`, so the equation is satisfied, not approximated. This matters because if I had instead just required the *empirical* risk `≤ r*` I would happily certify `r* = 0.125` on this data while the true error could be anywhere up to `0.47` at this confidence — the in-sample number understates, and the bound is the correction. Is Clopper-Pearson actually worth the inverse-CDF machinery over a one-line concentration bound? Hoeffding's one-sided bound here is `r̂ + sqrt(ln(1/δ)/(2m)) = 0.125 + sqrt(ln 20 / 16) ≈ 0.558`. So Hoeffding would force me to certify `0.558` where the exact inversion certifies `0.471` — looser by almost nine points of risk, which at a fixed `r*` directly costs coverage. Clopper-Pearson is the inversion of the *exact* Binomial tail rather than a sub-Gaussian over-approximation of it, so it is the tightest distribution-free statement available here, and the gap I just measured is why I pay for it.

But I am going to evaluate this bound at every distinct threshold the binary search visits, and a binary search over `m` sorted positions needs at most `k = ⌈log₂ m⌉` distinct cuts before the search interval collapses; an implementation can do one terminal repeat without creating a new event. If I tested each distinct cut at confidence `δ` and then picked the best, I would be data-snooping across correlated tests, and the joint guarantee would be weaker than any single one. The fix is a union bound: spend `δ/k` on each candidate cut, so that the probability that *any* of the `k` tested thresholds has its true risk exceed its own bound is at most `k · (δ/k) = δ`. Then whichever threshold I select, its bound holds simultaneously with the rest, at overall confidence `δ`.

Let me actually prove that the union-bound construction delivers what I claimed, because the bound `B*` was derived for a *fixed* sample size, and here the number of accepted points `m_i` at the `i`-th threshold is itself random — it depends on how the threshold falls relative to the data — and that is exactly the kind of subtlety that quietly breaks a guarantee if I do not handle it. Fix one threshold `g_i` and let `m_i = |g_i(S_m)|` be the random count of accepted calibration points. Condition on `m_i = n`. Given that count, the `n` accepted points are an i.i.d. sample of size `n` from the projected distribution `P_{g_i}(X,Y) = P(X,Y | g_i(X)=1)` — the original distribution restricted to the accept region — so the fixed-sample bound applies verbatim:

  Pr_{S_n ∼ (P_{g_i})^n} { R(f | P_{g_i}) > B*(r̂_i, δ/k, n) } < δ/k.

Now I need to connect `R(f | P_{g_i})` — the true error over the accept-region distribution — to the selective risk `R(f, g_i)` I actually care about. They are equal:

  R(f | P_{g_i}) = E_{P_{g_i}}[ ℓ(f(x),y) ] = E_P[ ℓ(f(x),y) g_i(x) ] / Φ(f,g_i) = R(f, g_i),

because conditioning on the accept region is the same as masking by `g_i` and renormalizing by the coverage — that is precisely the definition of selective risk. So the conditional statement reads `Pr{ R(f,g_i) > B* | m_i = n } < δ/k` for every `n`. Marginalize over the random count by the law of total probability:

  Pr_{S_m}{ R(f,g_i) > B*(r̂_i, δ/k, m_i) }
    = Σ_{n=0}^{m} Pr{ R(f,g_i) > B* | g_i(S_m)=n } · Pr{ g_i(S_m)=n }
    ≤ (δ/k) Σ_{n=0}^{m} Pr{ g_i(S_m)=n } = δ/k,

since each conditional term is below `δ/k` and the `Pr{m_i = n}` sum to one. The randomness of the accepted count washes out cleanly — that was the worry, and it is handled by conditioning first and summing after. Finally union-bound over the `k = ⌈log₂ m⌉` thresholds the search examines:

  Pr_{S_m}{ ∃ i : R(f,g_i) > B*(r̂_i, δ/k, m_i) } ≤ Σ_{i=1}^{k} (δ/k) = δ.

So with probability at least `1-δ` every distinct candidate threshold's true selective risk is at most its computed bound, and in particular the returned candidate is covered by that same event. If the returned bound is `≤ r*`, then the classifier it induces has true selective risk `≤ r*` with probability `≥ 1-δ`. If `κ` is badly skewed, the bound can stay above the target at useful coverages; the certificate is still honest, but it certifies only the returned `b*`, not magic coverage at any requested risk. The guarantee is about honesty; the ranking is about how much you keep.

Two modes, then, off the same two ingredients: rank by `κ = max softmax`, and either cut at the coverage quantile (target coverage given) or binary-search the cut against the Clopper-Pearson risk bound with a `δ/k` union correction (target risk given). The coverage mode is the bounded-abstention operating rule; the risk mode is the same selection rule plus a returned high-probability bound, and when that bound clears the requested target the thresholded-softmax policy has the corresponding risk certificate.

How do I want to judge the score itself, separately from any one operating point? The threshold is just where I stand on the curve; the *curve* is set by how well `κ` ranks, and I want a single threshold-free number for ranking quality. That is the AUROC of `κ` as a predictor of correctness: label each answered input positive if the classifier was right and negative if wrong, and measure the area under the ROC of `κ`. It equals `P( κ on a correct input > κ on a wrong input )` — the probability the score ranks a true positive above a false one — which is exactly the ranking property the whole construction needs, summarized in one number, chance at 0.5. A high AUROC means thresholding `κ` will peel off the wrong predictions before the right ones at every coverage; the risk-coverage curve and its area are the same ranking quality read off against coverage instead of against a probability. So I will report AUROC of the acceptance score alongside the selective risk at the operating coverage.

Now I land the coverage-mode policy as the code I would actually ship, filling the empty body of the selection harness. The score is the max over the class axis of the softmax matrix. Fit takes the calibration probabilities, scores them, and stores the `(1 - coverage)` quantile as the threshold. The accept decision is score-at-or-above-threshold. Everything is a sort/quantile and a comparison — offline, one pass, no retraining of the fixed classifier:

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Global confidence threshold tuned on the calibration set."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "confidence_thresholding"

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        # accept a `coverage` fraction => reject the lowest-confidence `1 - coverage`
        # => threshold at the (1 - coverage) quantile of the calibration scores.
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray,
                         X: np.ndarray | None = None) -> np.ndarray:
        # kappa(x) = max_j softmax(x)_j; it is used as a ranking signal.
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        # accept (predict) iff confidence is at or above the calibrated cut; else defer.
        return self.acceptance_score(probs, groups, X) >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {"threshold": float(self.threshold_)}
```

And the risk-mode sibling — same `κ`, but the threshold comes from the binary search against the Clopper-Pearson bound with the `δ/k` union correction, so the routine returns a high-probability risk bound for the selected cut:

```python
import math
import random
import numpy as np
import scipy.stats


class risk_control:
    def calculate_bound(self, delta, m, erm):
        # Invert Binom.cdf(int(m * erm), m, b) = delta.  The CDF decreases in b,
        # so a positive residual means the upper endpoint can move upward.
        precision = 1e-7

        def func(b):
            return (-1 * delta) + scipy.stats.binom.cdf(int(m * erm), m, b)

        a = erm
        c = 1.0
        b = (a + c) / 2
        funcval = func(b)
        while abs(funcval) > precision:
            if a == 1.0 and c == 1.0:
                b = 1.0
                break
            elif funcval > 0:
                a = b
            else:
                c = b
            b = (a + c) / 2
            funcval = func(b)
        return b

    def bound(self, rstar, delta, kappa, residuals, split=True):
        # kappa: higher means more confident; residuals: 0 correct, 1 error.
        valsize = 0.5
        probs = kappa
        FY = residuals

        if split:
            idx = list(range(len(FY)))
            random.shuffle(idx)
            slice_ = round(len(FY) * (1 - valsize))
            FY_val = FY[idx[slice_:]]
            probs_val = probs[idx[slice_:]]
            FY = FY[idx[:slice_]]
            probs = probs[idx[:slice_]]

        m = len(FY)
        probs_idx_sorted = np.argsort(probs)
        a = 0
        b = m - 1
        k = math.ceil(math.log2(m))
        deltahat = delta / k

        for _ in range(k + 1):
            # The loop has k + 1 iterations but only k distinct candidate cuts.
            mid = math.ceil((a + b) / 2)
            accepted = probs_idx_sorted[mid:]
            mi = len(FY[accepted])
            theta = probs[probs_idx_sorted[mid]]
            risk = sum(FY[accepted]) / mi
            if split:
                testrisk = sum(FY_val[probs_val >= theta]) / len(FY_val[probs_val >= theta])
                testcov = len(FY_val[probs_val >= theta]) / len(FY_val)
            bound = self.calculate_bound(deltahat, mi, risk)
            coverage = mi / m
            if bound > rstar:
                a = mid
            else:
                b = mid

        return [theta, bound]
```

So the causal chain. I started with a fixed classifier that is sometimes confidently wrong and a need to abstain on its weakest inputs while answering on as many as possible — a trade-off between coverage and selective risk on the answered set. Writing the abstaining classifier as `(f,g)` with coverage `E[g]` and selective risk `E[ℓg]/E[g]`, the whole performance profile became the risk-coverage curve, and I reduced the design to two pieces: a per-input reliability score `κ`, and a threshold on it. A prefix-of-the-sorted-list argument showed that if `κ` ranks inputs by reliability, a single global threshold is the optimal selection function for it, not a shortcut. Chow's known-posterior analysis pointed at the maximum posterior as the thing to threshold; with a trained net the available surrogate is the maximum softmax response, and although softmax is miscalibrated as a probability, I only ever need it to *rank*, which the calibration critique does not touch and which both the documented correct-vs-wrong separation and the broad-coalition-of-neurons picture support. The provable version-space strategy assumes a noise-free, enumerable hypothesis class and so does not reach deep nets, leaving thresholded confidence as the rule to make precise. Given a target coverage, the threshold is just the `(1 - coverage)` quantile of `κ` on a held-out calibration set, with the inequality oriented to accept high confidence — that is the coverage-mode policy I ship. Given a target risk instead, the same selection rule is set by binary-searching the cut against the tightest distribution-free bound — the Clopper-Pearson inverse-Binomial upper limit — with a `δ/k` union correction over the `⌈log₂ m⌉` distinct candidate thresholds, and the conditioning-then-marginalizing argument proves that the returned candidate's true selective risk is at most its returned bound with probability `≥ 1-δ`; when that bound is `≤ r*`, the requested target-risk guarantee follows, while `κ`'s ranking quality (summarized threshold-free by the AUROC of the score against correctness) sets how much coverage that guarantee keeps.
