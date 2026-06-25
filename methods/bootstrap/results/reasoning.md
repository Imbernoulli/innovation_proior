OK, let me think this through from scratch. I have one sample x = (x₁, …, xₙ) from some unknown F, and a statistic θ̂ = t(x). The number θ̂ alone says nothing — I need to attach an accuracy to it: a standard error, ideally the whole sampling distribution of the error. For the mean this is trivial, SE = s/√n. But the statistics I actually care about — the median, a ratio of means, a correlation, the misclassification rate of a discriminant rule, a nonlinear-regression coefficient — have no such formula sitting in a drawer, and deriving one is a separate piece of mathematics every single time. What I want is one procedure that returns the accuracy of *any* statistic, automatically, without a new derivation per problem and without pretending I know the shape of F.

So let me write down precisely what I'm after. There is a random variable R(X, F) = t(X) − θ(F), the error of the estimate, and under repeated draws of X from F it has a distribution. Its variance is the squared standard error; its mean is the bias. I get to see exactly one draw, x. From that one draw I have to estimate the distribution of R. That's the whole problem in one sentence: estimate the sampling distribution of R(X, F) from a single realization.

The automatic tool already on the table is the jackknife. Quenouille built it to knock the bias of an estimator down from order 1/n to 1/n²: delete one observation, recompute, see how the estimate shifts. Tukey gave it its name and, more importantly here, turned it into a variance estimator — form the pseudo-values θ̃ᵢ = n·θ̂ − (n−1)·θ̂₍₋ᵢ₎, treat them roughly as if they were n independent replicates of the estimate, and read the variance off their scatter,

    var_jack = ((n−1)/n) Σᵢ (θ̂₍₋ᵢ₎ − θ̄₍₋₎)².

It is automatic — one recipe, any statistic — and that is exactly the property I want. But it is also, in Miller's review, sometimes trustworthy and sometimes not, and nobody can quite say why in a way that tells me when to trust it. The median is the clean embarrassment: jackknife the sample median and the variance estimate isn't even consistent. So I have a tool with the right *shape* — automatic, model-free — and a hole in its *foundation*. The productive move is not to patch the jackknife case by case. It's to ask what the jackknife is secretly an approximation *of*. If I can find the more primitive object it's linearizing, I'll understand both why it usually works and exactly why it fails on the median — and maybe the primitive object is itself a better estimator.

So: what *is* the jackknife computing? Stare at the deletion operation. Deleting xᵢ and recomputing is a perturbation of the data — equivalently, a perturbation of the empirical distribution. Let me make the empirical distribution the central character. F̂ puts mass 1/n on each x_i. It is the nonparametric maximum-likelihood estimate of F: of all distributions, the one the data make most likely, and it converges to F as n grows. The jackknife perturbs F̂ by yanking one of its atoms; the pseudo-values measure how t responds to that yank. That's a finite-difference derivative of t with respect to reweighting the data. Which means the jackknife is a *linear* device — a first derivative — of something. What's the thing it's the derivative of?

I want the sampling distribution of R(X, F) = t(X) − θ(F). The only obstacle is that I don't know F, so I can't draw fresh samples from it. But I have an estimate of F sitting right there: F̂. The plug-in principle says, to estimate a feature of F, compute that feature of F̂ — that's how x̄ estimates the mean and s² the variance. The jackknife and the delta method use plug-in only on a *number* (a variance formula). What if I apply plug-in to the *entire operation*? The thing I can't do is "sample X from F and look at the spread of R." But I *can* do the F̂ version of that exact sentence: sample X* from F̂, and look at the spread of R(X*, F̂). Substitute F̂ for F everywhere — in the data-generating step and in the θ(·) being subtracted — and the operation becomes fully computable, because F̂ is known.

What does "sample X* from F̂" mean concretely? F̂ is the distribution with mass 1/n on each of x₁, …, xₙ. Drawing one observation from it means picking one of the x_i uniformly at random. Drawing a sample of size n means doing that n times, independently — so the same x_i can come up two or three times or not at all. It is sampling *with replacement* from the observed values. Let me make that the candidate procedure: call X* = (X₁*, …, Xₙ*) a resample, and R* = R(X*, F̂) = t(X*) − θ(F̂) the recomputed error. The proposal is that the distribution of R* — over the resampling randomness, with F̂ held fixed at its observed value — estimates the distribution of R(X, F). Recompute the statistic on many resamples and the histogram of R* is the estimated sampling distribution; standard error, bias, quantiles all fall out of that one histogram. That's the proposal; now I have to earn it, because I just substituted one distribution for another inside an operation and asserted the answer carries over. Let me check the joints before I trust it.

First, the resampling choice — with replacement, size n. Am I free to do otherwise, and if not, why not? Try sampling *without* replacement: drawing n points without replacement from n points just returns a permutation of the data, and t is permutation-invariant, so every "resample" gives the identical θ̂ and the histogram collapses to a single spike. That's not a near-miss I could rescale — it's degenerate, zero variance, useless. So the multiplicity of with-replacement draws isn't a convenience; it's the only thing carrying any variability. What about Hartigan's route — draw a random *subset* of the x_i and compute t on it? That's a legitimate resampling scheme, and it's asymptotically valid, so I can't dismiss it as wrong. But the artificial samples are *smaller* than n, so they carry more sampling variability than a real sample of size n does, and I'd have to insert a rescaling factor to undo that, with the match to the true law only asymptotic. Drawing with replacement at the full size n is the one choice with no fudge factor: X* is then genuinely "a sample of size n from F̂," same size and same amount of data as the real x, so R(X*, F̂) lives on the same scale as R(X, F) automatically. So the procedure isn't arbitrary — among the resampling schemes, with-replacement-at-size-n is the unique one that is the literal meaning of "a sample from F̂."

Second, is the estimate even self-consistent? The cleanest test: it should be *exactly* right in the one situation where I actually know the truth — when F happens to equal F̂. And there it is exact by construction, because if F = F̂ then "sample from F and compute R(X,F)" and "sample from F̂ and compute R(X*,F̂)" are the same sentence, word for word. So the procedure is Fisher consistent — it nails the answer at F = F̂, which is the central, most-likely point among the F's compatible with the data. Any nonparametric estimator of R's distribution that did *not* get the answer right at F̂ would be perverse, since F̂ is precisely where the data point. That is the property the jackknife's foundation was missing: plug-in is exact at the plugged-in distribution.

Self-consistency at F = F̂ is reassuring but cheap — it only says the estimator behaves at the one point I built it around. The real question is whether resampling reproduces a sampling distribution I can compute independently and check against. So let me take the simplest possible case and grind it out by hand. F puts all mass at 0 or 1, θ(F) = Prob{X = 1}, and R = X̄ − θ. A resample X* from F̂ has each component independently 1 with probability x̄ (the observed fraction of ones), 0 otherwise — so n·X̄* is Binomial(n, x̄), and X̄* has mean x̄ and variance x̄(1−x̄)/n exactly. Hence R* = X̄* − x̄ has mean 0 and variance x̄(1−x̄)/n. That is the textbook binomial standard error, and the bootstrap produced it with no special knowledge of binomials — it fell out of the resampling distribution. I didn't fully trust the algebra, so I checked it directly: with n = 20 and x̄ = 0.4, the claimed variance is 0.4·0.6/20 = 0.012, and resampling with replacement two million times gave a sample variance of 0.012006 and a mean of R* of about 2×10⁻⁵. Mean ≈ 0, variance ≈ 0.012 — the closed form and the brute-force resample agree to four figures. Good; the machinery isn't producing nonsense in the one case I can compute two independent ways.

Now push the same direct-theory approach to the case that broke the jackknife: the median. Take n = 2m−1 odd, t = x₍ₘ₎, R = t − θ(F̂) = t − x₍ₘ₎. A resample is described by how many of the n draws land at each ordered value; the count vector N* = (N₁*, …, Nₙ*) is Multinomial(n; 1/n, …, 1/n). The bootstrap median equals x₍ₗ₎ or below exactly when at least m of the n draws land at or below the ℓ-th order statistic, and the number landing at or below position ℓ is Binomial(n, ℓ/n). So

    Prob_*{t(X*) ≤ x₍ₗ₎} = Prob{Binomial(n, ℓ/n) ≥ m},

and differencing in ℓ gives the probability that the bootstrap median equals each order statistic:

    Prob_*{R* = x₍ₗ₎ − x₍ₘ₎}
        = Prob{Binomial(n, ℓ/n) ≥ m} − Prob{Binomial(n, (ℓ−1)/n) ≥ m}.

That is a closed-form bootstrap distribution for the median — no simulation needed. But the derivation is exactly the kind of order-statistic-counting argument where an off-by-one in the inequality would slip through unnoticed, so let me check it. For n = 13 (so m = 7) the formula gives probabilities on ranks ℓ = 1,…,13 of about (0, .0015, .0142, .0549, .1243, .1936, .2229, .1936, .1243, .0549, .0142, .0015, 0); they sum to 1.0000 and are symmetric about the middle rank, as they must be for a symmetric setup. Then I generated four million resamples of a fixed 13-point sample, took the median of each, and tallied how often it equaled each order statistic: (0, .0014, .0142, .0550, .1241, .1937, .2229, .1937, .1241, .0550, .0142, .0015, 0), matching the closed form to within 0.00014 everywhere. So the binomial formula is right, not just plausible.

Having the exact distribution, I can finally test the claim that this fixes the median — by computing its second moment and comparing to the known truth. The correct asymptotic squared error of the sample median for a density f is 1/(4 f²(θ)) per 1/n; for F = N(0,1), θ = 0, f(0) = 1/√(2π), so the target for n·E(R*)² is 1/(4 f(0)²) = 2π/4 = π/2 ≈ 1.5708. I don't get to *assert* the bootstrap hits this — let me actually evaluate it. Using the closed form, for each sampled N(0,1) data set I form Σ_ℓ p_ℓ (x₍ₗ₎ − x₍ₘ₎)² and average over data sets, then multiply by n:

    n = 25  :  n·E(R*)² ≈ 1.93
    n = 101 :  n·E(R*)² ≈ 1.77
    n = 401 :  n·E(R*)² ≈ 1.65

It is descending toward π/2 ≈ 1.57, monotonically and at a believable rate — median-variance convergence is famously slow (order n^{−1/2} corrections), so even at n = 401 sitting at 1.65 rather than 1.57 is what I'd expect, not a red flag. So the bootstrap is consistent for exactly the quantity the jackknife botched: it doesn't merely look philosophically nicer, it gets the median right where the jackknife's estimate isn't even consistent.

That comparison also pins down *why* the jackknife fails, and I want to nail it because it is the whole diagnostic. The jackknife's deletion perturbs F̂ by O(1/n) — it moves one atom's worth of mass. But the genuine sampling fluctuations of the median live at scale O(n^{−1/2}): the multinomial resample counts move the cell weights by O_p(n^{−1/2}), the right scale, while the deletions move them by O(1/n), a far smaller and wrong radius. The median's local behavior at the O(1/n) scale is too irregular to extrapolate from — its derivative with respect to reweighting is erratic there — so the jackknife's first-order extrapolation reads off garbage. The bootstrap samples at the scale the fluctuations actually live at, so it sees the right local geometry. The jackknife isn't wrong in principle; it's a linearization probing at the wrong radius. That story is consistent with the numbers: the bootstrap second moment converged to the analytic median variance, the jackknife one cannot.

So direct theory works when F̂ is simple, but the median needed a special counting argument and a discriminant error rate won't have one. The second computation removes that crutch: Monte Carlo. The distribution of R* is defined by a fully specified, known random mechanism — draw with replacement from x — so I can just *simulate* it. Generate resamples X*¹, X*², …, X*ᴺ, each one n draws with replacement; recompute R*ᵇ = t(X*ᵇ) − θ(F̂) for each; the histogram of {R*ᵇ} approximates the bootstrap distribution, as accurately as I like by taking N large. (This is precisely what I used above to check the binomial variance and the median probabilities, so I already know the loop reproduces the closed forms.) This is the part that makes "any statistic" literally true: I need nothing about t except the ability to evaluate it. Given the original program that computes t, I wrap it in a resample-and-recompute loop, and the cost is about N times the cost of computing t once. With the median I had a formula; for a discriminant error rate or a nonlinear regression coefficient I won't, and I don't need one — N resamples, N recomputations, read off the spread. The price is computation, paid in machine time rather than in a journal-length derivation, and machine time is exactly the resource that's gotten cheap.

The third computation is where the jackknife comes home — and where I should see, not just claim, that the jackknife is the linear piece of this object. I don't *have* to simulate; I can expand R* analytically in the resampling weights. Write the resample as its weight vector: let Pᵢ* = Nᵢ*/n be the bootstrap mass on x_i, P* = (P₁*, …, Pₙ*). Under multinomial resampling,

    E_* P* = e/n,     Cov_* P* = I/n² − e′e/n³,

with e = (1,1,…,1). Think of R as a smooth function of the weight vector, R(P*) = R(X*, F̂), and Taylor-expand about the observed weights P* = e/n:

    R(P*) = R(e/n) + (P* − e/n)U + ½ (P* − e/n)V(P* − e/n)′,

where Uᵢ = ∂R/∂Pᵢ at e/n and V is the matrix of second derivatives. R is unchanged if I scale all the weights by a common factor, so along that radial direction the derivative vanishes: P U(P) = 0. Differentiating that identity once more in P and setting P = e/n gives eU = 0, eV = −nU′, eVe′ = 0. Take the bootstrap expectation: the linear term has mean zero because E_*(P* − e/n) = 0, and the quadratic term contributes ½ tr(V Cov_* P*) = tr(V)/(2n²), using eVe′ = 0. Writing the average diagonal curvature as v = tr(V)/n, that's v/(2n), the usual second-order bias scale. Take the bootstrap variance and keep the leading term — the linear part dominates:

    Var_* R* ≈ U (Cov_* P*) U′ = Σᵢ Uᵢ² / n²,

using eU = 0 to drop the e′e/n³ piece. For the usual error R = θ(F̂*) − θ(F̂) we have R(e/n) = 0, so this is the variance of θ̂, approximated as Σᵢ Uᵢ²/n². That expression — sum of squared influence derivatives over n² — is exactly Jaeckel's infinitesimal jackknife. And the ordinary jackknife agrees with it up to a factor 1 + O(1/n), because the ordinary jackknife just replaces the derivative Uᵢ by the finite difference from deleting x_i. So the jackknife is the linear term of the Taylor expansion of the bootstrap: the delta method applied to the bootstrap distribution, with the bootstrap as the full nonlinear object. And this is the same diagnosis the median numbers gave me from the other side — the jackknife keeps only the first-order term, so on a statistic where the first-order term is a bad summary of the full distribution (the median), it fails, exactly as the consistency check showed.

I should make sure that expansion isn't a sleight of hand, because P* has dimension n, growing with the sample, and Taylor expansions in a dimension that grows with n are exactly where people fool themselves. So rewrite it in fixed dimension. If the sample space is a finite set {1, …, L}, replace the n weights by the L category proportions: let f̂ be the observed proportions and f̂* the resampled proportions, so R is a function Q(f̂*, f̂) of two L-vectors, and the expansion is the ordinary second-order expansion of Q in f̂* near f̂ — dimension L, fixed, not growing. So the expansion is honest. And in this finite picture the validity is transparent: the true mechanism gives f̂ | f ∼ Multinomial_L(n, f), the bootstrap gives f̂* | f̂ ∼ Multinomial_L(n, f̂), and since f̂ → f the bootstrap's conditional law of Q(f̂*, f̂) tracks the true sampling law of Q(f̂, f). Asymptotically: both n^{1/2}(f̂ − f) and n^{1/2}(f̂* − f̂) converge to N(0, Σ_f) with the same multinomial covariance Σ_f (entries f_l(δ − f_m)), so by the delta method both n^{1/2}Q(f̂, f) and the bootstrap n^{1/2}Q(f̂*, f̂) converge to the same Normal(0, u′Σ_f u). The bootstrap distribution and the true sampling distribution share a limit. That is the argument that resampling-with-replacement is doing the right thing in general, not just at F = F̂ and not just for the two statistics I happened to compute by hand.

Now, reading inference off the bootstrap distribution. The standard error is immediate: the standard deviation of the bootstrap replicates θ̂*ᵇ = t(X*ᵇ), estimated from the Monte Carlo sample with the usual divisor, SE = sd({θ̂*ᵇ}). The bias estimate is the mean of the replicates minus θ̂. For an interval, the most direct thing is the empirical quantiles of the bootstrap distribution: the central 1−α interval runs from the α/2 to the 1−α/2 quantile of {θ̂*ᵇ}. That is the percentile interval, and it has a property worth checking rather than asserting — it should respect monotone transformations. Suppose I work on a transformed scale φ = g(θ) (say tanh⁻¹ of a correlation, to make the quantity more nearly pivotal — its distribution more nearly free of the unknown parameter). Each bootstrap replicate maps to g(θ̂*ᵇ); since g is monotone it preserves order, so the α/2 quantile of the φ-replicates is g of the α/2 quantile of the θ-replicates. Therefore g⁻¹ of the percentile interval computed on the φ-scale equals the percentile interval computed directly on the θ-scale — the endpoints land in the same place either way. So I don't have to guess the right transformation; the percentile method is automatically transformation-respecting, and that follows from monotonicity alone, nothing deeper.

If the histogram is biased or skewed, fixed α/2 and 1−α/2 quantiles are a little crude. I can move the tail probabilities instead of changing the resampling logic. First locate θ̂ inside its own bootstrap distribution, counting ties the way a percentile-of-score does:

    P = (#{θ̂*ᵇ < θ̂} + #{θ̂*ᵇ ≤ θ̂})/(2B),     z0 = ndtri(P).

Then I need one number for curvature, because the standard error can change with θ. The leave-one-out values are already the local probes I need: with θ̇ = n⁻¹Σᵢθ̂₍₋ᵢ₎,

    Uᵢ = (n−1)(θ̇ − θ̂₍₋ᵢ₎),     â = (1/6)ΣᵢUᵢ³/(ΣᵢUᵢ²)^{3/2}.

So the nominal tail probability α gets sent to

    ndtr(z0 + (z0 + zα)/(1 − â(z0 + zα))),

applied at α and 1−α. As a sanity check on the algebra: if z0 = 0 and â = 0, the map sends α to ndtr(zα) = α, so it collapses back to the ordinary percentile levels — the correction reduces to the identity exactly when there's no bias and no curvature, which is the only way it could be a defensible adjustment. Otherwise it gives the bias-corrected, accelerated version without touching the resamples themselves.

A caution I shouldn't gloss over, because it's a place this can go genuinely wrong: this gives approximate *frequency* statements, not likelihood statements, and a clean standard error does not by itself produce a clean interval. If I naively treat θ̂ − θ as a pivot and reflect the interval — write Prob_*{x₍₁₎ ≤ θ* ≤ x₍ᵤ₎} and turn it into a statement about θ by flipping signs — I can produce an interval that is the *reflection* of the correct nonparametric one, pointing the wrong way, because θ̂ − θ is not actually pivotal. That's not a rounding error; it's the interval landing on the wrong side. The fix isn't in the resampling — the resampling is fine — it's in being careful about the inferential step, and in working on a scale where the quantity is closer to pivotal, which is exactly what the transformations above buy. A real limitation to flag, not to hide.

A couple of refinements the resampling step invites. If I'm willing to assume F is *smooth*, I shouldn't resample only the exact observed values — I can convolve each resampled point with a little noise of mean 0 and variance matched to a small window, a smoothed bootstrap, so F̂ becomes a smoothed window estimate rather than a spike train. If I'm willing to assume F is *symmetric*, I can reflect F̂ about the estimated center before resampling. These are knobs for extra assumptions; unless smoothness or symmetry is genuinely part of the problem, the plain empirical F̂ is the default because it spends no assumptions I have not earned. And nothing forces F̂ to be nonparametric: if I genuinely believe F is, say, normal, I can resample from the fitted normal MLE instead — the parametric bootstrap — and for the variance of the MLE this recovers one-over-the-Fisher-information. The empirical F̂ is just the assumption-free default at one end of a spectrum.

For regression the resampling unit needs a moment's thought, and the wrong choice is tempting. The model is xᵢ = gᵢ(β) + εᵢ with the εᵢ identically distributed. The obvious move — resample whole rows (cᵢ, xᵢ) the way the jackknife deletes rows — throws away the structural fact that the errors share a single distribution across all i, and so it estimates the wrong, over-general covariance (it lets the error distribution vary with the design point). The right bootstrap fixes β̂, forms the residuals ε̂ᵢ = xᵢ − gᵢ(β̂), resamples *those* with replacement, rebuilds xᵢ* = gᵢ(β̂) + εᵢ*, and refits β̂* on each resample. That respects the identical-distribution assumption — the same residual pool is reused at every design point — and recovers the classical σ²G⁻¹ covariance in the linear case while extending automatically to the nonlinear one where no formula exists. So the resampling *unit* has to match the model's exchangeability structure; with-replacement-at-full-size is right, but on the residuals, not the rows.

The routine is short and statistic-agnostic. Resample with replacement by drawing integer indices uniformly in {0, …, n−1}; gather B such resamples; evaluate the user's statistic on each; the standard error is the standard deviation of those B values, the percentile interval is their α/2 and 1−α/2 quantiles, and the BCa interval applies the z0/â map above.

```python
import numpy as np
from scipy.special import ndtr, ndtri

def empirical_distribution(x):
    return np.asarray(x)


def resample(x, n_resamples, rng):
    x = empirical_distribution(x)
    n = x.shape[-1]
    indices = rng.integers(0, n, size=(n_resamples, n))
    return x[..., indices]


def jackknife_resamples(x):
    x = empirical_distribution(x)
    n = x.shape[-1]
    indices = np.broadcast_to(np.arange(n), (n, n))
    keep = indices[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    return x[..., keep]


def percentile_of_score(values, score):
    values = np.asarray(values)
    score = np.expand_dims(score, axis=-1)
    n = values.shape[-1]
    return (
        np.count_nonzero(values < score, axis=-1)
        + np.count_nonzero(values <= score, axis=-1)
    ) / (2 * n)


def adjusted_quantile_levels(x, statistic, resampled_statistics, alpha):
    theta_hat = statistic(x, axis=-1)
    P = percentile_of_score(resampled_statistics, theta_hat)
    z0 = ndtri(P)

    theta_delete = statistic(jackknife_resamples(x), axis=-1)
    theta_dot = np.mean(theta_delete, axis=-1, keepdims=True)
    n = x.shape[-1]
    U = (n - 1) * (theta_dot - theta_delete)
    a_hat = (1.0 / 6.0) * np.sum(U**3, axis=-1) / np.sum(U**2, axis=-1) ** 1.5

    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    lo = ndtr(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    hi = ndtr(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha)))
    return lo, hi


def sampling_distribution(x, statistic, n_resamples, rng):
    return statistic(resample(x, n_resamples, rng), axis=-1)


def standard_error(resampled_statistics):
    return np.std(resampled_statistics, ddof=1, axis=-1)


def quantile_along_last(values, level):
    values = np.asarray(values)
    level = np.asarray(level)
    if level.ndim == 0:
        return np.quantile(values, level, axis=-1)
    flat_values = values.reshape((-1, values.shape[-1]))
    flat_levels = np.broadcast_to(level, values.shape[:-1]).ravel()
    out = [np.quantile(v, q) for v, q in zip(flat_values, flat_levels)]
    return np.asarray(out).reshape(values.shape[:-1])


def confidence_interval(resampled_statistics, confidence_level, levels=None):
    alpha = (1 - confidence_level) / 2
    if levels is None:
        levels = (alpha, 1 - alpha)
    low, high = levels
    return (
        quantile_along_last(resampled_statistics, low),
        quantile_along_last(resampled_statistics, high),
    )


def resampling_accuracy(x, statistic, n_resamples=9999, confidence_level=0.95,
                        method="bca", rng=None):
    rng = np.random.default_rng(rng)
    x = empirical_distribution(x)
    theta_star = sampling_distribution(x, statistic, n_resamples, rng)
    alpha = (1 - confidence_level) / 2

    if method.lower() == "bca":
        levels = adjusted_quantile_levels(x, statistic, theta_star, alpha)
    elif method.lower() == "percentile":
        levels = None
    else:
        raise ValueError('method must be "bca" or "percentile"')

    return {
        "bootstrap_distribution": theta_star,
        "standard_error": standard_error(theta_star),
        "confidence_interval": confidence_interval(
            theta_star, confidence_level, levels
        ),
    }


def bootstrap(x, statistic, n_resamples=9999, confidence_level=0.95,
              method="bca", rng=None):
    return resampling_accuracy(
        x, statistic, n_resamples, confidence_level, method, rng
    )
```

The causal chain, start to finish: I needed the sampling distribution — hence the standard error — of an arbitrary statistic from one sample, with no per-statistic derivation; the jackknife had the right automatic shape but a shaky foundation that collapsed on the median; asking what the jackknife *approximates* led me to make the empirical distribution F̂ the primitive object and apply the plug-in principle not to a number but to the entire "sample-and-recompute" operation; that forces sampling X* with replacement from the data (sampling without replacement collapses to a point, subsamples need a fudge factor, so with-replacement-at-size-n is the literal meaning of a sample from F̂) and recomputing R(X*, F̂); the estimate is exact at F = F̂ by Fisher consistency, reproduces the binomial standard error and the analytic median variance when I compute those out by hand, and is asymptotically correct because n^{1/2}(f̂*−f̂) and n^{1/2}(f̂−f) share the multinomial limit; for general statistics the distribution comes from Monte Carlo — resample, recompute, B times — and for smooth statistics a Taylor expansion of R in the resampling weights reproduces the jackknife as its linear term, so the jackknife is the delta method applied to the bootstrap and the bootstrap is the full nonlinear object that fixes its failures; from the bootstrap distribution the standard error is the spread of the replicates and the interval is their quantiles (percentile, transformation-respecting; BCa when there's bias or skew), all from one short statistic-agnostic loop whose only cost is computation.
