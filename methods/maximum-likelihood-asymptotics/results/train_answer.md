I propose to organize the asymptotics of maximum likelihood estimation around the regular local-quadratic theorem, which is the canonical name I use for maximum likelihood asymptotics. The starting point is a parametric family of distributions indexed by an unknown parameter vector theta_0, and an estimator defined by choosing the parameter that makes the observed sample most compatible with the model, usually by maximizing the log-likelihood ell_n(theta) = sum_{i=1}^n log p_theta(X_i). The first thing to notice is that this objective is a random surface: every new sample reshapes its peaks, changes the location of its maximizers, and sometimes removes any useful peak at all. So the real scientific problem is not simply how to compute an argmax, but when maximizing the likelihood behaves like a stable, well-understood statistical estimate as the sample size grows.

I do not treat a likelihood maximizer as automatically trustworthy. A global maximum can chase a single-observation spike, run off toward a shifted support endpoint, climb a false faraway peak, or lock onto an incidental nuisance feature. It may fail to exist as an attained value, depend on the chosen version of the density, or even improve after information is deliberately discarded. Any claim that the MLE is asymptotically normal without qualification papers over exactly the cases where it is not. What is needed is a result that separates the coarse global step that finds the right neighborhood from the fine local step that describes uncertainty inside that neighborhood, and that states precisely which conditions each step requires.

The global step comes first. Before any local uncertainty calculation is meaningful, the random objective must point toward the truth at a coarse scale. Under correct specification and identifiability, the population log-likelihood contrast separates theta_0 from every other parameter value by a Kullback-Leibler loss. A uniform law of large numbers transfers that separation from the population criterion to the sample criterion, so distant false peaks lose mass and a well-behaved maximizer is forced to localize near theta_0. Without this global separation, a formal local calculation can describe the wrong neighborhood entirely, which is why I insist that localization precede approximation.

Once localization is established, I stop thinking of the estimator as a hard global optimizer and start treating the log-likelihood as a local stochastic surface. The right local coordinate is theta = theta_0 + h / sqrt(n). The reason for this scaling is that the derivative of a sum of n centered terms fluctuates on the sqrt(n) scale, while curvature accumulates on the n scale, so this is the unique zoom at which both the random push and the deterministic penalty stay of order one. In that coordinate the load-bearing expansion is ell_n(theta_0 + h / sqrt(n)) - ell_n(theta_0) = h^T Delta_n - (1/2) h^T I(theta_0) h + o_p(1), uniformly for h in each fixed compact set, where Delta_n = n^{-1/2} dot ell_n(theta_0). The distinctive structure is right here: the local likelihood is not merely smooth, it is a random linear tilt plus a deterministic quadratic penalty.

Both pieces of that quadratic are governed by Fisher information I(theta_0), which does two jobs at once. The linear term h^T Delta_n is the normalized score. At the true parameter, the individual score has mean zero and covariance I(theta_0), so the central limit theorem gives Delta_n converging in distribution to N(0, I(theta_0)); Fisher information appears as the covariance of the random tangent force. The quadratic term is curvature. Under the regularity that lets me differentiate and average the Hessian, n^{-1} ddot ell_n(theta_0) converges to -I(theta_0), which is why, after the minus sign from the second-order Taylor term, the penalty is the strictly convex -(1/2) h^T I(theta_0) h; Fisher information appears as the positive curvature scale.

With these two facts the estimator's limit is forced by geometry, provided the local maximizer is tight, meaning that hat h_n = sqrt(n)(hat theta_n - theta_0) is O_p(1). Maximizing the limiting objective h^T Delta - (1/2) h^T I h means solving Delta - I h = 0, so the local optimizer is h = I^{-1} Delta. Since Delta is Gaussian with covariance I, the curvature-scaled displacement I^{-1} Delta is Gaussian with covariance I^{-1} I I^{-1} = I^{-1}. Translating back from h to theta yields sqrt(n)(hat theta_n - theta_0) converging in distribution to N(0, I(theta_0)^{-1}). This is the core statement of maximum likelihood asymptotics.

The classical score-equation argument is not a separate trick; it is the differential form of the same local quadratic approximation. Expanding dot ell_n(hat theta_n) about theta_0, the left side vanishes at an interior optimizer, the leading term is the score, and the next term is the Hessian times hat theta_n - theta_0. After scaling, the equation reads 0 = Delta_n - I(theta_0) sqrt(n)(hat theta_n - theta_0) + o_p(1), which rearranges to the same limit. The sign check is also immediate: the Hessian contributes -I, the Taylor coefficient contributes 1/2, and together they give the negative quadratic -(1/2) h^T I h. There is no missing factor of n or 2.

The efficiency claim follows from the same local Gaussian-shift viewpoint, but only for regular estimators. Once likelihood ratios in shrinking neighborhoods behave like those of a Gaussian shift experiment, the lower bounds of that limit experiment apply back to the original sequence of experiments. No regular estimator can improve on the inverse-information covariance in the first-order local sense, and the localized maximum likelihood estimator attains that bound under the stated regularity assumptions. The word regular carries the conclusion, not the act of maximizing alone.

I keep the caveats inside the result rather than appended after it, because each one marks a place where a movement fails. If the optimizer is not localized, the quadratic neighborhood is irrelevant. If the score has no Gaussian limit, the linear term changes. If I(theta_0) is singular or nearly flat, curvature can no longer convert score noise into a stable displacement. Single observations can manufacture infinite or dominating peaks in mixture, shifted-support, lognormal, gamma, or singular-density models. Boundary problems such as uniform-endpoint estimation have nonnormal rates and different constants. Incidental parameters, as in the Neyman-Scott construction, can drive the estimator to the wrong value. Bounded likelihood ratios do not by themselves prevent false-peak failure. Ordinary Wald quadratics can be parametrization-dependent or nonuniform near boundaries, so the safer local object is the likelihood-ratio or Hellinger or Gaussian-shift geometry.

The distinctive insight is therefore not that argmax implies normality. It is that a localized likelihood with nonsingular information curvature becomes a Gaussian quadratic, and that geometry, global separation to reach the truth followed by a curvature-controlled Gaussian linear-plus-quadratic locally, is what controls the estimator's uncertainty. The final deliverable is the theorem: in a regular correctly specified dominated parametric model with true interior parameter theta_0, and with hat h_n = sqrt(n)(hat theta_n - theta_0) assumed O_p(1) and I(theta_0) nonsingular, if the uniform local expansion ell_n(theta_0 + h / sqrt(n)) - ell_n(theta_0) = h^T Delta_n - (1/2) h^T I(theta_0) h + o_p(1) holds with Delta_n converging to N(0, I(theta_0)), then any local approximate maximizer satisfies hat h_n = I(theta_0)^{-1} Delta_n + o_p(1) and therefore sqrt(n)(hat theta_n - theta_0) converges in distribution to N(0, I(theta_0)^{-1}).

Because this is a theorem rather than a production algorithm, the accompanying code is a small runnable verification rather than a reference implementation. The script below simulates a simple Gaussian model with known variance, computes the maximum likelihood estimator of the mean, checks empirically that sqrt(n)(mu_hat - mu_0) has variance close to the inverse Fisher information, and compares the empirical distribution to the predicted asymptotic normal. This confirms the local-quadratic geometry in a concrete numerical example.

```python
import numpy as np
from scipy import stats

np.random.seed(0)
mu0 = 2.5          # true mean
sigma2 = 1.5       # known variance
n = 400            # sample size
B = 20000          # number of simulation repetitions
info_per_obs = 1.0 / sigma2
asymptotic_variance = sigma2

scaled_errors = np.empty(B)
for b in range(B):
    sample = np.random.normal(loc=mu0, scale=np.sqrt(sigma2), size=n)
    mu_hat = np.mean(sample)
    scaled_errors[b] = np.sqrt(n) * (mu_hat - mu0)

empirical_variance = np.var(scaled_errors, ddof=1)
print(f"Sample size n = {n}")
print(f"Inverse Fisher information (theoretical variance) = {asymptotic_variance:.6f}")
print(f"Empirical variance of sqrt(n) * (mu_hat - mu0) = {empirical_variance:.6f}")

ks_stat, p_value = stats.kstest(scaled_errors, "norm", args=(0.0, np.sqrt(asymptotic_variance)))
print(f"KS statistic vs N(0, {asymptotic_variance:.4f}) = {ks_stat:.4f}, p-value = {p_value:.4f}")

sample = np.random.normal(loc=mu0, scale=np.sqrt(sigma2), size=n)
ll0 = -0.5 * np.sum((sample - mu0) ** 2) / sigma2
score = np.sum(sample - mu0) / sigma2
observed_info = n / sigma2

hs = np.linspace(-3.0, 3.0, 7)
print("\nh   loglik(theta0 + h/sqrt(n))   quadratic approx")
for h in hs:
    mu = mu0 + h / np.sqrt(n)
    ll = -0.5 * np.sum((sample - mu) ** 2) / sigma2
    approx = ll0 + h * score / np.sqrt(n) - 0.5 * h * h * observed_info / n
    print(f"{h:+.2f}   {ll:+.6f}                {approx:+.6f}")
```
