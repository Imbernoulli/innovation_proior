# Context: comparing models of differing complexity

## Research question

In almost every data-modelling task — interpolating a noisy curve, fitting a regression, classifying, estimating a density — we do not work with a single model. We invent several: polynomials of various degrees, splines, radial basis functions, a feedforward network; or one functional form under several different priors / regularisers; or several noise models. Two distinct inferences then have to be done. The first is *fitting*: assume one model is true and infer its parameters from the data — standard, well-understood, summarised by the most probable parameters and their error bars. The second is *model comparison*: in the light of the data, decide which of the competing models to prefer, or how to rank them.

The second inference is the hard one, and the one for which no general principled method exists. The obvious recipe — pick the model that fits the data best — is wrong: a more flexible model can always fit at least as well as a less flexible one, so choosing on best fit drives you inexorably to the most over-parameterised model on the table, which then generalises terribly. Something must penalise unnecessary complexity. The pain point is sharp: *what quantity, derived from the rules of probability alone, ranks models and automatically discounts a model for being more complex than the data warrant — with no hand-tuned penalty term bolted on?* A solution would have to (a) be a single transportable number computable for any model class, (b) make over-flexible models lose to adequate simpler ones when the data don't demand the flexibility, and (c) fall out of inference itself rather than being added by fiat.

There is a nested version of the same problem inside interpolation. Maximum-likelihood interpolation is ill-posed: the parameters that minimise the data misfit are underdetermined and chase the noise, so the interpolant oscillates wildly. The cure is a regulariser (a smoothness prior) with a strength α, plus a noise level β. But then α and β themselves must be set, and that is again a complexity trade-off — large α gives a stiff, simple interpolant that underfits, small α gives a flexible one that overfits — for which misfit criteria, held-out test sets, and cross-validation are the standard but unsatisfying tools.

## Background

The logical ground is Bayesian probability as a calculus of plausibility. Cox (1946) showed that any consistent system of degrees of belief over a closed hypothesis space can be mapped onto probabilities obeying the sum and product rules, so probabilities are the unique consistent measure of plausibility. On that ground sit two levels of inference, both governed by Bayes' theorem: inferring parameters within a model, and inferring which model is more plausible. The two are distinct from decision theory, which uses the resulting probabilities together with a loss function to choose actions; the question here is inference alone — assigning probabilities to hypotheses.

Bayesian model comparison was first developed in depth by the geophysicist Harold Jeffreys, whose *Theory of Probability* (1939) is the foundational text. Jeffreys sharply separated estimation from testing, and for testing he built what later became the *Bayes factor*: to decide whether one extra parameter is justified by the data, compare the probabilities the rival models assign to the data, not their best fits. He applied this to simple geophysical model-comparison problems — for instance whether a single additional parameter earns its place. After Jeffreys the mainstream Bayesian emphasis drifted toward formally encoding prior information (insufficient reason, maximum entropy, "the right prior"); the model-comparison strand — which is not about injecting prior information but about extracting the most from the data — stayed comparatively dormant, carried forward by a few (Box & Tiao 1973; Zellner in econometrics; Kashyap 1977, who rediscovered that Bayesian model comparison embodies a complexity penalty in the time-series setting; Patrick & Wallace 1982, comparing theories of megalithic stone-circle geometry within a description-length framework equivalent to Bayes).

The marginal likelihood that does the work is

P(D | H) = ∫ P(D | w, H) P(w | H) dw,

the probability the model assigns to the observed data after averaging over its own parameters under the prior. As a normalised distribution over the space of possible datasets, it cannot be large everywhere: a model that places appreciable predictive probability on many different datasets must place less on any particular one. This is the seed of an automatic complexity penalty, and it is purely a consequence of normalisation, not an added term.

Two background ideas make the marginal likelihood computable and interpretable. The first is **Laplace's method** (Laplace 1774): an integral ∫ exp(−g(w)) dw whose integrand has a sharp peak at the minimiser w* is approximated by the peak value times a Gaussian volume — in k dimensions, ∫ exp(−½ Δwᵀ A Δw) dᵏw = (2π)^{k/2} |A|^{−1/2}, with A the Hessian of g at the peak. Applied to the marginal likelihood this turns the integral into best-fit likelihood × posterior accessible volume × prior density, and the volume ratio is exactly the object that will penalise complexity. The second is the well-known **Bayesian reading of regularisation** (Poggio et al. 1985; Titterington 1985): a quadratic regulariser αEw is a Gaussian prior on the parameters, and minimising misfit-plus-regulariser is finding the most probable parameters. This recasts smoothing as inference and lets the same machinery price the regulariser.

The motivating empirical fact that sets the whole problem is the overfitting phenomenon: across model classes, increasing the number of free parameters monotonically reduces the achievable training misfit while degrading generalisation. The maximum-likelihood interpolant through noisy data oscillates to chase the noise; the minimised misfit keeps dropping as basis functions are added long past the point where the extra functions describe anything real. A model-comparison criterion must counteract exactly this failure.

A second, subtler diagnostic fact concerns estimating a scale parameter. For N samples of a Gaussian with unknown mean μ and standard deviation σ, the maximum-likelihood estimate of σ is the σ_N of the calculator key (dividing by N); but if μ is itself fitted from the data, one degree of freedom is "used up" and the correct, less biased estimate is σ_{N−1} (dividing by N−1). The reduction by one happens because the fitted μ inevitably absorbs some of the noise. This distinction is routinely ignored, yet in interpolation — where the number of parameters can approach the number of data — its generalisation is unavoidable. Critically, the correction comes not from any prior but from *integrating out* the nuisance parameter rather than maximising over it.

## Baselines

**Maximum-likelihood / best-fit model selection.** Choose the model maximising P(D | w_ML, H). Core failure: P(D | w_ML, H) is non-decreasing in model flexibility, so this always prefers the most complex model and overfits. The same defect afflicts maximised-posterior (MAP) model choice. Using only the marginal likelihood for model comparison is, conversely, the model-level analogue of using maximum likelihood for parameter estimation — appropriate only when model priors are taken equal.

**Ad-hoc penalised-likelihood criteria.** AIC (Akaike 1970) scores a model by −2 log L_max + 2k; the Bayesian Information Criterion (Schwarz 1978) uses −2 log L_max + k log N. Both add a fixed penalty that depends only on the parameter count k (and, for BIC, N). Gap: the penalty is bolted on, not derived; it ignores the actual prior the model places on its parameters and the geometry of the fit, and it is at best an asymptotic approximation to a more fundamental quantity.

**Minimum description length (Rissanen 1978; Wallace & Boulton 1968; Wallace & Freeman 1987).** Choose the model giving the shortest two-part code for the data. Closely related to the Bayesian account — the ideal shortest message length is essentially −log₂ P(D | H) — but any implementation must approximate the ideal code length, and it offers no clear advantage over approximating the marginal likelihood directly.

**Structural complexity measures (VC dimension).** Bound generalisation by a combinatorial capacity of the model class. Gap: VC capacity is a worst-case property of the function class alone; it does not depend on the data set actually seen, on the prior, or on how the model distributes predictive probability across data space, so it cannot give the data-dependent, prior-dependent ranking the problem demands.

**Misfit criteria, test sets and cross-validation for the regularisation strength.** To set α and β, orthodox practice forces χ² to a target (the discrepancy principle χ²=N, or χ²=N−k), or minimises error on held-out data, or cross-validates. Gaps: misfit criteria have been argued to set the parameters incorrectly; test-set and cross-validation estimates are noisy and unreliable unless data are plentiful, and their optima are broad and ambiguous compared with what a principled, data-driven setting would give. These are exactly the methods a marginal-likelihood treatment of α and β would replace.

## Evaluation settings

The natural testbed is one-dimensional interpolation of a noisy data set D = {x_m, t_m}, m = 1…N, with the dependent variable corrupted by additive Gaussian noise of standard deviation σ_ν (so β = 1/σ_ν²). Two qualitatively different data sets are appropriate: one with discontinuities in derivative and one smoother. The competing models are the standard interpolation families — fixed-basis linear models y(x) = Σ_h w_h φ_h(x) with bases of Legendre polynomials of varying degree, radial basis functions (e.g. Gaussian or Cauchy bumps) at varying density, splines under regularisers of varying order, and feedforward networks — each fitted under a quadratic regulariser such as Ew = ½ Σ_h w_h². The yardsticks against which a model-comparison criterion would be judged are: agreement with the qualitative Occam's-razor preference (adequate-but-simple over needlessly-flexible), the sharpness and stability of the chosen complexity / chosen α compared with the broad, sample-noisy minima of held-out test error, and recovery of the correct noise level. For a linear model with a quadratic regulariser and Gaussian noise the relevant integrals are exact for any N, which makes this an unusually clean setting in which to check a criterion.

## Code framework

What already exists before any model-comparison criterion is the *fitting* machinery for a linear-in-parameters interpolation model: build the design matrix from a chosen basis, solve for the most probable parameters under a quadratic regulariser, and report error bars from the curvature of the objective. The model-comparison score is the empty slot.

```python
import numpy as np

def design_matrix(x, basis):
    """Phi[m, h] = phi_h(x_m) for the chosen fixed basis (polynomials / RBFs / ...)."""
    raise NotImplementedError

def fit(Phi, t, alpha, beta):
    """First-level inference: most probable parameters of a linear model with
    quadratic regulariser E_w = (1/2) ||w||^2 and Gaussian noise.

        M(w) = alpha E_w + beta E_n,   A = alpha I + beta Phi^T Phi
        w_MP = beta A^{-1} Phi^T t
    Returns w_MP and the Hessian A (whose inverse gives the parameter error bars).
    """
    A = alpha * np.eye(Phi.shape[1]) + beta * (Phi.T @ Phi)
    w_mp = beta * np.linalg.solve(A, Phi.T @ t)
    return w_mp, A

def error_bars(A):
    """Posterior covariance of the parameters = A^{-1}; pointwise bars on y(x)."""
    return np.linalg.inv(A)

def score_model(Phi, t, alpha, beta):
    """Second-level inference: a single transportable number by which competing
    models / values of (alpha, beta) are to be ranked.
    The fit already gives w_MP and A; this score must turn them into a preference
    that does NOT simply grow with the number of basis functions.
    """
    # TODO: the second-level score.
    raise NotImplementedError

def choose(models, t, alpha, beta):
    """Rank candidate models (bases / regularisers / noise models) by score_model."""
    return max(models, key=lambda Phi: score_model(Phi, t, alpha, beta))
```
