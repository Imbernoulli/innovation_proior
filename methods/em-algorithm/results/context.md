# Context: maximum likelihood when part of the data is hidden

## Research question

I have a parametric model with density $f(x \mid \phi)$, but I never get to see $x$ in full. What I observe is $y$, a many-to-one image of $x$ through a known map $x \mapsto y(x)$: counts that have been collapsed across categories, measurements that were censored or truncated, survey responses with cells left blank, samples whose group-of-origin label was never recorded. The complete data $x$ would have made estimation easy; the *incomplete* data $y$ is what I actually hold.

The likelihood I am allowed to maximize is therefore the marginal

$$g(y \mid \phi) = \int_{\mathcal{X}(y)} f(x \mid \phi)\, dx,$$

where $\mathcal{X}(y)$ is the set of complete data consistent with the observed $y$. I want $\hat\phi = \arg\max_\phi \log g(y\mid \phi)$.

The pain is concrete. Setting $\partial_\phi \log g(y\mid\phi)=0$ produces equations in which the integral (or sum) over the hidden coordinates sits *inside* a logarithm. For a mixture, $\log g = \sum_i \log\sum_k \pi_k\, p_k(y_i)$ — a sum of logs of sums — and the score equations couple every component's parameters to every data point through denominators that themselves depend on the unknowns. There is no closed form. Each specialty had built its own iterative patch for its own version of this, derived independently and stated in its own notation, and in most cases without any proof that the patch actually climbs the likelihood it claims to.

## Background

**Maximum likelihood and the score equations.** Fisher's program: choose $\phi$ to maximize $\log g(y\mid\phi)$; for regular models the estimator solves the score equation $\partial_\phi \log g = 0$, and for a *regular exponential family* $f(x\mid\phi)=b(x)\exp(\phi\, t(x)^{\mathsf T})/a(\phi)$ the complete-data likelihood equation has the especially clean form $E(t\mid\phi)=t(x)$: set the model's expected sufficient statistic equal to its observed value. When $x$ is fully observed this is routine. The whole difficulty is that with only $y$ in hand, $t(x)$ is not observed.

**A relation between full and marginal scores (Fisher; Sundberg).** Writing the conditional density of the complete data given the observed data as $k(x\mid y,\phi)=f(x\mid\phi)/g(y\mid\phi)$, differentiation of $\log g = \log f - \log k$ and taking conditional expectations yields, for exponential families,

$$\partial_\phi \log g(y\mid\phi) = -E(t\mid\phi) + E(t\mid y,\phi):$$

the marginal score is the difference between the *unconditional* and the *conditional-on-$y$* expectation of the sufficient statistic. R. A. Fisher used the first-derivative form long ago in the context of inefficient statistics; the general identity was written down by Sundberg (1974), who ascribed it to unpublished 1966 lecture notes of Martin-Löf. At a stationary point the two expectations coincide.

**The "missing information principle" (Orchard & Woodbury 1972).** The intuition, stated qualitatively before it was a theorem: estimate the missing parts of the data by their conditional expectation given what you observed and the current parameters, treat the filled-in data as if real, re-estimate, repeat. Orchard and Woodbury named this the missing information principle and discussed it in a non-exponential-family framework. It was a heuristic recipe without a general proof that the observed-data likelihood improves.

**Mixture estimation — the running sore.** Estimating a finite mixture $\sum_k \pi_k\, p_k(y\mid\theta_k)$ by maximum likelihood had been attacked repeatedly. Pearson (1894) resorted to the method of moments precisely because the likelihood was intractable. Hasselblad (1966, 1969) treated mixtures of normals, Poissons, binomials and exponentials; Day (1969) did two multivariate normals with common covariance; Wolfe (1970) did latent-structure / clustering mixtures. Strikingly, these authors worked largely independently, each manipulating the likelihood equations into the same "estimate membership, then re-weight" iteration and each *reporting* that in practice the likelihood went up at every step — but, as was noted at the time, none of them proved it.

**Censored and truncated data; grouping.** Parallel iterative schemes existed for arbitrarily censored data (Efron 1967), grouped/censored/truncated survival data (Turnbull 1974, 1976), and grouping into narrow class intervals (Grundy 1952; Blight 1970, who treated exponential families generally and explicitly recognised the two-step interpretation). Again, one ad-hoc iteration per problem.

## Baselines

- **Hartley (1958), "Maximum likelihood estimation from incomplete data" (Biometrics 14:174–194).** The closest direct ancestor. Hartley gave three multinomial examples in which observed counts are collapses of a finer multinomial; he split the observed categories back into latent sub-categories, used the current parameter to apportion each observed count into expected sub-counts (the fill-in), then re-estimated the parameter by ordinary multinomial maximum likelihood from those filled-in counts, and iterated. It is exactly the two-step loop — for multinomial counts. *Gap:* tied to the multinomial / discrete-count setting; no general formulation across model families, and no general monotonicity theorem.

- **Baum, Petrie, Soules & Weiss (1970), probabilistic functions of finite Markov chains (the Baum–Welch / forward–backward procedure).** For a hidden Markov chain they derived an iteration that, given current transition/emission parameters, computes posterior occupation and transition probabilities of the hidden states via forward–backward recursions, then re-estimates the parameters as posterior-weighted counts — and they *proved* the likelihood increases each iteration for their model, using an auxiliary function. *Gap:* the proof and construction were presented as specific to probabilistic functions of Markov chains; the wide applicability of the underlying argument to incomplete-data maximum likelihood in general was not recognised.

- **The independent mixture iterations (Hasselblad 1966/1969; Day 1969; Wolfe 1970).** Each computes, for every data point, its current posterior probability of belonging to each component (responsibilities), then re-estimates each component's parameters as a responsibility-weighted fit and the mixing weights as average responsibilities. *Gap:* derived case by case, not unified, not differentiated with respect to the natural exponential-family parameters that would have exposed the common structure, and unproven (improvement observed, not established).

- **Orchard & Woodbury (1972), missing information principle.** The qualitative statement of "replace the missing data by its conditional expectation and iterate." *Gap:* a principle, not an algorithm with a convergence theorem; non-exponential-family and informal.

- **Direct numerical maximization (Newton–Raphson / Fisher scoring on $\log g$).** Always available in principle: form the marginal score and information and step. *Gap:* requires the marginal likelihood's gradient and Hessian, which are exactly the awkward log-of-integral quantities; needs second derivatives or their approximations, step-size control, and gives no built-in guarantee of staying in the parameter space or of monotone improvement.

## Evaluation settings

The natural proving grounds are the standard incomplete-data problems of statistics, treated as worked derivations rather than benchmark contests: a multinomial whose observed categories collapse a finer five-category model (the genetic linkage example of Rao 1965, $197$ animals in four observed cells); finite mixtures of normal / Poisson / binomial / exponential families; multivariate normal data with values missing at random; arbitrarily censored, truncated, and grouped data; variance-component estimation in mixed-model ANOVA; factor analysis; iteratively reweighted least squares for robust estimation; and hidden-Markov-chain models. The yardsticks are the ones maximum likelihood already supplies — whether each iteration increases $\log g(y\mid\phi)$, whether the sequence converges, and to what kind of point of the likelihood surface — together with the rate of convergence near a fixed point. The map of the territory matters: mixtures are known to have multiple likelihood maxima, so "converges to a stationary point" is the honest target, not "finds the global optimum."

## Code framework

The scaffold is the apparatus that already exists for complete-data maximum likelihood in a finite mixture: evaluate component log densities, combine them with mixing weights in log space, fit ordinary weighted Gaussian parameters, and leave an empty iterative slot for the hidden labels.

```python
import numpy as np

def log_gaussian_prob(y, means, covariances):
    """Component log densities log p_k(y_i | theta_k)."""
    ...

def weighted_log_prob(y, weights, means, covariances):
    """log pi_k + log p_k(y_i | theta_k), before marginalizing labels."""
    return log_gaussian_prob(y, means, covariances) + np.log(weights)

def observed_data_loglik(y, weights, means, covariances):
    """sum_i log sum_k pi_k p_k(y_i | theta_k): the hard objective."""
    ...

def weighted_gaussian_fit(y, soft_membership, reg_covar=0.0):
    """Ordinary weighted Gaussian MLE from component-membership weights."""
    ...

def fit_mixture(y, initial_parameters, n_iter):
    weights, means, covariances = initial_parameters
    for _ in range(n_iter):
        # TODO: choose how to infer the unobserved labels from current parameters.
        soft_membership = infer_membership(y, weights, means, covariances)
        # TODO: choose how those inferred memberships should drive the next fit.
        weights, means, covariances = update_parameters(y, soft_membership)
    return weights, means, covariances

def infer_membership(y, weights, means, covariances):
    pass

def update_parameters(y, soft_membership):
    pass
```
