## Research question

We have a randomized prediction rule: a distribution `ρ` over a hypothesis class `H`, that
predicts by drawing a fresh `h ∼ ρ` for each input and applying it. Its true risk is
`E_ρ[L(h)] = E_{h∼ρ}[L(h)]`, its empirical risk `E_ρ[L̂(h,S)]` on an i.i.d. sample `S` of
size `n` with a `[0,1]`-bounded loss. PAC-Bayesian theory gives high-probability upper bounds
on `E_ρ[L(h)]` of the form *empirical risk + a complexity term*, where complexity is the
Kullback-Leibler divergence `KL(ρ‖π)` to a data-independent prior `π`. These are among the
tightest generalization certificates in statistical learning.

The practical goal is to *learn* `ρ` by directly **minimizing the bound** — turning a
certificate into a training objective — so that the same quantity certifies generalization
and drives the fit. The tightest known bound is the solution of an implicit relation that is
not convex in `ρ`; a common alternative is to optimize a *linear* surrogate
`E_ρ[L̂] + β·KL(ρ‖π)/n` whose trade-off coefficient `β` is picked by cross-validation, which
for models with super-quadratic training cost such as kernel SVMs means retraining many times
on almost the whole dataset. The question is how to formulate a bound on `E_ρ[L(h)]` that can
itself serve as the training objective for `ρ`, with the empirical-risk / complexity trade-off
determined from the data.

## Background

**The change-of-measure lemma.** Almost every PAC-Bayesian bound rests on one inequality.
For any function `f : H × (X×Y)^n → R` and any prior `π` independent of `S`, with probability
at least `1-δ` over `S`, simultaneously for all `ρ`,
```
E_{h∼ρ}[f(h,S)] ≤ KL(ρ‖π) + ln(1/δ) + ln E_{h∼π} E_{S'}[ e^{f(h,S')} ].
```
It comes from Donsker & Varadhan's (1975) variational characterization of the KL divergence,
`KL(ρ‖π) = sup_f { E_ρ[f(h)] + ln E_π[e^{f(h)}] }` — equivalently `ln E_π[e^{f}] =
sup_ρ { E_ρ[f] - KL(ρ‖π) }`, with the supremum attained at the *Gibbs/Boltzmann measure*
`dπ_f/dπ ∝ e^{f}`. One then applies Markov's inequality to the `π`-expectation of the moment
generating function and, using independence of `π` and `S`, swaps the order of the two
expectations. The art of a PAC-Bayes bound is the choice of `f` and the bound on its
moment generating function `E_S[e^{f(h,S')}]`.

**The PAC-Bayes-kl bound (Seeger 2002; Maurer 2004; Langford 2005).** The classical choice is
`f(h,S) = n·kl(L̂(h,S)‖L(h))`, where `kl(p‖q) = p ln(p/q) + (1-p) ln((1-p)/(1-q))` is the
binary KL divergence between two Bernoullis. Maurer (2004) proved the sharp moment-generating
bound `E_S[e^{n·kl(L̂,L)}] ≤ 2√n` for `n ≥ 8` (his Theorem 1; this *halves* the logarithmic
sample-size dependence in the bound, replacing an older `ln(2n)` by `ln(2√n)`). Substituting
gives, with probability `≥ 1-δ`, simultaneously for all `ρ`,
```
kl( E_ρ[L̂(h,S)] ‖ E_ρ[L(h)] ) ≤ ( KL(ρ‖π) + ln(2√n/δ) ) / n.
```
This is the tightest of the standard bounds, and the binary `kl` is invertible numerically:
solving for the largest `L` with `kl(L̂‖L) ≤ c` recovers a certificate on `E_ρ[L(h)]` from
`L̂` and `c = (KL + ln(2√n/δ))/n`. As a function of `ρ`, `E_ρ[L(h)]` is defined implicitly
through the `kl`.

**Relaxations via Pinsker.** Tolstikhin & Seldin (2013) and McAllester (2003) record the
standard route to an explicit bound. The lower bound on the binary KL, `kl(p‖q) ≥ 2(p-q)²`
(Pinsker), inverts to the *PAC-Bayes-classic* bound
`E_ρ[L] ≤ E_ρ[L̂] + √((KL + ln(2√n/δ))/(2n))`. A refinement of Pinsker valid for `p < q`,
`kl(p‖q) ≥ (q-p)²/(2q)`, gives instead
```
E_ρ[L(h)] - E_ρ[L̂(h,S)] ≤ √( 2·E_ρ[L(h)]·(KL(ρ‖π) + ln(2√n/δ))/n ).
```
Here `E_ρ[L(h)]` — the quantity being bounded — sits inside the square root on the right.
Viewed as a quadratic in `√(E_ρ[L])` it can be solved explicitly (the *PAC-Bayes-quadratic*
form), `E_ρ[L] ≤ ( √(E_ρ[L̂] + κ) + √κ )²` with `κ = (KL + ln(2√n/δ))/(2n)`.

**Catoni's trade-off bound (Catoni 2007).** Catoni took a different exponential-tilt
choice of `f`, engineered so that its moment generating function is exactly `1`.
Substituted into the change-of-measure lemma it yields a bound with the
linear-in-`L̂`-plus-`KL` structure, *convex in `ρ`*, carrying a free trade-off parameter `λ`:
```
E_ρ[L] ≤ (1/(1-e^{-λ}))·(1 - exp(-(λ E_ρ[L̂] + (KL + ln(1/δ))/n))).
```
The companion exact-minimization story (recounted e.g. in Alquier's introduction) is clean:
minimizing `E_ρ[L̂] + KL(ρ‖π)/η` over `ρ` is, by Donsker-Varadhan with `h = -η L̂`, solved in
closed form by the *Gibbs posterior* `ρ̂_η(dh) ∝ π(dh)·e^{-η L̂(h,S)}`; when the coefficient is
written as `KL/(nλ)`, the same calculation gives the temperature `η = nλ`. Catoni's bound
holds for a single fixed `λ`, chosen before seeing the data. Keshet, McAllester &
Hazan (2011) used a related parametrized convex bound, with the trade-off parameter set in
practice by the same linear surrogate and cross-validation.

**Convexity under reparametrization.** Two practical considerations recur. First, to keep
`KL(ρ‖π)` tractable, the posterior and prior are often restricted to a parametric family
(Gaussian posteriors with Gaussian priors is the popular choice), and convexity in the
abstract `ρ` can behave differently under such a reparametrization. Second, on an infinite
hypothesis space the Gibbs posterior's normalizer (the partition function
`E_π[e^{-λ n L̂}]`) is generally intractable, which is one reason parametric restrictions are
imposed.

## Baselines

These are the bound formulations a new bound is measured against and reacts to.

**PAC-Bayes-kl (Seeger 2002; Maurer 2004).** `kl(E_ρ[L̂] ‖ E_ρ[L]) ≤ (KL + ln(2√n/δ))/n`,
inverted numerically for a certificate on `E_ρ[L]`. *Core idea:* the sharpest concentration of
the empirical loss around the true loss, via the binary-KL moment bound. As a function of
`ρ`, the certificate is defined implicitly through the inverted `kl`.

**PAC-Bayes-classic / McAllester (McAllester 1999, 2003).**
`E_ρ[L] ≤ E_ρ[L̂] + √((KL + ln(2√n/δ))/(2n))`, from Pinsker `kl(p‖q) ≥ 2(p-q)²`. *Core idea:*
trade the implicit bound for an explicit additive one with a `√(complexity)` penalty.

**PAC-Bayes-quadratic (Seeger 2002; Maurer 2004; Rivasplata et al. 2019).** From the refined
Pinsker `kl(p‖q) ≥ (q-p)²/(2q)`, solving the resulting quadratic in `√(E_ρ[L])`:
`E_ρ[L] ≤ ( √(E_ρ[L̂] + κ) + √κ )²` with `κ = (KL + ln(2√n/δ))/(2n)`. *Core idea:* keep the
`E_ρ[L]`-inside-the-root form and solve it out.

**Catoni's parametrized bound (Catoni 2007; Keshet-McAllester-Hazan 2011).** A bound
`E_ρ[L] ≤ (1/(1-e^{-λ}))·(1 - exp(-(λ E_ρ[L̂] + (KL + ln(1/δ))/n)))`, convex in `ρ`, with a
free, data-independent `λ`. *Core idea:* introduce a trade-off parameter to obtain convexity in
the posterior, with the Gibbs posterior as the closed-form `ρ`-optimum.

## Evaluation settings

The randomized classifier (or its `ρ`-weighted majority vote, whose error is at most twice the
randomized error) is evaluated on standard UCI binary classification datasets (Mushrooms,
Skin, Waveform, Adult, Ionosphere, AvsB, Haberman, Breast cancer; sizes from ~150 to ~2000,
feature dimension `d` from 3 to 122), held-out test sets, with RBF-kernel SVMs tuned by
5-fold cross-validation (soft-margin `C` over `log₁₀C ∈ {-3,…,3}`, bandwidth `γ` on a
geometric grid around a median-distance heuristic) as the strong baseline. The primary metric
is the zero-one **risk certificate** itself — the
high-probability upper bound on `E_ρ[L(h)]` obtained by inverting the PAC-Bayes-kl inequality
on the learned `ρ` — with **lower meaning tighter**; alongside it, test error, the value of
`KL(ρ‖π)`, and running time. For a stochastic-neural-network harness, the same certificate
can be evaluated with Monte Carlo estimates of the stochastic predictor's empirical risk,
the posterior-mean predictor's loss, the ensemble loss, `KL(ρ‖π)`, and the bounded surrogate
loss. If a data-dependent prior is used, the prior-training data must be separated from the
data used to evaluate the bound.

## Code framework

A generic PAC-Bayes bound-optimization harness is available: a
stochastic predictor whose weights are drawn from a posterior `ρ` (a diagonal Gaussian per
parameter, reparametrized so a forward pass with `sample=True` draws fresh weights and
`sample=False` uses the posterior mean), a routine that sums the analytic `KL(ρ‖π)` across the
probabilistic layers, a bounded surrogate loss in `[0,1]`, the numerical binary-KL inverter
`inv_kl(q,c)` (largest `p` with `kl(q‖p) ≤ c`), and a Monte-Carlo estimator of the zero-one
risk. The bound itself is the open part: the functional that turns empirical risk and
`KL` into a certificate, the training objective derived from it, and how the final certificate
is read off. Those are the empty slots.

```python
import math
import torch
import torch.nn.functional as F


# --- existing harness primitives (already available) ---
# model(x, sample=True/False, clamping=True, pmin=...) : PBB-style stochastic forward pass
# get_total_kl(model)          : sum of analytic KL(ρ‖π) over probabilistic layers
# inv_kl(q, c)                 : largest p such that kl(Ber(q) || Ber(p)) <= c (bisection)
# compute_01_risk(model, loader, device, mc_samples) : MC estimate of stochastic 0-1 risk
# F.nll_loss on clamped log-probability outputs : the bounded surrogate loss


class BoundOptimizer:
    """Owns whatever state the bound needs and exposes three slots:
    a bound functional, a training step that minimizes it over the posterior,
    and a final risk-certificate evaluation. The bound functional itself is
    exactly what is to be designed."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-4):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        # TODO: any state the bound functional we design will need.

    def compute_bound(self, empirical_risk, kl, n, delta):
        """Combine empirical risk and KL into a high-probability upper bound
        on the true risk. The exact functional form is what we will design."""
        # TODO: the bound functional.
        pass

    def train_step(self, model, data, target, device, n_bound, delta):
        """One optimization step that decreases the bound over the posterior ρ."""
        log_probs = model(data, sample=True, clamping=True, pmin=self.pmin)
        nll = F.nll_loss(log_probs, target) / math.log(1.0 / self.pmin)
        kl = get_total_kl(model)
        # TODO: form the objective from (nll, kl) and take the step(s) that
        #       minimize the bound; return the objective value.
        pass

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        """Evaluate the final high-probability certificate on the learned ρ."""
        n_bound = len(bound_loader.dataset)
        emp_risk_01 = compute_01_risk(model, bound_loader, device, mc_samples=mc_samples)
        # ... estimate empirical risk and KL on the bound set ...
        # TODO: turn the empirical risk + KL into the final certificate.
        pass
```

The scaffold leaves the bound functional `compute_bound` and the optimization it induces
in `train_step` open; everything else (the stochastic layers, the analytic `KL`, `inv_kl`, the
MC risk estimator) is pre-existing scaffolding.
