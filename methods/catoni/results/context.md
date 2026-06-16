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
and drives the fit, with no held-out tuning. That is what makes the certificate non-vacuous
and the procedure self-contained. The obstacle is that the tightest known bound is awkward to
minimize over `ρ`: it is the solution of an implicit relation that is not convex in `ρ`, so
the optimization is hard and people fall back on a *linear* surrogate `E_ρ[L̂] + β·KL(ρ‖π)/n`
whose trade-off coefficient `β` must be picked by cross-validation. Cross-validation is
expensive — for models with super-quadratic training cost such as kernel SVMs it means
retraining many times on almost the whole dataset — and it is known to be able to mislead.
A solution would have to be (1) a bound nearly as tight as the best one, (2) **convex in `ρ`**
so the posterior can be optimized rigorously, and (3) able to set the empirical-risk /
complexity trade-off **from the data without cross-validation**, ideally with a guarantee
that the optimizer reaches the true minimum of the bound.

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
expectations. The whole art of a PAC-Bayes bound is the choice of `f` and the bound on its
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
This is the tightest of the standard bounds, and the binary `kl` is trivially invertible
numerically: solving for the largest `L` with `kl(L̂‖L) ≤ c` recovers a certificate on
`E_ρ[L(h)]` from `L̂` and `c = (KL + ln(2√n/δ))/n`. The catch is its *shape* as a function of
`ρ`: `E_ρ[L(h)]` is defined implicitly through the `kl`, and the resulting upper bound is
**not convex in `ρ`**.

**Relaxations via Pinsker.** Tolstikhin & Seldin (2013) and McAllester (2003) record the
standard route to an explicit bound. The lower bound on the binary KL, `kl(p‖q) ≥ 2(p-q)²`
(Pinsker), inverts to the *PAC-Bayes-classic* bound
`E_ρ[L] ≤ E_ρ[L̂] + √((KL + ln(2√n/δ))/(2n))`. A refinement of Pinsker valid for `p < q`,
`kl(p‖q) ≥ (q-p)²/(2q)`, is tighter when the risk is below `1/4` and gives instead
```
E_ρ[L(h)] - E_ρ[L̂(h,S)] ≤ √( 2·E_ρ[L(h)]·(KL(ρ‖π) + ln(2√n/δ))/n ).
```
This is closer to the truth at low risk, but `E_ρ[L(h)]` — the very quantity being bounded —
now sits *inside the square root on the right*. Viewed as a quadratic in `√(E_ρ[L])` it can
be solved explicitly (the *PAC-Bayes-quadratic* form), but the square root couples the unknown
to the empirical and complexity terms in a way that is inconvenient for optimizing over `ρ`.

**Catoni's trade-off bound (Catoni 2007).** Catoni took a different exponential-tilt
choice of `f`, engineered so that its moment generating function is exactly `1`.
Substituted into the change-of-measure lemma it yields a bound *convex in `ρ`* —
it has the linear-in-`L̂`-plus-`KL` structure that makes the optimum tractable — at the price
of carrying a free trade-off parameter `λ`. The companion exact-minimization story (recounted
e.g. in Alquier's introduction) is clean: minimizing `E_ρ[L̂] + KL(ρ‖π)/η` over `ρ` is, by
Donsker-Varadhan with `h = -η L̂`, solved in closed form by the *Gibbs posterior*
`ρ̂_η(dh) ∝ π(dh)·e^{-η L̂(h,S)}`; when the coefficient is written as `KL/(nλ)`, the same
calculation gives the temperature `η = nλ`. Catoni's bound, however, **holds for a single fixed `λ`**,
chosen before seeing the data. If one wants to tune `λ` to the sample, one must take a union
bound over a (geometrically spaced) grid of `λ`-values, which both pays an extra logarithmic
penalty and is a discretization rather than a continuous optimization. Keshet, McAllester &
Hazan (2011) used a related parametrized convex bound but, as with Catoni's, the additional
trade-off parameter was in practice replaced by the same linear surrogate and tuned by
cross-validation; rigorous tuning of the trade-off parameter through bound minimization had
not been demonstrated.

**Why convexity keeps breaking.** Two further obstacles recur. First, to keep `KL(ρ‖π)`
tractable, the posterior and prior are often restricted to a parametric family (Gaussian
posteriors with Gaussian priors is the popular choice); even a bound convex in the abstract
`ρ` can lose convexity under such a reparametrization. Second, on an infinite hypothesis space
the Gibbs posterior's normalizer (the partition function `E_π[e^{-λ n L̂}]`) is generally
intractable, which is part of why parametric restrictions are imposed in the first place.

## Baselines

These are the bound formulations a new bound is measured against and reacts to.

**PAC-Bayes-kl (Seeger 2002; Maurer 2004).** `kl(E_ρ[L̂] ‖ E_ρ[L]) ≤ (KL + ln(2√n/δ))/n`,
inverted numerically for a certificate on `E_ρ[L]`. *Core idea:* the sharpest concentration of
the empirical loss around the true loss, via the binary-KL moment bound. *Gap:* the implicit
`kl`-inverted upper bound is not convex in `ρ`, so it cannot be used directly as an objective
for learning the posterior; in practice it is reserved for the *final* certificate while
training is done on something else.

**PAC-Bayes-classic / McAllester (McAllester 1999, 2003).**
`E_ρ[L] ≤ E_ρ[L̂] + √((KL + ln(2√n/δ))/(2n))`, from Pinsker `kl(p‖q) ≥ 2(p-q)²`. *Core idea:*
trade the tight implicit bound for an explicit additive one. *Gap:* loose — the
`√` term does not shrink with the empirical risk, so at low risk it is far above PAC-Bayes-kl;
the additive `√(complexity)` over-penalizes a confident posterior.

**PAC-Bayes-quadratic (Seeger 2002; Maurer 2004; Rivasplata et al. 2019).** From the refined
Pinsker `kl(p‖q) ≥ (q-p)²/(2q)`, solving the resulting quadratic in `√(E_ρ[L])`:
`E_ρ[L] ≤ ( √(E_ρ[L̂] + κ) + √κ )²` with `κ = (KL + ln(2√n/δ))/(2n)`. *Core idea:* keep the
`E_ρ[L]`-inside-the-root tightness at low risk and solve it out. *Gap:* tighter than classic
at small risk, but the way the unknown risk enters under the square root makes the bound a
nonlinear coupling of the empirical and complexity terms, which is less convenient as a
trade-off to optimize and offers no separate, tunable empirical/complexity balance.

**Catoni's parametrized bound (Catoni 2007; Keshet-McAllester-Hazan 2011).** A bound
`E_ρ[L] ≤ (1/(1-e^{-λ}))·(1 - exp(-(λ E_ρ[L̂] + (KL + ln(1/δ))/n)))`, convex in `ρ`, with a
free `λ`. *Core idea:* introduce a trade-off parameter to buy convexity in the posterior, with
the Gibbs posterior as the closed-form `ρ`-optimum. *Gap:* it holds only for a *fixed*,
data-independent `λ`; selecting `λ` from the data requires a union bound over a grid of `λ`,
paying a logarithmic penalty and giving a discretized rather than continuous trade-off, and in
practice the parameter was still set by cross-validation rather than by minimizing the bound.

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
risk. What is *not* settled is the bound itself: the functional that turns empirical risk and
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
