## Research question

Given a probabilistic model that ties hidden variables `z` to observed data `x` through a
joint distribution `p(x, z)` we can write down cheaply — for a graphical model, a product of
local conditional factors — we want two things that look like they should be easy and turn
out not to be:

1. the **posterior** `p(z | x) = p(x, z) / p(x)`, which is what every downstream
   question (point estimates of the latent variables, predictive densities, decisions) is
   really asking for; and
2. the **evidence** (marginal likelihood) `p(x) = ∫ p(x, z) dz`, which we need both to
   normalize the posterior and to score / learn the model.

The obstruction is the same object in both: the evidence integral. For discrete `z` it is a
sum over exponentially many configurations of the hidden variables; for continuous `z` it is
a high-dimensional integral with no closed form. The numerator `p(x, z)` is cheap; the
denominator is what makes inference hard. A method that resolved this would have to produce a
usable stand-in for `p(z | x)` *without* ever evaluating `p(x)` — and, ideally, hand back a
computable estimate of `p(x)` as a by-product, deterministically and fast enough to fit many
models or very large data.

The pain is not hypothetical. In the QMR-DT medical-diagnosis network — roughly 600 disease
nodes over 4000 symptom nodes, with noisy-OR conditional probabilities — the diseases couple
through every observed positive finding ("explaining away"). Across standard diagnostic cases
the median size of the largest clique in the moralized graph is about 151.5 nodes; an exact
junction-tree calculation is exponential in that clique size, so it is simply infeasible.
Dense graphical models routinely produce large cliques, so exact inference is off the table
exactly where the models are most interesting.

## Background

**Graphical models and exact inference.** A directed graphical model (Bayesian network)
factorizes the joint as `p(S) = ∏_i p(S_i | S_{π(i)})`, a product of local conditionals over
each node given its parents; an undirected model factorizes over clique potentials. Exact
inference — computing `p(H | E)` for hidden `H` given evidence `E` — is done by the junction-
tree algorithm (moralize, triangulate, propagate). It is exact and exploits all the
conditional independencies in the graph, but its time complexity is exponential in the size
of the largest clique of the triangulated graph. Natural modeling choices (dense connectivity,
layered networks, "explaining away" in noisy-OR models) force large cliques, and then exact
inference becomes intractable. This structural cost follows from the graph itself.

**Monte Carlo.** The dominant route around intractability has been sampling: build an ergodic
Markov chain whose stationary distribution is the posterior (Metropolis–Hastings 1953/1970;
the Gibbs sampler, Geman & Geman 1984; its use for Bayesian inference, Gelfand & Smith 1990),
run it, and approximate the posterior with the collected samples. The Gibbs sampler in
particular iterates over variables, repeatedly drawing each one from its **complete
conditional** `p(z_j | z_{−j}, x)` — the distribution of one latent variable given all the
others and the data. MCMC is asymptotically exact and simple to implement, but it is
stochastic, can be slow to converge, and convergence is hard to diagnose. When an approximate
posterior is needed quickly, or on very large data, or when many candidate models must be
fit, sampling is an awkward fit.

**The Kullback–Leibler divergence.** For two distributions over the same variables,
`KL(q ‖ p) = Σ_z q(z) log [q(z) / p(z)]` (Kullback & Leibler 1951). It is non-negative, zero
iff `q = p`, and asymmetric: `KL(q ‖ p) ≠ KL(p ‖ q)`. The two directions ask different
computational questions — `KL(q ‖ p)` is an expectation under `q`, while `KL(p ‖ q)` is an
expectation under `p`. This asymmetry is load-bearing for anything that wants a *computable*
measure of how far an approximation is from a target it cannot evaluate.

**Jensen's inequality and convex duality.** For a concave function (the logarithm), Jensen's
inequality gives `log E[Y] ≥ E[log Y]`, turning the log of an average into an average of logs
— a lower bound. Convex-duality theory supplies the parallel view for log-sum-exp: if
`f(u) = log Σ_z exp(u_z)`, then its Fenchel conjugate is `f*(q) = Σ_z q_z log q_z` on the
probability simplex and `+∞` outside it, so `f(u) = sup_q {qᵀu − f*(q)}`. These are the
standard tools for replacing an intractable nonlinear object by a tractable bounded one, fit
by optimizing the variational parameter.

**Mean-field theory from statistical physics.** In statistical mechanics, intractable
interacting-spin systems are approximated by replacing each spin's interaction with its
neighbors by an interaction with the *average* ("mean") field they produce, decoupling the
system into independent single-spin problems whose self-consistent averages solve a set of
fixed-point "mean-field equations" (Parisi 1988). Peterson & Anderson (1987) carried this
into neural networks: for a Boltzmann machine they replaced the stochastic correlations
estimated by Gibbs sampling with deterministic mean-field fixed-point equations, approximating
the correlation between two units by the product of their individual mean activations. On
their test cases the deterministic mean-field computation ran 10–30 times faster than Gibbs
sampling at roughly equivalent accuracy. The crucial pre-existing idea here is the **fully
factorized approximation** — treat coupled variables as independent, each described by a
single average — together with the observation that this can be both fast and accurate when
averaging phenomena dominate.

**The EM algorithm as a bound.** Expectation–Maximization (Dempster, Laird & Rubin 1977) finds
maximum-likelihood parameters `θ` in latent-variable models by alternating an E step
(compute the posterior `p(z | x, θ)` over the latent variables) and an M step (re-estimate
`θ` to maximize the expected complete-data log-likelihood `E_{p(z|x,θ)}[log p(x, z | θ)]`),
with each iteration provably non-decreasing in the likelihood. Neal & Hinton (1998; first
circulated 1993) recast this as maximizing a single function jointly in `θ` and in a
free distribution `P̃` over the latent variables:
`F(P̃, θ) = E_{P̃}[log p(x, z | θ)] + H(P̃)`, where `H(P̃)` is the entropy of `P̃`. They
showed `F(P̃, θ) = −KL(P̃ ‖ p(z | x, θ)) + log p(x | θ)`, so for fixed `θ` the unique `P̃`
maximizing `F` is exactly the posterior `p(z | x, θ)` (the E step), and the M step maximizes
`F` over `θ` — EM is **coordinate ascent on `F`**. `F` is, up to a sign, the variational free
energy of statistical physics, with `−log p(x, z | θ)` playing the role of the energy of a
state. This decomposition — an objective equal to "expected log-joint plus entropy" and also
to "log-evidence minus a KL to the true posterior" — supplies a general inference objective.
Its limitation as stated is that the E step *assumes the posterior `p(z | x, θ)` is
computable*; precisely the assumption that fails for the dense graphical models above.

## Baselines

The existing approaches line up as follows.

- **Exact junction-tree inference (Jensen 1996; Shachter, Andersen & Szolovits 1994).** Core
  idea: moralize and triangulate the graph, build a clique tree, and propagate local messages
  to get exact marginals and exact `p(E)`. Math: complexity exponential in the maximal clique
  of the triangulated graph. Gap: that exponent is unbounded for dense models — at clique
  size ~151 (QMR-DT) it cannot run. It also makes no use of the *numerical* values in the
  conditional probability tables, so it pays the same exponential cost even when the
  distribution is "nearly" factorized and a cheap approximation would suffice.

- **Pruning / bounded-conditioning / localized partial evaluation (Kjærulff 1994; Horvitz,
  Suermondt & Cooper 1989; Draper & Hanks 1994).** Core idea: stay tied to the exact
  algorithm but prune low-probability configurations or condition on a subset of variables to
  shrink the work. Gap: because they ride on the exact algorithm, they inherit its exponential
  growth; they delay the blow-up rather than removing it, and provide no global accuracy
  guarantee.

- **Gibbs sampling and general MCMC (Geman & Geman 1984; Gelfand & Smith 1990; Metropolis–
  Hastings 1953/1970).** Core idea: simulate a Markov chain with the posterior as stationary
  distribution; for Gibbs, repeatedly resample each variable from its complete conditional
  `p(z_j | z_{−j}, x)`; estimate posterior quantities from the samples. Gap: stochastic and
  asymptotically (not finitely) exact; convergence can be slow and is hard to diagnose; it
  returns samples rather than a closed-form density or a deterministic estimate of `p(x)`, and
  it scales awkwardly to massive data or to fitting many models.

- **EM (Dempster, Laird & Rubin 1977) and the free-energy view (Neal & Hinton 1998).** Core
  idea: coordinate ascent on `F(P̃, θ) = E_{P̃}[log p(x, z | θ)] + H(P̃)`; the E step sets
  `P̃ = p(z | x, θ)`, the M step maximizes the expected complete-data log-likelihood. Gap: the
  exact E step is only available when `p(z | x, θ)` is itself tractable — the very condition
  that breaks for dense graphical models. EM gives the right *objective* but not a way to run
  it when the posterior is intractable.

- **Mean-field neural networks (Peterson & Anderson 1987).** Core idea: approximate an
  intractable Boltzmann distribution by a fully factorized distribution and solve deterministic
  mean-field fixed-point equations instead of sampling correlations. Gap as it stood: it was
  derived as a physics-style approximation for a *specific* model (the Boltzmann machine),
  justified by analogy to statistical mechanics rather than by a single global measure of
  approximation accuracy, and not connected to likelihood bounds or to a general recipe for
  arbitrary graphical models.

## Evaluation settings

The natural yardsticks that already exist for an approximate-inference method:

- **Models / datasets.** The QMR-DT diagnostic network (≈600 diseases, ≈4000 symptoms,
  noisy-OR conditionals) over standard diagnostic ("CPC") cases; Boltzmann machines and sigmoid
  belief networks; factorial and higher-order hidden Markov models; and, as a clean fully
  worked instance, a Bayesian mixture of unit-variance Gaussians with a Gaussian prior on the
  component means (`K` components, `n` observations) — small enough to derive every update by
  hand and to compare against an exactly computable answer.
- **Metrics.** The marginal likelihood / evidence `p(E)` (or its log) as both the quantity to
  bound and the score for model comparison; the achieved value of any bound on `log p(x)`,
  used to monitor convergence and to compare approximations (a tighter bound is a better
  approximation); the KL divergence from the approximation to the true posterior where the
  posterior is available; held-out predictive likelihood on a test set; and wall-clock time to
  a usable answer (the axis on which deterministic methods are meant to beat sampling).
- **Comparators / protocol.** Gibbs sampling (and Hamiltonian Monte Carlo on the continuous
  examples) as the sampling baselines; exact inference wherever the model is small enough to
  run it, to measure approximation error directly; the same model fit from several random
  initializations, since any non-convex objective over the approximating family will have
  multiple local optima and the protocol must report sensitivity to initialization.

## Code framework

The reusable pieces are a cheaply evaluated log-joint, per-coordinate log-factors, and an
outer loop with a convergence check. The missing pieces are a tractable stand-in for the
posterior, a computable score for it, and an update rule.

```python
import numpy as np

# ---- The model: a joint p(x, z) evaluated cheaply as local factors ----
# Concretely, a Bayesian latent-variable model with observed x and hidden z.
# We can compute log p(x, z) and each local log-factor in closed form.
# The evidence integral / sum p(x) = ∫ p(x, z) dz is intractable in general,
# so normalizing p(z | x) = p(x, z) / p(x) is not directly available.

def log_joint(x, z, params):
    """log p(x, z): a sum of local log-factors. Cheap to evaluate pointwise."""
    raise NotImplementedError  # TODO: model-specific product of local conditionals

def log_evidence_exact(x, params):
    """log p(x) = log ∫ p(x, z) dz. Intractable in general — this is the wall."""
    raise NotImplementedError  # exponential in the worst case


# ---- A tractable surrogate for p(z | x) ----
# Its form, score, and update rule are the open slots.

class PosteriorApproximation:
    """A tractable stand-in q for the intractable posterior p(z | x).

    We do not yet know:
      - what family q should live in (its factorization / parametric form),
      - what objective tells us q is 'close' to p(z | x) without touching p(x),
      - the rule that improves q.
    """
    def __init__(self, model_params):
        pass  # TODO: parameters of the approximating family

    def objective(self, x):
        # TODO: a score we can actually compute (no p(x)) and push uphill
        raise NotImplementedError

    def update(self, x):
        # TODO: one step that provably improves the objective
        raise NotImplementedError


def fit(x, model_params, max_iters=100, tol=1e-6):
    """Outer loop: improve the approximation until the objective stops moving."""
    q = PosteriorApproximation(model_params)
    prev = -np.inf
    for _ in range(max_iters):
        q.update(x)               # TODO: one improvement step
        score = q.objective(x)    # TODO: computable progress measure
        if abs(score - prev) < tol:
            break
        prev = score
    return q
```
