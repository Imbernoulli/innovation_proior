## Research question

I have a space of hypotheses (concepts, classifiers, predictors) and a bounded loss `ℓ(h, z) ∈ [0,1]`. Data
`z` arrives i.i.d. from an unknown distribution `D`. For any hypothesis `h` I can measure its **empirical
risk** `ℓ̂(h, S) = (1/m) Σ_{z∈S} ℓ(h, z)` on a sample `S` of size `m`, and I care about its **true risk**
`ℓ(h) = E_{z∼D} ℓ(h, z)`, which I cannot see. The generalization question is: how far can `ℓ(h)` be above
`ℓ̂(h, S)`, and what controls that gap?

Two mature but separate theories answer versions of this, and each pays a price the other doesn't.

- The **PAC / uniform-convergence** answer is distribution-free: it holds for *every* data distribution `D`,
  with no assumption. Its price is that the complexity of the hypothesis class enters pessimistically — a VC
  dimension, a covering number, or a union bound over the class — and for rich or continuous classes the
  resulting bound is *vacuous* (infinite, or larger than 1).
- The **Bayesian** answer lets me pour in domain knowledge through an informative prior, and is optimal *when
  the prior is correct*. Its price is that it offers **no guarantee at all** once the prior is wrong, which in
  practice it always partly is.

The goal that would resolve this: a generalization bound that (a) accepts an informative, data-independent
prior the way Bayes does, so I can encode which hypotheses I expect to be good; yet (b) holds
**distribution-free**, like PAC, assuming nothing about whether the prior matches reality; and (c) does not
become vacuous merely because the hypothesis class is continuous. A solution would have to find a complexity
measure that is simultaneously prior-shaped, finite for continuous classes, and provably controllable without
assuming the prior is true.

## Background

The field state rests on a small number of load-bearing facts and one observed failure mode.

**Concentration of a single empirical mean.** For one fixed hypothesis `h`, `ℓ̂(h, S)` is an average of `m`
i.i.d. `[0,1]` variables with mean `ℓ(h)`. Hoeffding's inequality (1963) and the Chernoff method (1952) give
`Pr_S( ℓ(h) − ℓ̂(h, S) ≥ x ) ≤ e^{−2 m x²}`, and more generally control exponential moments of the deviation,
`E_S exp(λ·deviation)`. When `ℓ` is the 0–1 loss, `m·ℓ̂` is `Binomial(m, ℓ(h))` and one can do better than
Hoeffding using the binomial structure directly, in terms of the **Bernoulli relative entropy**
`kl(q, p) = q ln(q/p) + (1−q) ln((1−q)/(1−p))`. This is the raw material: deviations of a *single* hypothesis
are exponentially unlikely.

**From one hypothesis to a countable class: the Occam / union bound.** Fix a probability distribution `P`
over a countable hypothesis class — a weighting summing to 1, which Linial–Mansour–Rivest (1991) used and
Shawe-Taylor–Williamson (1997) reinterpreted as "a kind of Bayesian prior." Apply the single-hypothesis tail
bound to each `h_i` with failure budget `P(h_i)·δ`, and union-bound: the total failure probability is
`Σ_i P(h_i)·δ = δ`. The result is that, with probability at least `1 − δ`, simultaneously for all `i`,

```
ℓ(h_i) ≤ ℓ̂(h_i, S) + sqrt( ( ln(1/P(h_i)) + ln(1/δ) ) / (2m) ).
```

The complexity of a hypothesis is `−ln P(h_i)` — a "description length." A hypothesis the prior deems likely
(large `P`) pays little; an a-priori unlikely one pays a lot. This is structural risk minimization with the
prior playing the role of the bias, and it justifies selecting the single hypothesis that minimizes the
right-hand side — a maximum-a-posteriori (MAP) rule.

**The diagnostic failure of MAP / selection.** Two known facts about this selection picture set up the
problem. First, from a Bayesian standpoint the MAP hypothesis is *not* the optimal predictor: when the prior
is correct, the optimal rule is the **posterior-weighted vote** over all hypotheses consistent with the data,
`P(label | S) = Σ_{h consistent} P(h)·[h says label] / Σ P(h)` — averaging beats selecting. Second,
Kearns–Mansour–Ng–Ron (1995) gave experimental and theoretical evidence that Bayesian and MDL (MAP-style)
algorithms **overfit** in settings where the Bayesian modeling assumptions fail, precisely because they trust
the prior. So the selection bound above is tied to an algorithm (MAP) that is both Bayes-suboptimal and
fragile, and the `−ln P(h)` complexity is `+∞` for continuous classes where every singleton has prior mass
zero — the bound there is vacuous.

**The change-of-measure identity.** A piece of machinery from large-deviations theory and statistical physics:
for any reference measure `P`, any measurable `h`, the Donsker–Varadhan / Gibbs variational formula states

```
log E_{c∼P} e^{h(c)} = sup_Q { E_{c∼Q} h(c) − KL(Q‖P) },
```

with the supremum attained at the Gibbs distribution `dQ ∝ e^{h} dP`. Equivalently, for *any* `Q` absolutely
continuous w.r.t. `P`, `E_{c∼Q} h(c) ≤ KL(Q‖P) + log E_{c∼P} e^{h(c)}`. Here `KL(Q‖P) = E_{c∼Q} ln(dQ/dP)` is
the Kullback–Leibler divergence (relative entropy). This identity transports an expectation under one measure
to an expectation under another, and the price of transport is exactly the divergence between them.

**Bayesian model averaging / Gibbs predictors.** Averaging predictions under a posterior, and the special case
of an exponentially-weighted ("Gibbs") posterior `dQ_β ∝ e^{−β ℓ̂} dP`, were established devices — weighted
majority and exponentially-weighted aggregation in online learning, mixtures of subtrees / suffix-tree models
in language modeling, where a single huge posterior implicitly weights exponentially many submodels and
smoothing is naturally a *mixture* (a model average), not a *selection*.

## Baselines

**Single-hypothesis Occam bound (Blumer et al.; Shawe-Taylor–Williamson 1997; Shawe-Taylor–Bartlett–
Williamson–Anthony 1996).** Core idea and math as above: per-hypothesis Chernoff + prior-weighted union
bound, complexity `−ln P(h)`. It is empirical (computable from data), tunable through `P`, and distribution-
free. Gap it leaves: the complexity is a *single* hypothesis's prior mass; the union bound is loose; it
applies only to countable classes (vacuous for continuous ones); and it scores a deterministic *selected*
hypothesis, inheriting MAP's Bayes-suboptimality.

**VC / uniform-convergence bounds (Vapnik–Chervonenkis; Valiant 1984).** Distribution-free bounds whose
complexity is the VC dimension of the whole class. Gap: they make no use of an informative prior — the bound
is the same whether or not I have good prior knowledge of which hypotheses are likely to work — and the VC
term can be enormous or infinite for expressive classes.

**Index-based SRM (Linial–Mansour–Rivest 1991; Lugosi–Zeger 1996).** Guarantees in terms of a concept's index
in a fixed sequence of concepts, or the index of a class in a sequence of classes of increasing VC dimension.
They give SRM-style guarantees with *some* prior structure but cannot accommodate an arbitrary prior measure
over concepts.

**Bayesian model averaging for prediction / density estimation (Barron 1991; Barron–Cover 1991; Catoni;
Yang 2000).** Bound a distance (KL divergence, squared loss) between the posterior-averaged predictor and the
truth, distribution-free, with the guarantee better when a simple model fits well. Gap: the clean countable-
mixture versions are typically expectation bounds (not high-probability / large-deviation), are stated in terms
of *unknown* quantities so a learning algorithm cannot output its own certificate, and still charge singleton
or index complexity in a way that goes vacuous for continuous model classes; continuous versions exist but do
not give the same simple empirical certificate.

**Online weighted mixtures (Littlestone–Warmuth weighted majority; Cesa-Bianchi et al.; Freund–Schapire).**
Compete with the best expert on an arbitrary sequence, using a Gibbs-style weighting `Q_β`. Gap: the
inverse-temperature `β` must be fixed *before* seeing data, so the algorithm is not guaranteed against the
optimal accuracy-vs-complexity tradeoff, and the guarantee needs the algorithm to find *all* well-performing
hypotheses, not a single simple one.

## Evaluation settings

The natural yardstick is a distribution-free generalization guarantee on a stochastic predictor:
draw `h ∼ Q`, predict with `h`, and measure the expected risk `E_{h∼Q} ℓ(h)` against its empirical
counterpart `E_{h∼Q} ℓ̂(h, S)`. The relevant regimes to check a candidate bound against: a finite class of
`M` hypotheses; a countable class with a prior `P`; and — the decisive case — a **continuous** class
(parameters in `R^n` with a prior density), where singleton-mass baselines go vacuous. The settings of interest
include realizable concept learning (0–1 loss with a target in the class), the agnostic / unrealizable case
(arbitrary bounded loss), and bounded log loss for density models. Sample sizes range from moderate (where
one would like the bound to be numerically tight, not merely asymptotic) upward. The quality criteria are:
does the bound hold simultaneously for all posteriors `Q`, including data-dependent ones; is it computable
from the sample so it can be reported as a certificate alongside the predictor; and is it tight enough in the
small-empirical-loss regime to be non-vacuous.

## Code framework

The starting point is a generic risk-estimation harness over a hypothesis space with a prior. What is missing is
a data-dependent generalization certificate for randomized predictors.

```python
import numpy as np

# --- pre-existing primitives ---------------------------------------------

def empirical_risk(h, S, loss):
    # mean bounded loss of hypothesis h on sample S; loss(h, z) in [0,1]
    return np.mean([loss(h, z) for z in S])

def kl_divergence(Q, P):
    # KL(Q || P) between two distributions over the (here finite) hypothesis space
    Q = np.asarray(Q, float); P = np.asarray(P, float)
    if np.any((Q > 0) & (P == 0)):
        return float("inf")
    mask = Q > 0
    return float(np.sum(Q[mask] * np.log(Q[mask] / P[mask])))

def posterior_risk(Q, risks):
    # E_{h~Q} of a per-hypothesis risk vector (empirical or true)
    return float(np.dot(Q, risks))

# --- pre-existing baseline: single-hypothesis Occam / union bound --------

def occam_bound(emp_risk_h, prior_mass_h, m, delta):
    # complexity = -ln P(h); vacuous when prior_mass_h -> 0 (continuous classes)
    return emp_risk_h + np.sqrt((np.log(1.0 / prior_mass_h) + np.log(1.0 / delta)) / (2 * m))

# --- open certificate interface -----------------------------------------

def generalization_certificate(Q, P, emp_risks, m, delta):
    """Upper bound, holding with prob >= 1 - delta simultaneously for ALL Q,
    on the true risk of the randomized predictor drawn from Q.
    Must stay finite for continuous classes and be computable from the sample.
    """
    # TODO: finite, empirical complexity-controlled bound
    pass
```
