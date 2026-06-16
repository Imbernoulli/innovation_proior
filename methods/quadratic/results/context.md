## Research question

A trained neural network produces a weight vector that fits the training data, but the
quantity we actually care about is its risk on unseen examples drawn from the same
distribution. The standard learning-theory machinery for controlling that risk —
VC-dimension, Rademacher complexity, uniform convergence — gives bounds that scale with
model capacity, and a modern over-parameterized network has far more parameters than
training examples. Plugged into those bounds, the capacity term dwarfs the sample size and
the certificate exceeds 1, i.e. it is *vacuous*: it tells you the error rate is at most some
number above 100%, which is no guarantee at all. Yet these same networks generalize well in
practice. So the question is sharp: can we compute a *non-vacuous* numerical upper bound on
the true risk of a deep network — a number well below the trivial 1, valid with high
probability over the draw of the training sample — and can we make that number *tight*,
close enough to the test error that it is actually informative?

A solution has to clear several bars at once. (1) The certificate must hold with high
probability `1 - delta` over the random training sample, simultaneously over whatever
predictor the learner ends up choosing (the learner looks at the data, so the bound must be
uniform over data-dependent choices). (2) It must be *computable* — every term must be
either evaluated exactly or upper-bounded by something we can compute, including the
empirical loss of a randomized predictor, which is itself an expectation we can only sample.
(3) It must be *optimizable*: ideally the very expression we certify can be minimized by
gradient descent during training, so the learner directly drives down the guarantee rather
than minimizing a proxy and hoping the bound follows. (4) For the regime that matters — a
network that has driven its empirical error close to zero — the bound must stay tight; a
guarantee that is only good when the training error is large is useless for the models we
build. (5) Ideally it is *self-certified*: the certificate should not consume a held-out
test set as part of the guarantee.

## Background

The framework that survives the over-parameterization problem is **PAC-Bayes**. Instead of
certifying a single fixed predictor `h_w`, it certifies a *randomized* (stochastic)
predictor: a distribution `Q` over weight space, where each prediction draws a fresh weight
vector `W ~ Q` and applies `h_W`. The `Q`-weighted population and empirical losses are

```
L(Q)    = E_{W~Q}[ L(W) ]       (true risk of the randomized predictor)
Lhat_S(Q) = E_{W~Q}[ Lhat_S(W) ]  (empirical risk, S the training sample)
```

PAC-Bayes inequalities relate `L(Q)` to `Lhat_S(Q)` and a complexity term — but the
complexity term is *not* a capacity count. It is the Kullback-Leibler divergence
`KL(Q || Q0)` between the learned distribution `Q` (the "posterior") and a reference
distribution `Q0` (the "prior") fixed before seeing the data on which the bound is
evaluated. Crucially `KL(Q || Q0)` depends on how far the learned distribution has moved
from the reference, not on the parameter count — so a billion-parameter network whose
posterior stays close to its prior pays a small complexity, which is exactly why PAC-Bayes
yields non-vacuous numbers where uniform-convergence bounds blow up. ("Prior" and
"posterior" here are PAC-Bayes terminology: `Q0` is just a data-free reference distribution
and `Q` an unrestricted data-dependent one — there is no likelihood factor connecting them,
they are not Bayesian objects.)

Two further pieces of standard apparatus are load-bearing.

**The binary KL divergence and its role.** For `q, p in [0,1]`,
`kl(q || p) = q log(q/p) + (1-q) log((1-q)/(1-p))` is the relative entropy between the
Bernoulli(`q`) and Bernoulli(`p`) distributions. The strongest classical PAC-Bayes
inequality, the **PAC-Bayes-kl bound** (Langford & Seeger 2001, "Bounds for averaging
classifiers", CMU-CS-01-102; Seeger 2002, "PAC-Bayesian generalisation bounds for Gaussian
processes", JMLR 3:233-269), states that with probability at least `1 - delta` over size-`n`
i.i.d. samples, simultaneously for all `Q`,

```
kl( Lhat_S(Q) || L(Q) )  <=  ( KL(Q || Q0) + log( 2 sqrt(n) / delta ) ) / n .
```

This controls the *binary KL between the empirical and true risks*, which is a tighter,
asymmetric measure of their gap than the absolute difference. The sharp `2 sqrt(n)` constant
in the numerator is due to Maurer (2004, "A Note on the PAC-Bayesian Theorem",
arXiv:cs/0411099), who proved the underlying moment inequality
`E_S[ e^{ n kl( M(h(S)), h(D) ) } ] <= 2 sqrt(n)` for `n >= 8` (and showed `sqrt(n)` is a
matching lower bound, so the order cannot be improved). His proof route — a change of
measure (Donsker-Varadhan / Csiszár 1975) that moves the data-dependent `Q` onto a fixed
`P`, the moment inequality applied per-hypothesis, then Markov's inequality — is the standard
template every PAC-Bayes bound below is a relaxation of. Standard tools for lower-bounding a
binary KL by an elementary expression of its arguments include the Pinsker inequality
`kl(qh || p) >= 2(p - qh)^2` and refined relative-entropy inequalities such as
`kl(qh || p) >= (p - qh)^2 / (2p)` (valid for `qh < p`); which one is tighter depends on the
regime of `p`.

**Loss bounding.** PAC-Bayes inequalities require a loss with range `[0,1]`, but networks are
trained with the cross-entropy (negative log-likelihood) loss, which is unbounded above. The
standard fix (Dziugaite & Roy 2017) lower-bounds each predicted class probability by a
constant `pmin > 0`: replacing `sigma(u)_y` with `max(sigma(u)_y, pmin)` caps the
cross-entropy at `log(1/pmin)`, and rescaling by `1/log(1/pmin)` produces a surrogate loss in
`[0,1]` usable inside the bounds.

**The empirical-risk-of-a-distribution problem.** `Lhat_S(Q) = E_{W~Q}[Lhat_S(W)]` is an
integral over `Q`, not directly computable. The standard recipe (Langford & Caruana 2001,
"(Not) bounding the true error") is Monte-Carlo: draw `m` weight samples `W_1, ..., W_m ~ Q`,
average the empirical loss, then account for the Monte-Carlo error itself with a high-
probability binary-kl (Chernoff) tail bound — `kl( Lhat_S(Qhat_m) || Lhat_S(Q) ) <=
log(2/delta')/m` with probability `1 - delta'`. Inverting the binary kl in its second
argument turns each such inequality into a numerical upper bound on the quantity of
interest.

**The motivating empirical observation.** Dziugaite & Roy (2017, "Computing Nonvacuous
Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than
Training Data", arXiv:1703.11008) showed that optimizing a PAC-Bayes bound by SGD over a
Gaussian distribution on the weights of an over-parameterized MNIST network yields the first
*non-vacuous* bounds — bound values well below 1. But their reported certificate (on the
order of 0.16-0.22 on MNIST) sits far above the actual test error (a few percent): the bound
is non-vacuous but loose. The diagnostic feature visible in the objective is that, as the
empirical loss approaches zero, the remaining complexity contribution is still an additive
square-root term. Closing that gap between the certificate and the test error is the open
problem.

## Baselines

**PAC-Bayes-classic / McAllester bound (McAllester 1999, "Some PAC-Bayesian Theorems",
COLT; with Maurer's sharp constant).** Relax the PAC-Bayes-kl bound with the standard Pinsker
inequality `kl(qh || p) >= 2(p - qh)^2`. Lower-bounding the left side of pb-kl and solving
the resulting `2(L - Lhat)^2 <= (KL + log(2 sqrt(n)/delta))/n` for `L` gives

```
L(Q)  <=  Lhat_S(Q) + sqrt( ( KL(Q || Q0) + log(2 sqrt(n)/delta) ) / (2n) ) .
```

As a training objective (replacing the 0-1 loss with the bounded cross-entropy) this is
`f_classic`, essentially the objective Dziugaite & Roy optimized. It holds simultaneously
over all `Q`, so it can be minimized over `Q`, and the complexity term is the genuine
KL-to-prior rather than a capacity count. **Limitation:** the complexity contribution enters
as an *additive* square-root term that does not shrink as the empirical loss goes to zero. In
the regime a trained network reaches — empirical loss near zero — the certificate is
dominated by this `sqrt(complexity/2n)` term and stays at the square-root scale of the
complexity, no matter how small the empirical loss becomes.

**PAC-Bayes-lambda bound (Thiemann, Igel, Wintenberger & Seldin 2017, "A Strongly Quasiconvex
PAC-Bayesian Bound", ALT).** Built from a relative-entropy relaxation of pb-kl combined with
the arithmetic-geometric inequality `sqrt(ab) <= (lambda a + b/lambda)/2` for a free
`lambda > 0`. The result holds uniformly over `Q` and over `lambda in (0,2)`:

```
L(Q)  <=  Lhat_S(Q) / (1 - lambda/2)
        + ( KL(Q || Q0) + log(2 sqrt(n)/delta) ) / ( n lambda (1 - lambda/2) ) .
```

It is jointly quasiconvex in `(Q, lambda)`, so one alternates minimization over `Q` and
`lambda` (or grid-searches `lambda` without weakening the guarantee). As `f_lambda` it is a
strong *training* objective. **Limitation:** it carries an extra parameter to tune, and
because the AM-GM step inserts a `lambda`-dependent slack, the risk certificates it yields are
generally not as tight as the underlying relative-entropy relaxation could in principle
deliver; the slack is the price of arriving at a clean linear-in-`Lhat` objective.

**Bayes by Backprop (Blundell et al. 2015, "Weight Uncertainty in Neural Networks", ICML).**
Not a certificate at all, but the optimization substrate. It learns a diagonal-Gaussian
distribution `Q_theta = N(mu, diag(sigma^2))` over weights by minimizing a variational
objective (expected negative log-likelihood plus `KL(Q || Q0)`), using the reparameterization
`W = mu + sigma odot V` with `V` standard noise and `sigma = log(1 + exp(rho))` (softplus, so
`sigma > 0` under unconstrained `rho`). The pathwise gradient estimator — differentiate
through the sampled weights — gives low-variance unbiased gradients of an expectation-over-`Q`
objective, and the gradients for `mu` and `rho` are the ordinary backprop gradients, shifted
and scaled. **Limitation:** the objective is an ELBO with a free KL-trade-off coefficient;
minimizing it certifies nothing about risk on unseen data.

**Data-dependent priors (Ambroladze et al. 2007; Parrado-Hernández et al. 2012; the prior-
mean-by-ERM construction).** The KL term dominates the bound, so shrinking it is the lever for
tightness. If the prior `Q0` is centered far from where the posterior ends up, `KL(Q || Q0)`
is large. The remedy is to learn the prior from data — but the PAC-Bayes-kl bound requires
`Q0` to be *data-free with respect to the sample on which the bound is evaluated*. The clean
construction splits the training set: use one part to learn the prior mean by empirical risk
minimization, and the disjoint remainder to learn the posterior and evaluate the certificate;
on that remainder the prior is still data-free, so pb-kl applies. **Limitation/cost:** the
construction must keep the bound-evaluation subset disjoint from the prior-learning subset, or
else recover validity with much heavier machinery (differential-privacy / max-information
corrections, as in the two-stage SGLD approach of Dziugaite & Roy 2018) that lacked finite-
sample guarantees.

## Evaluation settings

The natural yardsticks already in use for non-vacuous neural-network certificates:

- **MNIST** (60k training, 28×28 grayscale, 10 classes) with a fully connected network
  (`784 - 600 - 600 - 600 - 10`, three hidden layers) and with a convolutional network
  (two conv layers + two fully connected). The over-parameterized regime: parameter count far
  exceeds `n`.
- **FashionMNIST** (same dimensions/sizes as MNIST, harder) with the same CNN architecture.
- A reduced MNIST subset (10% of training data) used to spread out the range of achievable
  error/certificate values for diagnostic correlation plots.
- Distributions over weights: diagonal Gaussian and diagonal Laplace, with closed-form
  per-coordinate KL (summed over coordinates by independence).
- Both data-free priors (centered at random initialization) and data-dependent priors
  (prior mean trained by ERM on a 50% split, with dropout used only while learning the prior
  to avoid overfitting it). Posterior always initialized at the prior (both `mu` and `rho`).
- Optimizer: vanilla SGD with momentum; identical weight initialization, prior, optimizer,
  and architecture across the objectives being compared. Confidence `delta = 0.025` for the
  bound, `delta' = 0.01` for the Monte-Carlo tail; `m` on the order of 1000 weight samples for
  the empirical-risk estimate.
- Metrics that exist as the yardstick: the **risk certificate** (the PAC-Bayes-kl upper bound
  on the 0-1 loss of the stochastic predictor, evaluated via kl-inversion — *lower is
  tighter*), plus the empirical 0-1 risk, the KL divergence, a cross-entropy-style bound, and
  the test error of the stochastic / deterministic / ensemble predictors.

## Code framework

The bound machinery plugs into an existing probabilistic-network training harness. A
probabilistic layer already holds a distribution over its weights and biases (a `mu` and a
`rho` per weight, `sigma = softplus(rho)`); the network exposes a stochastic forward pass
`net(data, sample=True, clamping=True, pmin=pmin)` that draws fresh weights and returns
log-probabilities; `net.compute_kl()` sums the closed-form per-layer `KL(Q || Q0)`;
Monte-Carlo loops over the stochastic forward pass estimate empirical 0-1 and bounded
cross-entropy risks; and `inv_kl(q, c)` numerically inverts the binary KL by bisection. The
standard bounded cross-entropy is available through `F.nll_loss` on log-probabilities
clamped at `log(pmin)`.

What is *not* settled is the scalar upper-bound formula, the differentiable training
objective, and the final certificate-evaluation procedure. That is the single empty slot.

```python
import math
import torch
import torch.nn.functional as F


class BoundObjective:
    """Owns the bound machinery for a probabilistic network."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        # TODO: any loss-scaling constant needed by the bounded surrogate.

    def compute_bound(self, empirical_risk, kl, n, delta):
        # TODO: fill in the scalar upper-bound expression.
        pass

    def train_step(self, net, data, target, n_posterior, delta):
        # TODO: draw a stochastic forward pass, form the bounded surrogate
        #       empirical loss and KL, and return a scalar to minimize.
        pass

    def compute_risk_certificate(self, net, input=None, target=None, data_loader=None,
                                 delta=0.025, delta_test=0.01, mc_samples=1000):
        # TODO: estimate empirical risks by stochastic forward passes, get the
        #       KL, and produce numerical certificate values.
        pass


# existing harness the BoundObjective plugs into
def compute_losses(net, data, target, clamping=True):
    ...

def inv_kl(q, c):
    ...

def mc_sampling(net, input=None, target=None, data_loader=None):
    ...
```
