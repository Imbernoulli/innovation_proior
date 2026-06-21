## Research question

A learner receives an i.i.d. sample `S = (z_1, ..., z_n)` from an unknown distribution
and returns a distribution `Q` over classifiers rather than a single classifier. At prediction
time a classifier is drawn from `Q`, so the natural empirical quantity is the average
training loss `E_{h~Q}[r(h)]`, where `r(h) = (1/n) sum_i l(h,z_i)`. The quantity to certify is
the corresponding true loss `E_{h~Q}[R(h)]`, with `R(h) = E_z[l(h,z)]`.

The distribution `Q` is chosen after seeing `S`, so a single-hypothesis concentration
inequality does not apply to the returned predictor. A useful certificate must hold with
high probability over the sample draw and simultaneously for every posterior distribution
the learner might choose.

The central design problem is: how should empirical risk, a fixed reference
distribution `P`, the divergence from `Q` to `P`, the bound sample size, and the confidence
level combine into a valid upper bound on the true risk of the randomized predictor?

## Background

The classical PAC setup for a fixed hypothesis `h` and bounded loss `l(h,z) in [0,1]`
uses Hoeffding's inequality to give a high-probability upper bound on `R(h)` in terms of
`r(h)`. This is a statement about a hypothesis fixed before the sample is drawn.

A finite hypothesis class can be handled by a union bound: make the fixed-`h` statement
hold for every `h` at once and pay a `log |H|` penalty. For a continuously parameterized
classifier, or for a neural network with many real-valued weights, this penalty reflects
the size of the whole class rather than the complexity of the returned solution.

The alternative object is a randomized classifier. Fix a reference distribution `P` over
hypotheses before the bound-evaluation data are seen, and allow the learner to choose a
sample-dependent posterior `Q`. The reference is often called a prior, but no Bayesian data
model is required. It is simply the distribution with respect to which the learner's
specialization will be measured. The natural complexity quantity is the relative entropy
`KL(Q || P)`: it is zero if the posterior stays at the reference, grows as `Q` concentrates
away from `P`, and becomes infinite if `Q` puts mass where `P` assigns none.

The key technical tool available in this setting is the change-of-measure, or
Donsker-Varadhan, inequality:

```text
E_{h~Q}[phi(h)] <= KL(Q || P) + log E_{h~P}[exp(phi(h))].
```

It converts an exponential-moment statement under the fixed reference `P` into a statement
about an arbitrary posterior `Q`, paying exactly the `KL(Q || P)` transport cost.

## Baselines

The single-hypothesis Hoeffding bound is the simplest baseline. It is valid for a hypothesis
chosen independently of the sample, and its penalty scales like `sqrt(log(1/delta)/n)`.

The finite-class Occam or union-bound baseline repairs adaptivity by holding uniformly over
all hypotheses. Its complexity penalty is the description length or log prior weight of a
single chosen classifier. This is the right shape for countable models and finite-precision
rules.

Linear change-of-measure certificates use an exponential moment for the raw gap
`R(h)-r(h)` and introduce a temperature parameter. They replace `log |H|` by a
posterior-to-reference divergence.

Implicit relative-entropy certificates keep the Bernoulli geometry of empirical and true
risk rather than immediately relaxing to a raw difference. They can be much tighter when
empirical risk is small, and the final risk is recovered by inverting a binary-KL constraint.

Direct bound optimization for neural networks adds a practical constraint: the posterior
over weights is parameterized, the reference distribution must be placed before evaluating
the bound data, and the training loss used inside the certificate has to be bounded in
`[0,1]`. Cross-entropy is differentiable but unbounded; 0-1 loss is bounded but has no useful
gradient.

## Evaluation settings

The intended experimental setup is a stochastic neural-network certificate on image
classification data.

- Datasets: MNIST and FashionMNIST, with standard normalization.
- Architectures: a fully connected network `784-600-600-600-10` and a compact two-conv,
  two-linear convolutional network.
- Reference placement: split the training data. One part trains a deterministic network
  whose weights initialize the reference mean; the other part is the bound-evaluation set
  of size `n`, which the reference must not have used.
- Posterior family: independent Gaussian weights with learned posterior means and scales
  and fixed reference scales.
- Reported metrics: the 0-1 risk certificate, empirical 0-1 risk on the bound set,
  divergence `KL(Q || P)`, a differentiable surrogate-bound value, and deterministic test
  error of the posterior mean.
- Sampling issue: the empirical risk of `Q` is itself an expectation over random weights, so
  any Monte Carlo estimate of that expectation needs its own confidence correction before it
  is inserted into the outer generalization certificate.

## Code framework

The surrounding harness already supplies stochastic Gaussian layers, data splitting for the
reference and bound-evaluation sets, analytic per-layer Gaussian KLs, a binary-KL inversion
utility, stochastic forward passes, and the training loop. The missing component is the
bound optimizer: the function that combines empirical risk and KL, the differentiable
training objective for the posterior, and the final certificate evaluation.

```python
import math
import torch
import torch.nn.functional as F


# --- already-existing primitives the slot may call ---
# get_total_kl(model)             -> sum of per-layer KL(Q||P)
# inv_kl(q, c)                    -> largest p in [q,1] with kl_bernoulli(q || p) <= c
# model(data, sample=True)        -> stochastic forward pass with h ~ Q
# F.nll_loss / F.log_softmax      -> surrogate-loss machinery


class BoundOptimizer:
    """Fill the certificate, posterior-training objective, and final read-off."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin

    def compute_bound(self, empirical_risk, kl, n, delta):
        # TODO: combine empirical_risk, KL(Q||P), n, and delta into a
        # high-probability true-risk upper bound that holds for every posterior Q.
        pass

    def train_step(self, model, data, target, device, n_bound, delta):
        # TODO: choose a bounded differentiable empirical-risk surrogate, combine it
        # with the KL term, and return the scalar objective for backpropagation.
        pass

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        # TODO: estimate the stochastic predictor's empirical 0-1 risk, account for
        # Monte Carlo error, combine it with KL, and return the reported certificate.
        pass
```
