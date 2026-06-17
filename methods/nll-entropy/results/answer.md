# Entropy regularization (the maximum-entropy / NLL-minus-entropy loss), distilled

Entropy regularization augments a fit/reward objective with a term proportional to the
**Shannon entropy of the model's output distribution**, weighted by a small coefficient. It
trains a stochastic policy (or any distribution-emitting model) to do well on its primary
objective while keeping its output distribution spread out, so the optimizer cannot collapse
it prematurely onto a near-deterministic, suboptimal solution. The maximum-entropy variant of
policy-gradient learning built this way is sometimes called **MENT** (maximum-entropy
exploration).

## Problem it solves

A policy made of stochastic units is trained by following the gradient of expected reward
(or, in a supervised/imitation setting, of the data log-likelihood). The randomness that
enables search is itself an optimized parameter, and climbing the objective almost always
sharpens the output distribution — so plain gradient ascent drives the distribution toward a
deterministic output *early*, before it has established that the committed-to choice is best.
Once the distribution is peaked, exploration is gone and the agent is stuck at a local
optimum. The goal: still climb the objective, but keep enough spread alive that better regions
remain reachable.

## Key idea

Maximize the **entropy-regularized objective**

```
J(pi) = E_{a~pi}[ r(a) ] + tau * H(pi),   H(pi) = E_{a~pi}[ -log pi(a) ],   tau > 0,
      = E_{a~pi}[ r(a) - tau * log pi(a) ].
```

The first term fits/rewards; the second penalizes collapse. `tau` is a temperature.

- **The bonus makes the optimization target a Boltzmann distribution.** Maximizing `J` over
  a free distribution `pi` (Lagrange multipliers on `sum pi = 1`) gives
  `pi*_tau(a) = exp(r(a)/tau) / Z`, `Z = sum_a exp(r(a)/tau)`. As `tau -> 0`, `pi*_tau -> `
  a delta on `argmax_a r(a)` (greedy — and `tau = 0` recovers the un-regularized objective);
  as `tau -> infinity`, `pi*_tau -> ` uniform. `tau` slides continuously from exploit to
  explore.
- **It is exactly minimizing KL to that spread-out target.** Substituting the Boltzmann form
  back in, `J(pi) = -tau * D_KL(pi || pi*_tau) + tau * log Z`. Since KL >= 0 with equality
  iff `pi = pi*_tau`, maximizing `J` is minimizing KL to a *full-support* target — which
  cannot collapse to a delta. That is the mechanism that defeats premature collapse.
- **It is adaptive, unlike a fixed noise floor.** The entropy gradient steepens exactly as
  the distribution approaches collapse: for a Gaussian, `H = (1/2) log(2*pi*e*sigma^2)`, so
  `dH/dsigma = 1/sigma -> infinity` as `sigma -> 0`. Near a local optimum the reward gradient
  is flat while the entropy gradient blows up, so the sum points firmly back toward spread,
  precisely where it is needed.

## Gradient

The regularized policy gradient is REINFORCE with an entropy-augmented reward:

```
grad_theta J = E_{a~pi_theta}[ ( r(a) - tau*log pi_theta(a) ) * grad_theta log pi_theta(a) ].
```

The entropy part alone is `grad_theta H(pi_theta) = -E_{a~pi_theta}[ log pi_theta(a) *
grad_theta log pi_theta(a) ]` (using the zero-mean-score identity
`E[grad_theta log pi_theta] = 0`). If the undropped derivative is written as
`r(a) - tau*log pi_theta(a) - tau`, the extra `-tau` is only a constant baseline and has
zero expectation against the score. For a non-reparameterized mixture sample, detached
`-log_prob(sample)` estimates the entropy value but not this gradient; the correct PyTorch
surrogate carries `log_prob.detach()` as the score-function multiplier.

## Entropy estimation

- **Closed form when available.** Gaussian: `H = (1/2) log(2*pi*e*sigma^2)` per dimension.
  Bernoulli: `H = -p*log p - (1-p)*log(1-p)`.
- **Monte-Carlo for anything sampleable** (e.g. a mixture of Gaussians, whose entropy
  `-E[log sum_k w_k N_k]` has no closed form): draw `a^(k) ~ pi_theta`, take
  `H ~= (1/K) sum_k -log pi_theta(a^(k))`. For `MixtureSameFamily`, use `.sample(...)`
  for the value estimate and a score-function surrogate for the backward pass.

## Choosing tau

Small and positive. Too small (`-> 0`): back to premature collapse. Too large: the Boltzmann
target is near-uniform and the model stops fitting the data. A value on the order of `0.01`
keeps just enough spread while letting the fit term dominate where the data is informative.

## Final form for a likelihood / behavioral-cloning objective

When the "reward" is the data log-likelihood, maximizing `J = E[log-lik] + tau*H` is, as a
loss to *minimize*, the negative log-likelihood **minus** the entropy bonus:

```
loss = NLL - alpha * H(pi),   NLL = -E[ log pi_theta(target) ],   alpha = tau.
```

The entropy enters with a **minus** sign because minimizing `-alpha*H` maximizes entropy; a
plus sign would minimize entropy and *cause* the collapse.

## Working code (mixture-output / GMM behavioral cloning)

```python
import torch
import torch.nn as nn


class EntropyRegularizedLoss(nn.Module):
    """NLL of the target plus a maximum-entropy bonus on the predicted distribution.
    Minimizing (nll - alpha*H) maximizes  log-likelihood + alpha*H(pi): the fit term
    concentrates the distribution while the entropy term keeps it from collapsing to a
    near-deterministic solution. `dist` is any distribution exposing .log_prob and
    .sample (e.g. torch.distributions.MixtureSameFamily, a GMM over actions)."""

    def __init__(self, alpha=0.01, num_entropy_samples=1):
        super().__init__()
        self.alpha = alpha                                # entropy weight (temperature tau)
        self.num_entropy_samples = num_entropy_samples

    def forward(self, dist, target):
        # Fit term: NLL = -(log-likelihood of the target under the predicted distribution).
        nll = -dist.log_prob(target).mean()

        # Entropy bonus H(pi) = E_{a~pi}[ -log pi(a) ], estimated by Monte-Carlo because a
        # mixture has no closed-form entropy. MixtureSameFamily.sample is not reparameterized,
        # so the backward pass uses the score-function identity:
        # grad H = -E[log pi(a) * grad log pi(a)].
        with torch.no_grad():
            samples = dist.sample((self.num_entropy_samples,))
        log_prob = dist.log_prob(samples)
        entropy_value = -log_prob.mean()
        entropy_score_grad = -(log_prob.detach() * log_prob).mean()
        entropy = entropy_value.detach() + entropy_score_grad - entropy_score_grad.detach()

        # Subtract: minimizing nll - alpha*H maximizes H. (A plus sign would *minimize*
        # entropy and drive the collapse this is meant to prevent.)
        return nll - self.alpha * entropy
```

For a single-Gaussian output, the Monte-Carlo block can be replaced by the closed form
`entropy = 0.5 * (math.log(2*math.pi) + 1 + 2*log_sigma).sum(-1).mean()` (i.e.
`H = (1/2) log(2*pi*e*sigma^2)` per dimension), which is exact and lower-variance; the
sampling estimator above is the general fallback used when the distribution (such as a GMM)
has no closed-form entropy.
