## Research Question

I need samples from a continuous target density `pi(q)` on `R^d`, usually a Bayesian posterior or a statistical-mechanics density. The normalizing constant is unknown, but pointwise log density is available up to an additive constant, and the gradient of that log density can be computed almost everywhere. The numerical goal is to estimate expectations

```text
E_pi[f] = integral f(q) pi(q) dq
```

by averaging over a Markov chain whose stationary distribution is `pi`.

The integral is high dimensional. In high dimension, probability mass is concentrated in a narrow typical region that balances density against volume: the mode alone is not representative, and the far tails have large volume but tiny density. The question is how to build a Markov transition for `pi` that moves efficiently through that typical region using the available log density and its gradient.

## Background

The canonical-distribution view turns density evaluation into energy evaluation. If a density is proportional to `exp(-U(q))`, then adding any constant to `U` changes only the unknown normalizer, not the distribution. This is why a Markov-chain method can work from energy differences or density ratios rather than from a normalized density.

Classical mechanics supplies a vocabulary for coherent motion. A mechanical state has positions `q` and momenta `p`, and a total energy

```text
H(q, p) = U(q) + K(p)
```

where `U` is potential energy and `K` is kinetic energy. The associated equations of motion preserve the total energy, are reversible, and preserve phase-space volume.

Numerical simulation of these equations is done with discretized integrators. Existing mechanics practice favors reversible, volume-preserving discretizations, which keep numerical energy error controlled over a simulated trajectory.

## Baselines

The classical local-move sampler starts at a configuration, proposes a symmetric random displacement, and accepts the move when it lowers the energy. If the proposed energy is higher by `Delta E`, the move is accepted with probability `exp(-Delta E / T)`. Rejected configurations are counted again. This gives the desired canonical distribution because opposing proposal flows between two states are balanced by the acceptance rule.

The more general Markov-chain correction allows an arbitrary proposal density `Q(q' | q)` as long as the reverse proposal density is known. The acceptance probability has the form

```text
min(1, pi(q') Q(q | q') / (pi(q) Q(q' | q)))
```

so unknown normalizing constants cancel. A proposal can be complicated or directed, provided the reverse move and any volume change can be accounted for.

Simple random proposals move by diffusion. To keep acceptance high, the proposal scale is set comparable to the narrowest direction of the typical region, and motion along broad directions accumulates as a random walk over accepted moves.

Coordinate-wise conditional updates apply when conditionals are available, updating the joint space one local conditional adjustment at a time.

## Evaluation Settings

The natural controlled tests are smooth continuous densities where both the density and gradient are available: correlated multivariate Gaussians, anisotropic Gaussians with one narrow and one broad direction, and higher-dimensional independent Gaussian products with a range of scales. These reveal whether a transition diffuses slowly or moves coherently across broad directions.

In real statistical applications, the same setting appears in posterior simulation for models with many continuous parameters. The transition is judged by effective sample size per gradient evaluation, acceptance probability, autocorrelation time, robustness to scale differences, and whether independent chains explore the same typical region.

## Code Framework

The implementation starts with a target energy and its gradient. The open slot is the transition rule used once per Markov-chain iteration.

```r
U = function (q)
  stop("supply potential energy")

grad_U = function (q)
  stop("supply gradient of potential energy")

transition = function (U, grad_U, step_size, n_steps, current_q)
{
  # TODO: the update rule we'll define
  current_q
}

sample_chain = function (U, grad_U, step_size, n_steps, q0, n_iter)
{
  q = q0
  chain = matrix(NA, nrow = n_iter + 1, ncol = length(q0))
  chain[1, ] = q
  for (i in 1:n_iter)
  {
    q = transition(U, grad_U, step_size, n_steps, q)
    chain[i + 1, ] = q
  }
  chain
}
```
</content>
</invoke>
