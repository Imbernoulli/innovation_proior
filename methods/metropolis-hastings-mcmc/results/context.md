# Context

## Research Question

Compute expectations against a high-dimensional probability law when the density can be evaluated only up to an unknown constant. The statistical-mechanics version is the canonical ensemble average

`E(F) = int F(x) exp(-E(x)/kT) dx / int exp(-E(x)/kT) dx`,

where `x` is the configuration of many interacting particles. The momentum variables factor out for velocity-independent forces, leaving a configuration integral in hundreds of dimensions. The denominator is the partition function, another high-dimensional integral, so the practical input is an unnormalized weight such as `exp(-E(x)/kT)`, not the normalized probability density.

The computational target is a way to estimate averages using local quantities that can be evaluated on configurations: energies, energy differences, proposed local displacements, and observable values. The same shape appears outside physics whenever a posterior or likelihood-weighted distribution is available only up to a multiplicative constant.

## Existing Tools

Monte Carlo integration replaces grids by random samples. Its root-mean-square error falls like `1/sqrt(N)` rather than exploding directly with the number of dimensions, which makes it attractive when deterministic quadrature is hopeless.

The canonical ensemble says that important configurations have probability proportional to a Boltzmann weight. In a dense particle system, arbitrary configurations usually put particles too close together, causing extremely high energy and nearly zero weight. The useful configurations occupy a very small part of the raw configuration volume.

Importance sampling draws from a more convenient density `q` and corrects by `p/q`. It can be powerful when `q` tracks the target well. Uniform Monte Carlo over configuration space uses

`sum_i F(x_i) exp(-E(x_i)/kT) / sum_i exp(-E(x_i)/kT)`.

The normalizer cancels, and the effective sample is concentrated on the useful region.

Finite Markov chains have stationary distributions. If an irreducible chain has transition matrix `P` and stationary row vector `pi`, then `pi = pi P`, and long-run averages along the chain estimate expectations under `pi`. This gives a different kind of sampling object: correlated draws whose limiting distribution is controlled by the transition rule.

Acceptance-rejection sampling proposes a candidate from a dominating density and keeps it with a probability involving the target-to-envelope ratio. It gives independent samples when a global envelope is available.

Time averaging by integrating the equations of motion is another physical route to ensemble averages; it follows the detailed coupled dynamics for long simulated times.

## Evaluation Setting

A natural testbed is a periodic box of interacting particles. Periodic boundaries reduce surface effects so a few hundred particles can represent a bulk phase. The hard-disk case strips the interaction down to excluded volume: a proposed configuration is valid unless disks overlap.

The observable can be the equation of state, obtained through a contact-density or virial calculation. Reference behavior is available from free-volume theory at high density and virial expansions at low density. Useful diagnostics include running averages, autocorrelation, whether different run segments agree, and how proposed-move scale affects movement through configuration space.

The sampler itself is judged by whether retained configurations behave as if they came from the intended weight and whether the run explores all important regions in feasible time.

## Code Frame

The available ingredients are an unnormalized log weight, a random-number generator, a candidate generator, and an averaging loop. The unresolved design point is the transition rule.

```python
import numpy as np

def log_weight(state):
    """Return an unnormalized log density such as -E(state) / (k*T)."""
    ...

def candidate_rule(state, rng):
    """Return a tentative next state from the current state."""
    ...

def transition(state, log_weight, candidate_rule, rng):
    """Return the next state of a chain whose long-run averages are useful."""
    # TODO: fill in the transition mechanism
    pass

def sample(state0, log_weight, candidate_rule, n_steps, rng):
    state = state0
    states = []
    for _ in range(n_steps):
        state = transition(state, log_weight, candidate_rule, rng)
        states.append(np.copy(state))
    return states

def estimate(states, observable):
    return np.mean([observable(x) for x in states], axis=0)
```
