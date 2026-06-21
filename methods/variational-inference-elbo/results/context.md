## Posterior Wall

A probabilistic model can make the joint `p(x, z)` cheap while making the posterior
`p(z | x)` hard. The numerator is a product of local factors from the model. The denominator
is the evidence `p(x)`, a sum or integral over all hidden configurations. In discrete models it
can be exponentially large; in continuous models it can be a high-dimensional integral with no
closed form.

The graphical-model setting makes this concrete. Exact inference can exploit conditional
independencies by building a junction tree, but its complexity is exponential in the largest
clique. Dense networks and explaining-away effects can create cliques too large to handle. A
medical-diagnosis network with hundreds of disease variables and thousands of findings already
forces clique sizes on the order of hundreds.

## Existing Tools

Exact inference returns normalized probabilities and evidence values by exploiting conditional
independencies in the graphical structure. When the graph forces large cliques, the exact
algorithm pays the exponential cost.

Monte Carlo methods construct a Markov chain whose stationary distribution is the desired
posterior and estimate expectations from samples. They are broadly applicable and
asymptotically exact.

Expectation-Maximization provides a useful precedent for latent-variable models with
parameters. It alternates between a distribution over hidden variables and a parameter update,
and the free-energy view rewrites the likelihood objective as
`E_Q[log p(x,z | theta)] + H(Q)` plus a nonnegative gap. The exact E step computes
the posterior over hidden variables given current parameters.

## Bound Ingredients

The Kullback-Leibler divergence measures the discrepancy between two distributions on the
same variables:

`KL(q || p) = E_q[log q(z) - log p(z)]`.

It is nonnegative and asymmetric. An expectation under a chosen distribution `q` can be
tractable, while an expectation under the unknown posterior is not.

Jensen's inequality supplies a lower bound on the log of an average. If a positive quantity is
written as an expectation under a chosen distribution, then `log E_q[Y] >= E_q[log Y]`.
Convex duality gives the same kind of handle for log-sum-exp: its conjugate is the negative
entropy term on the probability simplex.

## Mean-Field Clue

Statistical physics had already shown one way to make coupled binary systems tractable.
Instead of tracking every joint configuration of interacting variables, replace the neighbors of
each variable by their average field and solve deterministic fixed-point equations for the
individual means.

In Boltzmann-machine learning this replaces stochastic measurements of correlations with
deterministic mean-field equations. A complicated joint distribution may be replaced by a
factorized distribution whose parameters are solved self-consistently.

## Code Framework

The reusable computational pieces before the final idea are a cheap log joint, a hidden-variable
family to be designed, and an outer loop with a scalar progress measure.

```python
import numpy as np


def log_joint(x, z, params):
    """Pointwise log p(x,z), implemented from local model factors."""
    raise NotImplementedError


def log_evidence_exact(x, params):
    """The unavailable normalization log p(x)."""
    raise NotImplementedError


class ApproximatePosterior:
    def __init__(self, params):
        self.params = params

    def step(self, x):
        raise NotImplementedError

    def score(self, x):
        raise NotImplementedError


def fit(x, params, max_iters=200, tol=1e-6):
    q = ApproximatePosterior(params)
    previous = -np.inf
    for _ in range(max_iters):
        q.step(x)
        value = q.score(x)
        if value < previous - 1e-8:
            raise FloatingPointError("progress measure decreased")
        if abs(value - previous) < tol:
            break
        previous = value
    return q
```
