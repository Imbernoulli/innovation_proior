## Research Question

Direct policy search asks for a good controller by adjusting the parameters of a stochastic policy, not by first building a value function and acting greedily from it. The attraction is clear in large or partially observed control problems: a small parameter change can make a small change in the policy, stochastic policies remain available, and convergence arguments can be phrased for the policy parameters themselves.

The weak point is the update direction. The conventional rule follows the raw derivative of long-run reward with respect to the chosen coordinates. That rule silently treats the parameter vector as a flat Euclidean object. If the same policy class is reparameterized, the derivative transforms as a covector, while the applied parameter displacement is being used as a tangent vector. Two coordinate descriptions of the same policy can therefore produce different physical moves in policy space. A usable replacement must make the step a property of the policy being changed, while still allowing the direction to be estimated from rollouts.

## Background

In the average-reward setting, a finite Markov decision process has state set `S`, action set `A`, transition law `P`, reward `R`, and a stochastic policy `pi(a|s)`. Under an ergodicity assumption, each policy has a stationary distribution `rho^pi(s)`. The objective is the average reward

`eta(pi) = sum_{s,a} rho^pi(s) pi(a|s) R(s,a)`.

The differential state-action value `Q^pi(s,a)` measures the expected excess reward after starting from `(s,a)` and then following the policy. The advantage subtracts the state's policy-average value, so only relative action quality remains.

Information geometry had already changed the interpretation of gradients in statistical models. When parameters index probability distributions, the coordinate space inherits a Riemannian metric from the distribution family. With squared length `d theta^T G(theta) d theta`, the steepest direction of a scalar objective is `G(theta)^{-1}` times the ordinary derivative. For probability models, the Fisher information matrix gives a reparameterization-invariant local metric and is the second-order local form of KL divergence. This gives a principled way to ask for a small change in distributions rather than a small change in coordinate labels.

## Baselines

The policy-gradient theorem gives the key direct-search baseline:

`grad eta(theta) = sum_{s,a} rho^pi(s) grad pi(a|s,theta) Q^pi(s,a)`.

The crucial feature is what is absent: no explicit derivative of the stationary state distribution appears. That cancellation makes sample-based gradient estimation possible without differentiating the environment dynamics.

Sutton, McAllester, Singh, and Mansour also show how to use a learned critic without biasing that gradient. A linear critic is compatible when its features are the policy score,

`psi(s,a) = grad log pi(a|s,theta)`.

For Gibbs policies, these score features are the action features centered under the current policy, so the critic is naturally advantage-like. The baseline still moves along the raw policy-gradient direction; compatibility solves estimation bias, not the geometry of the step.

## Tensions

Greedy policy iteration offers a different promise: evaluate a policy, then choose actions that look best under the current value estimates. With exact values this is powerful, but approximate value functions can make greedy changes that are non-monotone and brittle. Smooth policy gradients are safer but can be slow and can get trapped by state-visitation weights and plateaus.

Second-order optimization is also tempting, but the reward Hessian in a control problem is not merely a property of the policy parameterization. It is coupled to values and to how values change as the policy changes. A curvature matrix for the objective may be indefinite far from a maximum, while a metric for policy displacement should remain a valid positive notion of local length. The open slot is a direction that is coordinate-respecting like information geometry, sample-estimable like policy gradients, and still aimed toward the improvement behavior that makes policy iteration attractive.

## Code Framework

```python
import numpy as np


class Policy:
    def __init__(self, theta):
        self.theta = theta

    def prob(self, state, action):
        raise NotImplementedError

    def score(self, state, action):
        # grad_theta log pi(action | state, theta)
        raise NotImplementedError


def estimate_q(policy, mdp):
    raise NotImplementedError


def policy_gradient(policy, mdp, q_value):
    grad = np.zeros_like(policy.theta)
    for state in mdp.states:
        rho = mdp.stationary_prob(state, policy)
        for action in mdp.actions:
            prob = policy.prob(state, action)
            grad += rho * prob * policy.score(state, action) * q_value(state, action)
    return grad


def ascent_direction(policy, mdp, grad):
    # The unresolved problem is how to turn the coordinate derivative into
    # the policy-space direction that should actually be stepped along.
    raise NotImplementedError
```
