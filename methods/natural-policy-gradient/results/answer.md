# The Natural Policy Gradient

## Core Move

A vanilla policy-gradient step treats parameter coordinates as Euclidean:

`Delta theta = alpha grad eta(theta)`.

That is steepest ascent only under the arbitrary metric `G = I`. A policy is a distributional object, so the step should instead be steepest under a metric that measures local change in the policy distributions. For a probability family, that metric is the Fisher information, equivalently the local second-order form of KL divergence.

## Construction

For an ergodic average-reward MDP with differentiable stochastic policy `pi(a|s,theta)`,

`eta(theta) = sum_{s,a} rho^pi(s) pi(a|s,theta) R(s,a)`.

The policy-gradient theorem gives

`g(theta) = grad eta(theta) = sum_{s,a} rho^pi(s) grad pi(a|s,theta) Q^pi(s,a)`.

Define the score

`psi(s,a) = grad log pi(a|s,theta)`.

For each state, use the Fisher metric of the conditional action distribution:

`F_s(theta) = E_{a~pi(.|s,theta)}[psi(s,a) psi(s,a)^T]`.

Average those per-state metrics by the current stationary distribution:

`F(theta) = E_{s~rho^pi}[F_s(theta)]`

`= sum_{s,a} rho^pi(s) pi(a|s,theta) psi(s,a) psi(s,a)^T`.

The natural policy-gradient direction is the metric steepest-ascent direction:

`v = F(theta)^{-1} g(theta)`,

using a pseudoinverse or ridge-regularized solve when the score covariance is singular.

## Compatible Critic Theorem

Let the compatible critic be

`f(s,a;w) = w^T psi(s,a)`.

Fit it by minimizing

`sum_{s,a} rho^pi(s) pi(a|s,theta) (w^T psi(s,a) - Q^pi(s,a))^2`.

The normal equations are

`[sum rho^pi pi psi psi^T] w = sum rho^pi pi psi Q^pi`.

The left side bracket is `F(theta)`. The right side is `g(theta)` because `pi psi = grad pi`. Therefore

`F(theta) w = g(theta)`,

so the compatible critic weights are exactly the natural policy-gradient direction:

`w = v = F(theta)^{-1} g(theta)`.

## Greedy-Improvement Link

For an exponential-family policy

`pi(a|s,theta) proportional to exp(theta^T phi_sa)`,

the compatible score is `psi(s,a) = phi_sa - E_pi[phi_s.]`. The centered term is independent of the action being maximized, so

`argmax_a f(s,a;w) = argmax_a w^T phi_sa`.

After stepping `theta' = theta + alpha v`,

`pi(a|s,theta') proportional to exp(theta^T phi_sa + alpha v^T phi_sa)`.

As `alpha -> infinity`, probability concentrates exactly on actions maximizing the compatible critic. For general smooth policies,

`pi(a|s,theta + alpha v) = pi(a|s,theta)(1 + alpha f(s,a;w)) + O(alpha^2)`.

So the update locally scales action probabilities by the compatible critic's advantage-like value and, in the exponential-family limit, performs the greedy move suggested by approximate policy iteration.

## Not Newton

The Fisher matrix here is not the Hessian of average reward. The reward Hessian contains value-coupled terms:

`grad^2 eta = sum_{s,a} rho^pi(s)(grad^2 pi Q^pi + grad pi grad Q^pi^T + grad Q^pi grad pi^T)`.

The Fisher metric contains only policy score outer products. It is chosen because it measures local policy-distribution displacement and is invariant to reparameterization, not because it approximates the objective curvature. This is why the insight is more than recombining policy gradients with second-order optimization.

## Reference Step

```python
def natural_policy_gradient_step(policy, mdp, q_value, epsilon=1e-2, ridge=1e-8):
    grad = 0.0
    fisher = 0.0
    for s in mdp.states:
        rho = mdp.stationary_prob(s, policy)
        for a in mdp.actions:
            p = policy.prob(s, a)
            psi = policy.score(s, a)
            grad += rho * p * psi * q_value(s, a)
            fisher += rho * p * outer(psi, psi)

    direction = solve(fisher + ridge * I, grad)
    step = sqrt(2 * epsilon / dot(grad, direction))
    policy.theta = policy.theta + step * direction
    return policy
```
