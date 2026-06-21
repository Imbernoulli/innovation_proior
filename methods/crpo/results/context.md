# Context

## Constrained Control Objective

A reinforcement-learning agent is no longer judged only by expected reward. Each policy also induces
one or more expected discounted cost returns, and those costs must stay under prescribed limits:

```text
max_pi J_0(pi)    subject to    J_i(pi) <= d_i,  i = 1, ..., p.
```

Here `J_0` is the discounted reward return, `J_i` is the discounted return for cost signal `c_i`, and
`d_i` is the corresponding budget. The feasible set is `Omega_C = {pi : J_i(pi) <= d_i for every i}`,
and the target is the reward-maximizing policy inside that set. In a parameterized policy class, both
the reward objective and the constraints can be nonconcave or nonconvex, even though the underlying
occupancy-measure view has a linear-programming structure.

## Standard Safe-RL Machinery

The common approach is to introduce nonnegative multipliers and optimize a Lagrangian such as
`J_0(pi) - sum_i lambda_i (J_i(pi) - d_i)`. The policy parameters are updated by a policy-gradient,
natural-gradient, or trust-region step on the Lagrangian, while the multipliers are projected back to
the nonnegative orthant after being increased by observed constraint violation. This family includes
actor-critic and policy-gradient primal-dual methods, and its appeal is supported by CMDP duality
results showing that constrained reinforcement learning can have zero duality gap under suitable
representations. Implementations carry dual learning rates, multiplier initialization choices, and
projection thresholds or normalizations.

## Available Optimization Building Blocks

Unconstrained policy-optimization methods provide local update rules and, in tabular or
well-parameterized settings, finite-time global-convergence analyses. Natural policy gradient has a
simple tabular form: for each reward or cost return `J_i`, the natural-gradient direction is proportional
to the corresponding action-value function, with factor `(1 - gamma)^(-1)`. TRPO and related methods can
be viewed as adaptive-step versions of the same natural-gradient idea, and ordinary actor-critic code can
estimate reward and cost advantages from the same rollout batch.

## Research Question

The setting asks how to drive a parameterized policy toward the reward-maximizing point of the feasible
set `Omega_C`, using the same kind of reward and cost action-value estimates that an actor-critic rollout
already produces, across one or more constraints and from an arbitrary starting policy.
