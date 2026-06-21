# Context: finite Markov decision planning by optimization

## Research Question

Given a finite Markov decision process with states `S`, actions `A`, transition probabilities `P(s'|s,a)`, one-step rewards `r(s,a)`, an initial distribution `mu`, and discount `gamma in [0,1)`, find a stationary policy `pi(a|s)` that maximizes

```text
J(pi) = E[ sum_{t>=0} gamma^t r(s_t,a_t) | s_0 ~ mu, pi ].
```

The question is how to formulate this planning problem as a finite mathematical program.

## Background

A stationary policy maps each state to a distribution over actions. Once the policy is fixed, the controlled process becomes an ordinary Markov chain with transition matrix

```text
P_pi(s,s') = sum_a pi(a|s) P(s'|s,a).
```

The value of that policy satisfies the linear evaluation equation

```text
V^pi(s) = sum_a pi(a|s)[ r(s,a) + gamma sum_{s'} P(s'|s,a)V^pi(s') ].
```

The optimal value satisfies the Bellman optimality equation

```text
V*(s) = max_a [ r(s,a) + gamma sum_{s'} P(s'|s,a)V*(s') ].
```

The Bellman operator is monotone and is a `gamma`-contraction in max norm, so this equation has a unique fixed point.

Markov chains also have a balance viewpoint. In a stationary chain, probability mass flowing into a state equals mass flowing out of that state. In a discounted process, the same idea must be modified: mass is injected from the initial distribution and then discounted as it flows through future transitions.

## Baselines

Value iteration applies the Bellman optimality operator repeatedly. It is simple and globally convergent.

Policy iteration alternates between solving a linear system for the current policy and taking a greedy improvement step. It often converges quickly.

Direct policy search optimizes over `pi(a|s)`. In the tabular case, the return depends on `(I - gamma P_pi)^-1` as a function of the policy parameters.

The value-function inequality relaxation replaces the max equation by one inequality per action, giving linear constraints in `V`.

## Evaluation Setting

The clean test case is a fully known finite MDP. The transition matrix and rewards are small enough that dynamic programming can provide a reference solution, but the formulation should not depend on enumerating all histories or all policies.

A classic operations-research instance is a finite inventory system. The state is stock on hand at the beginning of a period, the action is a bounded production quantity, demand is random, and the next stock level is determined by stock plus production minus demand. The objective may be discounted reward or long-run average cost. The useful formulation should tolerate arbitrary tabulated costs, not only convex or quadratic costs.

The metric is exact agreement with dynamic programming in the finite model: the optimization program should attain the same optimal start value and should produce an optimal stationary policy.
