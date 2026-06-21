# LSVI-UCB

## Method

Assume a finite-horizon MDP with known features `phi(x,a) in R^d` and a linear MDP structure:

`P_h(.|x,a)=<phi(x,a),mu_h(.)>`, `r_h(x,a)=<phi(x,a),theta_h>`,

with `||phi(x,a)||<=1` and `max{||theta_h||, ||mu_h(S)||}<=sqrt(d)`. This is a structural assumption on the reward and transition kernel, not a linear-policy assumption.

The Bellman factorization gives, for every policy `pi`,

`Q_h^pi(x,a)=<phi(x,a), theta_h + int V_{h+1}^pi dmu_h>`.

So the action-value functions are linear in `phi`, and a Bellman backup can be estimated by ridge regression from sampled transitions.

## Algorithm

For episode `k`, run a backward least-squares value-iteration sweep, then act greedily:

```text
Input: feature map phi, lambda=1, beta=c d H sqrt(log(2dT/p))

for episode k = 1,...,K:
    receive x_1^k

    for h = H,...,1:
        Lambda_h = sum_{tau<k} phi_h^tau (phi_h^tau)^T + lambda I
        w_h = Lambda_h^{-1} sum_{tau<k} phi_h^tau
              [ r_h^tau + max_a Q_{h+1}(x_{h+1}^tau,a) ]

        Q_h(x,a) = min{
            <w_h, phi(x,a)>
            + beta sqrt(phi(x,a)^T Lambda_h^{-1} phi(x,a)),
            H
        }

    for h = 1,...,H:
        a_h^k = argmax_a Q_h(x_h^k,a)
        receive r_h^k and observe x_{h+1}^k
```

The ridge term makes `Lambda_h` invertible and bounds the capacity of the value class. The bonus is the self-normalized elliptical uncertainty width: it is large along feature directions with few effective samples and small along well-observed directions. The canonical implementation artifact is the algorithm pseudocode itself; no official code release was found.

## Theorem

With `lambda=1`, `beta=c d H sqrt(iota)`, and `iota=log(2dT/p)`, the algorithm satisfies, with probability at least `1-p`,

`Regret(K) = O(sqrt(d^3 H^3 T iota^2)) = tilde O(sqrt(d^3 H^3 T))`,

where `T=KH`. The bound is independent of the cardinality of the state space. With finite-action maximization, the direct implementation has runtime `O(d^2 A K T)` and space `O(d^2 H + d A T)`.

For an approximately linear MDP with total-variation transition error and reward error at most `zeta`, using

`beta_k=c(d sqrt(iota)+zeta sqrt(kd))H`

gives, with the same `iota`,

`Regret(K)=O(sqrt(d^3 H^3 T iota^2) + zeta d H T sqrt(iota))`.

The approximate case is not exact optimism: the guarantee is `Q_h^k >= Q_h^* - 4H(H+1-h)zeta`, so at `h=1` the regret proof pays a deterministic `4H^2 zeta` per episode in addition to the variable-bonus term.

## Why The Proof Works

The key deviation relation is

`|<phi,w_h^k> - Q_h^pi - P_h(V_{h+1}^k - V_{h+1}^pi)| <= beta ||phi||_{(Lambda_h^k)^{-1}}`.

Its proof has three load-bearing parts.

First, the linear transition model turns the Bellman recursion term into a linear object. This is why the regression coefficient can represent a backed-up value function even when the state space is infinite.

Second, the stochastic term is not a fixed-function regression noise term. The target contains `V_{h+1}^k`, which is learned from the same data. This is handled by uniform self-normalized concentration over the algorithm's value class

`V(.)=min{max_a [phi(.,a)^T w + beta sqrt(phi(.,a)^T Lambda^{-1} phi(.,a))],H}`.

The bonus can be written as `sqrt(phi^T A phi)` with `A=beta^2 Lambda^{-1}`. Covering this matrix parameter costs `d^2`, which is why the confidence radius has scale `dH`, not just `sqrt(d)H`.

Third, optimism propagates backward through Bellman induction. If `V_{h+1}^k >= V_{h+1}^*`, then `P_h(V_{h+1}^k - V_{h+1}^*) >= 0`, so the recursive term helps the upper-confidence bound. This keeps the horizon dependence polynomial rather than exponential.

## Why It Is More Than Regression Plus UCB

The method combines two ideas only because the linear MDP factorization makes them compatible:

- Least-squares value iteration supplies Bellman regression targets.
- Linear-bandit confidence geometry supplies the bonus `sqrt(phi^T Lambda^{-1} phi)`.
- Uniform concentration over a data-dependent value class justifies applying that bonus to learned future values.
- Bellman induction turns local confidence intervals into global optimism.
- The elliptical-potential lemma bounds the total exploration cost across episodes.

Without the transition factorization, the Bellman backup would not remain linear. Without the uniform concentration step, the learned future value in the regression target would invalidate the fixed-function self-normalized argument. Without recursive optimism, a stagewise bandit view would compound uncertainty through the horizon.

The regret proof finally telescopes the optimistic value gap into a martingale term plus cumulative bonuses. Azuma controls the martingale term, and the elliptical-potential lemma gives `sum_k phi_h^k^T (Lambda_h^k)^{-1} phi_h^k = O(d log T)` for each step `h`. Multiplying the width sum `H sqrt(dK)` by `beta=dH sqrt(iota)` yields `tilde O(sqrt(d^3 H^3 T))` after using `T=KH`.
