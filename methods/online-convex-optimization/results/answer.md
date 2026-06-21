# Online Convex Optimization And Online Gradient Descent

## Method

Given a fixed closed bounded convex feasible set `F`, choose `x^1 in F`. On
round `t`, play `x^t`. After the convex cost `c^t` is revealed, compute
`g^t = grad c^t(x^t)` and update

`x^{t+1} = P(x^t - eta_t g^t)`,

where `P(y) = argmin_{x in F} ||x - y||` is Euclidean projection onto `F`.
Zinkevich calls this Greedy Projection; it is the standard Online Gradient
Descent update. A subgradient can replace the gradient.

## Guarantee

Let

- `D = max_{x,y in F} ||x - y||`,
- `G = sup_{t,x in F} ||grad c^t(x)||`,
- `R(T) = sum_{t=1}^T c^t(x^t) - min_{x in F} sum_{t=1}^T c^t(x)`.

For nonincreasing positive step sizes,

`R(T) <= D^2/(2 eta_T) + (G^2/2) sum_{t=1}^T eta_t`.

With `eta_t = t^{-1/2}`, Zinkevich's exact form is

`R(T) <= (D^2/2) sqrt(T) + (sqrt(T) - 1/2) G^2`.

If `D,G > 0`, the scaled anytime schedule `eta_t = D/(G sqrt(t))` gives

`R(T) <= (3/2) D G sqrt(T)`.

With known horizon `T` and constant `eta = D/(G sqrt(T))`,

`R(T) <= D G sqrt(T)`.

If `D = 0` or `G = 0`, the learner and the best fixed comparator incur the same
cost each round, so regret is zero.

Thus average regret goes to zero for every adversarial sequence of convex costs
with bounded gradients.

## Proof Kernel

Fix any comparator `x* in F`. Convexity gives

`c^t(x^t) - c^t(x*) <= g^t dot (x^t - x*)`.

So it is enough to bound the regret for the linearized losses. Let
`y^{t+1} = x^t - eta_t g^t`. Expanding the squared distance to `x*`,

`||y^{t+1} - x*||^2 = ||x^t - x*||^2 - 2 eta_t g^t dot (x^t - x*) + eta_t^2 ||g^t||^2`.

Projection onto a convex set is non-expansive toward feasible points, so
`||x^{t+1} - x*|| <= ||y^{t+1} - x*||`. Therefore

`g^t dot (x^t - x*) <= (||x^t - x*||^2 - ||x^{t+1} - x*||^2)/(2 eta_t) + (eta_t/2)G^2`.

For nonincreasing `eta_t`, Abel summation bounds the potential part by
`D^2/(2 eta_T)`, giving the regret bound. The step-size choice balances that
term against `(G^2/2) sum_t eta_t`.

## Consequences

- Expert advice is the special case where `F` is the probability simplex and
  costs are linear. The method competes with the best fixed expert or fixed
  distribution using Euclidean geometry rather than the entropic geometry of
  multiplicative weights.
- Repeated games use the ascent version on utility, or equivalently descent on
  loss `-u`. On a mixed-strategy simplex this is Generalized Infinitesimal
  Gradient Ascent and gives no external regret for any number of actions.
- The `sqrt(T)` order is tight in the worst case. Random sign linear losses on
  the hypercube force `Omega(D G sqrt(T))` regret for any online algorithm.
- A moving comparator sequence with path length at most `L` satisfies, for fixed
  `eta`, the dynamic-regret bound
  `R(T,L) <= 7D^2/(4 eta) + LD/eta + T eta G^2/2`.

## Artifact

The method-local implementation artifact is
`methods/online-convex-optimization/code/online_gradient_descent.py`; its
`reveal_cost(t, x_t)` callback is called only after `x_t` has been committed, so
the code follows the online timing in the source pseudocode. The theorem/proof
artifact is
`methods/online-convex-optimization/refs/final_artifact/online_gradient_descent_theorem.md`.
