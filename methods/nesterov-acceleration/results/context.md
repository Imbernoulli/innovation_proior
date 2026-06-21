# Context

## Research Question

I need to minimize a convex differentiable function using only a first-order oracle. At a query point I can read the function value and gradient; I cannot use Hessians, exact global structure, or a solver for the whole problem. The target class is smooth convex optimization: the gradient is Lipschitz with constant `L`, so every point supplies both a tangent lower bound by convexity and a quadratic upper model by smoothness.

The practical question is an oracle-complexity question. How many gradient calls are needed, in the worst case over all smooth convex functions, to guarantee `f(x_k)-f* <= eps`? In the strongly convex subcase, with curvature parameter `mu` and condition number `kappa=L/mu`, the analogous question is how the logarithmic convergence constant should scale with `kappa`.

## Geometry In Hand

Convexity gives a global first-order lower model:

`f(x) >= f(y) + <grad f(y), x-y>`.

If the function is also `mu`-strongly convex, the lower model becomes quadratic:

`f(x) >= f(y) + <grad f(y), x-y> + (mu/2)||x-y||^2`.

Smoothness gives the dual upper model:

`f(x) <= f(y) + <grad f(y), x-y> + (L/2)||x-y||^2`.

Minimizing that upper model gives the plain gradient step `y-(1/L)grad f(y)` and the descent inequality

`f(y-(1/L)grad f(y)) <= f(y) - (1/(2L))||grad f(y)||^2`.

The unresolved issue is how to combine many such local models without paying more than one gradient per iteration.

## Baselines

Plain gradient descent is cheap and robust. With `x_{k+1}=x_k-(1/L)grad f(x_k)`, smoothness gives the one-step decrease and convexity bounds the gradient by the current distance to a minimizer. The standard proof yields

`f(x_t)-f* <= 2L||x_1-x*||^2/(t-1)`.

For smooth strongly convex objectives, the same method converges linearly but with exponent scale `1/kappa`; with the usual fixed step the oracle complexity is `O(kappa log(1/eps))`. This is dependable but slow when the level sets are elongated.

Classical step-length rules improve constants but not the order. A sufficient-decrease backtracking test is cheaper than exact one-dimensional minimization and guarantees a finite number of reductions under Lipschitz-gradient assumptions. It still leaves a method whose analysis is local and descent-based.

## Pressure From Older Methods

Momentum and conjugate-gradient behavior on quadratics show that the condition-number scale `kappa` is not sacred. On a strongly convex quadratic, spectral methods can achieve a `sqrt(kappa)` dependence, and conjugate gradients can terminate in at most the dimension many exact arithmetic steps. The difficulty is that these arguments use linearity or exact subspace minimization. They do not automatically certify a general non-quadratic smooth convex function under a one-gradient-per-step oracle.

Worst-case lower bounds sharpen the target. Tridiagonal quadratic examples reveal one new coordinate per first-order query, so no method in the black-box span model can beat order `1/k^2` on smooth convex objectives. In the strongly convex case, the hard minimizer has a geometric tail controlled by `(sqrt(kappa)-1)/(sqrt(kappa)+1)`, implying a `sqrt(kappa) log(1/eps)` floor. The open slot is therefore a plain first-order rule that reaches these floors without relying on a quadratic spectral proof.

## Code Framework

The code framework has an objective with `value(x)` and `grad(x)`, a known or estimated smoothness constant `L`, optionally a strong-convexity parameter `mu`, and the ordinary gradient step. A line-search routine can enforce the smoothness descent inequality when `L` is unknown. The missing part is a stateful first-order update that decides where to query the next gradient and how to combine the new step with previous iterates while preserving a global certificate.
