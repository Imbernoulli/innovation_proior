# The Frank-Wolfe Method

## Problem

Minimize a differentiable convex function over a compact convex set:

```text
minimize f(x) subject to x in D.
```

The domain primitive is a Linear Minimization Oracle:

```text
LMO(c) returns s in argmin_{v in D} <v, c>.
```

## Algorithm

```text
input x_0 in D
for k = 0, 1, 2, ...
    g_k = grad f(x_k)
    s_k = argmin_{s in D} <s, g_k>
    gap_k = <x_k - s_k, g_k>
    if gap_k <= epsilon: stop
    gamma_k = 2/(k+2)
    x_{k+1} = (1 - gamma_k) x_k + gamma_k s_k
```

The method replaces a projection step with linear minimization over the feasible set. The update stays feasible because it is a convex combination of feasible points.

## Gap Certificate

The oracle gives a computable certificate:

```text
gap(x) = max_{v in D} <x - v, grad f(x)> = <x - s, grad f(x)>.
```

By convexity,

```text
f(x*) >= f(x) + <x* - x, grad f(x)>,
```

so

```text
f(x) - f(x*) <= <x - x*, grad f(x)> <= gap(x).
```

Thus `gap(x) <= epsilon` proves epsilon-optimality.

## Curvature And Rate

Define the curvature constant

```text
C_f = sup (2/gamma^2) [f(y) - f(x) - <y - x, grad f(x)>],
```

where `x,s in D`, `gamma in (0,1]`, and `y = x + gamma(s - x)`.

For one step,

```text
f(x_next) <= f(x) - gamma gap(x) + (gamma^2/2) C_f.
```

Since `gap(x) >= f(x) - f(x*)`, the open-loop rule `gamma_k = 2/(k+2)` gives, for `k >= 1`,

```text
f(x_k) - f(x*) <= 2 C_f/(k+2).
```

For `K >= 2`, some iterate `k_hat in {1,...,K}` has

```text
gap(x_{k_hat}) <= 2 beta C_f/(K+2),  beta = 27/8.
```

If `grad f` is `L`-Lipschitz in a norm, then `C_f <= L diam(D)^2`.

## Properties

- Projection-free: each iteration solves a linear problem over `D`, not a nearest-point projection.
- Sparse or low-rank iterates: one atom is added per step, so `x_k` is a convex combination of at most `k+1` atoms when started from an atom.
- Affine-invariant analysis: the curvature form and the LMO/convex-combination update do not depend on choosing a Euclidean projection geometry.
- Built-in stopping rule: the Frank-Wolfe gap is computed from the same oracle call as the direction.

## Minimal Implementation

The mathematical oracle above minimizes `<s, grad f(x)>`. A common equivalent sign convention calls the LMO with `-grad`, returns the feasible update direction `d = s - x`, computes the certificate as `dot(d, -grad)`, and updates `x += step_size * d`.

```python
import numpy as np

def frank_wolfe(f_grad, lmo, x0, max_iter, tol=1e-8):
    x = np.asarray(x0, dtype=float).copy()
    _, grad = f_grad(x)
    certificate = np.inf

    for it in range(max_iter):
        direction, max_step_size = lmo(-grad, x)
        certificate = float(np.dot(direction, -grad))
        if certificate <= tol:
            break

        step_size = min(2.0 / (it + 2.0), max_step_size)
        x = x + step_size * direction
        _, grad = f_grad(x)

    return x, certificate

def lmo_simplex(neg_grad, x):
    s = np.zeros_like(x, dtype=float)
    s[np.argmax(neg_grad)] = 1.0
    return s - x, 1.0

def solve_simplex_least_squares(b, max_iter=1000, tol=1e-10):
    b = np.asarray(b, dtype=float)
    if b.ndim != 1 or b.size == 0:
        raise ValueError("b must be a nonempty 1-D array")
    x0 = np.zeros_like(b, dtype=float)
    x0[0] = 1.0

    def f_grad(x):
        r = x - b
        return 0.5 * float(np.dot(r, r)), r

    return frank_wolfe(f_grad, lmo_simplex, x0, max_iter, tol)
```
