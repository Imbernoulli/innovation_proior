We are asked to minimize a differentiable convex function over a compact convex set, using only a first-order oracle and a cheap primitive for the feasible domain. The obstacle is feasibility: the naive gradient step x - eta grad f(x) usually leaves the feasible region, and restoring feasibility by Euclidean projection is itself an expensive optimization problem on structured domains such as the nuclear-norm ball, assignment polytopes, or general atomic convex hulls. Projected gradient descent and mirror-descent variants therefore inherit a costly inner solve every iteration, while second-order and interior-point methods require Hessians, linear-system solves, or explicit barrier representations. None of these match the requirement that the domain be accessed only through a cheap first-order primitive.

The key observation is that many feasible sets admit a much cheaper operation than projection: linear minimization. On a simplex, minimizing a linear function is a single coordinate choice; on a nuclear-norm ball it is a leading singular-vector pair; on an assignment polytope it is a linear assignment problem. This suggests that the gradient should be used to define a linear objective over the original domain, not to generate an infeasible point that must later be projected back.

The method that captures this idea is the Frank-Wolfe method, also called the conditional gradient method. At each iteration it calls a Linear Minimization Oracle (LMO) on the current gradient and then takes a convex-combination step toward the returned atom. Because the step stays inside the convex hull of two feasible points, every iterate remains feasible by construction. The same oracle call also yields the Frank-Wolfe gap, a nonnegative certificate that upper-bounds the true primal error without requiring knowledge of the optimal value. Convergence is governed by the curvature constant C_f, and with the open-loop step size gamma_k = 2/(k+2) the primal error decays as O(C_f/k).

The update also produces sparse or low-rank iterates naturally. Starting from an atom, x_k is a convex combination of at most k+1 atoms after k iterations, so the representation stays compact on atomic domains. The analysis is affine-invariant in the sense that the curvature constant and the LMO step do not depend on a particular Euclidean geometry. The entire procedure is projection-free: it never computes the nearest feasible point, relying instead on a linear subproblem matched to the geometry of the feasible set.

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
