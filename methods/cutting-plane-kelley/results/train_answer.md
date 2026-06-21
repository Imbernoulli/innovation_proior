The problem is to minimize a continuous convex function over a compact convex set. Convexity makes global optimization tractable in principle, but the objective need not be smooth: the canonical example is Chebyshev (minimax) fitting, where the objective is a pointwise maximum of linear functions and has kinks. Smooth descent methods such as gradient projection fail here because they require a gradient and a tractable projection onto a possibly curved feasible set. Linear programming, by contrast, is already a mature, fast, reusable technology, but it only solves linear objectives over polyhedra. The missing piece is a way to reduce the curved convex program to a sequence of linear programs.

Existing ideas point toward the right reduction but do not complete it. Gradient projection handles nonlinear programs but assumes differentiability and a feasible projection oracle. Gomory's cutting planes for integer programming show that one can repeatedly solve an LP, cut off the unwanted solution with a separating linear inequality, and re-solve, but Gomory's cuts are derived from integrality, not from convexity of a curved constraint. The Chebyshev exchange and support-plane idea for semi-infinite approximation uses supporting hyperplanes of a convex function, yet it is not formulated as a general convex programming method with a convergence guarantee on a compact domain. What is needed is a clean loop that replaces the curved feasible set by a growing family of halfspaces and proves that the resulting LP iterates converge.

The method is Kelley's cutting-plane method. The starting observation is that any convex program can be written, via the epigraph trick, as minimizing a linear form over a convex set: minimize c^T x subject to x in R = {x : G(x) <= 0}, where G is convex. This is the form I will work with. The objective is already linear, so the entire difficulty is the curved feasible set R. The key fact is that a closed convex set is the intersection of all its supporting halfspaces. Therefore R can be approximated by intersecting finitely many of these halfspaces, each obtained from a first-order oracle call.

At any query point t, convexity gives a supporting affine minorant p(x; t) = G(t) + g(t)^T (x - t), where g(t) is a subgradient of G at t (or the gradient if G is smooth there). This affine function satisfies p(x; t) <= G(x) for all x. Consequently, if x is feasible then p(x; t) <= 0, while at t itself we have p(t; t) = G(t) > 0 whenever t is infeasible. Thus the halfspace {x : p(x; t) <= 0} contains every feasible point and excludes t. It is a valid cut. The oracle need only return G(t) and one subgradient, both of which are available even when G is nonsmooth: for a max-type function G = max_i g_i, any active index h gives a valid subgradient as the gradient of g_h.

The algorithm begins with a simple outer approximation S_0, typically a box known to contain R. At iteration k it solves the LP that minimizes c^T x over the current polyhedron S_k, obtaining a candidate t_k. If t_k is feasible, it is optimal, because R is contained in S_k and no point of R can improve on the minimum over S_k. If t_k is infeasible, the oracle generates a cut p(x; t_k) <= 0, which is intersected with S_k to form S_{k+1}, and the LP is re-solved. Each step is therefore the previous LP plus one new linear inequality. The same idea applies directly to minimizing a convex objective f by epigraph reformulation: introduce t, minimize t subject to f(x) <= t, and generate cuts for f using its subgradients. The master LP then minimizes the piecewise-linear lower model of f built from all collected support planes.

Convergence follows from two assumptions: the working set S is compact, and the subgradients of G are uniformly bounded on S. The lower bounds f_k = min_{S_k} c^T x are monotone nondecreasing because the sets S_k shrink, and they are bounded above by the true optimum f^* because R is always contained in S_k. If some iterate remained infeasible with violation bounded away from zero, the proof shows that any two such high-violation iterates would have to be separated by a fixed positive distance. An infinite separated sequence cannot live inside the compact set S, so the violations must converge to zero. A convergent subsequence then has a limit in R, and because f_k converges upward to a limit no larger than f^*, that limit must equal f^*. In practice one stops when the gap between the best feasible objective found so far and the master-LP lower bound falls below a tolerance.

```python
import numpy as np
from scipy.optimize import linprog


def kelley_cutting_plane(f, subgrad, x_lo, x_hi, eps=1e-6, max_iter=500):
    """Minimize a convex (possibly nonsmooth) function f over a box.

    f(x)       -> scalar value of the convex objective
    subgrad(x) -> a subgradient of f at x

    Uses the epigraph variable t; decision vector is [x; t].
    Master LP:  min t  s.t.  f(x_i) + g_i^T (x - x_i) <= t  for all queries i,
                and x_lo <= x <= x_hi.
    """
    n = len(x_lo)
    c = np.concatenate([np.zeros(n), [1.0]])          # minimize t
    bounds = list(zip(x_lo, x_hi)) + [(None, None)]   # x in box, t free

    A_ub, b_ub = [], []
    best_x, best_ub = None, np.inf
    x = 0.5 * (np.asarray(x_lo, dtype=float) + np.asarray(x_hi, dtype=float))

    for _ in range(max_iter):
        fx = f(x)
        g = np.asarray(subgrad(x), dtype=float)

        if fx < best_ub:
            best_x, best_ub = x.copy(), fx

        # Add cut: f(x_i) + g^T (x - x_i) <= t  ->  [g, -1] @ [x; t] <= g^T x_i - f(x_i)
        A_ub.append(np.concatenate([g, [-1.0]]))
        b_ub.append(g @ x - fx)

        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(res.message)

        x, lb = res.x[:n], res.x[n]  # next query and lower bound
        if best_ub - lb <= eps:
            return best_x, best_ub, lb

    return best_x, best_ub, lb


# Example: minimize max_i |a_i^T x - b_i| (Chebyshev / minimax fitting)
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m, n = 20, 3
    A = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    x_true = np.linalg.lstsq(A, b, rcond=None)[0]

    def f(x):
        return np.max(np.abs(A @ x - b))

    def subgrad(x):
        r = A @ x - b
        i = np.argmax(np.abs(r))
        return A[i] * np.sign(r[i])

    x_lo = x_true - 5.0
    x_hi = x_true + 5.0

    x_opt, ub, lb = kelley_cutting_plane(f, subgrad, x_lo, x_hi)
    print("x_opt:", x_opt)
    print("upper bound:", ub)
    print("lower bound:", lb)
    print("certified gap:", ub - lb)
```
