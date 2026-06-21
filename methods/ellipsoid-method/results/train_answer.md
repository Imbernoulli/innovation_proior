The task is to decide, in time polynomial in the encoding length of the data, whether a system of integer-coefficient linear inequalities has a real solution. This is the feasibility core of linear programming, and the simplex method, for all its practical success, can visit exponentially many vertices on Klee-Minty perturbed cubes, so it offers no worst-case guarantee. Earlier localization ideas maintain a polyhedron that grows a new facet with every cut, so the per-iteration cost increases and no dimension-only shrink rate is available. Space-dilation methods carry a variable metric, but they were not analyzed as exact decision procedures with bit-length bounds.

The right method is the ellipsoid method. It replaces the growing polyhedral localization set with a single ellipsoid, described forever by a center and a positive-definite shape matrix. At each step we query the center of the current ellipsoid. If the center satisfies a small feasibility tolerance, we accept; otherwise a violated inequality gives a separating hyperplane through the center, and the target body must lie in the resulting half-ellipsoid. We then re-enclose that half-ellipsoid in the minimum-volume ellipsoid and repeat. The volume shrinks by a fixed factor depending only on the dimension, yielding a polynomial iteration count.

Integer data make this exact. If a solution exists, determinant bounds put one inside the ball of radius 2^L, and if the system is infeasible the residual is at least 2^{-L} everywhere. Choosing ε < 2^{-L} and searching the relaxed body {x : Ax ≤ b + ε} separates the two cases: feasible systems gain a full-dimensional volume floor, while infeasible systems remain empty. Starting from the ball of radius 2^L and using the central-cut update, the ellipsoid volume drops by at most e^{-1/(2(n+1))} each step, so O(n^2 L) iterations drive it below any volume floor. Rounding entries to O(nL) bits and inflating the update slightly restores containment despite rounding, keeping the method sound and polynomial. Optimization then follows by binary-searching the objective value as an additional inequality.

```python
import math
import numpy as np


def as_integer_system(Arows, brhs):
    Arows = np.asarray(Arows, dtype=float)
    brhs = np.asarray(brhs, dtype=float).reshape(-1)
    if Arows.ndim != 2 or brhs.shape[0] != Arows.shape[0]:
        raise ValueError("expected Arows with shape (m, n) and brhs with shape (m,)")
    if Arows.shape[0] == 0:
        raise ValueError("expected at least one inequality")
    if not np.all(np.isfinite(Arows)) or not np.all(np.isfinite(brhs)):
        raise ValueError("all coefficients must be finite")
    Aint, bint = np.rint(Arows), np.rint(brhs)
    if not np.array_equal(Arows, Aint) or not np.array_equal(brhs, bint):
        raise ValueError("this decision wrapper expects integer coefficients")
    return Aint.astype(float), bint.astype(float)


def encoding_length(Arows, brhs):
    Arows, brhs = as_integer_system(Arows, brhs)
    s = sum(math.log2(abs(int(a)) + 1) for a in np.ravel(Arows))
    s += sum(math.log2(abs(int(bi)) + 1) for bi in np.ravel(brhs))
    s += math.log2(max(1, Arows.shape[0] * Arows.shape[1]))
    return int(math.ceil(s)) + 1


def separation_for_system(Arows, brhs, tol=0.0):
    Arows, brhs = as_integer_system(Arows, brhs)

    def separate(x):
        x = np.asarray(x, dtype=float)
        slack = Arows @ x - brhs
        row = int(np.argmax(slack))
        if slack[row] <= tol:
            return None
        return Arows[row].copy()

    return separate


def decide_feasibility(separate, n, R, max_iters, inflate=1.0):
    if n < 2:
        raise ValueError("the central-cut update below assumes n >= 2")
    x = np.zeros(n, dtype=float)
    A = (float(R) ** 2) * np.eye(n)
    scale = n**2 / (n**2 - 1)

    for _ in range(max_iters):
        a = separate(x)
        if a is None:
            return x

        a = np.asarray(a, dtype=float)
        Aa = A @ a
        q = float(a @ Aa)
        if q <= 0:
            return None

        b = Aa / np.sqrt(q)
        x = x - b / (n + 1)
        A = inflate * scale * (A - (2.0 / (n + 1)) * np.outer(b, b))
        A = 0.5 * (A + A.T)

    return None


def decide_linear_inequalities(Arows, brhs, max_iters=None, tol=None):
    Arows, brhs = as_integer_system(Arows, brhs)
    n = Arows.shape[1]
    L = encoding_length(Arows, brhs)
    R = 2.0**L
    max_iters = 16 * n * n * L if max_iters is None else max_iters
    tol = 0.5 * 2.0 ** (-L) if tol is None else tol
    sep = separation_for_system(Arows, brhs, tol=tol)
    x = decide_feasibility(sep, n, R, max_iters)
    return ("feasible", x) if x is not None else ("infeasible", None)


def maximize_linear(base_separate, c, n, L, tol=None):
    c = np.asarray(c, dtype=float)
    R = 2.0**L
    max_iters = 16 * n * n * L
    tol = 2.0 ** (-L) if tol is None else tol
    bound = R * float(np.linalg.norm(c))
    if bound == 0.0:
        return decide_feasibility(base_separate, n, R, max_iters), 0.0
    lo, hi = -bound, bound
    best = None

    def with_objective_floor(d):
        def separate(x):
            cut = base_separate(x)
            if cut is not None:
                return cut
            if float(c @ x) < d:
                return -c
            return None

        return separate

    while hi - lo > tol:
        d = 0.5 * (lo + hi)
        x = decide_feasibility(with_objective_floor(d), n, R, max_iters)
        if x is None:
            hi = d
        else:
            lo, best = d, x
    return best, lo
```
