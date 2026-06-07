# The Simplex Method for Linear Programming

## Problem

Choose activity levels to optimize an explicit linear objective subject to linear constraints:

```
minimize   z = c·x
subject to A x = b,   x ≥ 0,
```

with inequalities put into equality form by adding non-negative slack variables. This is the linear program that arises when military deployment/supply scheduling, the diet problem, transportation, and assignment are modeled as activities consuming and producing items in fixed proportions, with a goal stated as a linear form rather than as ad-hoc ground rules. Exhaustive enumeration is not viable: a 70×70 assignment has 70! > 10^100 feasible assignments.

## Key Idea

The feasible set is a convex polyhedron, and if a finite optimum is attained in standard form, at least one optimum is attained at a **vertex**. If an optimal point lies in a higher-dimensional optimal face, linearity lets us move within that face until reaching an extreme point. Algebraically, after redundant equality rows are removed, a vertex is a **basic feasible solution**: choose `m` linearly independent columns of `A` as a basis `B`, set all other variables to zero, and solve `B x_B = b`; if `x_B >= 0`, this is a feasible vertex. Conversely, every vertex has at least one basis, with degeneracy allowing several bases for the same vertex. The simplex method walks from one vertex to an adjacent one by swapping one column into the basis and one column out.

The column-geometry view explains the local rule. Fit a plane through the current basic columns-as-points:

`z = π·y + π0`.

For every column, compute its vertical gap below that plane. The entering column is

`s = argmin_j [c_j - (π A_j + π0)]`.

If the minimum gap is non-negative, the current plane supports the point cloud from below and the current vertex is optimal. If it is negative, that column enters; the leaving column is the first current basic weight driven to zero.

## Algorithm

Maintain a full tableau: constraint rows, an objective row of reduced costs, and the current basis.

1. **Pricing.** Reduced costs are `cbar_j = c_j - c_B B^{-1} A_j = c_j - π A_j`. The current basic value is `c_B B^{-1} b`, and every feasible `x` satisfies `c^T x = c_B B^{-1} b + sum_j cbar_j x_j`. If every `cbar_j >= 0`, no non-negative feasible `x` can improve the objective, so the current vertex is optimal. Otherwise Dantzig pricing chooses the most negative reduced cost as the entering column.
2. **Minimum-ratio test.** Raise the entering variable. The current basic variables change as `x_B - t B^{-1} A_q`, so only positive entries in the pivot column can block the move. The leaving row is `argmin_i b_i / a_i` over `a_i > 0`. If no entry is positive, the objective decreases without bound.
3. **Pivot.** Divide the pivot row by the pivot entry and subtract multiples of it from every other row, including the objective row. This Gaussian-elimination step moves to the adjacent basic feasible solution.
4. **Phase 1.** If a starting basis is not already present, add artificial variables, minimize their sum, and proceed to the real objective only if that sum can be driven to zero. Artificial basic rows that remain at zero are pivoted out when possible or dropped as redundant. Otherwise the constraints are infeasible.
5. **Degeneracy.** Zero-valued basic variables can cause zero-length pivots. Perturbation, lexicographic bookkeeping, or Bland-style smallest-index pivoting prevents repeated bases in exact arithmetic; the code exposes that smallest-index option.

## Code

```python
import numpy as np

def to_standard_form(A_ub, b_ub, c):
    """Convert A_ub @ x <= b_ub, x >= 0 into A @ x == b, x >= 0."""
    c = np.asarray(c, dtype=float).reshape(-1)
    A_ub = np.asarray(A_ub, dtype=float)
    b = np.asarray(b_ub, dtype=float).reshape(-1)
    if A_ub.ndim != 2:
        raise ValueError("A_ub must be a two-dimensional array")
    m, n = A_ub.shape
    if c.size != n or b.size != m:
        raise ValueError("incompatible dimensions for c, A_ub, and b_ub")

    A = np.hstack((A_ub, np.eye(m)))
    c_ext = np.concatenate((c, np.zeros(m)))
    negative = b < 0
    A[negative, :] *= -1
    b[negative] *= -1
    return c_ext, A, b


def choose_improving_column(tableau, tol=1e-9, smallest_index_ties=False):
    """Return an entering column, or None when all reduced costs are non-negative."""
    objective = tableau[-1, :-1]
    candidates = np.where(objective < -tol)[0]
    if candidates.size == 0:
        return None
    if smallest_index_ties:
        return int(candidates[0])
    return int(candidates[np.argmin(objective[candidates])])


def choose_blocking_row(tableau, basis, column, n_rows,
                        tol=1e-9, smallest_index_ties=False):
    """Minimum-ratio test over positive pivot-column entries."""
    pivot_column = tableau[:n_rows, column]
    rhs = tableau[:n_rows, -1]
    rows = np.where(pivot_column > tol)[0]
    if rows.size == 0:
        return None
    ratios = rhs[rows] / pivot_column[rows]
    best = ratios.min()
    tied = rows[ratios <= best + tol]
    if smallest_index_ties:
        basis = np.asarray(basis)
        return int(tied[np.argmin(basis[tied])])
    return int(tied[0])


def apply_row_operation(tableau, basis, row, column):
    """Gaussian elimination pivot: make the pivot 1 and clear its column."""
    tableau[row, :] = tableau[row, :] / tableau[row, column]
    for other in range(tableau.shape[0]):
        if other != row:
            tableau[other, :] -= tableau[other, column] * tableau[row, :]
    basis[row] = column


def solve_tableau(tableau, basis, n_rows, tol=1e-9,
                  smallest_index_ties=False, maxiter=1000, nit0=0):
    """Run tableau pivots against the last row as the active objective."""
    nit = nit0
    while True:
        column = choose_improving_column(tableau, tol, smallest_index_ties)
        if column is None:
            return "optimal", nit

        row = choose_blocking_row(
            tableau, basis, column, n_rows, tol, smallest_index_ties
        )
        if row is None:
            return "unbounded", nit
        if nit >= maxiter:
            return "iteration_limit", nit

        apply_row_operation(tableau, basis, row, column)
        nit += 1


def solve_linear_program(c, A, b, tol=1e-9, maxiter=1000,
                         smallest_index_ties=False):
    """Minimize c @ x subject to A @ x == b and x >= 0.

    Returns (x, objective_value, status), where status is one of
    "optimal", "infeasible", "unbounded", or "iteration_limit".
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array")
    m, n = A.shape
    if c.size != n or b.size != m:
        raise ValueError("incompatible dimensions for c, A, and b")

    # Standard-form rows need b >= 0 so artificial variables start feasible.
    A = A.copy()
    b = b.copy()
    negative = b < 0
    A[negative, :] *= -1
    b[negative] *= -1

    # Phase 1 tableau: constraints, true objective, and feasibility objective.
    tableau = np.zeros((m + 2, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[m, :n] = c
    basis = list(range(n, n + m))
    tableau[-1, :] = -tableau[:m, :].sum(axis=0)
    tableau[-1, n:n + m] = 0

    status, nit = solve_tableau(
        tableau, basis, m, tol, smallest_index_ties, maxiter
    )
    if status == "iteration_limit":
        return None, None, status
    if abs(tableau[-1, -1]) > tol:
        return None, np.inf, "infeasible"

    # Remove artificial columns; pivot any zero-valued artificial basic out first.
    keep_rows = []
    for row in range(m):
        if basis[row] >= n:
            choices = np.where(np.abs(tableau[row, :n]) > tol)[0]
            if choices.size:
                apply_row_operation(tableau, basis, row, int(choices[0]))
                keep_rows.append(row)
        else:
            keep_rows.append(row)

    phase2 = np.vstack((tableau[keep_rows, :], tableau[[m], :]))
    phase2 = np.delete(phase2, np.s_[n:n + m], axis=1)
    basis = [basis[row] for row in keep_rows]

    status, nit = solve_tableau(
        phase2, basis, len(keep_rows), tol, smallest_index_ties, maxiter, nit
    )
    if status == "iteration_limit":
        return None, None, status
    if status == "unbounded":
        return None, -np.inf, "unbounded"

    x = np.zeros(n)
    for row, variable in enumerate(basis):
        value = phase2[row, -1]
        x[variable] = 0.0 if abs(value) < tol else value
    return x, float(c @ x), "optimal"
```
