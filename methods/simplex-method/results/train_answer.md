The task in front of me is to mechanize military program planning. A "program" in the military sense is a time-staged schedule of training, supply, and deployment, and the relations a planner juggles are genuinely linear: each activity consumes some items and produces others, the supplies of items must balance the demands, activities run at non-negative levels, and the whole thing unfolds over time stages. Today this is done by hand, on desk calculators, and the way a "best" plan gets chosen is a thick book of ground rules and edicts. There is no statement of what makes one plan better than another — only a procedure for producing *some* feasible plan. I want to replace that with a mathematical model, an explicit notion of "best," and a computable procedure that finds the best plan, not merely a feasible one. The cleanest existing model I know is Leontief's interindustry input-output system, but it is the wrong shape for planning in three concrete ways: it is steady-state, my problem is nothing but time stages; it ties each item to exactly one producing process, so there is no notion of *choosing* among alternative ways to make a thing, and choice is the whole point of planning; and it carries no objective — it computes *the* solution of a determined system, not the *best* of many. So I generalize: an activity is any process consuming and producing items in fixed non-negative proportions, many alternative activities compete to supply each item, the model runs over time stages (giving a staircase structure), and everything stays linear because the relations really are. That gives a big system of linear equalities and inequalities in non-negative variables.

The genuinely new move is not the constraints — those are bookkeeping — but pulling the goal out of the ground rules and stating it explicitly as one more linear form to extremize. The planner's trap, which I have fallen into myself, is to promote a means to a goal: ask the commander the objective and he says "win the war," press him and he says "build bombers," but bombers are a means, and that means spawns sub-rules confused with goals all the way down. The ground rules are at best a clumsy way of groping toward an objective that was never written down. And making the objective explicit is forced, not merely tidy, because the alternative — find the best plan by listing feasible plans and comparing them — is hopeless. The cleanest sub-problem, assigning 70 men to 70 jobs with each man to one job and each job to one man, has $70! > 10^{100}$ feasible assignments, and the real programs are larger. Enumeration is not a slow method; it is not a method. So I need both an explicit objective and an algorithm that reaches the optimum without ever enumerating the feasible set. Written cleanly, with a non-negative slack added to each inequality so every constraint becomes an equation, the problem is to minimize $z = c\cdot x$ subject to $A x = b$, $x \ge 0$, with $A$ an $m\times n$ matrix.

I propose the **simplex method**. The starting point is geometric: the feasible set $\{x : Ax = b,\, x \ge 0\}$ is a convex polyhedron, pointed once redundant equality rows are removed, and $c\cdot x$ is a linear form constant on parallel hyperplanes. If a point in an optimal face is not extreme, it is a convex combination of two other feasible points whose objective values average to the same minimum, so neither is worse and I can keep moving inside the optimal face until no convex splitting remains. Hence some vertex is optimal, and vertices are algebraic objects: pick $m$ linearly independent columns of $A$ as a basis $B$, set the other $n-m$ variables to zero, solve $B x_B = b$, and if $x_B \ge 0$ this *basic feasible solution* is a vertex. Every vertex has at least one such basis, degeneracy allowing several. There are at most $\binom{n}{m}$ of them, so the continuum collapses to a finite search over basic feasible solutions.

The obvious algorithm — start at a vertex, find an edge that goes downhill in the objective, walk to the adjacent vertex, repeat until no leaving edge descends — I distrusted on sight, because the boundary of a high-dimensional polyhedron is an enormous convoluted thing and a path along its outside could meander astronomically far. What rescued it is a change of eyes drawn from my thesis: instead of the row geometry, whose dimension is the number of variables $n$ (possibly enormous, even infinite), I work in the geometry of a *column*, whose dimension is set by the number of constraints $m$. Each column becomes a point $(y_j, z_j)$, the non-negative weights $x_j$ on these points must form a convex combination whose $y$-coordinate is $b$ — a center of gravity sitting on the requirement line $y = b$ — and I want its height $z$ as low as possible. A "vertex" in this picture is a simplex of $m$ points whose convex hull the requirement line pierces; this is what gives the method its name, moving from one simplex of points to a neighboring one. The local rule reads directly off the picture. Fit the plane through the current basic points,
$$z = \pi\cdot y + \pi_0,$$
where $\pi, \pi_0$ solve $\pi\cdot A_{j_i} + \pi_0 = c_{j_i}$ over the $m$ basic columns, and measure each other column's vertical gap below that plane. The entering column is the one most below it,
$$s = \arg\min_j\,\bigl[\,c_j - (\pi\cdot A_j + \pi_0)\,\bigr].$$
If that minimum is non-negative, the plane already supports the whole cloud from below; since any feasible center of gravity is a convex combination and the average of non-negative gaps is non-negative, no feasible point can sit below the plane's height at $y=b$, and the current vertex is optimal. This optimality test is purely local — read off the $m$ basic points and one pass over the columns — and never enumerates vertices. If the minimum is negative, point $s$ enters, forming an $m+1$ point body that the requirement line pierces along a segment with a strictly lower lower end; that lower end lies on the face opposite exactly one old point $r$, which is dropped. Algebraically $r$ is the first old basic weight driven to zero as $s$ enters with increasing weight, since pushing past that would make its weight negative.

Why this is fast becomes visible in the two-dimensional version: take the *underbelly* of the convex hull of the points, the lower piecewise-linear boundary $f$, with the optimum at $f(b)$. The algorithm only ever moves to points on that underbelly — the rare extreme points of the lower hull — while most points lie well above it, so the number of real choices is small, about $m$. I made this quantitative by tracking the gap $\Delta_t = z_t - z^*$. Eliminating the basic variables from the objective row gives reduced costs $\bar c_j$, zero on basic columns and $\bar c_s < 0$ on the entering one. Because the optimal weights $x_j^*$ are non-negative, sum to one, and reproduce the gap, the most negative reduced cost dominates the whole remaining gap,
$$\Delta_{t-1} = \sum_j (-\bar c_j)\,x_j^* \;\le\; (-\bar c_s)\sum_j x_j^* = -\bar c_s,$$
and the one-step decrease is that reduced cost times the entering value $\theta_t$, so $\Delta_{t-1} - \Delta_t = (-\bar c_s)\,\theta_t \ge \Delta_{t-1}\theta_t$, giving $\Delta_t \le (1-\theta_t)\Delta_{t-1} \le e^{-\theta_t}\Delta_{t-1}$. Chaining over iterations,
$$\frac{\Delta_t}{\Delta_0} \le \prod_i (1-\theta_i) \le e^{-\sum_i \theta_i},$$
independent of $n$: the gap decays geometrically in the sum of the entering values, and with $\theta$ averaging about $1/m$ a thousand-fold reduction needs under $7m$ iterations. When I strip the convexity row, work out the same column-geometry algorithm for the general $Ax=b,\,x\ge0$, and run it on Stigler's diet problem, I see that this "new" algorithm *is* the edge-descent I had discarded — now visibly efficient, because the points it can move to are the rare extreme points of the underbelly.

Made arithmetic, it is the tableau method. A basis is $m$ columns $B$ with $x_B = B^{-1}b$ and the rest zero, kept in a tableau whose basic columns form an identity, with an extra row of reduced costs. The reduced cost of column $j$ is $\bar c_j = c_j - c_B B^{-1} A_j = c_j - \pi\cdot A_j$ with $\pi = c_B B^{-1}$ the dual prices — the same plane comparison, now "pricing out" the column. Since every feasible $x$ satisfies $c\cdot x = c_B B^{-1} b + \sum_j \bar c_j x_j$, all $\bar c_j \ge 0$ certifies global optimality, not just a flat local edge; otherwise Dantzig pricing takes the most negative reduced cost as the entering column, which is cheap because the columns are sparse. The leaving rule is the ratio test, "first basic weight to zero" turned arithmetic: raising the entering variable to $t$ changes basic variables as $x_B - t\,B^{-1}A_q$, so only positive pivot-column entries can block, and the leaving row is
$$t^* = \min_{i:\,a_i > 0}\frac{b_i}{a_i}.$$
If no entry is positive the entering variable goes to infinity and the problem is unbounded. The pivot — divide the pivot row by its pivot entry, subtract multiples to clear that column elsewhere including the objective row — is Gaussian elimination carrying the tableau to the adjacent basic feasible solution. To start when no basis is at hand, Phase 1 adds an artificial variable to each row and minimizes the sum of artificials; driving that sum to zero yields a genuine basic feasible solution (remaining zero artificial rows are pivoted out or dropped as redundant), and a positive minimum proves the constraints infeasible. Finally, degeneracy — zero-valued basic variables, which Air Force programs produce routinely rather than with probability zero — allows zero-length pivots that can cycle through repeated bases forever; the fix is infinitesimal perturbation, equivalent lexicographic bookkeeping, or a Bland-style smallest-index pivot, which the code exposes as an option.

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
