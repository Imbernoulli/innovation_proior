Linear programming is solved: the simplex method walks the vertices of a convex polytope and returns a globally optimal continuous solution. But many real problems demand whole-number answers—number of flights, machine setups, or binary yes/no decisions. The moment integrality is required, the feasible set becomes a scattered cloud of lattice points inside the polytope, no longer convex, so simplex has no vertex to walk to. Rounding the continuous optimum often violates constraints or produces a suboptimal point, and total enumeration of all integer points grows exponentially with the number of variables. What is needed is a general, automatic procedure that respects integrality exactly and returns a provably optimal integer point without inspecting the whole lattice.

The key insight is that dropping the integrality requirement gives the LP relaxation, whose feasible set contains every integer-feasible point and possibly more. Because enlarging the feasible set can only raise a maximum, the relaxation's optimum value is a certified upper bound on the true integer optimum. When the relaxation itself happens to be integral, it is immediately optimal. When it is fractional in some marked variable, say x_j = 2.6, no integer point lies in the open strip between 2 and 3, so the problem can be split into two subproblems: x_j <= 2 and x_j >= 3. This loses no integer point and removes the fractional optimum from both children. Repeating this recursively creates a tree of LP relaxations whose leaves are pinned-down integer candidates.

The method is called Branch and Bound. It combines three operations: bound, branch, and fathom. At each node of the tree, solve the LP relaxation to obtain an upper bound on what that subtree can achieve. If the relaxation is fractional, branch on the most-fractional marked variable by tightening its bounds to the nearest integers on either side. Track the best integer-feasible solution found anywhere, called the incumbent; its value is a lower bound on the optimum. A node is fathomed—discarded without further exploration—when its relaxation is infeasible, when its relaxation upper bound is no better than the incumbent, or when its relaxation is already integral. The bound, not enumeration, prunes entire subtrees that cannot contain the optimum.

The algorithm is correct because branching partitions the integer-feasible set exactly and fathoming only discards subtrees whose upper bound proves they cannot beat the incumbent. As the search proceeds, the incumbent rises and the outstanding ceilings fall, squeezing the optimum from both sides. The optimality gap, defined as the largest remaining upper bound minus the incumbent value, gives a live certificate of progress: when it reaches zero, optimality is proved. For a maximization, the relaxation provides the ceiling and the incumbent provides the floor; for minimization the roles reverse. Node selection is a practical choice: depth-first dives for an early incumbent on small memory, while best-first always expands the most promising node to minimize nodes explored. Branching on the variable closest to half-integer tends to make the most decisive split.

```python
import numpy as np
from scipy.optimize import linprog
import math, itertools


def solve_lp(c, A_ub, b_ub, bounds):
    """Continuous relaxation at one node; returns (x, upper_bound) for a maximization."""
    res = linprog(-np.asarray(c, float), A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method="highs")
    if res.status == 2:
        return None, None
    if res.status == 3:
        raise ValueError("LP relaxation is unbounded; no finite upper bound")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x, -res.fun


def select_fractional_integer_var(x, int_vars):
    """Most-fractional marked variable: nearest to a half-integer."""
    return max((abs(x[k] - round(x[k])), k) for k in int_vars)


def solve_integer_lp(c, A_ub, b_ub, n, int_vars=None, bounds=None, tol=1e-6):
    """Maximize c'x s.t. A_ub x <= b_ub, bounds l_j<=x_j<=u_j, x_j integer for j in int_vars."""
    if int_vars is None:
        int_vars = list(range(n))
    else:
        int_vars = list(int_vars)
    if bounds is None:
        bounds = [(0, None)] * n

    best_val = -np.inf
    best_x = None
    stack = [list(bounds)]
    nodes = 0
    while stack:
        bnds = stack.pop()
        nodes += 1
        x, ub = solve_lp(c, A_ub, b_ub, bnds)
        if x is None:
            continue
        if ub <= best_val + tol:
            continue
        frac, j = select_fractional_integer_var(x, int_vars)
        if frac <= tol:
            if ub > best_val + tol:
                best_val, best_x = ub, x.copy()
            continue
        lo, hi = bnds[j]
        down = list(bnds); down[j] = (lo, math.floor(x[j]))
        up   = list(bnds); up[j]   = (math.ceil(x[j]), hi)
        if hi is None or math.ceil(x[j]) <= hi:
            stack.append(up)
        if lo is None or lo <= math.floor(x[j]):
            stack.append(down)
    return best_x, best_val, nodes


if __name__ == "__main__":
    # 0/1 knapsack
    vals = np.array([8, 11, 6, 4, 7, 3])
    wts  = np.array([5,  7, 4, 3, 5, 2])
    cap  = 14
    n = len(vals)
    x, val, nodes = solve_integer_lp(vals, wts.reshape(1, -1), [cap],
                                     n=n, int_vars=list(range(n)), bounds=[(0, 1)] * n)
    print("B&B  :", x.round().astype(int), "value", val, "nodes", nodes)
    best = (-1, None)
    for combo in itertools.product([0, 1], repeat=n):
        a = np.array(combo)
        if wts @ a <= cap and vals @ a > best[0]:
            best = (int(vals @ a), a)
    print("brute:", best[1], "value", best[0])
    assert abs(val - best[0]) < 1e-6

    # general-integer LP
    c2, A2, b2 = [4, -1], [[7, -2], [0, 1], [2, -2]], [14, 3, 3]
    x2, v2, nodes2 = solve_integer_lp(c2, A2, b2, n=2,
                                      int_vars=[0, 1], bounds=[(0, None), (0, None)])
    print("B&B  :", x2, "value", v2, "nodes", nodes2)
    best2 = (-1e9, None)
    for a in range(0, 11):
        for bb in range(0, 4):
            p = np.array([a, bb])
            if all(np.array(A2) @ p <= b2):
                if c2[0]*a + c2[1]*bb > best2[0]:
                    best2 = (c2[0]*a + c2[1]*bb, p)
    print("brute:", best2[1], "value", best2[0])
    assert abs(v2 - best2[0]) < 1e-6
```
