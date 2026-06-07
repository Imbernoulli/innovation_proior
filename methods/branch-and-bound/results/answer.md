# Branch and Bound for (mixed-)integer programming

## Problem

Solve a linear program in which some variables must be integer:

```
maximize  c'x   subject to   Ax <= b,   l <= x <= u,   x_j integer for j in J
```

(`J` = all variables for a pure integer program, a subset for a mixed one; binary variables are the
case `[l_j, u_j] = [0, 1]`). The integer-feasible set is the lattice points inside the polytope —
**non-convex**, so the simplex method cannot be aimed at it directly, and **exponentially large**
(`2^n` for `n` binaries), so total enumeration is hopeless. Rounding the continuous optimum is
generally infeasible or suboptimal. The goal is a *provably* optimal integer point, found
automatically, without walking the whole lattice.

## Key idea

Drop integrality to get the **LP relaxation**. Because relaxing (enlarging the feasible set) can
only raise a maximum, the relaxation's value is a certified **upper bound** on the integer optimum.
Two outcomes at a subproblem (node):

- The relaxation optimum `x*` is already integer in all of `J` → it is feasible and achieves the
  upper bound, hence **optimal for that subtree**; record it.
- Some `x*_j` is fractional → **branch**. No integer lies in the open strip `(floor(x*_j),
  ceil(x*_j))`, so split into two children, `x_j <= floor(x*_j)` and `x_j >= ceil(x*_j)`, losing no
  integer point while cutting `x*` out of both. Each child's relaxation is over a smaller feasible
  region, so its bound cannot be higher than the parent's, though it can tie if another optimum
  remains.

Carry the best integer point found as the **incumbent** (value `z_inc`, a *lower* bound on the
optimum). **Fathom** (prune) a node without exploring its subtree when it is **infeasible**, when
its relaxation upper bound `<= z_inc` (the whole subtree cannot beat the incumbent), or when its
relaxation is **integral** (a leaf — update the incumbent). The bound, not enumeration, discards the
subtrees that cannot contain the optimum. The **optimality gap** `z_bar - z_inc`, where `z_bar` is
the largest upper bound over still-open nodes, traps the optimum `z_inc <= z* <= z_bar`; gap zero is
a proof of optimality, a small gap is a principled early stop.

For minimization, flip every direction: the relaxation gives a *lower* bound, the incumbent an
*upper* bound, and a node is fathomed when its bound `>= z_inc`.

## Algorithm

```
incumbent z_inc = -inf,  x_inc = none
open nodes = { root: bounds [l_j, u_j] }                # depth-first or best-first
while open nodes remain:
    pop a node (its per-variable bounds)
    solve the LP relaxation -> (x*, u)                  # u = node upper bound
    if infeasible:                       fathom (empty)
    elif u <= z_inc:                     fathom (bound: cannot beat incumbent)
    elif x* integral on J:                              # leaf candidate
        if u > z_inc: z_inc, x_inc = u, x*              # update incumbent
        fathom
    else:                                               # branch
        pick fractional j in J (most-fractional: argmax_j |x*_j - round(x*_j)|)
        push child with x_j <= floor(x*_j)
        push child with x_j >= ceil(x*_j)
return x_inc, z_inc                       # provably optimal when open set empties
```

**Node selection.** *Best-first* expands the open node with the best bound — fewest nodes, but a
large open frontier in memory. *Depth-first* dives to a leaf — tiny memory and an early incumbent
that makes bound-pruning fire sooner. Practical solvers mix the two.

**Cut strengthening.** When a node's relaxation bound is loose, an existing cutting-plane routine
can add a valid inequality — e.g. a Gomory cut from the simplex tableau — that is violated by `x*`
but satisfied by every integer point. Re-solving the tightened relaxation can lower the node's
ceiling before the split.

## Code

A self-contained LP-relaxation solver using `scipy.optimize.linprog` (HiGHS) as the node solver;
branching simply tightens a variable's bound. The small checks compare against exhaustive
enumeration only because those toy instances are small enough to enumerate.

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

    best_val = -np.inf            # incumbent value = LOWER bound on the optimum
    best_x = None

    stack = [list(bounds)]        # open nodes = bound vectors; pop = depth-first
    nodes = 0
    while stack:
        bnds = stack.pop()
        nodes += 1
        x, ub = solve_lp(c, A_ub, b_ub, bnds)
        if x is None:                 # FATHOM: infeasible
            continue
        if ub <= best_val + tol:      # FATHOM by bound: ceiling <= incumbent floor
            continue
        frac, j = select_fractional_integer_var(x, int_vars)
        if frac <= tol:               # relaxation integral -> leaf candidate
            if ub > best_val + tol:
                best_val, best_x = ub, x.copy()    # update incumbent
            continue
        lo, hi = bnds[j]
        down = list(bnds); down[j] = (lo, math.floor(x[j]))  # x_j <= floor(x*_j)
        up   = list(bnds); up[j]   = (math.ceil(x[j]), hi)   # x_j >= ceil(x*_j)
        if hi is None or math.ceil(x[j]) <= hi:
            stack.append(up)
        if lo is None or lo <= math.floor(x[j]):
            stack.append(down)
    return best_x, best_val, nodes


if __name__ == "__main__":
    # 0/1 knapsack: maximize value, total weight <= capacity
    vals = np.array([8, 11, 6, 4, 7, 3])
    wts  = np.array([5,  7, 4, 3, 5, 2])
    cap  = 14
    n = len(vals)
    x, val, nodes = solve_integer_lp(vals, wts.reshape(1, -1), [cap],
                                     n=n, int_vars=list(range(n)), bounds=[(0, 1)] * n)
    print("B&B  :", x.round().astype(int), "value", val, "nodes", nodes)

    best = (-1, None)                 # brute-force ground truth
    for combo in itertools.product([0, 1], repeat=n):
        a = np.array(combo)
        if wts @ a <= cap and vals @ a > best[0]:
            best = (int(vals @ a), a)
    print("brute:", best[1], "value", best[0])
    assert abs(val - best[0]) < 1e-6
    print("knapsack matches brute force")

    # general-integer LP (fractional relaxation -> branching fires)
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
    print("integer-LP matches brute force")
```

The solver returns a candidate only after every live node has been fathomed by infeasibility, by the
incumbent bound, or by an integral relaxation. Divide the discrete space, conquer each piece with a
convex LP bound, and let the bound prune everything that cannot hold the optimum.
