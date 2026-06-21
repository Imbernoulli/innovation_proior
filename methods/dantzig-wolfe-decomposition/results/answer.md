# Dantzig-Wolfe decomposition

## Problem

We want to solve a large LP with block structure:

`min sum_k c_k^T x_k`

subject to

`sum_k A_k x_k = b`, and `x_k in X_k`.

Each block `X_k` contains local constraints, while `sum_k A_k x_k = b` is the small set of global linking constraints. Solving the compact LP directly forces one solver model to manage every local variable and every local constraint at once.

## Key idea

Dantzig-Wolfe decomposition changes the unit of decision. If `X_k` is a bounded polyhedron with extreme points `p_kr`, then every feasible block decision can be represented as

`x_k = sum_r lambda_kr p_kr`, `sum_r lambda_kr = 1`, `lambda_kr >= 0`.

Substituting this into the original LP gives the master problem:

`min sum_k sum_r (c_k^T p_kr) lambda_kr`

subject to

`sum_k sum_r (A_k p_kr) lambda_kr = b`,

`sum_r lambda_kr = 1` for each block `k`,

`lambda_kr >= 0`.

The master variables are no longer raw coordinates of `x_k`. A variable `lambda_kr` means “use this complete feasible extreme-point solution of subproblem `k` with this weight.” The local feasibility of a block is baked into the column itself.

## Column generation

The full master may have exponentially many columns, because a subproblem may have exponentially many extreme points. Dantzig-Wolfe therefore solves a restricted master over a small column pool and generates new columns only when the current dual prices say they are valuable.

For a minimization problem, let `pi` be the dual vector for the linking constraint and `mu_k` the dual value for block `k`'s convexity constraint. A candidate point `p in X_k` has reduced cost

`(c_k - A_k^T pi)^T p - mu_k`.

So the pricing problem for block `k` is

`min_{x in X_k} (c_k - A_k^T pi)^T x`.

Because this is a linear objective over a polyhedron, an optimal solution is an extreme point. If its reduced cost is negative, it becomes a new master column. If every block fails to produce a negative reduced-cost point, then no omitted extreme point can improve the restricted master, so the current solution is optimal for the full LP.

## Algorithm

1. Start with a few feasible columns for each block.
2. Solve the restricted master problem.
3. Read the master dual prices `pi` and `mu_k`.
4. For each block, solve the pricing subproblem.
5. Add every negative reduced-cost column.
6. Stop when no pricing subproblem can generate a negative reduced-cost column.

For integer optimization, this LP machinery becomes the relaxation engine inside branch-and-price.

## Code illustration

```python
import numpy as np
from scipy.optimize import linprog

# Toy LP: min 2*x1 + x2  s.t.  x1 + x2 = 1,  0 <= xk <= 1.
c, A, b = [2.0, 1.0], [1.0, 1.0], 1.0
bounds = [(0.0, 1.0), (0.0, 1.0)]

# (cost, linking_value) columns; start with block 0 x=0/1 and block 1 x=0.
columns = [(0.0, 0.0), (c[0], A[0]), (0.0, 0.0)]
block = [0, 0, 1]

for it in range(10):
    costs = [col[0] for col in columns]
    A_eq = [[col[1] for col in columns]]
    for k in range(2):
        A_eq.append([1.0 if block[j] == k else 0.0 for j in range(len(columns))])
    res = linprog(costs, A_eq=A_eq, b_eq=[b, 1.0, 1.0],
                  bounds=(0, None), method="highs")
    pi, mu = res.eqlin.marginals[0], res.eqlin.marginals[1:]

    added = False
    for k in range(2):
        coeff = c[k] - A[k] * pi
        x = bounds[k][0] if coeff >= 0 else bounds[k][1]
        if coeff * x - mu[k] < -1e-8:
            columns.append((c[k] * x, A[k] * x))
            block.append(k)
            added = True
    if not added:
        print(f"Converged in {it} iterations, objective = {res.fun:.4f}")
        break
```

## Why this is the insight

The method is not just “solve subproblems.” Its distinctive move is to replace direct manipulation of the enormous original variable set with dynamically generated structural variables. Each generated column is a whole locally feasible configuration: a cutting pattern, a route, a crew duty, a production schedule, or another block-level plan.

The master coordinates these configurations through global shadow prices. The subproblem reacts to those prices by constructing the best missing configuration. This turns the impossible task “store and price every variable” into the iterative task “generate the next variable that matters.” When pricing can no longer find such a variable, the method has a dual certificate for the full LP, including all columns that were never explicitly listed.
