We are given a large linear program whose variables fall naturally into independent blocks. Each block has its own local feasibility polyhedron, and only a small set of linking constraints couples the blocks together. Solving the original LP directly is wasteful because the solver must juggle every local variable and every local constraint at the same time, even though the real difficulty is the global coordination. A simpler idea is to optimize each block independently and then try to patch the results so the linking constraints are satisfied, but such heuristics have no optimality certificate and often produce locally attractive decisions that are globally infeasible or suboptimal. Enumerating all feasible block configurations is also impossible: a polyhedron can have exponentially many extreme points, so an explicit list of all candidate columns would be unmanageable.

The right approach is Dantzig-Wolfe decomposition. It keeps the linking constraints in a master problem but changes the decision unit from individual coordinates to entire block-level feasible configurations. The master problem does not ask how much of each original variable to set; instead, it decides how much weight to place on each complete feasible extreme-point plan of a block. Because any point in a bounded polyhedron can be written as a convex combination of its extreme points, this reformulation is exact. The master problem coordinates these plans through the linking constraints and convexity constraints, while the discovery of useful plans is delegated to smaller pricing subproblems.

Here is how the method works. Suppose the original LP is minimize the sum over blocks k of c_k^T x_k, subject to the linking constraints sum over k of A_k x_k equals b, and each x_k lies in its local polyhedron X_k. For each block k, let p_kr denote the extreme points of X_k. Writing x_k as a convex combination of these extreme points, with weights lambda_kr, transforms the problem into a master LP over the lambda variables. The objective becomes minimize sum over k and r of (c_k^T p_kr) lambda_kr, the linking constraints become sum over k and r of (A_k p_kr) lambda_kr equals b, and for each block k we require sum over r of lambda_kr equals one with all lambda_kr nonnegative. Each column of the master is therefore not a single original variable but an entire locally feasible extreme point, together with its cost and its contribution to the linking constraints.

The master LP may still have an enormous number of columns, so Dantzig-Wolfe solves it with column generation. We begin with a small pool of feasible columns for each block and solve the restricted master. From its dual solution we obtain pi, the dual vector for the linking constraints, and mu_k, the dual value for each block's convexity constraint. The reduced cost of a candidate extreme point p in X_k is (c_k - A_k^T pi)^T p minus mu_k. Finding the most negative reduced cost over all extreme points of X_k is equivalent to solving the pricing subproblem minimize (c_k - A_k^T pi)^T x over x in X_k. Because this is a linear program over X_k, an optimal solution is an extreme point, and if its reduced cost is negative it is added to the restricted master. If no block can produce a negative reduced-cost column, the current restricted master solution is optimal for the full master and hence for the original LP. This iterative interplay between a global coordinator and local pricing oracles is the heart of the method.

```python
import numpy as np
from scipy.optimize import linprog

# Toy demonstration of Dantzig-Wolfe with two single-variable blocks.
# Original LP: min 2*x0 + 1*x1  s.t.  x0 + x1 = 1,  0 <= xk <= 1.
c = [2.0, 1.0]
A_link = [1.0, 1.0]
b_link = 1.0
bounds = [(0.0, 1.0), (0.0, 1.0)]

# Each column is (cost, linking_value).  Block membership is stored separately.
columns = [(0.0, 0.0),   # x0 = 0
           (c[0], A_link[0]),  # x0 = 1
           (0.0, 0.0)]   # x1 = 0
block = [0, 0, 1]

for it in range(20):
    costs = [col[0] for col in columns]
    A_eq = [[col[1] for col in columns]]
    for k in range(2):
        A_eq.append([1.0 if block[j] == k else 0.0
                     for j in range(len(columns))])
    res = linprog(costs, A_eq=A_eq, b_eq=[b_link, 1.0, 1.0],
                  bounds=(0, None), method="highs")
    pi = res.eqlin.marginals[0]
    mu = res.eqlin.marginals[1:]

    added = False
    for k in range(2):
        coeff = c[k] - A_link[k] * pi
        # Optimal extreme point for a single bounded variable.
        x = bounds[k][0] if coeff >= 0 else bounds[k][1]
        if coeff * x - mu[k] < -1e-8:
            columns.append((c[k] * x, A_link[k] * x))
            block.append(k)
            added = True
    if not added:
        print(f"Converged in {it} iterations, objective = {res.fun:.4f}")
        break
```

Dantzig-Wolfe decomposition is therefore not merely a way to split a large problem into smaller pieces. Its defining insight is to replace a static, enormous set of original variables with a dynamically generated set of structural variables, each of which represents a complete locally feasible configuration. The master problem coordinates these configurations using global shadow prices, and the subproblems react to those prices by constructing the next configuration that matters. When no subproblem can find an improving configuration, the method terminates with a dual certificate of optimality for the full linear program.
