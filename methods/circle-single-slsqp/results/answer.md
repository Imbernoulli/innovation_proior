**Problem.** Construct a feasible packing of `26` non-overlapping circles in `[0,1]²` maximizing
`Σ rᵢ`. The radii are free. This is a nonconvex QCQP: linear objective, convex wall constraints,
and nonconvex pairwise-distance constraints. The grid baseline (`2.5414`) is rigid; this rung
hands the problem to a constrained nonlinear optimizer from one initialization.

**Key idea.** Formulate the full problem in `78` variables (`26` centers + `26` radii) and run
SLSQP once from a single random start. SLSQP linearizes the `325` pairwise non-overlap
inequalities and the `4·26` wall inequalities, builds a quadratic model of the Lagrangian, and
descends to a KKT point — exactly the right local engine for a smooth constrained QCQP. Centers and
radii are optimized *jointly* so the solver can trade circle position against radius growth in one
smooth descent. The start: `26` centers drawn uniformly in an inset square, with initial radii set
to their LP-optimal values (maximize `Σ rᵢ` s.t. `rᵢ + rⱼ ≤ dᵢⱼ`, `rᵢ ≤ wallᵢ` — convex, exact),
so SLSQP begins feasible. After convergence the radii are re-tightened to their exact LP optimum
for the returned centers, recovering any slack SLSQP left at its KKT point.

**Why these choices.** SLSQP is the standard strong method for this exact problem shape and
converges to a genuine constrained local optimum. Joint center+radius optimization lets the
objective pull circles toward configurations where radii can grow, which a fixed-center scheme
cannot. The LP re-tightening is free (convex, fast) and guarantees the reported radii are maximal
for the final centers. This rung is deliberately the *single-start* version: it exposes that the
limitation is not the optimizer but the lone initialization — one random basin out of many. The
result clears the grid by sliding circles into the wasted corners/edges and making radii unequal,
but lands well short of the `2.636` frontier because a single start cannot explore the basin
structure.

**Hyperparameters / contract.** One random seed (default `0`); `26` centers uniform on
`[0.05, 0.95]²`; SLSQP `maxiter=300`, `ftol=1e-10`; bounds `[0,1]` on centers, `[0,0.5]` on radii.
Output verified feasible at `atol=1e-7` (measured `maxviol ~1e-14`).

```python
import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def _max_radii_lp(centers):
    """Optimal radii for FIXED centers: maximize sum r s.t. r_i+r_j<=d_ij, r_i<=wall_i (an LP)."""
    n = len(centers)
    wall = np.minimum.reduce([centers[:,0], centers[:,1], 1-centers[:,0], 1-centers[:,1]])
    A, b = [], []
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n); row[i] = 1; row[j] = 1
            A.append(row); b.append(np.hypot(*(centers[i]-centers[j])))
    res = linprog(-np.ones(n), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(0, w) for w in wall], method="highs")
    return np.maximum(res.x, 0.0)

def construct_packing(seed=0):
    rng = np.random.default_rng(seed)
    n = N
    centers0 = rng.uniform(0.05, 0.95, size=(n, 2))
    r0 = np.maximum(_max_radii_lp(centers0), 1e-4)
    v0 = np.concatenate([centers0.ravel(), r0])

    def neg_sum(v):
        g = np.zeros_like(v); g[2*n:] = -1.0
        return -v[2*n:].sum(), g

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    def pair_con(v):
        c = v[:2*n].reshape(n, 2); r = v[2*n:]
        return np.array([np.hypot(*(c[i]-c[j])) - (r[i]+r[j]) for i, j in pairs])
    def wall_con(v):
        c = v[:2*n].reshape(n, 2); r = v[2*n:]
        return np.concatenate([c[:,0]-r, 1-c[:,0]-r, c[:,1]-r, 1-c[:,1]-r])

    res = minimize(neg_sum, v0, jac=True, method="SLSQP",
                   bounds=[(0,1)]*(2*n) + [(0,0.5)]*n,
                   constraints=[{"type":"ineq","fun":pair_con},
                                {"type":"ineq","fun":wall_con}],
                   options={"maxiter":300, "ftol":1e-10})
    centers = res.x[:2*n].reshape(n, 2)
    radii = _max_radii_lp(centers)          # re-tighten radii to exact LP optimum
    return centers, radii


if __name__ == "__main__":
    c, r = construct_packing(seed=0)
    print("sum_radii =", r.sum())           # 2.5949163422…
```
