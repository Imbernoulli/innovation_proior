The task is to pack $26$ circles into the unit square $[0,1]^2$, pairwise non-overlapping, so as to
maximize the sum of radii $\sum_i r_i$, with the radii free and unequal. A structured $5\times 5$
grid of equal circles plus an interstitial filler is a clean, feasible, parameter-free baseline,
but it reaches only $\approx 2.5414$, and its own rigidity tells me what is wrong: every circle is
locked to radius $0.1$ on a lattice, the corners and edges are wasted, and the sum-of-radii
objective clearly wants something the grid cannot give — unequal radii and freely-placed centers.
So the move is to stop constructing the packing by hand and start optimizing it: write the
feasibility constraints as nonlinear inequalities and hand the whole thing to a constrained
optimizer.

The structure of the optimization problem dictates the method. The variables are the $26$ centers
$(x_i,y_i)$ and the $26$ radii $r_i$ — $78$ in all. The objective $\sum_i r_i$ is linear. The
constraints are the wall constraints $x_i \ge r_i$, $x_i \le 1-r_i$, $y_i \ge r_i$, $y_i \le 1-r_i$
(linear), and the $\binom{26}{2}=325$ pairwise non-overlap constraints
$\sqrt{(x_i-x_j)^2+(y_i-y_j)^2} \ge r_i + r_j$, which are nonconvex. So this is a nonconvex QCQP
with no global convexity to exploit; the realistic goal is a strong local optimum. I propose to
solve it with a single run of SLSQP — Sequential Least-Squares Quadratic Programming — from one
random initialization. SLSQP linearizes the constraints, builds a quadratic model of the Lagrangian
at each iterate, solves that QP for a step, and iterates to a KKT point; it handles the $325$
distance inequalities and the wall inequalities directly and is the standard strong engine for
exactly this smooth-objective, smooth-constraint shape.

Two design choices make the method work. First, centers and radii are optimized *jointly*, all $78$
variables together, so the optimizer can trade a circle's position against radius growth in a single
smooth descent — a fixed-center scheme cannot pull circles toward configurations where the radii can
grow. Second, I exploit a structural fact about the radii: for fixed centers, the optimal radii are
the solution of a linear program, maximize $\sum_i r_i$ subject to $r_i + r_j \le d_{ij}$ and
$r_i \le \text{wall}_i$, all linear and solved exactly. I use this twice. At initialization I draw
$26$ centers uniformly in a slightly inset square $[0.05,0.95]^2$ (to keep them off the walls) and
set the initial radii to their LP-optimal values for that scatter, so SLSQP begins strictly feasible
and climbs from inside the region rather than from a constraint corner where it could stall. After
SLSQP converges, I re-tighten the radii to their exact LP optimum for the returned centers: SLSQP
stops at a KKT point where the radii may sit slightly inside the feasible boundary, and the LP
recovers that last bit of slack for free and guarantees the radii are maximal for those centers.

This is deliberately the single-start version, and its limitation is the lone initialization, not
the optimizer. A single SLSQP run finds *a* local optimum — it beats the grid comfortably by making
the radii unequal and sliding circles into the wasted corners and edges, reaching $\approx 2.595$ —
but it lands in whatever basin its one random scatter happened to fall into, and this nonconvex
landscape has many basins of quite different quality. One initialization samples one basin, so the
run falls short of the $\approx 2.636$ frontier. The SLSQP engine is correct; the fix is to wrap it
in many restarts, which is the natural next step.

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
