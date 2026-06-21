The grid baseline did what a baseline should: it put $2.5414$ on the board and then, by its own rigidity, named the flaw. Every circle is locked to radius $0.1$ on a lattice, the corners and edges are wasted, and the sum-of-radii objective plainly wants something a grid cannot give — unequal radii and freely placed centers. So the move now is to stop constructing the packing by hand and start *optimizing* it: write the feasibility conditions as nonlinear inequalities and hand the whole thing to a constrained optimizer, letting it find an irregular arrangement no grid can express.

The structure of the problem dictates the method, so let me set it down precisely. The variables are the $26$ centers $(x_i, y_i)$ together with the $26$ radii $r_i$ — $78$ variables. The objective $\Sigma r_i$ is linear. The constraints are the wall constraints $x_i \ge r_i$, $x_i \le 1-r_i$, $y_i \ge r_i$, $y_i \le 1-r_i$, which are linear, and the pairwise non-overlap constraints

$$\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} \;\ge\; r_i + r_j \qquad \text{for all } \binom{26}{2}=325 \text{ pairs},$$

which are *nonconvex*: the feasible region is the complement of a union of "too close" regions, which is not convex. So this is a nonconvex QCQP with no global convexity to exploit, and there is no hope of a global guarantee — the realistic target is a strong local optimum.

I propose to solve it with a *single joint (centers + radii) SLSQP run from one random initialization, with the radii LP-re-tightened at the end*. SLSQP — Sequential Least-Squares Quadratic Programming — is the standard strong tool for exactly this shape: a smooth nonlinear objective with smooth nonlinear inequality constraints. At each iterate it linearizes the $325$ distance inequalities and the wall inequalities, builds a quadratic model of the Lagrangian, solves that QP for a step, and iterates, converging to a KKT point of the constrained problem. It handles the distance and wall constraints directly, which is precisely what I need.

The one design decision I want to get right is how to treat the radii, because it determines both the initialization and how I read off the answer. There is a structural fact worth exploiting: *for fixed centers, the optimal radii are the solution of a linear program* — maximize $\Sigma r_i$ subject to $r_i + r_j \le d_{ij}$ and $r_i \le \text{wall}_i$, all linear in the radii, hence convex and solvable exactly and cheaply. One could therefore optimize only the centers and snap the radii to their LP optimum at every configuration. But for this rung I deliberately let SLSQP optimize centers and radii *jointly*, all $78$ together. The reason the joint form is right is that the two couple: the value of moving a center depends on how much radius that move unlocks across its neighbors, and only a solver that sees centers and radii in the same quadratic model can trade them off in a single coordinated smooth descent. A scheme that freezes one to optimize the other misses those coordinated moves.

I then add the LP only as a final polish. SLSQP stops at a KKT point where the radii may sit a hair inside the feasible boundary; after it converges I *re-tighten* the radii to their exact LP optimum for the returned centers. That step is free — convex, fast — and it recovers whatever slack the local solver left and guarantees the reported radii are maximal for those centers.

Initialization needs to be feasible and reasonable. I draw $26$ centers uniformly in a slightly inset square $[0.05, 0.95]^2$, which keeps the initial circles off the walls and gives the LP a non-degenerate feasible start, then set the initial radii to their LP optimum for that scatter. That start already respects every constraint, so SLSQP begins inside the feasible region and climbs from there.

I expect this single run to clear the grid comfortably — it can make the radii unequal and slide circles into the corners and edges the grid wasted, and a linear objective with a good local solver reliably tightens a random feasible packing into a locally maximal one — but to fall short of the $2.636$ frontier, because one initialization lands in whatever basin its random draw happened to fall into, and this nonconvex landscape has many basins of very different quality. The limitation this rung exposes is therefore not the optimizer but the *single start*: SLSQP is the right local engine, and the next rung has to wrap it in many restarts to find a better basin than one random draw can offer.

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
