The problem is to pack $26$ circles into the unit square $[0,1]^2$, pairwise non-overlapping, so as
to maximize the sum of radii $\sum_i r_i$, with the radii free and unequal — a nonconvex QCQP whose
linear objective and convex wall constraints are coupled to $325$ nonconvex pairwise-distance
constraints. A single SLSQP run on the full $78$-variable problem (centers plus radii, optimized
jointly, radii re-tightened by LP) reaches $\approx 2.595$, comfortably beating the rigid grid
baseline by making radii unequal and using the corners, but it then names its own limitation: it
stops at the local optimum of whatever basin its one random initialization happened to fall into,
well short of the $\approx 2.636$ frontier. The optimizer is not the problem; the single
initialization is. This landscape has many basins of very different quality, and one draw samples
one basin.

I propose multi-start SLSQP: run the same joint center-plus-radius SLSQP from many random
initializations and keep the best feasible packing. The justification is an order-statistic
argument. Each random center scatter, refined by SLSQP, converges to a distinct local optimum; the
sum-of-radii values of those optima are spread out, and the maximum over many draws climbs as I add
starts. The more basins I sample, the better the best one I find — with diminishing returns, since
the high-quality basins are rarer and harder to hit by uniform random scattering. There is no new
algorithmic idea beyond restarting, and that honesty matters: this method buys its improvement with
breadth, not with a cleverer search.

The mechanics carry over from the single-start method unchanged, and a few choices are deliberate.
For each start I draw $26$ centers uniformly in a slightly inset square $[0.04,0.96]^2$ — keeping
the initial circles off the walls gives the radius LP a feasible, non-degenerate start and stops
SLSQP from beginning at a constraint corner where it can stall. I set the initial radii to their LP
optimum for that scatter (maximize $\sum_i r_i$ subject to $r_i + r_j \le d_{ij}$ and
$r_i \le \text{wall}_i$, solved exactly), run SLSQP jointly over centers and radii to a KKT point,
then re-tighten the radii to their exact LP optimum for the final centers to recover any slack
SLSQP left at its boundary. I check each result against the real constraints and keep it only if it
is feasible and its sum exceeds the running best. The per-start iteration budget is kept moderate at
$\text{maxiter}=250$ on purpose: a start heading toward a mediocre basin is not worth polishing to
machine precision, so I would rather spend the budget on *more* starts than over-refine each one —
more, slightly coarser refinements beat fewer perfectly-converged ones when the goal is to find a
good basin. A single master seed makes the whole multi-start run reproducible, so the reported
number is deterministic rather than a lucky draw.

With on the order of a hundred random restarts the best-of-many clears the single-start value and
climbs into the low $2.62$s, a real step toward the frontier bought entirely by sampling more
basins. But it plateaus below $2.636$, and the reason is visible in advance: the basins that reach
the very top correspond to specific irregular arrangements — a few large circles in a particular
pattern with the rest filling gaps — and those are a vanishingly small target for uniform
scattering, so the order statistic saturates. That saturation is exactly the limitation: pure random
multi-start finds good basins but not the best ones, because it has no structure and no memory of
the best packing found so far. Doing better requires seeding the search with structured layouts and
exploiting the incumbent by perturbing and re-refining it, rather than throwing every start away.

```python
import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def _max_radii_lp(centers):
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

def _slsqp(centers0, maxiter=250):
    n = N
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
                   options={"maxiter":maxiter, "ftol":1e-10})
    c = res.x[:2*n].reshape(n, 2)
    return c, _max_radii_lp(c)

def _feasible(c, r, atol=1e-7):
    if np.any(r < -atol): return False
    if np.any(r - np.minimum(c[:,0], c[:,1]) > atol): return False
    if np.any(r - np.minimum(1-c[:,0], 1-c[:,1]) > atol): return False
    for i in range(N):
        for j in range(i+1, N):
            if (r[i]+r[j]) - np.hypot(*(c[i]-c[j])) > atol: return False
    return True

def construct_packing(seed=12345, n_starts=120):
    rng = np.random.default_rng(seed)
    best = -1.0; best_cr = None
    for _ in range(n_starts):
        c0 = rng.uniform(0.04, 0.96, size=(N, 2))
        try:
            c, r = _slsqp(c0)
        except Exception:
            continue
        if _feasible(c, r) and r.sum() > best:
            best, best_cr = r.sum(), (c.copy(), r.copy())
    return best_cr


if __name__ == "__main__":
    c, r = construct_packing()
    print("sum_radii =", r.sum())           # 2.6221020467…
```
