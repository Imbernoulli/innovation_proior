**Problem.** Feasible packing of `26` circles in `[0,1]²` maximizing `Σ rᵢ` (nonconvex QCQP).
A single SLSQP run (`2.5949`) lands in one random basin; this rung wraps the same solver in many
random restarts and keeps the best feasible packing.

**Key idea.** Multi-start SLSQP. Draw many random center scatters; refine each with the same joint
center+radius SLSQP as the previous rung; re-tighten the radii to their LP optimum; verify
feasibility; keep the best `Σ rᵢ` across all starts. Each refined start is a distinct local
optimum, and the maximum over many draws is an order statistic that climbs with the number of
starts — converting compute into packing quality. A fixed master seed makes the whole run
reproducible.

**Why these choices.** Restarting is the standard cure for the single-start limitation of a
nonconvex problem: one initialization samples one basin, many sample many, and the best improves.
Centers are drawn in an inset square `[0.04, 0.96]²` so the LP radii-start is feasible and SLSQP
does not begin at a constraint corner. The per-start iteration budget is kept moderate
(`maxiter=250`) on purpose — better to afford *more* coarse refinements (more basins) than fewer
perfectly-converged ones, with the final LP re-tightening recovering the radii cleanly. This rung
buys its gain with breadth, not a new algorithm, and it will plateau below the frontier: uniform
random scatters are a blunt seed that rarely hits the rare top-quality basins, so the order
statistic saturates — the motivation for the structured-init + perturbation-chain endpoint.

**Hyperparameters / contract.** Master seed (default `12345`); `K=120` starts; each: `26` centers
uniform on `[0.04, 0.96]²`, SLSQP `maxiter=250`, `ftol=1e-10`, radii LP-re-tightened; keep best
feasible (`atol=1e-7`). Reproducible. Measured `maxviol ~3e-15`.

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
