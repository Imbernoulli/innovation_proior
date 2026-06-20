**Problem.** Place `n = 11` points in `[0,1]^2` to maximize the minimum area over all `165`
triangles. Score = exact min triangle area. Record `1/27 = 0.037037` (Goldberg 1972, conjectured
optimal). ENDPOINT: heavy multi-restart simulated annealing to find the right basin, then a soft-min
gradient polish to snap onto the exact optimum, with basin-hopping for robustness.

**Key idea.** Annealing plateaus just short of the record because near the optimum several triangles
are simultaneously near-tight and a random single-point move cannot grow them all at once. Replace
the non-differentiable `min` with a smooth **soft-minimum** surrogate `-(1/β)·log Σ exp(-β·area_t)`,
whose gradient is the softmax-weighted sum over *all* near-tight triangles — so maximizing it inflates
every tight triangle together. Each area is bilinear in the point coordinates (cross product), giving
a clean analytic gradient assembled by scattering each triangle's contribution onto its three points.
Feed the negative soft-min and its gradient to **L-BFGS-B** (box constraints `[0,1]`), and **anneal
`β` upward** through a ladder (`200 → 120000`): start smooth and easy, finish sharp enough that the
surrogate tracks the true minimum. Seed the polish from heavy annealing — and especially from the
*previous rung's best annealed configuration*, which already sits at the edge of the record's basin.
After polishing, always recompute the exact minimum over all `165` triangles. Finish with
basin-hopping (perturb-and-re-polish) to land on the best of the cluster of near-equivalent optima.

**Why these choices.** The soft-min is the differentiable stand-in for the hard objective; its
softmax-weighted gradient is exactly the coordinated push annealing could not make. `β`-annealing is
forced: large `β` from the start gives a near-degenerate gradient and stalls, small `β` optimizes a
soft proxy that lies below the true minimum, so the ladder warm-starts each sharper stage from the
previous. L-BFGS-B fits a smooth analytic-gradient box-constrained objective. Seeding from the
annealing best is the load-bearing choice: the polish is *local*, so it only reaches `1/27` if it
starts in that basin, and the rung-3 best is the configuration most likely there. This is the
single-machine analogue of AlphaEvolve's search-plus-refine recipe on the sibling Heilbronn
containers. The endpoint *matches* the record (the conjectured optimum) to within floating-point
precision; beating tabulated records requires large-scale evolutionary search on cases where the
tabulated value is not already optimal, which `n = 11` in the square is believed not to be.

**Hyperparameters / contract.** `48` SA restarts (`300,000` steps each) feeding the polish; `β`
ladder `[200, 500, 1000, 3000, 8000, 20000, 50000, 120000]`, L-BFGS-B `maxiter 4000` per stage;
seed the polish also from `rung3_best.npy` if present; `80` rounds of basin-hopping (`σ = 0.03`
kick). Seed `2`. Output is the best configuration found (exact min-area verified, all coordinates in
`[0,1]`).

```python
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))
I, J, K = TRIPLES[:, 0], TRIPLES[:, 1], TRIPLES[:, 2]
INVOLVES = [np.array([t for t, tr in enumerate(TRIPLES) if p in tr]) for p in range(N)]

def all_areas(P):
    a, b, c = P[I], P[J], P[K]
    cr = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (c[:,0]-a[:,0])*(b[:,1]-a[:,1])
    return 0.5*np.abs(cr), cr

def min_area(P):
    return all_areas(P)[0].min()

def sa_run(rng, iters=300_000, T0=0.02, Tend=1e-5):
    P = rng.random((N, 2)); areas, _ = all_areas(P); cur = areas.min()
    best, bestP = cur, P.copy(); cool = (Tend/T0)**(1.0/iters); T = T0
    for _ in range(iters):
        p = rng.integers(N); old = P[p].copy()
        P[p] = np.clip(old + rng.normal(0, 0.05+0.5*T, size=2), 0.0, 1.0)
        rows = INVOLVES[p]; a, b, c = P[I[rows]], P[J[rows]], P[K[rows]]
        cr = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (c[:,0]-a[:,0])*(b[:,1]-a[:,1])
        ca = areas.copy(); ca[rows] = 0.5*np.abs(cr); cand = ca.min(); d = cand - cur
        if d >= 0 or rng.random() < np.exp(d/max(T, 1e-12)):
            areas, cur = ca, cand
            if cur > best: best, bestP = cur, P.copy()
        else:
            P[p] = old
        T *= cool
    return bestP

def neg_softmin_and_grad(x, beta):
    P = x.reshape(N, 2); areas, cross = all_areas(P)
    s = np.sign(cross); s[s == 0] = 1.0
    amin = areas.min(); w = np.exp(-beta*(areas - amin)); Z = w.sum()
    m = amin - np.log(Z)/beta; coef = w/Z
    ax, ay = P[I,0], P[I,1]; bx, by = P[J,0], P[J,1]; cx, cy = P[K,0], P[K,1]
    f = 0.5*s*coef; g = np.zeros((N, 2))
    np.add.at(g, I, np.column_stack([f*(by-cy), f*(cx-bx)]))
    np.add.at(g, J, np.column_stack([f*(cy-ay), f*(ax-cx)]))
    np.add.at(g, K, np.column_stack([f*(ay-by), f*(bx-ax)]))
    return -m, -g.ravel()

def polish(P0):
    x = P0.ravel().copy(); bounds = [(0.0, 1.0)]*(2*N)
    for beta in [200, 500, 1000, 3000, 8000, 20000, 50000, 120000]:
        x = minimize(neg_softmin_and_grad, x, args=(beta,), jac=True, method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": 4000, "ftol": 1e-15, "gtol": 1e-13}).x
    return x.reshape(N, 2)

def basin_hop(P0, rng, rounds=80):
    best = min_area(np.clip(P0, 0, 1)); bestP = np.clip(P0, 0, 1).copy()
    for _ in range(rounds):
        cand = np.clip(polish(np.clip(bestP + rng.normal(0, 0.03, (N, 2)), 0, 1)), 0, 1)
        a = min_area(cand)
        if a > best: best, bestP = a, cand.copy()
    return bestP

def construct(seed_config=None):
    rng = np.random.default_rng(2); best, bestP = -1.0, None
    for _ in range(48):
        P = np.clip(polish(sa_run(rng)), 0.0, 1.0); a = min_area(P)
        if a > best: best, bestP = a, P.copy()
    if seed_config is not None:                       # e.g. the rung-3 annealing best
        P = np.clip(polish(seed_config), 0.0, 1.0); a = min_area(P)
        if a > best: best, bestP = a, P.copy()
    return basin_hop(bestP, rng)
```
