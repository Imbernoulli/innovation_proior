Random multi-start saturated around a third of the record because it finds a decent configuration and then forgets it — it has no memory and no way to improve a promising point set. So the move was to add both: keep the current configuration and nudge it toward a larger minimum triangle. The subtlety is that the naive version of this fails in a way specific to the objective. Plain greedy hill-climbing — move a point, keep the move only if the minimum triangle grew — stalls almost immediately, because the score is the minimum over 165 triangles and that structure manufactures shallow traps. At any configuration one or a few "worst" triangles pin the score; enlarging one of them by moving a point almost always shrinks another, which becomes the new worst, so most single moves do not improve the minimum and a greedy climber gets stuck on a knife-edge.

The method is simulated annealing on point positions. I keep the single-point move but change the acceptance rule: propose moving one random point by a small Gaussian step (clipped back into the square so the configuration stays legal); if the minimum triangle improves, take it; if it worsens, take it anyway with Metropolis probability `exp(Δ/T)`. A warm temperature early accepts worsening moves so the search walks out of the knife-edge traps and crosses ridges between basins; geometric cooling to a tiny floor lets it settle into a deep basin late. Three choices make it work in practice. First, I anneal directly on the change in area rather than its logarithm, because areas are already on an `O(0.01)` scale, so the temperature lives on the same scale and the schedule is transparent. Second, I tie the step size to the temperature — large jumps when hot to reshape globally, fine jumps when cold to refine locally. Third, and what makes the budget affordable: moving one point changes only the `C(10,2) = 45` triples that involve it, so I keep the full vector of 165 areas and recompute only those 45 per step, splice them in, and take the minimum — each step is a handful of small array operations instead of 165. Finally, since a single run gambles on its start and its sequence of accepted moves, I run thirty independent restarts and keep the best minimum triangle across all of them.

Measured, this jumped from random multi-start's `0.010872` to `0.035639` — from 0.294 to 0.962 of the Goldberg record `1/27 = 0.037037` — confirming that *improving* configurations beats merely sampling them by a wide margin, and that accepting downhill moves escapes the traps the minimum-of-165 objective creates. The per-restart spread was wide (`0.024` to `0.036`), so multi-restart did real work. But the best run plateaued just short of `1/27`: near the optimum several triangles are simultaneously near-tight, and a random single-point Gaussian move tends to grow one while shrinking another, so annealing cannot coordinate the final squeeze. That plateau is the opening for the endpoint — a smooth, differentiable soft-minimum whose gradient pushes all the near-tight triangles up at once.

```python
import numpy as np
from itertools import combinations

N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))
I, J, K = TRIPLES[:, 0], TRIPLES[:, 1], TRIPLES[:, 2]
INVOLVES = [np.array([t for t, tr in enumerate(TRIPLES) if p in tr]) for p in range(N)]

def all_areas(P):
    a, b, c = P[I], P[J], P[K]
    cr = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (c[:,0]-a[:,0])*(b[:,1]-a[:,1])
    return 0.5*np.abs(cr)

def sa_run(rng, iters=200_000, T0=0.02, Tend=1e-5):
    P = rng.random((N, 2)); areas = all_areas(P); cur = areas.min()
    best, bestP = cur, P.copy(); cool = (Tend/T0)**(1.0/iters); T = T0
    for _ in range(iters):
        p = rng.integers(N); old = P[p].copy()
        P[p] = np.clip(old + rng.normal(0, 0.05+0.5*T, size=2), 0.0, 1.0)
        rows = INVOLVES[p]
        a, b, c = P[I[rows]], P[J[rows]], P[K[rows]]
        cr = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (c[:,0]-a[:,0])*(b[:,1]-a[:,1])
        ca = areas.copy(); ca[rows] = 0.5*np.abs(cr); cand = ca.min(); d = cand - cur
        if d >= 0 or rng.random() < np.exp(d/max(T, 1e-12)):
            areas, cur = ca, cand
            if cur > best: best, bestP = cur, P.copy()
        else:
            P[p] = old
        T *= cool
    return best, bestP

def construct():
    rng = np.random.default_rng(1)
    best, bestP = -1.0, None
    for _ in range(30):
        b, bp = sa_run(rng)
        if b > best: best, bestP = b, bp.copy()
    return bestP
```
