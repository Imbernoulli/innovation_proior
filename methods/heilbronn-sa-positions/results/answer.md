**Problem.** Place `n = 11` points in `[0,1]^2` to maximize the minimum area over all `165`
triangles. Score = exact min triangle area. Record `1/27 = 0.037037` (Goldberg). This rung is
simulated annealing on point positions, one point moved at a time, with multi-restart.

**Key idea.** Add the two things random multi-start lacks — memory and improvement. Keep the current
configuration; propose moving one random point by a small Gaussian step (clipped into the square);
accept on the Metropolis rule applied directly to the change in minimum triangle area (the score is
already on an `O(0.01)` scale, so no logarithm is needed). A high starting temperature accepts
worsening moves so the search walks out of the shallow traps that the minimum-of-`165` objective
manufactures; geometric cooling to a tiny floor lets it settle into a deep basin. Moving one point
changes only the `C(10,2) = 45` triples that involve it, so keep the full vector of `165` areas and
recompute only those `45` per step — making each step a handful of small array ops. Run many
independent random restarts and keep the best minimum triangle across all of them.

**Why these choices.** Greedy hill-climbing stalls immediately here: enlarging the worst triangle by
moving a point usually shrinks another, so the minimum sits on a knife-edge and most single moves do
not improve it. Annealing's acceptance of downhill moves is exactly what crosses the ridge out of
such traps. Annealing directly on the area (not its log) is right because areas are already `O(0.01)`
and the temperature lives on the same scale, making the schedule transparent. The step size is tied
to temperature (large jumps hot, fine jumps cold) so the search reshapes globally early and refines
locally late. Incremental `45`-of-`165` scoring is what makes the budget — hundreds of thousands of
steps times dozens of restarts — affordable. Multi-restart turns a noisy single run into a reliable
best-of-many.

**Hyperparameters / contract.** `30` restarts; per run `200,000` steps, temperature `0.02 →
1e-5` geometric; Gaussian step `σ = 0.05 + 0.5·T` clipped to `[0,1]^2`; seed `1`. Output is the best
configuration over all restarts (exact min-area verified).

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
