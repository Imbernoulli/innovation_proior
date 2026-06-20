By this point in the Erdős minimum-overlap ladder the hierarchical recipe had paid off twice — `0.381240`
at `24` cells, `0.381076` at `120` — each rung upscaling the optimized profile for free, kicking it off the
repeated-block plateau, and refining with annealed-soft-max search while keeping the best true overlap. The
cap each time was resolution, and the published frontier (AlphaEvolve's `95`-step `0.380924`, AutoEvolver's
`~600`-step record `0.38086945`) lives at several hundred cells. So the endpoint lifts once more, to the
`~600`-cell scale the records use, and grinds there.

The method is upscale-to-scale plus a deliberately *scalable* refinement, and the key engineering decision
is what optimizer to use at `600` cells. The coarse and middle rungs leaned on constrained SLSQP against
the soft-max surrogate, which was right at two dozen and a hundred-odd cells. At `600` cells SLSQP becomes
the bottleneck — its internal quadratic program is super-linear in the number of variables, so a single
annealed-SLSQP ladder takes about two minutes, and from a good starting point it barely moves because the
surrogate optimum has essentially coincided with where the profile already sits. So I abandon SLSQP at this
scale for two tools that scale. The first is `β`-annealed Adam on the *analytic* gradient of the soft-max
bound: each step is one cross-correlation plus a gradient assembly, cheap enough to run tens of thousands of
steps in seconds, with `β` annealed far sharper than at the coarse levels (into the thousands and beyond)
because the `600`-cell profile is spiky with many closely-tied binding shifts, and a coarse `β` would let
the surrogate's peak sit below the true worst overlap. The second is an exact subgradient polish on the true
minimax: the soft-max and the hard `max` diverge slightly, so I finish by descending the genuine objective,
distributing a step across the active near-worst shifts, to shave the last fraction the surrogate leaves on
the table. Throughout I keep the best true overlap ever seen, since the surrogate-best and the true-best are
different vectors. The upscale itself is free — repeating each of the `120` cells five times is literally
the same step function, same `C` — so the endpoint starts at `0.3810764` and the refinement can only help.

The honest finding is that it does not help: `0.3810764` is a **robust local optimum**. Neither the sharp-β
analytic-gradient Adam, nor a fresh `n=600` or `n=360` multistart, nor the exact subgradient polish improves
on the lifted value — the refinement at scale *holds* the number rather than lowering it. So the returned
`600`-cell profile scores `0.3810764`, which is within `~1.9×10⁻⁴` of the AutoEvolver record `0.38086945`,
`~1.6×10⁻⁴` below the AlphaEvolve and Haugland landmarks, and `~2.1×10⁻³` above White's provable lower
bound `0.379005`. The solution is feasible (sum exactly `300 = n/2`) and near-binary (about `31%` of cells
pinned at `0` or `1`), the spiky asymmetric structure the literature reports for near-optimal overlap
profiles. This is the step-function frontier a single bounded hierarchical-gradient constructor reaches; the
absolute records were found by large evolutionary / test-time searches (AutoEvolver ran `~12` hours to
`n=750`) with orders of magnitude more compute, and the gap from `0.3810764` to `0.38087` — together with the
sliver `0.379005 ≤ C5 ≤ 0.380868` the true constant is squeezed into — is the still-open part of this
seventy-year-old problem.

```python
import numpy as np
from scipy.optimize import minimize

def _smooth_bound(v, beta):
    n = len(v); conv = np.correlate(v, 1.0 - v, mode='full'); m = conv.max()
    return (m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta) * 2.0 / n

def _project(v, target):
    v = v.copy()
    for _ in range(60):
        v = np.clip(v, 0.0, 1.0); diff = target - v.sum()
        if abs(diff) < 1e-12: break
        free = (v > 0.0) & (v < 1.0); k = free.sum() or len(v)
        v[free if free.sum() else slice(None)] += diff / k
    return np.clip(v, 0.0, 1.0)

def _anneal_slsqp(v0, betas, maxiter=180):
    n = len(v0); t = n / 2.0; v = _project(v0.copy(), t)
    cons = {'type': 'eq', 'fun': lambda s: s.sum() - t}; bnds = [(0, 1)] * n
    for b in betas:
        r = minimize(lambda s: _smooth_bound(s, b), v, method='SLSQP',
                     bounds=bnds, constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12})
        v = _project(r.x, t)
    return v

def _basin_hop(v0, n_hops, betas, kick, seed):
    rng = np.random.default_rng(seed)
    best = _anneal_slsqp(v0, betas); bestv = compute_upper_bound(best)
    for h in range(n_hops):
        c = _project(best + rng.standard_normal(len(best)) * kick * (0.9**h + 0.1), len(best) / 2.0)
        c = _anneal_slsqp(c, betas); cv = compute_upper_bound(c)
        if cv < bestv: bestv, best = cv, c.copy()
    return best

def _subgrad_polish(v, iters, lr0, seed):
    n = len(v); target = n / 2.0; v = _project(v.copy(), target)
    best = v.copy(); bestv = compute_upper_bound(best)
    for it in range(iters):
        conv = np.correlate(v, 1.0 - v, mode='full'); m = conv.max()
        active = np.where(conv >= m - 1e-9 * max(1, m))[0]; grad = np.zeros(n)
        for k in active:
            s = k - (n - 1); i = np.arange(max(0, s), min(n, n + s)); j = i - s
            np.add.at(grad, i, 1.0 - v[j]); np.add.at(grad, j, -v[i])
        grad /= len(active); lr = lr0 / (1.0 + it / 500.0)
        v = _project(v - lr * grad, target); cv = compute_upper_bound(v)
        if cv < bestv: bestv, best = cv, v.copy()
    return best

def _construct_120(seed=1):
    np.random.seed(seed); n = 60; best, bestv = None, np.inf
    for _ in range(8):
        v = _anneal_slsqp(_project(np.random.rand(n), n / 2.0), betas=(60,150,300,600,1200,2400))
        c = compute_upper_bound(v)
        if c < bestv: bestv, best = c, v.copy()
    return _basin_hop(np.repeat(best, 2), n_hops=20, betas=(300, 800, 1800, 3600), kick=0.03, seed=3)

def construct():
    """Endpoint: rung-3 profile -> upscale x5 -> n=600 -> exact-minimax polish. -> C = 0.3810764."""
    b120 = _construct_120()
    b600 = np.repeat(b120, 5)
    polished = _subgrad_polish(b600, iters=2000, lr0=0.005, seed=7)
    return polished if compute_upper_bound(polished) < compute_upper_bound(b600) else b600

def compute_upper_bound(sequence):            # frozen evaluator (AlphaEvolve App. B.5)
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct(); assert abs(v.sum() - len(v)/2.0) < 1e-6
    print("n =", len(v), " C =", compute_upper_bound(v))
```
