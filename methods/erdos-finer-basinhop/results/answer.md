**Problem.** Same evaluator: minimize `C(v) = max_k (Σ_i v_i (1 − v_{i−k}))·2/n` over feasible heights
(`v_i ∈ [0,1]`, `Σ v = n/2`); lower is a tighter bound on the Erdős constant `C5`. Here: lift the
optimized coarse profile to a middle resolution and refine with basin-hopping.

**Key idea.** Upscale the best coarse vector (repeat each cell, a free no-op that preserves `C` while
adding degrees of freedom), kick it slightly to break the repeated-block plateau, then run
**basin-hopping** around the annealed-SLSQP ladder: solve to a local optimum, perturb the best-so-far,
re-solve, accept only improvements, shrinking the perturbation over the hops. Anneal `β` *sharper* than at
the coarse level (into the thousands) so the soft-max surrogate hugs the true hard-max overlap at this
spikier resolution.

**Why these choices.** Upscaling transfers the coarse profile's gross structure for free; the kick gives
the optimizer traction off the degenerate plateau. Basin-hopping is the right tool for the non-convex
minimax — local SLSQP descent is cheap but the basins are many, and perturb-and-re-solve jumps between
them while keeping the good structure; this is the "perturbation search + basin-hopping" recipe the
agentic-search record (AutoEvolver) uses on this problem. The sharper `β` ladder is necessary because the
finer profile is spikier, so a coarse-level `β` lets the surrogate's max sit below the true max. The middle
resolution (`~120` cells) is chosen to confirm lifting helps and to tune the hop schedule before the
endpoint spends its budget on resolution.

**Hyperparameters / contract.** Lift path `24 → 60` (multistart, `8` starts, `β`-ladder up to `2400`) then
upscale `×2 → 120`; basin-hopping `20` hops with `β`-ladder `(300, 800, 1800, 3600)`, `180` SLSQP
iterations/level, initial kick `0.03` shrinking as `0.9^hop`; seeds `1/3`. Output is the best feasible
`~120`-cell vector under the frozen hard-max evaluator. Reproducible under the fixed seeds.

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
        cand = _project(best + rng.standard_normal(len(best)) * kick * (0.9**h + 0.1), len(best) / 2.0)
        cand = _anneal_slsqp(cand, betas); c = compute_upper_bound(cand)
        if c < bestv: bestv, best = c, cand.copy()
    return best

def construct(seed=1):
    rng = np.random.default_rng(seed)
    n = 60; best, bestv = None, np.inf
    for _ in range(8):                                       # multistart at n=60
        v = _anneal_slsqp(_project(rng.random(n), n / 2.0), betas=(60,150,300,600,1200,2400))
        c = compute_upper_bound(v)
        if c < bestv: bestv, best = c, v.copy()
    v120 = np.repeat(best, 2)                                # upscale x2 -> n=120 (free)
    return _basin_hop(v120, n_hops=20, betas=(300, 800, 1800, 3600), kick=0.03, seed=3)

def compute_upper_bound(sequence):                          # frozen evaluator (AlphaEvolve App. B.5)
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct(); assert abs(v.sum() - len(v)/2.0) < 1e-6
    print("n =", len(v), " C =", compute_upper_bound(v))
```
