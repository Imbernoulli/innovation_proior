The coarse rung dropped off the flat floor in a single step, landing at $0.381240$ with two dozen cells — already within $\sim 3\times 10^{-4}$ of the Haugland/AlphaEvolve landmarks. The cap was resolution: two dozen wide steps cannot resolve the fine structure of a near-optimal profile, so the worst overlap cannot be shaved any thinner. The natural next move is to lift that optimized profile to many more pieces and refine it there — but carefully, so the gross structure the coarse search already found is not thrown away.

The method is *upscale-and-basin-hop*. The clean way to add resolution for free is to upscale: replace each coarse cell by several identical finer cells of the same height. This is the same step function expressed on a finer grid — same $h$, same overlap envelope, same $C$ — so the upscaled point *starts* at the coarse value and now carries many more degrees of freedom. But the upscaled point is a degenerate plateau of repeated blocks: perturbing two cells of a former block in opposite directions often leaves the overlap unchanged to first order, so a gradient or SLSQP step from the exact plateau can stall. The remedy is a small kick — perturb the upscaled vector slightly and re-project to feasibility — which barely moves $C$ but breaks the block symmetry and gives the optimizer traction.

What makes the refinement actually descend at the larger $n$ is wrapping the annealed-SLSQP ladder of the previous rung inside a **basin-hopping** loop. The extra degrees of freedom that let me carve better structure also create *more* local minima, so one SLSQP ladder from one start is no longer enough. Basin-hopping solves the ladder to a local optimum, then perturbs the best-so-far vector by Gaussian noise of scale $\text{kick}\cdot(0.9^h + 0.1)$ at hop $h$, re-projects, re-solves, and accepts only improvements in the *true* overlap, for a budget of hops. Each hop is a constrained restart *near* the current best — far enough to jump basins, near enough to keep the good gross structure — and the shrinking schedule means early hops explore while late hops refine. This is precisely the "perturbation search + basin-hopping" recipe the agentic-search record (AutoEvolver) uses on this problem, and it is the right tool for a non-convex minimax where local descent is cheap but the basins are many.

The second necessary change is a *sharper* $\beta$ ladder than at the coarse level. With more cells the optimal profile is spikier and the cross-correlation has a finer set of near-binding shifts; a $\beta$ that was sharp enough at $24$ cells is too soft at $120$, letting the surrogate's peak sit below the true max so the optimizer chases a slightly wrong objective. So the basin-hopping ladder anneals $\beta$ up to $3600$ in the late hops, so the soft-max genuinely tracks the hard overlap I report. Concretely the lift path is $24\to 60$ through a fresh multistart (8 starts, $\beta$-ladder to $2400$), then a free upscale $\times 2$ to $120$, then $20$ basin-hops with $\beta$-ladder $(300,800,1800,3600)$ and initial kick $0.03$. The middle resolution is chosen deliberately — far enough above the coarse rung to confirm lifting helps, and the place to *tune* the hop schedule (perturbation scale, hop count, $\beta$ ceiling) so the endpoint rung can spend its budget on resolution rather than re-tuning. I expect only a modest gain, because the coarse profile already captured the gross structure; the win from $24\to 120$ cells is fine shaving, not a jump.

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
    np.random.seed(seed)
    n = 60; best, bestv = None, np.inf
    for _ in range(8):                                       # multistart at n=60
        v = _anneal_slsqp(_project(np.random.rand(n), n / 2.0), betas=(60,150,300,600,1200,2400))
        c = compute_upper_bound(v)
        if c < bestv: bestv, best = c, v.copy()
    v120 = np.repeat(best, 2)                                # upscale x2 -> n=120 (free)
    return _basin_hop(v120, n_hops=20, betas=(300, 800, 1800, 3600), kick=0.03, seed=3)
    # -> C = 0.3810764 at n=120

def compute_upper_bound(sequence):                          # frozen evaluator (AlphaEvolve App. B.5)
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct(); assert abs(v.sum() - len(v)/2.0) < 1e-6
    print("n =", len(v), " C =", compute_upper_bound(v))
```
